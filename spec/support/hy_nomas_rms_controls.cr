require "json"
require "../../src/ml/video/hy_nomas_block"
require "../../src/ml/video/wan_t5"
require "../../src/ml/video/metal_mps_command_chain"

module HyAttentionProbe
  # Separate-product RoPE control for the pinned tuple-frequency equation.
  # The pragma is deliberately scoped to this diagnostic kernel: it prevents
  # multiply-add contraction so the two products and final sum round as
  # separate F32 operations.
  RMS_ROPE_CONTROL_SOURCE = <<-METAL
  #pragma clang fp contract(off)
  kernel void hy_probe_rope_uncontracted(
      device const float *x [[buffer(0)]], device const float *cosine [[buffer(1)]],
      device const float *sine [[buffer(2)]], device float *y [[buffer(3)]],
      constant uint &tokens [[buffer(4)]], constant uint &heads [[buffer(5)]],
      constant uint &head_dim [[buffer(6)]], constant uint &batches [[buffer(7)]],
      uint id [[thread_position_in_grid]]) {
      const uint dim = heads * head_dim;
      const uint count = batches * tokens * dim;
      if (id >= count) return;
      const uint token = (id / dim) % tokens;
      const uint channel = id % head_dim;
      const uint pair = (channel & 1u) == 0u ? id + 1u : id - 1u;
      const float rotated = (channel & 1u) == 0u ? -x[pair] : x[pair];
      const uint table_index = token * head_dim + channel;
      const float product_x_cos = x[id] * cosine[table_index];
      const float product_rotated_sin = rotated * sine[table_index];
      y[id] = product_x_cos + product_rotated_sin;
  }
  METAL

  # Same-input reproduction of the pinned T5 MPSGraph RMS reduction for the
  # four frozen Hy-Nomas Q/K norm operations. The logical [B,T,H,D] input is
  # reshaped to [B,T*H,D]; this preserves each RMS row's values but deliberately
  # tests whether that flattening changes MPSGraph reduction arithmetic.
  # Keep this owner and its Arena buffers alive through the terminal wait.
  class MPSRMSControls
    private struct Entry
      getter index : Int32
      getter name : String
      getter prefix : String
      getter config : ML::Video::WanT5Config
      getter input : ML::MetalBuffer
      getter output : ML::MetalBuffer
      getter weight : ML::MetalBuffer
      getter square : ML::MetalBuffer
      getter mean : ML::MetalBuffer
      getter weight_dtype : ML::Video::WeightDType
      getter plan : ML::Video::WanT5RMSMeanPlan

      def initialize(@index, @name, @prefix, @config, @input, @output,
                     @weight, @square, @mean, @weight_dtype, @plan)
      end
    end

    @hy_config : ML::Video::HyNomasConfig
    @arena : ML::Video::Arena
    @weights : ML::Video::Arena
    @entries = {} of Int32 => Entry
    @released = false

    def initialize(@hy_config : ML::Video::HyNomasConfig, ops : Array(JSON::Any),
                   @arena : ML::Video::Arena, @weights : ML::Video::Arena)
      @hy_config.validate!
      raise ArgumentError.new("MPS RMS controls require operations") if ops.empty?
      ML::Video::WanT5.pipeline("wan_t5_rms_square")
      ML::Video::WanT5.pipeline("wan_t5_rms_from_mean")

      seen = Set(String).new
      begin
        ops.each_with_index do |op, raw_index|
          next unless op["kind"].as_s == "rms"
          name = op["name"].as_s
          raise ArgumentError.new("Duplicate MPS RMS operation") unless seen.add?(name)
          entry = prepare_entry!(op, raw_index.to_i32)
          @entries[entry.index] = entry
        end
        expected = {
          "img.attn.q.rms", "img.attn.k.rms",
          "txt.attn.q.rms", "txt.attn.k.rms",
        }.to_set
        unless seen == expected && @entries.size == expected.size
          raise ArgumentError.new("MPS RMS controls require exactly the four admitted Q/K norm operations")
        end
      rescue ex
        release
        raise ex
      end
    end

    def plan_count : Int32
      @entries.size.to_i32
    end

    def has_plan?(index : Int32) : Bool
      !@released && @entries.has_key?(index)
    end

    # Append square, MPSGraph mean, and affine RMS stages to the serial chain.
    # WanT5.rms_norm_mps_mean pauses/resumes around its prepared graph and
    # refreshes the command encoder internally; this method allocates no Metal
    # buffers, waits, or reads data back.
    def encode(chain : ML::Metal::MPSCommandChain, op : JSON::Any,
               index : Int32) : Nil
      raise "MPS RMS controls have been released" if @released
      entry = @entries[index]?
      raise ArgumentError.new("No MPS RMS plan for operation index #{index}") unless entry
      validate_operation_identity!(op, entry)
      validate_bindings!(entry)
      ML::Video::WanT5.rms_norm_mps_mean(
        chain, entry.plan, entry.config, entry.input, entry.weight, entry.output,
        entry.square, entry.mean, entry.weight_dtype,
      )
    end

    # Release plans only after the owning chain has drained. The Arena owns
    # every input/output/scratch buffer and releases those separately.
    def release : Nil
      return if @released
      @entries.each_value { |entry| entry.plan.release }
      @entries.clear
      @released = true
    end

    private def prepare_entry!(op : JSON::Any, index : Int32) : Entry
      name = op["name"].as_s
      parts = name.split('.')
      unless parts.size == 4 && parts[1] == "attn" &&
             {"img", "txt"}.includes?(parts[0]) && {"q", "k"}.includes?(parts[2]) && parts[3] == "rms"
        raise ArgumentError.new("Unsupported MPS RMS operation identity: #{name}")
      end
      stream = parts[0]
      part = parts[2]
      prefix = "#{stream}_attn_#{part}_norm"
      unless op["weight_prefix"].as_s == prefix
        raise ArgumentError.new("MPS RMS weight identity mismatch: #{name}")
      end
      inputs = op["inputs"].as_h
      unless inputs.keys.to_set == ["x"].to_set
        raise ArgumentError.new("MPS RMS input inventory mismatch: #{name}")
      end

      tokens = stream == "img" ? @hy_config.video_tokens : @hy_config.text_tokens
      rows = ML::Video::HyNomasDualStreamBlock.checked_count(
        @hy_config.batches, tokens, @hy_config.heads,
      )
      count = rows.to_i64 * @hy_config.head_dim
      unless count > 0 && count <= Int32::MAX
        raise ArgumentError.new("MPS RMS shape exceeds addressable storage: #{name}")
      end
      input_shape = [@hy_config.batches, tokens, @hy_config.heads, @hy_config.head_dim]
      require_shape!(inputs["x"], input_shape, "#{name}/x")
      require_shape!(op["expected"], input_shape, "#{name}/output")
      require_shape!(op["rounded_expected"], input_shape, "#{name}/rounded output")
      unless HyAttentionProbe.values(op["expected"]).size == count &&
             HyAttentionProbe.values(op["rounded_expected"]).size == count
        raise ArgumentError.new("MPS RMS tensor length mismatch: #{name}")
      end

      input_key = "#{index}.input.x"
      output_key = "#{index}.output"
      input = @arena[input_key]
      output = @arena[output_key]
      unless @arena.dtype(input_key).f32? && input.storage_mode.shared? &&
             @arena.dtype(output_key).f32? && output.storage_mode.shared?
        raise ArgumentError.new("MPS RMS controls require shared F32 activations: #{name}")
      end
      ML::Video::MetalOps.check(input, count)
      ML::Video::MetalOps.check(output, count)

      weight_key = "#{prefix}_weight"
      raise ArgumentError.new("Missing MPS RMS weight: #{prefix}") unless @weights.allocated?(weight_key)
      weight = @weights[weight_key]
      weight_dtype = @weights.dtype(weight_key)
      ML::Video::MetalOps.check_storage(weight, @hy_config.head_dim.to_i64, weight_dtype)

      scratch_key = "#{index}.mps_rms.square"
      mean_key = "#{index}.mps_rms.mean"
      square = @arena.alloc(scratch_key, count.to_i32)
      mean = @arena.alloc(mean_key, rows)
      ML::Video::MetalOps.check(square, count)
      ML::Video::MetalOps.check(mean, rows.to_i64)
      validate_distinct!([input, output, weight, square, mean], name)

      # WanT5Config uses tokens as the middle axis. Treat heads as additional
      # independent token rows so each contiguous head dimension is reduced.
      flattened_tokens = ML::Video::HyNomasDualStreamBlock.checked_count(tokens, @hy_config.heads)
      cfg = ML::Video::WanT5Config.new(
        vocab: 1, dim: @hy_config.head_dim, ffn_dim: @hy_config.head_dim,
        heads: 1, layers: 1, batches: @hy_config.batches,
        tokens: flattened_tokens, eps: @hy_config.norm_eps,
      )
      cfg.validate!
      unless cfg.rows == rows && cfg.dim == @hy_config.head_dim && cfg.eps == @hy_config.norm_eps
        raise ArgumentError.new("MPS RMS flattened row geometry mismatch: #{name}")
      end
      plan = ML::Video::WanT5RMSMeanPlan.new(cfg, square, mean)
      Entry.new(index, name, prefix, cfg, input, output, weight, square, mean, weight_dtype, plan)
    end

    private def validate_operation_identity!(op : JSON::Any, entry : Entry) : Nil
      unless op["kind"].as_s == "rms" && op["name"].as_s == entry.name &&
             op["weight_prefix"].as_s == entry.prefix &&
             op["eps"].as_f.to_f32 == @hy_config.norm_eps
        raise ArgumentError.new("MPS RMS operation identity changed")
      end
      input_shape = [@hy_config.batches,
                     entry.config.tokens // @hy_config.heads,
                     @hy_config.heads, @hy_config.head_dim]
      require_shape!(op["inputs"]["x"], input_shape, "#{entry.name}/x")
      require_shape!(op["expected"], input_shape, "#{entry.name}/output")
      require_shape!(op["rounded_expected"], input_shape, "#{entry.name}/rounded output")
      expected_stream = entry.name.starts_with?("img.") ? "img" : "txt"
      unless entry.prefix == "#{expected_stream}_attn_#{entry.name.split('.')[2]}_norm"
        raise ArgumentError.new("MPS RMS stream identity changed: #{entry.name}")
      end
    end

    private def validate_bindings!(entry : Entry) : Nil
      input_key = "#{entry.index}.input.x"
      output_key = "#{entry.index}.output"
      weight_key = "#{entry.prefix}_weight"
      square_key = "#{entry.index}.mps_rms.square"
      mean_key = "#{entry.index}.mps_rms.mean"
      unless @arena.allocated?(input_key) && @arena[input_key].same?(entry.input) &&
             @arena.dtype(input_key).f32? && entry.input.storage_mode.shared? &&
             @arena.allocated?(output_key) && @arena[output_key].same?(entry.output) &&
             @arena.dtype(output_key).f32? && entry.output.storage_mode.shared? &&
             @weights.allocated?(weight_key) && @weights[weight_key].same?(entry.weight) &&
             @weights.dtype(weight_key) == entry.weight_dtype &&
             @arena.allocated?(square_key) && @arena[square_key].same?(entry.square) &&
             @arena.dtype(square_key).f32? && @arena.allocated?(mean_key) &&
             @arena[mean_key].same?(entry.mean) && @arena.dtype(mean_key).f32?
        raise ArgumentError.new("MPS RMS buffer binding changed: #{entry.name}")
      end
      ML::Video::MetalOps.check(entry.input, entry.config.rows.to_i64 * entry.config.dim)
      ML::Video::MetalOps.check(entry.output, entry.config.rows.to_i64 * entry.config.dim)
      ML::Video::MetalOps.check_storage(entry.weight, entry.config.dim.to_i64, entry.weight_dtype)
      ML::Video::MetalOps.check(entry.square, entry.config.rows.to_i64 * entry.config.dim)
      ML::Video::MetalOps.check(entry.mean, entry.config.rows.to_i64)
      validate_distinct!([entry.input, entry.output, entry.weight, entry.square, entry.mean], entry.name)
    end

    private def validate_distinct!(buffers : Array(ML::MetalBuffer), label : String) : Nil
      buffers.each_with_index do |left, left_index|
        buffers.each_with_index do |right, right_index|
          next if right_index <= left_index
          if ML::Video::MetalOps.overlap?(left, right)
            raise ArgumentError.new("MPS RMS buffers must be distinct: #{label}")
          end
        end
      end
    end

    private def require_shape!(record : JSON::Any, expected : Array(Int32), label : String) : Nil
      unless record["shape"].as_a.map(&.as_i) == expected
        raise ArgumentError.new("MPS RMS layout mismatch: #{label}")
      end
    end
  end

  # Frozen-input control for upstream's tuple RoPE equation, kept separate
  # from MPS RMS results and from the production fused native kernel.
  class MPSRopeControls
    private struct Entry
      getter index : Int32
      getter name : String
      getter input : ML::MetalBuffer
      getter cosine : ML::MetalBuffer
      getter sine : ML::MetalBuffer
      getter output : ML::MetalBuffer
      getter tokens : Int32
      getter count : Int32

      def initialize(@index, @name, @input, @cosine, @sine, @output, @tokens, @count)
      end
    end

    @config : ML::Video::HyNomasConfig
    @arena : ML::Video::Arena
    @entries = {} of Int32 => Entry
    @released = false

    def initialize(@config : ML::Video::HyNomasConfig, ops : Array(JSON::Any),
                   @arena : ML::Video::Arena)
      @config.validate!
      seen = Set(String).new
      begin
        ops.each_with_index do |op, raw_index|
          next unless op["kind"].as_s == "rope"
          entry = prepare_entry!(op, raw_index.to_i32)
          raise ArgumentError.new("Duplicate source RoPE operation") unless seen.add?(entry.name)
          @entries[entry.index] = entry
        end
        expected = {"img.attn.q.rope", "img.attn.k.rope"}.to_set
        unless seen == expected && @entries.size == expected.size
          raise ArgumentError.new("Source RoPE control requires the two admitted image Q/K operations")
        end
        HyAttentionProbe.pipeline("hy_probe_rope_uncontracted")
      rescue ex
        @entries.clear
        @released = true
        raise ex
      end
    end

    def has_plan?(index : Int32) : Bool
      !@released && @entries.has_key?(index)
    end

    def encode(e : ML::Metal::ComputeEncoder, op : JSON::Any, index : Int32) : Nil
      raise "Source RoPE controls have been released" if @released
      entry = @entries[index]?
      raise ArgumentError.new("No source RoPE control for operation index #{index}") unless entry
      validate_operation_identity!(op, entry)
      validate_bindings!(entry)
      e.set_pipeline(HyAttentionProbe.pipeline("hy_probe_rope_uncontracted"))
      e.set_buffer(entry.input, 0).set_buffer(entry.cosine, 1).set_buffer(entry.sine, 2)
      e.set_buffer(entry.output, 3, ML::Metal::BufferAccess::Write)
      e.set_value(entry.tokens.to_u32, 4).set_value(@config.heads.to_u32, 5)
      e.set_value(@config.head_dim.to_u32, 6).set_value(@config.batches.to_u32, 7)
      e.dispatch_1d(entry.count)
    end

    def release : Nil
      return if @released
      @entries.clear
      @released = true
    end

    private def prepare_entry!(op : JSON::Any, index : Int32) : Entry
      name = op["name"].as_s
      unless {"img.attn.q.rope", "img.attn.k.rope"}.includes?(name)
        raise ArgumentError.new("Unsupported source RoPE operation identity: #{name}")
      end
      inputs = op["inputs"].as_h
      unless inputs.keys.to_set == ["x", "cos", "sin"].to_set
        raise ArgumentError.new("Source RoPE input inventory mismatch: #{name}")
      end
      tokens = @config.video_tokens
      shape = [@config.batches, tokens, @config.heads, @config.head_dim]
      require_shape!(inputs["x"], shape, "#{name}/x")
      require_shape!(op["expected"], shape, "#{name}/output")
      require_shape!(op["rounded_expected"], shape, "#{name}/rounded output")
      require_shape!(inputs["cos"], [tokens, @config.head_dim], "#{name}/cos")
      require_shape!(inputs["sin"], [tokens, @config.head_dim], "#{name}/sin")
      count = ML::Video::HyNomasDualStreamBlock.checked_count(
        @config.batches, tokens, @config.heads, @config.head_dim,
      )
      input = @arena["#{index}.input.x"]
      cosine = @arena["#{index}.input.cos"]
      sine = @arena["#{index}.input.sin"]
      output = @arena["#{index}.output"]
      {"#{index}.input.x", "#{index}.input.cos", "#{index}.input.sin", "#{index}.output"}.each do |key|
        buffer = @arena[key]
        unless @arena.dtype(key).f32? && buffer.storage_mode.shared?
          raise ArgumentError.new("Source RoPE control requires shared F32 buffers: #{name}/#{key}")
        end
      end
      {input, output}.each { |buffer| ML::Video::MetalOps.check(buffer, count.to_i64) }
      {cosine, sine}.each { |buffer| ML::Video::MetalOps.check(buffer, tokens.to_i64 * @config.head_dim) }
      validate_distinct!([input, cosine, sine, output], name)
      Entry.new(index, name, input, cosine, sine, output, tokens, count)
    end

    private def validate_operation_identity!(op : JSON::Any, entry : Entry) : Nil
      unless op["kind"].as_s == "rope" && op["name"].as_s == entry.name
        raise ArgumentError.new("Source RoPE operation identity changed")
      end
      shape = [@config.batches, entry.tokens, @config.heads, @config.head_dim]
      require_shape!(op["inputs"]["x"], shape, "#{entry.name}/x")
      require_shape!(op["inputs"]["cos"], [entry.tokens, @config.head_dim], "#{entry.name}/cos")
      require_shape!(op["inputs"]["sin"], [entry.tokens, @config.head_dim], "#{entry.name}/sin")
      require_shape!(op["expected"], shape, "#{entry.name}/output")
      require_shape!(op["rounded_expected"], shape, "#{entry.name}/rounded output")
    end

    private def validate_bindings!(entry : Entry) : Nil
      {"x" => entry.input, "cos" => entry.cosine, "sin" => entry.sine}.each do |name, bound|
        key = "#{entry.index}.input.#{name}"
        unless @arena.allocated?(key) && @arena[key].same?(bound) &&
               @arena.dtype(key).f32? && bound.storage_mode.shared?
          raise ArgumentError.new("Source RoPE input binding changed: #{entry.name}/#{name}")
        end
      end
      output_key = "#{entry.index}.output"
      unless @arena.allocated?(output_key) && @arena[output_key].same?(entry.output) &&
             @arena.dtype(output_key).f32? && entry.output.storage_mode.shared?
        raise ArgumentError.new("Source RoPE output binding changed: #{entry.name}")
      end
      validate_distinct!([entry.input, entry.cosine, entry.sine, entry.output], entry.name)
    end

    private def validate_distinct!(buffers : Array(ML::MetalBuffer), label : String) : Nil
      buffers.each_with_index do |left, left_index|
        buffers.each_with_index do |right, right_index|
          next if right_index <= left_index
          if ML::Video::MetalOps.overlap?(left, right)
            raise ArgumentError.new("Source RoPE buffers must be distinct: #{label}")
          end
        end
      end
    end

    private def require_shape!(record : JSON::Any, expected : Array(Int32), label : String) : Nil
      unless record["shape"].as_a.map(&.as_i) == expected
        raise ArgumentError.new("Source RoPE layout mismatch: #{label}")
      end
    end
  end
end
