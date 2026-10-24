require "json"
require "../../src/ml/video/hy_nomas_block"
require "../../src/ml/video/metal_linear_mps"

module HyAttentionProbe
  # Same-input MPS controls for the attention Q/K/V/output projections.
  #
  # The helper prepares one fixed-buffer MPSNDArray plan for each linear
  # operation. All plans share one F32 weight-expansion stage, so encode calls
  # must be serialized on the caller's command chain. Keep this object, its
  # Arenas, and every bound buffer alive until the chain's terminal wait.
  class MPSProjectionControls
    WEIGHT_STAGE_KEY = "hy_attention_probe.mps_projection_controls.weight_stage"

    private struct Candidate
      getter index : Int32
      getter name : String
      getter prefix : String
      getter bias_mode : String
      getter tokens : Int32
      getter rows : Int32
      getter input : ML::MetalBuffer
      getter output : ML::MetalBuffer
      getter weight : ML::MetalBuffer
      getter bias : ML::MetalBuffer
      getter weight_dtype : ML::Video::WeightDType
      getter bias_dtype : ML::Video::WeightDType

      def initialize(@index, @name, @prefix, @bias_mode, @tokens, @rows,
                     @input, @output, @weight, @bias, @weight_dtype, @bias_dtype)
      end
    end

    private struct Entry
      getter candidate : Candidate
      getter plan : ML::Video::WanT5MPSLinearPlan

      def initialize(@candidate, @plan)
      end
    end

    @config : ML::Video::HyNomasConfig
    @arena : ML::Video::Arena
    @weights : ML::Video::Arena
    @weight_stage : ML::MetalBuffer
    @entries = {} of Int32 => Entry
    @released = false

    def initialize(@config : ML::Video::HyNomasConfig, ops : Array(JSON::Any),
                   @arena : ML::Video::Arena, @weights : ML::Video::Arena)
      @config.validate!
      raise ArgumentError.new("MPS projection control requires operations") if ops.empty?
      raise ArgumentError.new("Duplicate MPS projection control weight-stage key") if @arena.allocated?(WEIGHT_STAGE_KEY)

      candidates = [] of Candidate
      seen_names = Set(String).new
      ops.each_with_index do |op, index|
        next unless op["kind"].as_s == "linear"
        raise ArgumentError.new("Duplicate MPS projection operation") unless seen_names.add?(op["name"].as_s)
        candidates << candidate!(op, index.to_i32)
      end
      validate_candidate_set!(candidates)

      count = ML::Video::HyNomasDualStreamBlock.checked_count(@config.dim, @config.dim)
      ML::Video::MetalLinearMPS.prepare!
      @weight_stage = @arena.alloc(WEIGHT_STAGE_KEY, count)
      begin
        candidates.each do |candidate|
          cfg = ML::Video::WanT5Config.new(
            vocab: 1, dim: @config.dim, ffn_dim: @config.ffn_dim,
            heads: @config.heads, layers: 1, batches: @config.batches,
            tokens: candidate.tokens,
          )
          cfg.validate!
          plan = ML::Video::WanT5MPSLinearPlan.new(
            cfg, @config.dim, @config.dim, candidate.input, @weight_stage, candidate.output,
          )
          @entries[candidate.index] = Entry.new(candidate, plan)
        end
      rescue ex
        release
        @arena.remove(WEIGHT_STAGE_KEY) if @arena.allocated?(WEIGHT_STAGE_KEY)
        raise ex
      end
    end

    def plan_count : Int32
      @entries.size.to_i32
    end

    def has_plan?(index : Int32) : Bool
      !@released && @entries.has_key?(index)
    end

    # Append an MPS projection to an already-open serial encoder. A "none"
    # bias mode executes only the linear operation; normal mode adds the
    # resident bias in a separate Metal dispatch, matching the native
    # projection's arithmetic boundary. This method does not wait or allocate
    # MetalBuffers.
    def encode(e : ML::Metal::ComputeEncoder, op : JSON::Any, index : Int32) : Nil
      raise ArgumentError.new("MPS projection controls have been released") if @released
      entry = @entries[index]?
      raise ArgumentError.new("No MPS projection plan for operation index #{index}") unless entry
      candidate = entry.candidate
      validate_operation_identity!(op, candidate)

      input = @arena["#{index}.input.x"]
      output = @arena["#{index}.output"]
      weight_name = "#{candidate.prefix}_weight"
      bias_name = "#{candidate.prefix}_bias"
      weight = @weights[weight_name]
      bias = @weights[bias_name]
      unless input.same?(candidate.input) && output.same?(candidate.output) &&
             weight.same?(candidate.weight) && bias.same?(candidate.bias)
        raise ArgumentError.new("MPS projection control buffer identity changed")
      end
      unless @weights.dtype(weight_name) == candidate.weight_dtype &&
             @weights.dtype(bias_name) == candidate.bias_dtype
        raise ArgumentError.new("MPS projection control weight dtype changed")
      end
      unless @arena.allocated?(WEIGHT_STAGE_KEY) &&
             @arena[WEIGHT_STAGE_KEY].same?(@weight_stage) &&
             @arena.dtype(WEIGHT_STAGE_KEY).f32?
        raise ArgumentError.new("MPS projection control weight stage changed")
      end

      ML::Video::MetalLinearMPS.project(
        e, input, weight, bias, output, candidate.rows,
        @config.dim, @config.dim, candidate.weight_dtype, candidate.bias_dtype,
        entry.plan, @weight_stage, apply_bias: candidate.bias_mode != "none")
    end

    # Call only after the caller's terminal command wait. The Arena owns the
    # shared stage and releases it when the Arena itself is released.
    def release : Nil
      return if @released
      @entries.each_value { |entry| entry.plan.release }
      @entries.clear
      @released = true
    end

    private def candidate!(op : JSON::Any, index : Int32) : Candidate
      name = op["name"].as_s
      parts = name.split('.')
      unless parts.size == 4 && parts[1] == "attn" &&
             {"img", "txt"}.includes?(parts[0]) && {"q", "k", "v", "proj"}.includes?(parts[2])
        raise ArgumentError.new("Unsupported MPS attention projection name: #{name}")
      end
      stream = parts[0]
      part = parts[2]
      suffix = parts[3]
      expected_mode = case suffix
                      when "linear"               then "normal"
                      when "linear_nobias"        then "none"
                      when "linear_explicit_bias" then "normal"
                      else                             raise ArgumentError.new("Unsupported MPS attention projection name: #{name}")
                      end
      bias_mode = op["bias_mode"]?.try(&.as_s) || "normal"
      unless bias_mode == expected_mode
        raise ArgumentError.new("MPS attention projection bias mode mismatch: #{name}")
      end
      prefix = "#{stream}_attn_#{part}"
      unless op["weight_prefix"].as_s == prefix
        raise ArgumentError.new("MPS attention projection weight identity mismatch: #{name}")
      end

      tokens = stream == "img" ? @config.video_tokens : @config.text_tokens
      rows = ML::Video::HyNomasDualStreamBlock.checked_count(@config.batches, tokens)
      input_record = op["inputs"]["x"]
      require_shape!(input_record["shape"], [@config.batches, tokens, @config.dim], name + "/x")
      expected_shape = if part == "proj"
                         [@config.batches, tokens, @config.dim]
                       else
                         [@config.batches, tokens, @config.heads, @config.head_dim]
                       end
      require_shape!(op["expected"]["shape"], expected_shape, name + "/output")
      input_count = rows.to_i64 * @config.dim
      raise ArgumentError.new("MPS attention projection shape exceeds Int32: #{name}") if input_count > Int32::MAX

      input = @arena["#{index}.input.x"]
      output = @arena["#{index}.output"]
      weight_name = "#{prefix}_weight"
      bias_name = "#{prefix}_bias"
      unless @weights.allocated?(weight_name) && @weights.allocated?(bias_name)
        raise ArgumentError.new("Missing MPS attention projection weights: #{prefix}")
      end
      weight = @weights[weight_name]
      bias = @weights[bias_name]
      weight_dtype = @weights.dtype(weight_name)
      bias_dtype = @weights.dtype(bias_name)
      ML::Video::MetalOps.check(input, input_count)
      ML::Video::MetalOps.check(output, input_count)
      ML::Video::MetalOps.check_storage(weight, @config.dim.to_i64 * @config.dim, weight_dtype)
      ML::Video::MetalOps.check_storage(bias, @config.dim, bias_dtype)
      ML::Video::MetalOps.distinct(output, input, weight, bias)
      Candidate.new(index, name, prefix, bias_mode, tokens, rows, input, output,
        weight, bias, weight_dtype, bias_dtype)
    end

    private def validate_candidate_set!(candidates : Array(Candidate)) : Nil
      expected = Set(String).new
      {"img", "txt"}.each do |stream|
        {"q", "k", "v", "proj"}.each do |part|
          {"linear", "linear_nobias", "linear_explicit_bias"}.each do |variant|
            expected << "#{stream}.attn.#{part}.#{variant}"
          end
        end
      end
      actual = candidates.map(&.name).to_set
      unless candidates.size == expected.size && actual == expected
        raise ArgumentError.new("MPS projection controls require exactly the admitted 24 attention linear operations")
      end
    end

    private def validate_operation_identity!(op : JSON::Any, candidate : Candidate) : Nil
      unless op["kind"].as_s == "linear" && op["name"].as_s == candidate.name &&
             op["weight_prefix"].as_s == candidate.prefix &&
             (op["bias_mode"]?.try(&.as_s) || "normal") == candidate.bias_mode
        raise ArgumentError.new("MPS projection control operation identity changed")
      end
    end

    private def require_shape!(shape : JSON::Any, expected : Array(Int32), label : String) : Nil
      unless shape.as_a.map(&.as_i) == expected
        raise ArgumentError.new("MPS projection control layout mismatch: #{label}")
      end
    end
  end
end
