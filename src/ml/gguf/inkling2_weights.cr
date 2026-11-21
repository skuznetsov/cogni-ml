require "./reader"
require "./compute"

# Native mmap-backed GGUF loader for Inkling 2 loop architectures.
module ML::GGUF
  struct Inkling2Config
    getter n_layer : Int32
    getter n_embd : Int32
    getter n_head : Int32
    getter n_head_kv : Int32
    getter head_dim : Int32
    getter rel_dim : Int32
    getter rel_extent : Int32
    getter sliding_window : Int32
    getter conv_size : Int32
    getter log_scale_floor : Int32
    getter log_scale_alpha : Float32
    getter qk_clip : Float32
    getter rel_clip : Float32
    getter rms_eps : Float32
    getter n_expert : Int32
    getter n_expert_used : Int32
    getter expert_dim : Int32
    getter expert_scale : Float32
    getter situ_beta : Float32
    getter situ_linear_beta : Float32
    getter res_block_size : Int32
    getter vocab_size : Int32
    getter padded_vocab_size : Int32
    getter context_length : Int32
    getter core_start : Int32
    getter core_end : Int32
    getter passes : Int32
    getter gate_rank : Int32
    getter logit_softcap : Float32
    getter varying : Bool

    @sliding_pattern : Array(Bool)

    def initialize(
      @n_layer : Int32,
      @n_embd : Int32,
      @n_head : Int32,
      @n_head_kv : Int32,
      @head_dim : Int32,
      @rel_dim : Int32,
      @rel_extent : Int32,
      @sliding_window : Int32,
      @conv_size : Int32,
      @log_scale_floor : Int32,
      @log_scale_alpha : Float32,
      @qk_clip : Float32,
      @rel_clip : Float32,
      @rms_eps : Float32,
      @n_expert : Int32,
      @n_expert_used : Int32,
      @expert_dim : Int32,
      @expert_scale : Float32,
      @situ_beta : Float32,
      @situ_linear_beta : Float32,
      @res_block_size : Int32,
      @vocab_size : Int32,
      @padded_vocab_size : Int32,
      @context_length : Int32,
      @core_start : Int32,
      @core_end : Int32,
      @passes : Int32,
      @gate_rank : Int32,
      @logit_softcap : Float32,
      @varying : Bool,
      @sliding_pattern : Array(Bool),
    )
    end

    def initialize(g : GGUFFile)
      arch = g.get_string("general.architecture")
      raise ArgumentError.new("inkling2_weights: expected general.architecture=\"inkling2_loop\", got #{arch.inspect}") unless arch == "inkling2_loop"

      prefix = "inkling2_loop"
      @n_layer = required_int(g, "#{prefix}.block_count")
      @n_embd = required_int(g, "#{prefix}.embedding_length")
      @context_length = required_int(g, "#{prefix}.context_length")
      @n_head = uniform_layer_int(g, "#{prefix}.attention.head_count", @n_layer)
      @n_head_kv = uniform_layer_int(g, "#{prefix}.attention.head_count_kv", @n_layer)
      @head_dim = required_int(g, "#{prefix}.attention.key_length")
      value_dim = required_int(g, "#{prefix}.attention.value_length")
      @rel_dim = required_int(g, "#{prefix}.attention.relative_length")
      @rel_extent = required_int(g, "#{prefix}.attention.relative_extent")
      @sliding_window = required_int(g, "#{prefix}.attention.sliding_window")
      @conv_size = required_int(g, "#{prefix}.shortconv.l_cache")
      @log_scale_floor = required_int(g, "#{prefix}.attention.log_scale_floor")
      @log_scale_alpha = required_float(g, "#{prefix}.attention.log_scale_alpha")
      @qk_clip = required_float(g, "#{prefix}.attention.qk_clip")
      @rel_clip = required_float(g, "#{prefix}.attention.relative_clip")
      @rms_eps = required_float(g, "#{prefix}.attention.layer_norm_rms_epsilon")
      @n_expert = required_int(g, "#{prefix}.expert_count")
      @n_expert_used = uniform_layer_int(g, "#{prefix}.expert_used_count", @n_layer)
      @expert_dim = uniform_layer_int(g, "#{prefix}.expert_feed_forward_length", @n_layer)
      @expert_scale = optional_float(g, "#{prefix}.expert_weights_scale", 1.0_f32)
      @situ_beta = required_float(g, "#{prefix}.activation.situ_beta")
      @situ_linear_beta = required_float(g, "#{prefix}.activation.situ_linear_beta")
      @res_block_size = required_int(g, "#{prefix}.attn_res.block_size")
      @padded_vocab_size = required_int(g, "#{prefix}.vocab_size")
      @vocab_size = required_int(g, "#{prefix}.vocab_size_unpadded")
      @core_start = required_int(g, "#{prefix}.loop.start")
      @core_end = required_int(g, "#{prefix}.loop.end")
      @passes = required_int(g, "#{prefix}.loop.count")
      @gate_rank = required_int(g, "#{prefix}.loop.gate_rank")
      @varying = optional_bool(g, "#{prefix}.loop.varying", false)
      @logit_softcap = optional_float(g, "#{prefix}.final_logit_softcapping", 30.0_f32)
      @sliding_pattern = required_bool_array(g, "#{prefix}.attention.sliding_window_pattern")

      validate!
      raise ArgumentError.new("inkling2_weights: attention key_length and value_length differ") unless @head_dim == value_dim
    end

    def varying? : Bool
      @varying
    end

    def sliding?(layer : Int32) : Bool
      raise IndexError.new("inkling2_weights: layer #{layer} is outside 0...#{@n_layer}") unless 0 <= layer < @n_layer
      @sliding_pattern[layer]
    end

    private def validate! : Nil
      positive = {
        {"block_count", @n_layer}, {"embedding_length", @n_embd}, {"context_length", @context_length},
        {"head_count", @n_head}, {"head_count_kv", @n_head_kv}, {"head_dim", @head_dim},
        {"relative_length", @rel_dim}, {"relative_extent", @rel_extent}, {"sliding_window", @sliding_window},
        {"shortconv.l_cache", @conv_size}, {"log_scale_floor", @log_scale_floor}, {"expert_count", @n_expert},
        {"expert_used_count", @n_expert_used}, {"expert_feed_forward_length", @expert_dim},
        {"attn_res.block_size", @res_block_size}, {"vocab_size", @padded_vocab_size}, {"vocab_size_unpadded", @vocab_size},
        {"loop.gate_rank", @gate_rank},
      }
      positive.each do |entry|
        raise ArgumentError.new("inkling2_weights: #{entry[0]} must be positive") unless entry[1] > 0
      end

      raise ArgumentError.new("inkling2_weights: invalid attention head dimensions") if @n_head % @n_head_kv != 0
      raise ArgumentError.new("inkling2_weights: sliding-window pattern has #{@sliding_pattern.size} entries for #{@n_layer} layers") unless @sliding_pattern.size == @n_layer
      raise ArgumentError.new("inkling2_weights: conv_size must be between 2 and 64") unless 2 <= @conv_size <= 64
      raise ArgumentError.new("inkling2_weights: unpadded vocabulary exceeds its padding") if @vocab_size > @padded_vocab_size
      raise ArgumentError.new("inkling2_weights: invalid expert count") if @n_expert_used > @n_expert
      raise ArgumentError.new("inkling2_weights: invalid loop core [#{@core_start}, #{@core_end})") unless 0 < @core_start < @core_end <= @n_layer
      raise ArgumentError.new("inkling2_weights: loop core is not aligned to residual block size #{@res_block_size}") if @core_start % @res_block_size != 0 || @core_end % @res_block_size != 0
      raise ArgumentError.new("inkling2_weights: loop count must be positive") unless @passes > 0
      unless @varying
        raise ArgumentError.new("inkling2_weights: fixed Inkling 2 checkpoints require exactly 3 passes") unless @passes == 3
      end
      expanded_layers = @n_layer.to_i64 + (@passes - 1).to_i64 * (@core_end - @core_start).to_i64
      raise ArgumentError.new("inkling2_weights: expanded loop has #{expanded_layers} slots, exceeding 512") if expanded_layers > 512

      if @varying && (@n_embd != 2304 || @core_start != 6 || @core_end != 18)
        raise ArgumentError.new("inkling2_weights: variable loops are admitted only for the 2304-wide core [6, 18) layout")
      end

      {
        {"rms_epsilon", @rms_eps}, {"qk_clip", @qk_clip}, {"relative_clip", @rel_clip},
        {"expert_weights_scale", @expert_scale}, {"situ_beta", @situ_beta}, {"situ_linear_beta", @situ_linear_beta},
      }.each do |entry|
        raise ArgumentError.new("inkling2_weights: #{entry[0]} must be finite and positive") unless entry[1].finite? && entry[1] > 0.0_f32
      end
      raise ArgumentError.new("inkling2_weights: log_scale_alpha must be finite and nonnegative") unless @log_scale_alpha.finite? && @log_scale_alpha >= 0.0_f32
      raise ArgumentError.new("inkling2_weights: logit_softcap must be finite and nonnegative") unless @logit_softcap.finite? && @logit_softcap >= 0.0_f32
    end

    private def required_int(g : GGUFFile, key : String) : Int32
      value = g.get_int(key) || raise ArgumentError.new("inkling2_weights: missing integer metadata #{key.inspect}")
      raise ArgumentError.new("inkling2_weights: integer metadata #{key.inspect} exceeds Int32") unless Int32::MIN <= value <= Int32::MAX
      value.to_i32
    end

    private def required_float(g : GGUFFile, key : String) : Float32
      value = g.get_float(key) || raise ArgumentError.new("inkling2_weights: missing float metadata #{key.inspect}")
      converted = value.to_f32
      raise ArgumentError.new("inkling2_weights: metadata #{key.inspect} is not finite Float32") unless converted.finite?
      converted
    end

    private def optional_float(g : GGUFFile, key : String, default : Float32) : Float32
      return default unless g.metadata.has_key?(key)
      required_float(g, key)
    end

    private def optional_bool(g : GGUFFile, key : String, default : Bool) : Bool
      return default unless g.metadata.has_key?(key)
      case value = g.metadata[key]?
      when Bool then value
      else           raise ArgumentError.new("inkling2_weights: metadata #{key.inspect} must be bool")
      end
    end

    private def required_bool_array(g : GGUFFile, key : String) : Array(Bool)
      g.get_bool_array(key) || raise ArgumentError.new("inkling2_weights: missing bool-array metadata #{key.inspect}")
    end

    private def uniform_layer_int(g : GGUFFile, key : String, n_layer : Int32) : Int32
      if value = g.get_int(key)
        raise ArgumentError.new("inkling2_weights: metadata #{key.inspect} exceeds Int32") unless value <= Int32::MAX
        return value.to_i32
      end
      values = g.get_int_array(key) || raise ArgumentError.new("inkling2_weights: missing integer or integer-array metadata #{key.inspect}")
      raise ArgumentError.new("inkling2_weights: metadata #{key.inspect} must have #{n_layer} values") unless values.size == n_layer
      first = values.first
      raise ArgumentError.new("inkling2_weights: metadata #{key.inspect} values must be uniform") unless values.all? { |v| v == first }
      raise ArgumentError.new("inkling2_weights: metadata #{key.inspect} exceeds Int32") unless Int32::MIN <= first <= Int32::MAX
      first.to_i32
    end
  end

  struct Inkling2AttentionWeights
    getter q : QuantWeight
    getter k : QuantWeight
    getter v : QuantWeight
    getter output : QuantWeight
    getter gate : QuantWeight
    getter sink : QuantWeight
    getter rel_proj : QuantWeight
    getter q_clip : Array(Float32)
    getter rel_bias : Array(Float32)
    getter rel_clip : Array(Float32)
    getter k_conv : Array(Float32)
    getter v_conv : Array(Float32)
    getter post_conv : Array(Float32)
    getter k_conv_proj : QuantWeight?
    getter v_conv_proj : QuantWeight?

    def initialize(
      @q : QuantWeight,
      @k : QuantWeight,
      @v : QuantWeight,
      @output : QuantWeight,
      @gate : QuantWeight,
      @sink : QuantWeight,
      @rel_proj : QuantWeight,
      @q_clip : Array(Float32),
      @rel_bias : Array(Float32),
      @rel_clip : Array(Float32),
      @k_conv : Array(Float32),
      @v_conv : Array(Float32),
      @post_conv : Array(Float32),
      @k_conv_proj : QuantWeight?,
      @v_conv_proj : QuantWeight?,
    )
    end
  end

  struct Inkling2ExpertWeights
    getter raw : Bytes
    getter type : TensorType
    getter in_dim : Int32
    getter out_dim : Int32
    getter n_expert : Int32
    getter expert_bytes : Int32

    def initialize(@raw : Bytes, @type : TensorType, @in_dim : Int32, @out_dim : Int32, @n_expert : Int32)
      @expert_bytes = (@raw.size // @n_expert).to_i32
    end

    def expert(id : Int32) : QuantWeight
      raise IndexError.new("inkling2_weights: expert id #{id} is outside 0...#{@n_expert}") unless 0 <= id < @n_expert
      start = id * @expert_bytes
      bytes = Bytes.new(@raw.to_unsafe + start, @expert_bytes, read_only: true)
      QuantWeight.new(bytes, @type, @out_dim, @in_dim, "inkling2:expert:#{id}")
    end
  end

  struct Inkling2LoopWeights
    getter query : Array(Float32)
    getter down : Array(QuantWeight)
    getter gain : Array(QuantWeight)
    getter forget : Array(QuantWeight)

    def initialize(@query : Array(Float32), @down : Array(QuantWeight), @gain : Array(QuantWeight), @forget : Array(QuantWeight))
    end
  end

  struct Inkling2LayerWeights
    getter attention : Inkling2AttentionWeights
    getter q_norm : Array(Float32)
    getter k_norm : Array(Float32)
    getter v_norm : Array(Float32)
    getter attn_res_query : Array(Float32)?
    getter ffn_res_query : Array(Float32)
    getter ffn_norm : Array(Float32)
    getter ffn_post_conv : Array(Float32)
    getter router : QuantWeight
    getter router_bias : Array(Float32)?
    getter gate_up : Inkling2ExpertWeights
    getter down : Inkling2ExpertWeights
    getter controller : Inkling2LoopWeights?

    def initialize(
      @attention : Inkling2AttentionWeights,
      @q_norm : Array(Float32),
      @k_norm : Array(Float32),
      @v_norm : Array(Float32),
      @attn_res_query : Array(Float32)?,
      @ffn_res_query : Array(Float32),
      @ffn_norm : Array(Float32),
      @ffn_post_conv : Array(Float32),
      @router : QuantWeight,
      @router_bias : Array(Float32)?,
      @gate_up : Inkling2ExpertWeights,
      @down : Inkling2ExpertWeights,
      @controller : Inkling2LoopWeights?,
    )
    end
  end

  class Inkling2Weights
    getter config : Inkling2Config
    getter gguf : GGUFFile
    getter layers : Array(Inkling2LayerWeights)
    getter output : QuantWeight
    getter output_norm : Array(Float32)
    getter output_res_query : Array(Float32)
    getter token_embd_norm : Array(Float32)

    getter token_embd : QuantWeight
    @validated_infos : Hash(String, TensorInfo)
    @file_size : Int64
    @closed : Bool
    @close_mutex : Mutex

    def initialize(path : String)
      @gguf = GGUFFile.new(path)
      @validated_infos = {} of String => TensorInfo
      @file_size = File.size(path)
      @closed = false
      @close_mutex = Mutex.new
      @layers = uninitialized Array(Inkling2LayerWeights)
      @output = uninitialized QuantWeight
      @output_norm = uninitialized Array(Float32)
      @output_res_query = uninitialized Array(Float32)
      @token_embd_norm = uninitialized Array(Float32)
      @token_embd = uninitialized QuantWeight
      begin
        config = Inkling2Config.new(@gguf)
      rescue ex
        close rescue nil
        raise ex
      end
      @config = config
      begin
        unless @gguf.mmap_region
          raise ArgumentError.new("inkling2_weights: a working GGUF mmap is required for zero-copy loading")
        end
        validate_inventory!

        @token_embd = load_qw("token_embd.weight")
        @output = load_qw("output.weight")
        @output_norm = load_f32("output_norm.weight")
        @output_res_query = load_f32("output_res_query.weight")
        @token_embd_norm = load_f32("token_embd_norm.weight")
        @layers = Array(Inkling2LayerWeights).new(@config.n_layer) { |il| load_layer(il) }
      rescue ex
        close rescue nil
        raise ex
      end
    end

    def close : Nil
      @close_mutex.synchronize do
        return if @closed
        begin
          @gguf.close
        ensure
          @closed = true
        end
      end
    end

    def finalize
      close
    end

    def embedding_row(id : Int32) : Array(Float32)
      ensure_open!
      raise IndexError.new("inkling2_weights: token id #{id} is outside 0...#{@config.vocab_size}") unless 0 <= id < @config.vocab_size
      row_bytes = bytes_for_elements(@token_embd.type, @token_embd.in_dim.to_i64)
      offset = id.to_i64 * row_bytes
      raise ArgumentError.new("inkling2_weights: embedding row exceeds token embedding extent") if offset < 0 || offset + row_bytes > @token_embd.raw.size
      row = Bytes.new(@token_embd.raw.to_unsafe + offset.to_i, row_bytes.to_i32, read_only: true)
      Dequant.dequantize(row, @token_embd.type, @config.n_embd)
    end

    private def ensure_open! : Nil
      raise "inkling2_weights: weights are closed" if @closed
    end

    private def validate_inventory! : Nil
      ensure_supported_mmap!
      c = @config
      validate_tensor("token_embd.weight", [c.n_embd, c.padded_vocab_size], :weight)
      validate_tensor("output.weight", [c.n_embd, c.padded_vocab_size], :weight)
      validate_tensor("token_embd_norm.weight", [c.n_embd], :float)
      validate_tensor("output_norm.weight", [c.n_embd], :float)
      validate_tensor("output_res_query.weight", [c.n_embd], :float)

      c.n_layer.times do |il|
        p = "blk.#{il}."
        n_heads = c.n_head
        kv_heads = c.n_head_kv
        q_dim = checked_dim_product(c.head_dim, n_heads, "query projection width")
        kv_dim = checked_dim_product(c.head_dim, kv_heads, "KV projection width")
        relative_width = checked_dim_product(c.rel_dim, n_heads, "relative projection width")
        conv_kv_width = checked_dim_product(c.conv_size, kv_heads, "convolution KV width")
        expert_gate_width = checked_dim_product(c.expert_dim, 2, "expert gate/up width")
        loop_down_width = checked_dim_product(c.n_embd, 2, "loop controller down width")
        extent = c.sliding?(il) ? c.sliding_window : c.rel_extent

        validate_tensor(p + "attn_q.weight", [c.n_embd, q_dim], :weight)
        validate_tensor(p + "attn_k.weight", [c.n_embd, kv_dim], :weight)
        validate_tensor(p + "attn_v.weight", [c.n_embd, kv_dim], :weight)
        validate_tensor(p + "attn_output.weight", [q_dim, c.n_embd], :weight)
        validate_tensor(p + "attn_gate.weight", [c.n_embd, q_dim], :weight)
        validate_tensor(p + "attn_q_clip.weight", [c.head_dim, n_heads], :float)
        validate_tensor(p + "attn_sink_proj.weight", [c.n_embd, n_heads], :weight)
        validate_tensor(p + "attn_rel.weight", [c.n_embd, relative_width], :weight)
        validate_tensor(p + "attn_rel_bias.weight", [c.rel_dim, extent], :float)
        validate_tensor(p + "attn_rel_clip.weight", [c.rel_dim, n_heads], :float)
        validate_tensor(p + "attn_k_conv.weight", [c.conv_size, kv_dim], :float)
        validate_tensor(p + "attn_v_conv.weight", [c.conv_size, kv_dim], :float)
        validate_tensor(p + "attn_post_conv.weight", [c.conv_size, c.n_embd], :float)
        validate_tensor(p + "ffn_post_conv.weight", [c.conv_size, c.n_embd], :float)

        if c.sliding?(il)
          validate_tensor(p + "attn_norm.weight", [c.n_embd], :float)
          reject_present(p + "attn_q_inp_norm.weight")
          reject_present(p + "attn_k_inp_norm.weight")
          reject_present(p + "attn_v_inp_norm.weight")
          reject_present(p + "attn_k_conv_proj.weight")
          reject_present(p + "attn_v_conv_proj.weight")
        else
          reject_present(p + "attn_norm.weight")
          validate_tensor(p + "attn_q_inp_norm.weight", [c.n_embd], :float)
          validate_tensor(p + "attn_k_inp_norm.weight", [c.n_embd], :float)
          validate_tensor(p + "attn_v_inp_norm.weight", [c.n_embd], :float)
          validate_tensor(p + "attn_k_conv_proj.weight", [c.n_embd, conv_kv_width], :weight)
          validate_tensor(p + "attn_v_conv_proj.weight", [c.n_embd, conv_kv_width], :weight)
        end

        if il == 0
          reject_present(p + "attn_res_query.weight")
        else
          validate_tensor(p + "attn_res_query.weight", [c.n_embd, c.sliding?(il) ? 1 : 3], :float)
        end
        validate_tensor(p + "ffn_res_query.weight", [c.n_embd], :float)
        validate_tensor(p + "ffn_norm.weight", [c.n_embd], :float)
        validate_tensor(p + "ffn_gate_inp.weight", [c.n_embd, c.n_expert], :weight)
        validate_tensor(p + "ffn_gate_up_exps.weight", [c.n_embd, expert_gate_width, c.n_expert], :expert_weight)
        validate_tensor(p + "ffn_down_exps.weight", [c.expert_dim, c.n_embd, c.n_expert], :expert_weight)
        validate_tensor(p + "exp_probs_b.bias", [c.n_expert], :float, required: false)

        if c.core_start <= il < c.core_end
          validate_tensor(p + "loop_msg_query.weight", [c.n_embd, 4], :float)
          {"q", "k", "v", "ffn"}.each do |kind|
            validate_tensor(p + "loop_#{kind}_ctrl_down.weight", [loop_down_width, c.gate_rank], :weight)
            validate_tensor(p + "loop_#{kind}_ctrl_gain.weight", [c.gate_rank, c.n_embd], :weight)
            validate_tensor(p + "loop_#{kind}_ctrl_forget.weight", [c.gate_rank, c.n_embd], :weight)
          end
        else
          {"loop_msg_query.weight", "loop_q_ctrl_down.weight", "loop_q_ctrl_gain.weight", "loop_q_ctrl_forget.weight",
           "loop_k_ctrl_down.weight", "loop_k_ctrl_gain.weight", "loop_k_ctrl_forget.weight",
           "loop_v_ctrl_down.weight", "loop_v_ctrl_gain.weight", "loop_v_ctrl_forget.weight",
           "loop_ffn_ctrl_down.weight", "loop_ffn_ctrl_gain.weight", "loop_ffn_ctrl_forget.weight"}.each do |suffix|
            reject_present(p + suffix)
          end
        end
      end
    end

    private def ensure_supported_mmap! : Nil
      region = @gguf.mmap_region || raise ArgumentError.new("inkling2_weights: a working GGUF mmap is required for zero-copy loading")
      raise ArgumentError.new("inkling2_weights: GGUF mmap size differs from file size") unless region[1] == @file_size.to_u64
    end

    private def reject_present(name : String) : Nil
      raise ArgumentError.new("inkling2_weights: unexpected tensor for this layout: #{name}") if @gguf.tensor(name)
    end

    private def validate_tensor(name : String, expected_dims : Array(Int32), kind : Symbol, *, required : Bool = true) : Nil
      info = @gguf.tensor(name)
      unless info
        raise ArgumentError.new("inkling2_weights: missing required tensor #{name.inspect}") if required
        return
      end
      actual_dims = normalized_dims(info.dims)
      wanted_dims = normalized_dims(expected_dims.map(&.to_i64))
      raise ArgumentError.new("inkling2_weights: tensor #{name.inspect} dimensions #{info.dims} != expected #{expected_dims}") unless actual_dims == wanted_dims
      raise ArgumentError.new("inkling2_weights: tensor #{name.inspect} has unsupported type #{info.type.name}") unless supported_type?(info.type)
      if kind == :float && !(info.type.f32? || info.type.f16? || info.type.bf16?)
        raise ArgumentError.new("inkling2_weights: tensor #{name.inspect} must use an unquantized float type, got #{info.type.name}")
      end
      validate_extent!(info)
      if kind == :weight || kind == :expert_weight
        raise ArgumentError.new("inkling2_weights: matrix tensor #{name.inspect} must have at least two dimensions") if info.dims.size < 2
        in_dim = info.dims[0]
        if info.type.block_elements > 1 && in_dim % info.type.block_elements != 0
          raise ArgumentError.new("inkling2_weights: tensor #{name.inspect} input dimension is not aligned to #{info.type.name} blocks")
        end
      end
      @validated_infos[name] = info
    end

    private def normalized_dims(dims : Array(Int64)) : Array(Int64)
      normalized = dims.dup
      while normalized.size > 1 && normalized.last == 1
        normalized.pop
      end
      normalized
    end

    private def supported_type?(type : TensorType) : Bool
      type.f32? || type.f16? || type.bf16? || type.q4_k? || type.q5_k? || type.q6_k? || type.q8_0? || type.iq4_nl?
    end

    private def checked_dim_product(left : Int32, right : Int32, label : String) : Int32
      product = left.to_i64 * right.to_i64
      raise ArgumentError.new("inkling2_weights: #{label} exceeds supported tensor dimensions") if product > Int32::MAX
      product.to_i32
    end

    private def validate_extent!(info : TensorInfo) : Nil
      elements = 1_i64
      raise ArgumentError.new("inkling2_weights: tensor #{info.name.inspect} has no dimensions") if info.dims.empty?
      info.dims.each do |dimension|
        raise ArgumentError.new("inkling2_weights: tensor #{info.name.inspect} has a nonpositive dimension") unless dimension > 0
        raise ArgumentError.new("inkling2_weights: tensor #{info.name.inspect} dimensions overflow") if elements > Int64::MAX // dimension
        elements *= dimension
      end
      raise ArgumentError.new("inkling2_weights: tensor #{info.name.inspect} is too large for the GGUF reader") if elements > Int32::MAX
      if info.type.block_elements > 1
        raise ArgumentError.new("inkling2_weights: tensor #{info.name.inspect} element count is not quantization-block aligned") if elements % info.type.block_elements != 0
      end
      bytes = bytes_for_elements(info.type, elements)
      raise ArgumentError.new("inkling2_weights: tensor #{info.name.inspect} byte count does not match GGUF metadata") unless bytes == info.data_bytes
      raise ArgumentError.new("inkling2_weights: tensor #{info.name.inspect} is too large for the GGUF reader") if bytes > Int32::MAX
      raise ArgumentError.new("inkling2_weights: tensor #{info.name.inspect} offset exceeds Int64") if info.offset > Int64::MAX.to_u64
      start = @gguf.data_offset.to_i64 + info.offset.to_i64
      raise ArgumentError.new("inkling2_weights: tensor #{info.name.inspect} has an invalid offset") if start < @gguf.data_offset || start > @file_size
      raise ArgumentError.new("inkling2_weights: tensor #{info.name.inspect} extends beyond the GGUF file") if bytes > @file_size - start
    end

    private def bytes_for_elements(type : TensorType, elements : Int64) : Int64
      if type.f32? || type.f16? || type.bf16?
        width = type.block_bytes.to_i64
        raise ArgumentError.new("inkling2_weights: tensor byte count overflows") if elements > Int64::MAX // width
        elements * width
      else
        block = type.block_elements.to_i64
        raise ArgumentError.new("inkling2_weights: quantized tensor is not block aligned") if elements % block != 0
        count = elements // block
        width = type.block_bytes.to_i64
        raise ArgumentError.new("inkling2_weights: tensor byte count overflows") if count > Int64::MAX // width
        count * width
      end
    end

    private def load_qw(name : String) : QuantWeight
      info = @validated_infos[name]
      raw = @gguf.read_tensor_raw(info)
      in_dim = info.dims[0].to_i32
      out_dim = (info.n_elements // in_dim).to_i32
      QuantWeight.new(raw, info.type, out_dim, in_dim, "inkling2:#{name}")
    end

    private def load_f32(name : String) : Array(Float32)
      @gguf.read_tensor_f32(@validated_infos[name])
    end

    private def load_expert_weights(name : String, in_dim : Int32, out_dim : Int32) : Inkling2ExpertWeights
      info = @validated_infos[name]
      Inkling2ExpertWeights.new(@gguf.read_tensor_raw(info), info.type, in_dim, out_dim, @config.n_expert)
    end

    private def load_layer(il : Int32) : Inkling2LayerWeights
      c = @config
      p = "blk.#{il}."
      sliding = c.sliding?(il)
      shared_norm = sliding ? load_f32(p + "attn_norm.weight") : nil
      q_norm = shared_norm || load_f32(p + "attn_q_inp_norm.weight")
      k_norm = shared_norm || load_f32(p + "attn_k_inp_norm.weight")
      v_norm = shared_norm || load_f32(p + "attn_v_inp_norm.weight")
      proj_k = sliding ? nil : load_qw(p + "attn_k_conv_proj.weight")
      proj_v = sliding ? nil : load_qw(p + "attn_v_conv_proj.weight")
      attention = Inkling2AttentionWeights.new(
        q: load_qw(p + "attn_q.weight"),
        k: load_qw(p + "attn_k.weight"),
        v: load_qw(p + "attn_v.weight"),
        output: load_qw(p + "attn_output.weight"),
        gate: load_qw(p + "attn_gate.weight"),
        sink: load_qw(p + "attn_sink_proj.weight"),
        rel_proj: load_qw(p + "attn_rel.weight"),
        q_clip: load_f32(p + "attn_q_clip.weight"),
        rel_bias: load_f32(p + "attn_rel_bias.weight"),
        rel_clip: load_f32(p + "attn_rel_clip.weight"),
        k_conv: load_f32(p + "attn_k_conv.weight"),
        v_conv: load_f32(p + "attn_v_conv.weight"),
        post_conv: load_f32(p + "attn_post_conv.weight"),
        k_conv_proj: proj_k,
        v_conv_proj: proj_v,
      )
      controller = if c.core_start <= il < c.core_end
                     load_controller(il)
                   else
                     nil
                   end
      Inkling2LayerWeights.new(
        attention: attention,
        q_norm: q_norm,
        k_norm: k_norm,
        v_norm: v_norm,
        attn_res_query: il == 0 ? nil : load_f32(p + "attn_res_query.weight"),
        ffn_res_query: load_f32(p + "ffn_res_query.weight"),
        ffn_norm: load_f32(p + "ffn_norm.weight"),
        ffn_post_conv: load_f32(p + "ffn_post_conv.weight"),
        router: load_qw(p + "ffn_gate_inp.weight"),
        router_bias: @validated_infos.has_key?(p + "exp_probs_b.bias") ? load_f32(p + "exp_probs_b.bias") : nil,
        gate_up: load_expert_weights(p + "ffn_gate_up_exps.weight", c.n_embd, c.expert_dim * 2),
        down: load_expert_weights(p + "ffn_down_exps.weight", c.expert_dim, c.n_embd),
        controller: controller,
      )
    end

    private def load_controller(il : Int32) : Inkling2LoopWeights
      c = @config
      p = "blk.#{il}."
      kinds = {"q", "k", "v", "ffn"}
      down = [] of QuantWeight
      gain = [] of QuantWeight
      forget = [] of QuantWeight
      kinds.each do |kind|
        down << load_qw(p + "loop_#{kind}_ctrl_down.weight")
        gain << load_qw(p + "loop_#{kind}_ctrl_gain.weight")
        forget << load_qw(p + "loop_#{kind}_ctrl_forget.weight")
      end
      Inkling2LoopWeights.new(load_f32(p + "loop_msg_query.weight"), down, gain, forget)
    end
  end
end
