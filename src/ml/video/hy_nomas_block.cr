# Adapted from Tencent Hy-Nomas at pinned revision
# 911cea6458a89c2663513ff7099561ec190dd491.
# Copyright (C) 2026 Tencent. Licensed under Apache-2.0.
# Modified for native Crystal/Metal execution.
require "../gguf/safetensors"
require "../metal/dispatch"
require "./metal_ops"
require "./checkpoint"

module ML::Video
  # F32Control preserves the initial native control path. BF16BoundaryControl
  # rounds concatenated Q/K/V and the SDPA output through BF16 while keeping
  # the intervening native attention arithmetic in F32.
  enum HyNomasAttentionPrecision
    F32Control
    BF16BoundaryControl
  end

  # One unmasked, bidirectional segment through a dual-stream MMDoubleStreamBlock.
  # A modulation row count of zero means one condition row per batch item
  # ([B,1,C]); set it to B*L for per-token conditions ([B,L,C]).
  record HyNomasConfig,
    batches : Int32,
    video_tokens : Int32,
    text_tokens : Int32,
    dim : Int32,
    heads : Int32,
    head_dim : Int32,
    ffn_dim : Int32,
    img_modulation_rows : Int32 = 0,
    txt_modulation_rows : Int32 = 0,
    segment_count : Int32 = 1,
    chunk_size_tokens : Int32 = 0,
    history_tokens : Int32 = 0,
    has_mask : Bool = false,
    has_camera : Bool = false,
    action_windows : Bool = false,
    norm_eps : Float32 = 1e-6_f32,
    attention_precision : HyNomasAttentionPrecision = HyNomasAttentionPrecision::F32Control do
    def effective_img_modulation_rows : Int32
      img_modulation_rows == 0 ? batches : img_modulation_rows
    end

    def effective_txt_modulation_rows : Int32
      txt_modulation_rows == 0 ? batches : txt_modulation_rows
    end

    def image_rows : Int32
      HyNomasDualStreamBlock.checked_count(batches, video_tokens)
    end

    def text_rows : Int32
      HyNomasDualStreamBlock.checked_count(batches, text_tokens)
    end

    def joint_tokens : Int32
      total = video_tokens.to_i64 + text_tokens
      raise ArgumentError.new("Hy-Nomas joint token count is too large") if total > Int32::MAX
      total.to_i32
    end

    def joint_rows : Int32
      HyNomasDualStreamBlock.checked_count(batches, joint_tokens)
    end

    def validate! : Nil
      unless attention_precision.f32_control? || attention_precision.bf16_boundary_control?
        raise ArgumentError.new("Unsupported Hy-Nomas attention precision control")
      end
      HyNomasDualStreamBlock.checked_count(batches, video_tokens, dim)
      HyNomasDualStreamBlock.checked_count(batches, text_tokens, dim)
      HyNomasDualStreamBlock.checked_count(batches, joint_tokens, dim)
      HyNomasDualStreamBlock.checked_count(dim, 6)
      raise ArgumentError.new("Hy-Nomas heads and head dimension must be positive") unless heads > 0 && head_dim > 0
      raise ArgumentError.new("Hy-Nomas attention head dimension exceeds the native kernel limit") if head_dim > 256
      raise ArgumentError.new("Hy-Nomas dim must equal heads * head_dim") unless dim.to_i64 == heads.to_i64 * head_dim
      raise ArgumentError.new("Hy-Nomas feed-forward width must be 4 * dim") unless ffn_dim.to_i64 == 4_i64 * dim
      raise ArgumentError.new("Hy-Nomas RoPE head dimension must be even") unless head_dim % 2 == 0
      unless norm_eps == 1e-6_f32
        raise ArgumentError.new("Pinned Hy-Nomas normalization epsilon must be 1e-6")
      end

      unless segment_count == 1 && chunk_size_tokens == 0 && history_tokens == 0 &&
             !has_mask && !has_camera && !action_windows
        raise ArgumentError.new("Hy-Nomas native block admits one full unmasked segment without history, camera, or action windows")
      end

      img_mod_rows = effective_img_modulation_rows
      txt_mod_rows = effective_txt_modulation_rows
      unless img_mod_rows == batches || img_mod_rows == image_rows
        raise ArgumentError.new("Image modulation rows must be B or B*video_tokens")
      end
      unless txt_mod_rows == batches || txt_mod_rows == text_rows
        raise ArgumentError.new("Text modulation rows must be B or B*text_tokens")
      end
      HyNomasDualStreamBlock.checked_count(img_mod_rows, dim)
      HyNomasDualStreamBlock.checked_count(txt_mod_rows, dim)
      HyNomasDualStreamBlock.checked_count(video_tokens, head_dim)
    end
  end

  # Encoder-only F32-math first slice of Hy-Nomas double_blocks.0.
  # This control uses F32 attention, not Magi's BF16 attention cast/compute path.
  # Inputs, weights, scratch and output buffers remain caller-owned and resident.
  # `encode` only validates, binds and dispatches into the borrowed encoder.
  class HyNomasDualStreamBlock
    SOURCE = {{ read_file("#{__DIR__}/kernels/hy_nomas_block.metal") }}

    struct LinearWeights
      getter weight : ML::MetalBuffer
      getter bias : ML::MetalBuffer
      getter weight_dtype : WeightDType
      getter bias_dtype : WeightDType

      def initialize(arena : Arena, prefix : String)
        @weight = arena["#{prefix}_weight"]
        @bias = arena["#{prefix}_bias"]
        @weight_dtype = arena.dtype("#{prefix}_weight")
        @bias_dtype = arena.dtype("#{prefix}_bias")
      end
    end

    struct NormWeight
      getter weight : ML::MetalBuffer
      getter dtype : WeightDType

      def initialize(arena : Arena, name : String)
        @weight = arena[name]
        @dtype = arena.dtype(name)
      end
    end

    struct StreamWeights
      getter modulation : LinearWeights
      getter query : LinearWeights
      getter key : LinearWeights
      getter value : LinearWeights
      getter projection : LinearWeights
      getter query_norm : NormWeight
      getter key_norm : NormWeight
      getter mlp_fc1 : LinearWeights
      getter mlp_fc2 : LinearWeights

      def initialize(arena : Arena, stream : String)
        @modulation = LinearWeights.new(arena, "#{stream}_mod")
        @query = LinearWeights.new(arena, "#{stream}_attn_q")
        @key = LinearWeights.new(arena, "#{stream}_attn_k")
        @value = LinearWeights.new(arena, "#{stream}_attn_v")
        @projection = LinearWeights.new(arena, "#{stream}_attn_proj")
        @query_norm = NormWeight.new(arena, "#{stream}_attn_q_norm_weight")
        @key_norm = NormWeight.new(arena, "#{stream}_attn_k_norm_weight")
        @mlp_fc1 = LinearWeights.new(arena, "#{stream}_mlp_fc1")
        @mlp_fc2 = LinearWeights.new(arena, "#{stream}_mlp_fc2")
      end
    end

    struct StreamBuffers
      getter vec_silu : ML::MetalBuffer
      getter modulation : ML::MetalBuffer
      getter attn_shift : ML::MetalBuffer
      getter attn_scale : ML::MetalBuffer
      getter attn_gate : ML::MetalBuffer
      getter ff_shift : ML::MetalBuffer
      getter ff_scale : ML::MetalBuffer
      getter ff_gate : ML::MetalBuffer
      getter attn_norm : ML::MetalBuffer
      getter attn_modulated : ML::MetalBuffer
      getter attn_q : ML::MetalBuffer
      getter attn_k : ML::MetalBuffer
      getter attn_v : ML::MetalBuffer
      getter attn_q_norm : ML::MetalBuffer
      getter attn_k_norm : ML::MetalBuffer
      getter attn_q_rope : ML::MetalBuffer?
      getter attn_k_rope : ML::MetalBuffer?
      getter attn_raw : ML::MetalBuffer
      getter attn_projected : ML::MetalBuffer
      getter after_attention : ML::MetalBuffer
      getter ff_norm : ML::MetalBuffer
      getter ff_modulated : ML::MetalBuffer
      getter ff_hidden : ML::MetalBuffer
      getter ff_activated : ML::MetalBuffer
      getter ff_projected : ML::MetalBuffer
      getter output : ML::MetalBuffer

      def initialize(scratch : Arena, stream : String, video : Bool)
        prefix = "hy_nomas.#{stream}"
        @vec_silu = scratch["#{prefix}.vec_silu"]
        @modulation = scratch["#{prefix}.modulation"]
        @attn_shift = scratch["#{prefix}.attn.shift"]
        @attn_scale = scratch["#{prefix}.attn.scale"]
        @attn_gate = scratch["#{prefix}.attn.gate"]
        @ff_shift = scratch["#{prefix}.ff.shift"]
        @ff_scale = scratch["#{prefix}.ff.scale"]
        @ff_gate = scratch["#{prefix}.ff.gate"]
        @attn_norm = scratch["#{prefix}.attn.norm"]
        @attn_modulated = scratch["#{prefix}.attn.modulated"]
        @attn_q = scratch["#{prefix}.attn.q"]
        @attn_k = scratch["#{prefix}.attn.k"]
        @attn_v = scratch["#{prefix}.attn.v"]
        @attn_q_norm = scratch["#{prefix}.attn.q_norm"]
        @attn_k_norm = scratch["#{prefix}.attn.k_norm"]
        @attn_q_rope = video ? scratch["#{prefix}.attn.q_rope"] : nil
        @attn_k_rope = video ? scratch["#{prefix}.attn.k_rope"] : nil
        @attn_raw = scratch["#{prefix}.attn.raw"]
        @attn_projected = scratch["#{prefix}.attn.projected"]
        @after_attention = scratch["#{prefix}.after_attention"]
        @ff_norm = scratch["#{prefix}.ff.norm"]
        @ff_modulated = scratch["#{prefix}.ff.modulated"]
        @ff_hidden = scratch["#{prefix}.ff.hidden"]
        @ff_activated = scratch["#{prefix}.ff.activated"]
        @ff_projected = scratch["#{prefix}.ff.projected"]
        @output = scratch["#{prefix}.output"]
      end
    end

    getter config : HyNomasConfig
    getter checkpoints : Hash(String, ML::MetalBuffer)
    getter image_output : ML::MetalBuffer
    getter text_output : ML::MetalBuffer

    @weights : Arena
    @scratch : Arena
    @bindings : Array(WeightBinding)
    @zero : ML::MetalBuffer
    @scratch_plan : Array(Tuple(String, Int32))
    @scratch_buffers : Array(ML::MetalBuffer)
    @image_weights : StreamWeights
    @text_weights : StreamWeights
    @image_buffers : StreamBuffers
    @text_buffers : StreamBuffers
    @joint_query : ML::MetalBuffer
    @joint_key : ML::MetalBuffer
    @joint_value : ML::MetalBuffer
    @joint_attention : ML::MetalBuffer

    def self.checked_count(*factors : Int32) : Int32
      total = 1_i64
      factors.each do |factor|
        raise ArgumentError.new("Hy-Nomas dimensions must be positive") unless factor > 0
        if total > Int32::MAX.to_i64 // factor
          raise ArgumentError.new("Hy-Nomas tensor exceeds the native Metal count limit")
        end
        total *= factor
      end
      total.to_i32
    end

    def self.checkpoint_bindings(config : HyNomasConfig,
                                 dtype : ML::GGUF::SafeTensorDType? = nil,
                                 block_index : Int32 = 0) : Array(WeightBinding)
      config.validate!
      raise ArgumentError.new("Hy-Nomas block index must be non-negative") if block_index < 0
      if dtype && !(dtype.not_nil!.f32? || dtype.not_nil!.bf16?)
        raise ArgumentError.new("Hy-Nomas checkpoints support only F32 or BF16 weights")
      end

      dim = config.dim.to_i64
      ffn = config.ffn_dim.to_i64
      bindings = [] of WeightBinding
      prefix = "double_blocks.#{block_index}"
      {"img", "txt"}.each do |stream|
        append_binding(bindings, "#{stream}_mod_weight", "#{prefix}.#{stream}_mod.linear.weight", [6_i64 * dim, dim], dtype)
        append_binding(bindings, "#{stream}_mod_bias", "#{prefix}.#{stream}_mod.linear.bias", [6_i64 * dim], dtype)
        {"q", "k", "v", "proj"}.each do |part|
          append_binding(bindings, "#{stream}_attn_#{part}_weight", "#{prefix}.#{stream}_attn_#{part}.weight", [dim, dim], dtype)
          append_binding(bindings, "#{stream}_attn_#{part}_bias", "#{prefix}.#{stream}_attn_#{part}.bias", [dim], dtype)
        end
        {"q", "k"}.each do |part|
          append_binding(bindings, "#{stream}_attn_#{part}_norm_weight", "#{prefix}.#{stream}_attn_#{part}_norm.weight", [config.head_dim.to_i64], dtype)
        end
        append_binding(bindings, "#{stream}_mlp_fc1_weight", "#{prefix}.#{stream}_mlp.fc1.weight", [ffn, dim], dtype)
        append_binding(bindings, "#{stream}_mlp_fc1_bias", "#{prefix}.#{stream}_mlp.fc1.bias", [ffn], dtype)
        append_binding(bindings, "#{stream}_mlp_fc2_weight", "#{prefix}.#{stream}_mlp.fc2.weight", [dim, ffn], dtype)
        append_binding(bindings, "#{stream}_mlp_fc2_bias", "#{prefix}.#{stream}_mlp.fc2.bias", [dim], dtype)
      end
      bindings
    end

    # Call before WeightPlan/PreparedWeights allocation to reject unsupported
    # safetensors types while the checkpoint is still header-only.
    def self.validate_source_dtypes!(files : Array(CheckpointFile), config : HyNomasConfig,
                                     block_index : Int32 = 0) : Nil
      bindings = checkpoint_bindings(config, block_index: block_index)
      tensors = {} of String => CheckpointTensor
      files.each do |file|
        file.unchanged!
        file.tensors.each do |name, info|
          raise ArgumentError.new("Duplicate checkpoint tensor across shards #{name}") if tensors.has_key?(name)
          tensors[name] = info
        end
      end
      bindings.each do |binding|
        next unless info = tensors[binding.source]?
        unless info.dtype.f32? || info.dtype.bf16?
          raise ArgumentError.new("Hy-Nomas checkpoint tensor must be F32 or BF16: #{binding.source}")
        end
      end
    end

    # Model-specific admission must precede the generic planner so source
    # tensors outside Hy-Nomas' supported F32/BF16 set fail before preparation.
    # CheckpointFile headers and backing bytes must remain immutable afterward.
    def self.weight_plan(files : Array(CheckpointFile), config : HyNomasConfig,
                         budget_bytes : Int64, reserve_bytes : Int64 = 0_i64,
                         resident_dtype : WeightDType = WeightDType::F32,
                         block_index : Int32 = 0, require_all : Bool = false,
                         allow_f32_to_bf16 : Bool = false) : WeightPlan
      validate_source_dtypes!(files, config, block_index)
      bindings = checkpoint_bindings(config, block_index: block_index)
      WeightPlan.new(files, bindings, budget_bytes, reserve_bytes, resident_dtype,
        require_all, allow_f32_to_bf16)
    end

    def self.prepare! : Nil
      MetalOps.prepare!
      {"hy_nomas_silu_f32", "hy_nomas_unpack_modulation_f32", "hy_nomas_rope_f32",
       "hy_nomas_concat_f32", "hy_nomas_split_f32"}.each do |name|
        pipeline(name)
      end
    end

    # Every intermediate is allocated once during setup and retained as a
    # named diagnostic checkpoint for parity comparison.
    def self.scratch_plan(config : HyNomasConfig) : Array(Tuple(String, Int32))
      config.validate!
      plan = [] of Tuple(String, Int32)
      add = ->(name : String, count : Int32) { plan << {"hy_nomas.#{name}", count} }
      add.call("zero_head_bias", config.head_dim)

      {"img", "txt"}.each do |stream|
        tokens = stream == "img" ? config.video_tokens : config.text_tokens
        rows = stream == "img" ? config.image_rows : config.text_rows
        mod_rows = stream == "img" ? config.effective_img_modulation_rows : config.effective_txt_modulation_rows
        add.call("#{stream}.vec_silu", checked_count(mod_rows, config.dim))
        add.call("#{stream}.modulation", checked_count(mod_rows, config.dim, 6))
        {"attn", "ff"}.each do |phase|
          {"shift", "scale", "gate"}.each do |field|
            add.call("#{stream}.#{phase}.#{field}", checked_count(rows, config.dim))
          end
        end
        add.call("#{stream}.attn.norm", checked_count(rows, config.dim))
        add.call("#{stream}.attn.modulated", checked_count(rows, config.dim))
        {"q", "k", "v"}.each do |part|
          add.call("#{stream}.attn.#{part}", checked_count(rows, config.dim))
        end
        {"q_norm", "k_norm"}.each do |part|
          add.call("#{stream}.attn.#{part}", checked_count(rows, config.dim))
        end
        if stream == "img"
          {"q_rope", "k_rope"}.each do |part|
            add.call("#{stream}.attn.#{part}", checked_count(rows, config.dim))
          end
        end
        add.call("#{stream}.attn.raw", checked_count(rows, config.dim))
        add.call("#{stream}.attn.projected", checked_count(rows, config.dim))
        add.call("#{stream}.after_attention", checked_count(rows, config.dim))
        add.call("#{stream}.ff.norm", checked_count(rows, config.dim))
        add.call("#{stream}.ff.modulated", checked_count(rows, config.dim))
        add.call("#{stream}.ff.hidden", checked_count(rows, config.ffn_dim))
        add.call("#{stream}.ff.activated", checked_count(rows, config.ffn_dim))
        add.call("#{stream}.ff.projected", checked_count(rows, config.dim))
        add.call("#{stream}.output", checked_count(rows, config.dim))
      end

      joint_count = checked_count(config.joint_rows, config.dim)
      {"joint.query", "joint.key", "joint.value", "joint.attention"}.each do |name|
        add.call(name, joint_count)
      end
      plan
    end

    def self.prepare_scratch!(arena : Arena, config : HyNomasConfig) : Nil
      plan = scratch_plan(config)
      names = plan.map(&.[0])
      if names.any? { |name| arena.allocated?(name) }
        raise ArgumentError.new("Hy-Nomas scratch Arena already contains a planned buffer")
      end
      created = [] of String
      begin
        plan.each do |name, count|
          if name == "hy_nomas.zero_head_bias"
            arena.upload(name, Array(Float32).new(count, 0.0_f32))
          else
            arena.alloc(name, count)
          end
          created << name
        end
      rescue ex
        created.reverse_each { |name| arena.remove(name) }
        raise ex
      end
    end

    def initialize(@config : HyNomasConfig, @weights : Arena, @scratch : Arena)
      @config.validate!
      @bindings = self.class.checkpoint_bindings(@config)
      @scratch_plan = self.class.scratch_plan(@config)
      validate_weights!
      validate_scratch!
      @scratch_buffers = @scratch_plan.map { |entry| @scratch[entry[0]] }
      @zero = @scratch["hy_nomas.zero_head_bias"]
      validate_scratch_weight_aliases!
      @image_weights = StreamWeights.new(@weights, "img")
      @text_weights = StreamWeights.new(@weights, "txt")
      @image_buffers = StreamBuffers.new(@scratch, "img", true)
      @text_buffers = StreamBuffers.new(@scratch, "txt", false)
      @joint_query = @scratch["hy_nomas.joint.query"]
      @joint_key = @scratch["hy_nomas.joint.key"]
      @joint_value = @scratch["hy_nomas.joint.value"]
      @joint_attention = @scratch["hy_nomas.joint.attention"]
      self.class.prepare!
      @image_output = @image_buffers.output
      @text_output = @text_buffers.output
      @checkpoints = build_checkpoints
    end

    # Enqueue one complete block into the existing encoder. The caller owns
    # command submission, completion and any later diagnostic readback.
    def encode(e : ML::Metal::ComputeEncoder, img : ML::MetalBuffer,
               txt : ML::MetalBuffer, vec : ML::MetalBuffer,
               vec_txt : ML::MetalBuffer, video_rope_cos : ML::MetalBuffer,
               video_rope_sin : ML::MetalBuffer) : Tuple(ML::MetalBuffer, ML::MetalBuffer)
      validate_inputs!(img, txt, vec, vec_txt, video_rope_cos, video_rope_sin)

      c = @config
      img_rows = c.image_rows
      txt_rows = c.text_rows
      img_mod_rows = c.effective_img_modulation_rows
      txt_mod_rows = c.effective_txt_modulation_rows
      encode_modulation(e, @image_buffers, @image_weights, vec, img_mod_rows, img_rows, c.video_tokens)
      encode_modulation(e, @text_buffers, @text_weights, vec_txt, txt_mod_rows, txt_rows, c.text_tokens)

      encode_stream_qkv(e, @image_buffers, @image_weights, img, img_rows, c.video_tokens,
        video_rope_cos, video_rope_sin)
      encode_stream_qkv(e, @text_buffers, @text_weights, txt, txt_rows, c.text_tokens)

      concat(e, @image_buffers.attn_q_rope.not_nil!, @text_buffers.attn_q_norm, @joint_query, c.video_tokens, c.text_tokens)
      concat(e, @image_buffers.attn_k_rope.not_nil!, @text_buffers.attn_k_norm, @joint_key, c.video_tokens, c.text_tokens)
      concat(e, @image_buffers.attn_v, @text_buffers.attn_v, @joint_value, c.video_tokens, c.text_tokens)
      MetalOps.attention(e, @joint_query, @joint_key, @joint_value, @joint_attention,
        c.joint_tokens, c.joint_tokens, c.heads, c.head_dim, c.batches)
      split(e, @joint_attention, @image_buffers.attn_raw, @text_buffers.attn_raw, c.video_tokens, c.text_tokens)

      encode_attention_residual(e, img, @image_buffers, @image_weights, img_rows)
      encode_attention_residual(e, txt, @text_buffers, @text_weights, txt_rows)
      encode_feed_forward(e, @image_buffers, @image_weights, img_rows)
      encode_feed_forward(e, @text_buffers, @text_weights, txt_rows)
      {@image_output, @text_output}
    end

    private def self.append_binding(bindings : Array(WeightBinding), target : String,
                                    source : String, shape : Array(Int64),
                                    dtype : ML::GGUF::SafeTensorDType?) : Nil
      bindings << WeightBinding.new(target, source, shape, dtype)
    end

    def self.pipeline(name : String) : ML::Metal::ComputePipeline
      ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
    end

    private def validate_weights! : Nil
      @bindings.each do |binding|
        count = binding.shape.reduce(1_i64) { |n, d| n * d }
        dtype = @weights.dtype(binding.target)
        unless dtype.f32? || dtype.bf16?
          raise ArgumentError.new("Hy-Nomas weights must be resident as F32 or BF16: #{binding.target}")
        end
        buffer = @weights[binding.target]
        MetalOps.check_storage(buffer, count, dtype)
        expected_bytes = count * dtype.bytes_per_element
        unless buffer.size == expected_bytes
          raise ArgumentError.new("Hy-Nomas weight buffer has an unexpected extent: #{binding.target}")
        end
      end
    end

    private def validate_scratch! : Nil
      @scratch_plan.each do |name, count|
        buffer = @scratch.buffers[name]?
        raise ArgumentError.new("Missing Hy-Nomas scratch buffer #{name}") unless buffer
        MetalOps.check(buffer, count)
        unless buffer.size == count.to_i64 * sizeof(Float32)
          raise ArgumentError.new("Hy-Nomas scratch buffer has an unexpected extent: #{name}")
        end
      end
      @scratch_plan.each_with_index do |(name, _), i|
        a = @scratch[name]
        other_index = 0
        while other_index < i
          other_name = @scratch_plan[other_index][0]
          if MetalOps.overlap?(a, @scratch[other_name])
            raise ArgumentError.new("Hy-Nomas scratch buffers must not alias")
          end
          other_index += 1
        end
      end
    end

    private def validate_scratch_weight_aliases! : Nil
      @bindings.each do |binding|
        weight = @weights[binding.target]
        @scratch_plan.each do |name, _|
          if MetalOps.overlap?(weight, @scratch[name])
            raise ArgumentError.new("Hy-Nomas scratch and checkpoint weights must not alias")
          end
        end
      end
    end

    private def validate_inputs!(img : ML::MetalBuffer, txt : ML::MetalBuffer,
                                 vec : ML::MetalBuffer, vec_txt : ML::MetalBuffer,
                                 cos : ML::MetalBuffer, sin : ML::MetalBuffer) : Nil
      c = @config
      exact_f32(img, c.image_rows.to_i64 * c.dim)
      exact_f32(txt, c.text_rows.to_i64 * c.dim)
      exact_f32(vec, c.effective_img_modulation_rows.to_i64 * c.dim)
      exact_f32(vec_txt, c.effective_txt_modulation_rows.to_i64 * c.dim)
      rope_count = c.video_tokens.to_i64 * c.head_dim
      exact_f32(cos, rope_count)
      exact_f32(sin, rope_count)
      validate_input_scratch_alias!(img)
      validate_input_scratch_alias!(txt)
      validate_input_scratch_alias!(vec)
      validate_input_scratch_alias!(vec_txt)
      validate_input_scratch_alias!(cos)
      validate_input_scratch_alias!(sin)
    end

    private def validate_input_scratch_alias!(input : ML::MetalBuffer) : Nil
      @scratch_buffers.each do |scratch_buffer|
        if MetalOps.overlap?(scratch_buffer, input)
          raise ArgumentError.new("Hy-Nomas scratch must not alias block inputs")
        end
      end
    end

    private def exact_f32(buffer : ML::MetalBuffer, count : Int64) : Nil
      MetalOps.check(buffer, count)
      unless buffer.size == count * sizeof(Float32)
        raise ArgumentError.new("Hy-Nomas input buffer has an unexpected extent")
      end
    end

    private def build_checkpoints : Hash(String, ML::MetalBuffer)
      result = {} of String => ML::MetalBuffer
      result["img.modulation"] = @image_buffers.modulation
      result["img.attn.norm"] = @image_buffers.attn_norm
      result["img.attn.modulated"] = @image_buffers.attn_modulated
      result["img.attn.q"] = @image_buffers.attn_q
      result["img.attn.k"] = @image_buffers.attn_k
      result["img.attn.v"] = @image_buffers.attn_v
      result["img.attn.q_norm"] = @image_buffers.attn_q_norm
      result["img.attn.k_norm"] = @image_buffers.attn_k_norm
      result["img.attn.q_rope"] = @image_buffers.attn_q_rope.not_nil!
      result["img.attn.k_rope"] = @image_buffers.attn_k_rope.not_nil!
      result["joint.query"] = @joint_query
      result["joint.key"] = @joint_key
      result["joint.value"] = @joint_value
      result["joint.attention"] = @joint_attention
      result["img.attn.raw"] = @image_buffers.attn_raw
      result["img.attn.projected"] = @image_buffers.attn_projected
      result["img.after_attention"] = @image_buffers.after_attention
      result["img.ff.norm"] = @image_buffers.ff_norm
      result["img.ff.modulated"] = @image_buffers.ff_modulated
      result["img.ff.hidden"] = @image_buffers.ff_hidden
      result["img.ff.activated"] = @image_buffers.ff_activated
      result["img.ff.projected"] = @image_buffers.ff_projected
      result["img.output"] = @image_buffers.output
      result["txt.modulation"] = @text_buffers.modulation
      result["txt.attn.norm"] = @text_buffers.attn_norm
      result["txt.attn.modulated"] = @text_buffers.attn_modulated
      result["txt.attn.q"] = @text_buffers.attn_q
      result["txt.attn.k"] = @text_buffers.attn_k
      result["txt.attn.v"] = @text_buffers.attn_v
      result["txt.attn.q_norm"] = @text_buffers.attn_q_norm
      result["txt.attn.k_norm"] = @text_buffers.attn_k_norm
      result["txt.attn.raw"] = @text_buffers.attn_raw
      result["txt.attn.projected"] = @text_buffers.attn_projected
      result["txt.after_attention"] = @text_buffers.after_attention
      result["txt.ff.norm"] = @text_buffers.ff_norm
      result["txt.ff.modulated"] = @text_buffers.ff_modulated
      result["txt.ff.hidden"] = @text_buffers.ff_hidden
      result["txt.ff.activated"] = @text_buffers.ff_activated
      result["txt.ff.projected"] = @text_buffers.ff_projected
      result["txt.output"] = @text_buffers.output
      result
    end

    private def encode_modulation(e : ML::Metal::ComputeEncoder, buffers : StreamBuffers,
                                  weights : StreamWeights, vector : ML::MetalBuffer,
                                  mod_rows : Int32, rows : Int32, tokens : Int32) : Nil
      c = @config
      e.set_pipeline(self.class.pipeline("hy_nomas_silu_f32"))
      e.set_buffer(vector, 0).set_buffer(buffers.vec_silu, 1, ML::Metal::BufferAccess::Write)
      e.set_value((mod_rows * c.dim).to_u32, 2)
      e.dispatch_1d(mod_rows * c.dim)
      linear(e, buffers.vec_silu, weights.modulation, buffers.modulation, mod_rows, c.dim, 6 * c.dim)
      unpack_modulation(e, buffers.modulation, buffers, mod_rows, rows, tokens)
    end

    private def encode_stream_qkv(e : ML::Metal::ComputeEncoder, buffers : StreamBuffers,
                                  weights : StreamWeights, input : ML::MetalBuffer,
                                  rows : Int32, tokens : Int32,
                                  cos : ML::MetalBuffer? = nil,
                                  sin : ML::MetalBuffer? = nil) : Nil
      c = @config
      MetalOps.norm(e, input, @zero, @zero, buffers.attn_norm, rows, c.dim, c.norm_eps)
      MetalOps.modulate(e, buffers.attn_norm, buffers.attn_shift, buffers.attn_scale, buffers.attn_modulated,
        rows, c.dim, rows)
      linear(e, buffers.attn_modulated, weights.query, buffers.attn_q, rows, c.dim, c.dim)
      linear(e, buffers.attn_modulated, weights.key, buffers.attn_k, rows, c.dim, c.dim)
      linear(e, buffers.attn_modulated, weights.value, buffers.attn_v, rows, c.dim, c.dim)
      MetalOps.norm(e, buffers.attn_q, weights.query_norm.weight, @zero, buffers.attn_q_norm,
        rows * c.heads, c.head_dim, c.norm_eps, rms: true, affine: true,
        weight_dtype: weights.query_norm.dtype)
      MetalOps.norm(e, buffers.attn_k, weights.key_norm.weight, @zero, buffers.attn_k_norm,
        rows * c.heads, c.head_dim, c.norm_eps, rms: true, affine: true,
        weight_dtype: weights.key_norm.dtype)
      unless cos.nil? && sin.nil?
        cosine = cos || raise ArgumentError.new("Hy-Nomas video RoPE requires both cosine and sine tables")
        sine = sin || raise ArgumentError.new("Hy-Nomas video RoPE requires both cosine and sine tables")
        q_rope = buffers.attn_q_rope
        k_rope = buffers.attn_k_rope
        raise ArgumentError.new("Hy-Nomas RoPE scratch is unavailable for this stream") unless q_rope && k_rope
        rope(e, buffers.attn_q_norm, q_rope.not_nil!, cosine, sine, tokens, rows)
        rope(e, buffers.attn_k_norm, k_rope.not_nil!, cosine, sine, tokens, rows)
      end
    end

    private def unpack_modulation(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
                                  buffers : StreamBuffers, mod_rows : Int32,
                                  rows : Int32, tokens : Int32) : Nil
      c = @config
      e.set_pipeline(self.class.pipeline("hy_nomas_unpack_modulation_f32"))
      e.set_buffer(input, 0)
      e.set_buffer(buffers.attn_shift, 1, ML::Metal::BufferAccess::Write)
      e.set_buffer(buffers.attn_scale, 2, ML::Metal::BufferAccess::Write)
      e.set_buffer(buffers.attn_gate, 3, ML::Metal::BufferAccess::Write)
      e.set_buffer(buffers.ff_shift, 4, ML::Metal::BufferAccess::Write)
      e.set_buffer(buffers.ff_scale, 5, ML::Metal::BufferAccess::Write)
      e.set_buffer(buffers.ff_gate, 6, ML::Metal::BufferAccess::Write)
      e.set_value(rows.to_u32, 7).set_value(tokens.to_u32, 8).set_value(c.dim.to_u32, 9)
      e.set_value(mod_rows.to_u32, 10).set_value(c.batches.to_u32, 11)
      e.dispatch_1d(rows * c.dim)
    end

    private def rope(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
                     output : ML::MetalBuffer, cos : ML::MetalBuffer,
                     sin : ML::MetalBuffer,
                     tokens : Int32, rows : Int32) : Nil
      c = @config
      e.set_pipeline(self.class.pipeline("hy_nomas_rope_f32"))
      e.set_buffer(input, 0).set_buffer(cos, 1).set_buffer(sin, 2)
      e.set_buffer(output, 3, ML::Metal::BufferAccess::Write)
      e.set_value(tokens.to_u32, 4).set_value(c.heads.to_u32, 5).set_value(c.head_dim.to_u32, 6)
      e.set_value(c.batches.to_u32, 7)
      e.dispatch_1d(rows * c.dim)
    end

    private def concat(e : ML::Metal::ComputeEncoder, left : ML::MetalBuffer,
                       right : ML::MetalBuffer, output : ML::MetalBuffer,
                       left_tokens : Int32, right_tokens : Int32) : Nil
      c = @config
      e.set_pipeline(self.class.pipeline("hy_nomas_concat_f32"))
      e.set_buffer(left, 0).set_buffer(right, 1)
      e.set_buffer(output, 2, ML::Metal::BufferAccess::Write)
      e.set_value(c.batches.to_u32, 3).set_value(left_tokens.to_u32, 4)
      e.set_value(right_tokens.to_u32, 5).set_value(c.dim.to_u32, 6)
      e.set_value(c.attention_precision.bf16_boundary_control? ? 1_u32 : 0_u32, 7)
      e.dispatch_1d(c.joint_rows * c.dim)
    end

    private def split(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
                      img : ML::MetalBuffer, txt : ML::MetalBuffer,
                      img_tokens : Int32, txt_tokens : Int32) : Nil
      c = @config
      e.set_pipeline(self.class.pipeline("hy_nomas_split_f32"))
      round_boundary = c.attention_precision.bf16_boundary_control?
      input_access = round_boundary ? ML::Metal::BufferAccess::ReadWrite : ML::Metal::BufferAccess::Read
      e.set_buffer(input, 0, input_access).set_buffer(img, 1, ML::Metal::BufferAccess::Write)
      e.set_buffer(txt, 2, ML::Metal::BufferAccess::Write)
      e.set_value(c.batches.to_u32, 3).set_value(img_tokens.to_u32, 4)
      e.set_value(txt_tokens.to_u32, 5).set_value(c.dim.to_u32, 6)
      e.set_value(round_boundary ? 1_u32 : 0_u32, 7)
      e.dispatch_1d(c.joint_rows * c.dim)
    end

    private def encode_attention_residual(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
                                          buffers : StreamBuffers, weights : StreamWeights,
                                          rows : Int32) : Nil
      c = @config
      linear(e, buffers.attn_raw, weights.projection, buffers.attn_projected, rows, c.dim, c.dim)
      MetalOps.add(e, input, buffers.attn_projected, buffers.after_attention,
        rows * c.dim, buffers.attn_gate, c.dim, rows)
    end

    private def encode_feed_forward(e : ML::Metal::ComputeEncoder, buffers : StreamBuffers,
                                    weights : StreamWeights, rows : Int32) : Nil
      c = @config
      MetalOps.norm(e, buffers.after_attention, @zero, @zero, buffers.ff_norm,
        rows, c.dim, c.norm_eps)
      MetalOps.modulate(e, buffers.ff_norm, buffers.ff_shift, buffers.ff_scale,
        buffers.ff_modulated, rows, c.dim, rows)
      linear(e, buffers.ff_modulated, weights.mlp_fc1, buffers.ff_hidden, rows, c.dim, c.ffn_dim)
      MetalOps.gelu(e, buffers.ff_hidden, buffers.ff_activated, rows * c.ffn_dim, approximate: true)
      linear(e, buffers.ff_activated, weights.mlp_fc2, buffers.ff_projected, rows, c.ffn_dim, c.dim)
      MetalOps.add(e, buffers.after_attention, buffers.ff_projected, buffers.output,
        rows * c.dim, buffers.ff_gate, c.dim, rows)
    end

    private def linear(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
                       weights : LinearWeights, output : ML::MetalBuffer,
                       rows : Int32, in_dim : Int32, out_dim : Int32) : Nil
      MetalOps.linear(e, input, weights.weight, weights.bias, output, rows, in_dim, out_dim,
        weights.weight_dtype, weights.bias_dtype)
    end
  end
end
