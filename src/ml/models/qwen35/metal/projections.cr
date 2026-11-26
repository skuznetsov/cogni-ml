# Dense and quantized projections for the Qwen35 Metal engine.

module ML
  module GGUF
    module Qwen35Metal
      Q4K_BLOCK_BYTES    = 144
      Q5K_BLOCK_BYTES    = 176
      Q6K_BLOCK_BYTES    = 210
      Q8_0_BLOCK_BYTES   =  34
      IQ4_NL_BLOCK_BYTES =  18
      QK_K               = 256
      Q8_0_QK            =  32
      IQ4_NL_QK          =  32

      # GEMV (decode) tiling — must match the quant-specific kernels.
      MV_Q4_NSG             =  2
      MV_Q4_NR0             =  2
      MV_Q5_NSG             =  2
      MV_Q5_NR0             =  1
      MV_Q6_NSG             =  2
      MV_Q6_NR0             =  1
      MV_Q8_NSG             =  4
      MV_Q8_NR0             =  1
      MV_IQ4_NL_NSG         =  2
      MV_IQ4_NL_NR0         =  2
      MV_F32_NSG            =  4
      MV_F32_NR0            =  1

      # GEMM (prefill) tiling — Q4_K only for now.
      MM_NR0          =    64
      MM_NR1          =    32
      MM_TG           =   128 # threads per threadgroup (4 simdgroups × 32)
      MM_SHMEM        = 12288 # bytes: 2 × (MM_SA_SIZE + MM_SB_SIZE) = 2 × 6144
      MM48_NR1        =    48
      MM48_TG         =   192 # threads per threadgroup (6 simdgroups × 32)
      MM48_SHMEM      = 14336 # bytes: 2 × (4096 + 3072), larger than 64×48 f32 edge scratch
      MM64_NR1        =    64
      MM64_TG         =   256 # threads per threadgroup (8 simdgroups × 32)
      MM64_SHMEM      = 16384 # bytes: 2 × (MM64_SA_SIZE + MM64_SB_SIZE)
      MM80_NR1        =    80
      MM80_TG         =   320 # threads per threadgroup (10 simdgroups × 32)
      MM80_SHMEM      = 20480 # bytes: max double-buffered tile and 64×80 f32 edge scratch
      MM96_NR1        =    96
      MM96_TG         =   384 # threads per threadgroup (12 simdgroups × 32)
      MM96_SHMEM      = 24576 # bytes: max double-buffered tile and 64×96 f32 edge scratch
      MM112_NR1       =   112
      MM112_TG        =   448 # threads per threadgroup (14 simdgroups × 32)
      MM112_SHMEM     = 28672 # bytes: max double-buffered tile and 64×112 f32 edge scratch
      MM128_SG8_NR1   =   128
      MM128_SG8_TG    =   256 # 8 simdgroups; each keeps two 16-row accumulator bands
      MM128_SG8_SHMEM = 24576 # bytes: 2 × (4096-byte weights + 8192-byte inputs)
      Q4_TENSOR_NR1   =   128
      Q4_TENSOR_TG    =   128 # 4 simdgroups × 32, matches simd_mm_q4k_tensor_f32out
      Q4_TENSOR_SHMEM =  4096 # one 64×32 H16 dequantized A tile

      # Above this batch, use GEMM. At or below, GEMV is faster.
      # The default is deliberately conservative; small speculative verifier
      # chunks can lower it via env for bounded A/B without changing the normal
      # prefill/decode policy.
      GEMM_BATCH_THRESHOLD = ENV["QWEN35_GEMM_BATCH_THRESHOLD"]?.try(&.to_i?) || 8

      # Reusing one F32->F16 activation conversion across FFN gate/up now
      # pays even at pp64 after the later H16 routing cleanups.
      Q4_PAIR_H16_MIN_BATCH = 64

      # Admit the tail-safe B64 FFN fusion automatically only when its padded
      # row count stays within 12.5% of the live batch. Explicit overrides retain
      # the historical min-batch semantics for experiments and rollback.
      def self.q4_h16_b64_tail_policy?(batch : Int32,
                                       device_name : String,
                                       override : String?) : Bool
        if raw = override
          min_batch = raw.to_i32
          return min_batch > 0 && batch >= min_batch
        end

        return false unless device_name == "Apple M2 Max" && batch >= 96
        live_batch = batch.to_i64
        padded_batch = ((live_batch + MM64_NR1 - 1) // MM64_NR1) * MM64_NR1
        padded_batch * 8_i64 <= live_batch * 9_i64
      end

      # M2 Max Qwen3.8 prefill alternative to the refuted 16-simdgroup B128
      # tile. Keep the threadgroup at 8 simdgroups and let each simdgroup
      # accumulate two independent 16-row bands. Automatic admission stays on
      # measured Qwen3.8 projection shapes and pp256+ exact tiles; explicit "1"
      # retains the wider exact-shape experiment, while "0" is the immediate
      # rollback.
      def self.q4_h16_b128_sg8_policy?(batch : Int32,
                                       in_dim : Int32,
                                       out_dim : Int32,
                                       device_name : String,
                                       override : String?) : Bool
        measured_shape = in_dim == 5120 && {1024, 6144, 10240, 12288, 17408}.includes?(out_dim)
        enabled = case override
                  when nil
                    device_name == "Apple M2 Max" && batch >= 256 && measured_shape
                  when "0" then false
                  when "1" then true
                  else
                    raise ArgumentError.new("QWEN35_Q4K_H16_B128_SG8 must be 0 or 1")
                  end
        enabled && batch >= MM128_SG8_NR1 && batch % MM128_SG8_NR1 == 0 &&
          in_dim % QK_K == 0 && out_dim % MM_NR0 == 0
      end

      # Convert one normalized activation row set to H16 once and reuse it
      # across sibling Q/K/V or recurrent projections. Automatic admission is
      # limited to the measured M2 Max route; 0 is rollback and 1 is an
      # explicit cross-device experiment.
      # The established per-matmul routes already round these inputs to H16;
      # this changes only where that identical conversion is materialized.
      def self.prefill_shared_projection_h16_policy?(qws : Array(QuantWeight),
                                                     batch : Int32,
                                                     checkpoint_requested : Bool,
                                                     device_name : String,
                                                     override : String?) : Bool
        enabled = case override
                  when "1" then true
                  when "0" then false
                  when nil then device_name == "Apple M2 Max"
                  else
                    raise ArgumentError.new("QWEN35_PREFILL_SHARED_PROJECTION_H16 must be 0 or 1")
                  end
        return false unless enabled && !checkpoint_requested && batch >= 256 && qws.size >= 2

        in_dim = qws[0].in_dim
        qws.all? do |qw|
          next false unless qw.in_dim == in_dim
          case qw.type
          when TensorType::Q4_K
            qw.out_dim > 64
          when TensorType::Q5_K, TensorType::Q6_K
            true
          else
            false
          end
        end
      end

      {% if flag?(:cpu_only) %}
        def self.matmul_q4k(x : Array(Float32),
                            w_raw : Bytes,
                            in_dim : Int32,
                            out_dim : Int32,
                            batch : Int32) : Array(Float32)
          raise "Metal disabled (cpu_only)"
        end

        def self.matmul_q5k(x : Array(Float32),
                            w_raw : Bytes,
                            in_dim : Int32,
                            out_dim : Int32,
                            batch : Int32) : Array(Float32)
          raise "Metal disabled (cpu_only)"
        end

        def self.matmul_q6k(x : Array(Float32),
                            w_raw : Bytes,
                            in_dim : Int32,
                            out_dim : Int32,
                            batch : Int32) : Array(Float32)
          raise "Metal disabled (cpu_only)"
        end

        def self.bf16_gemv(w_raw : Bytes,
                           in_dim : Int32,
                           out_dim : Int32,
                           x : Array(Float32)) : Array(Float32)
          raise "Metal disabled (cpu_only)"
        end

        def self.bf16_q_gate_gemv(w_raw : Bytes,
                                  in_dim : Int32,
                                  q_dim : Int32,
                                  head_dim : Int32,
                                  x : Array(Float32)) : Array(Float32)
          raise "Metal disabled (cpu_only)"
        end

        def self.clear_bf16_weight_cache : Nil
        end
      {% else %}
        GEMM_Q4K_SOURCE  = {{ read_file("#{__DIR__}/../../../metal/kernels/shared/gemm_q4k.metal") }}
        GEMM_Q56K_SOURCE = {{ read_file("#{__DIR__}/../../../metal/kernels/shared/gemm_q56k.metal") }}
        GEMM_MM_SOURCE   = {{ read_file("#{__DIR__}/../../../metal/kernels/shared/gemm_mm.metal") }}

        @@mv_pipeline   : ML::Metal::ComputePipeline?
        @@mv_q4_x16_pipeline : ML::Metal::ComputePipeline?
        @@mv_q4_dual_pipeline : ML::Metal::ComputePipeline?
        @@mv_add_pipeline : ML::Metal::ComputePipeline?

        @@mm_pipeline   : ML::Metal::ComputePipeline?
        @@mm_h16_pipeline : ML::Metal::ComputePipeline?
        @@mm_h16_b48_pipeline : ML::Metal::ComputePipeline?
        @@mm_h16_b64_pipeline : ML::Metal::ComputePipeline?
        @@mm_h16_b128_sg8_pipeline : ML::Metal::ComputePipeline?
        @@mm_h16_b128_sg8_swiglu_h16_pipeline : ML::Metal::ComputePipeline?
        @@mm_h16_b64_swiglu_pipeline : ML::Metal::ComputePipeline?
        @@mm_h16_b64_gelu_mul_pipeline : ML::Metal::ComputePipeline?
        @@mm_h16_b64_swiglu_h16_pipeline : ML::Metal::ComputePipeline?
        @@mm_h16_b80_pipeline : ML::Metal::ComputePipeline?
        @@mm_h16_b96_pipeline : ML::Metal::ComputePipeline?
        @@mm_h16_b112_pipeline : ML::Metal::ComputePipeline?
        @@mm_q4_tensor_f32out_pipeline : ML::Metal::ComputePipeline?
        @@mm5_pipeline  : ML::Metal::ComputePipeline?
        @@mm6_pipeline  : ML::Metal::ComputePipeline?
        @@mm5_f32out_pipeline : ML::Metal::ComputePipeline?
        @@mm6_f32out_pipeline : ML::Metal::ComputePipeline?
        @@mm6_f32out_add_pipeline : ML::Metal::ComputePipeline?
        @@mv5_pipeline  : ML::Metal::ComputePipeline?
        @@mv6_pipeline  : ML::Metal::ComputePipeline?
        @@mv6_add_pipeline : ML::Metal::ComputePipeline?
        @@mv8_pipeline  : ML::Metal::ComputePipeline?
        @@mv8_add_pipeline : ML::Metal::ComputePipeline?
        @@mv8_dual_pipeline : ML::Metal::ComputePipeline?
        @@mv_iq4_nl_pipeline : ML::Metal::ComputePipeline?
        @@mv_f32_pipeline : ML::Metal::ComputePipeline?
        @@bf16_gemv_pipeline : ML::Metal::ComputePipeline?
        @@bf16_q_gate_gemv_pipeline : ML::Metal::ComputePipeline?

        # Per-slot tags for `matmul_many` output buffers. Multiple outputs
        # are bound into one compute encoder, so they must not alias.
        # Add more tags here if a batched dispatch ever grows past 8.
        MANY_SLOT_TAGS = [
          :mv_many_out_0,
          :mv_many_out_1,
          :mv_many_out_2,
          :mv_many_out_3,
          :mv_many_out_4,
          :mv_many_out_5,
          :mv_many_out_6,
          :mv_many_out_7,
        ]

        @@bf16_weight_buffers = {} of String => ML::MetalBuffer
        @@bf16_weight_mutex = Mutex.new
        @@q4_fused_bench_mutex = Mutex.new

        private def self.mv_pipeline : ML::Metal::ComputePipeline
          @@mv_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q4k_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q4k_f32", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mv_q4_x16_pipeline : ML::Metal::ComputePipeline
          @@mv_q4_x16_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q4k_f32_x16") {
            ML::Metal::ComputePipeline.new("simd_mv_q4k_f32_x16", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mv_q4_dual_pipeline : ML::Metal::ComputePipeline
          @@mv_q4_dual_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q4k_dual_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q4k_dual_f32", GEMM_Q4K_SOURCE)
          }
        end

        private def self.normalized_route_tag(qw : QuantWeight?) : String?
          tag = qw.try(&.route_tag)
          tag.try { |value| value.gsub(/\.\d+\./, ".*.") }
        end

        # Default x16 routing is deliberately operator-, shape-, and model-family
        # scoped. Gemma4 keeps its established route-tag default. Qwen uses x16
        # only for the measured Qwen3.8-27B FFN and recurrent projections;
        # full-attention and unrelated Q4_K projections retain the base layout.
        def self.q4_gemv_x16_default_route?(route_tag : String?,
                                             in_dim : Int32,
                                             out_dim : Int32,
                                             capability : Q4GemvX16Capability) : Bool
          return false unless route_tag
          return true if route_tag.starts_with?("gemma4:")
          return false unless route_tag.starts_with?("qwen35:")
          return false unless capability.qwen38?

          shape = {in_dim, out_dim}
          return true if shape == {5120, 17408} &&
                         (route_tag.ends_with?(".ffn_gate.weight") ||
                           route_tag.ends_with?(".ffn_up.weight"))
          return true if shape == {5120, 10240} && route_tag.ends_with?(".attn_qkv.weight")
          return true if shape == {5120, 6144} && route_tag.ends_with?(".attn_gate.weight")
          return true if shape == {5120, 48} &&
                         (route_tag.ends_with?(".ssm_alpha.weight") ||
                           route_tag.ends_with?(".ssm_beta.weight"))
          return true if shape == {6144, 5120} && route_tag.ends_with?(".ssm_out.weight")

          false
        end

        private def self.q4_gemv_x16_enabled?(qw : QuantWeight?) : Bool
          if value = ENV["QWEN35_Q4K_GEMV_X16"]?
            return true if value == "1"
            return false if value == "0"
          end
          return false if ENV["QWEN35_Q4K_GEMV_X16_OFF"]? == "1"
          return false if ENV["GEMMA4_Q4K_GEMV_X16_OFF"]? == "1"
          tag = normalized_route_tag(qw)
          if qw && q4_gemv_x16_default_route?(tag, qw.in_dim, qw.out_dim, qw.q4_gemv_x16_capability)
            return true
          end
          return false unless patterns = ENV["QWEN35_Q4K_GEMV_X16_TAGS"]?
          return false unless tag

          patterns.split(',').any? do |pattern|
            needle = pattern.strip
            !needle.empty? && tag.includes?(needle)
          end
        end

        private def self.mv_q4_layout_pipeline(nsg : Int32, nr0 : Int32) : ML::Metal::ComputePipeline
          raise ArgumentError.new("Q4 layout nsg must be positive") unless nsg > 0
          raise ArgumentError.new("Q4 layout nr0 must be positive") unless nr0 > 0

          key = "simd_mv_q4k_f32_layout_nsg#{nsg}_nr#{nr0}"
          ML::Metal::PipelineCache.get(key) do
            source = GEMM_Q4K_SOURCE
              .gsub("constant short MV_NSG = 2;", "constant short MV_NSG = #{nsg};")
              .gsub("constant short MV_NR0 = 2;", "constant short MV_NR0 = #{nr0};")
            ML::Metal::ComputePipeline.new("simd_mv_q4k_f32", source)
          end
        end

        private def self.q4_gemv_shape_layout_enabled?(qw : QuantWeight?) : Bool
          return false if ENV["QWEN35_Q4K_GEMV_SHAPE_LAYOUT_OFF"]? == "1"
          if value = ENV["QWEN35_Q4K_GEMV_SHAPE_LAYOUT"]?
            return value != "0"
          end

          # Gemma4 decode corridors benchmark faster on the baseline Q4_K GEMV
          # route than on the older shared shape-layout table. Keep Qwen's
          # historic default while avoiding a hidden process-wide env mutation
          # in Gemma profile/benchmark surfaces.
          return false if qw.try(&.route_tag).try(&.starts_with?("gemma4:"))

          true
        end

        private def self.q4_gemv_shape_layout(in_dim : Int32, out_dim : Int32) : {Int32, Int32}?
          case {in_dim, out_dim}
          when {15360, 3840}
            {2, 3}
          when {3840, 4096}
            {2, 3}
          when {4096, 3840}
            {1, 3}
          when {3840, 2048}
            {4, 2}
          else
            nil
          end
        end

        private def self.q6_gemv_shape_layout_enabled? : Bool
          ENV["QWEN35_Q6K_GEMV_SHAPE_LAYOUT"]? == "1" &&
            ENV["QWEN35_Q6K_GEMV_SHAPE_LAYOUT_OFF"]? != "1"
        end

        private def self.q6_gemv_shape_layout(in_dim : Int32, out_dim : Int32) : {Int32, Int32}?
          case {in_dim, out_dim}
          when {15360, 3840}
            {4, 2}
          else
            nil
          end
        end

        private def self.profile_route_tag(qw : QuantWeight?) : String
          return "" unless ENV["QWEN35_PROFILE_ROUTE_TAGS"]? == "1"
          " tag=#{normalized_route_tag(qw) || "untagged"}"
        end

        private def self.mv_add_pipeline : ML::Metal::ComputePipeline
          @@mv_add_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q4k_f32_add") {
            ML::Metal::ComputePipeline.new("simd_mv_q4k_f32_add", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_pipeline : ML::Metal::ComputePipeline
          @@mm_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_f32") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_f32", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_h16_pipeline : ML::Metal::ComputePipeline
          @@mm_h16_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_h16") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_h16", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_h16_b48_pipeline : ML::Metal::ComputePipeline
          @@mm_h16_b48_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_h16_b48") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_h16_b48", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_h16_b64_pipeline : ML::Metal::ComputePipeline
          @@mm_h16_b64_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_h16_b64") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_h16_b64", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_h16_b128_sg8_pipeline : ML::Metal::ComputePipeline
          @@mm_h16_b128_sg8_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_h16_b128_sg8") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_h16_b128_sg8", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_h16_b128_sg8_swiglu_h16_pipeline : ML::Metal::ComputePipeline
          @@mm_h16_b128_sg8_swiglu_h16_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_h16_b128_sg8_swiglu_h16") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_h16_b128_sg8_swiglu_h16", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_h16_b64_swiglu_pipeline : ML::Metal::ComputePipeline
          @@mm_h16_b64_swiglu_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_h16_b64_swiglu") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_h16_b64_swiglu", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_h16_b64_gelu_mul_pipeline : ML::Metal::ComputePipeline
          @@mm_h16_b64_gelu_mul_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_h16_b64_gelu_mul") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_h16_b64_gelu_mul", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_h16_b64_swiglu_h16_pipeline : ML::Metal::ComputePipeline
          @@mm_h16_b64_swiglu_h16_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_h16_b64_swiglu_h16") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_h16_b64_swiglu_h16", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_h16_b80_pipeline : ML::Metal::ComputePipeline
          @@mm_h16_b80_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_h16_b80") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_h16_b80", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_h16_b96_pipeline : ML::Metal::ComputePipeline
          @@mm_h16_b96_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_h16_b96") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_h16_b96", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_h16_b112_pipeline : ML::Metal::ComputePipeline
          @@mm_h16_b112_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_h16_b112") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_h16_b112", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_q4_tensor_f32out_pipeline : ML::Metal::ComputePipeline
          @@mm_q4_tensor_f32out_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_tensor_f32out") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_tensor_f32out", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm5_pipeline : ML::Metal::ComputePipeline
          @@mm5_pipeline ||= ML::Metal::PipelineCache.get("qwen35_simd_mm_q5k") {
            ML::Metal::ComputePipeline.new("simd_mm_q5k", GEMM_MM_SOURCE)
          }
        end

        private def self.mm6_pipeline : ML::Metal::ComputePipeline
          @@mm6_pipeline ||= ML::Metal::PipelineCache.get("qwen35_simd_mm_q6k") {
            ML::Metal::ComputePipeline.new("simd_mm_q6k", GEMM_MM_SOURCE)
          }
        end

        private def self.mm5_f32out_pipeline : ML::Metal::ComputePipeline
          @@mm5_f32out_pipeline ||= ML::Metal::PipelineCache.get("qwen35_simd_mm_q5k_f32out") {
            ML::Metal::ComputePipeline.new("simd_mm_q5k_f32out", GEMM_MM_SOURCE)
          }
        end

        private def self.mm6_f32out_pipeline : ML::Metal::ComputePipeline
          @@mm6_f32out_pipeline ||= ML::Metal::PipelineCache.get("qwen35_simd_mm_q6k_f32out") {
            ML::Metal::ComputePipeline.new("simd_mm_q6k_f32out", GEMM_MM_SOURCE)
          }
        end

        private def self.mm6_f32out_add_pipeline : ML::Metal::ComputePipeline
          @@mm6_f32out_add_pipeline ||= ML::Metal::PipelineCache.get("qwen35_simd_mm_q6k_f32out_add") {
            ML::Metal::ComputePipeline.new("simd_mm_q6k_f32out_add", GEMM_MM_SOURCE)
          }
        end

        private def self.mv5_pipeline : ML::Metal::ComputePipeline
          @@mv5_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q5k_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q5k_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv6_pipeline : ML::Metal::ComputePipeline
          @@mv6_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q6k_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q6k_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv_q6_layout_pipeline(nsg : Int32, nr0 : Int32) : ML::Metal::ComputePipeline
          raise ArgumentError.new("Q6 layout nsg must be positive") unless nsg > 0
          raise ArgumentError.new("Q6 layout nr0 must be positive") unless nr0 > 0

          key = "simd_mv_q6k_f32_layout_nsg#{nsg}_nr#{nr0}"
          ML::Metal::PipelineCache.get(key) do
            source = GEMM_Q56K_SOURCE
              .gsub("constant short MV6_NSG = 2;", "constant short MV6_NSG = #{nsg};")
              .gsub("constant short MV6_NR0 = 1;", "constant short MV6_NR0 = #{nr0};")
            ML::Metal::ComputePipeline.new("simd_mv_q6k_f32", source)
          end
        end

        private def self.mv6_add_pipeline : ML::Metal::ComputePipeline
          @@mv6_add_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q6k_f32_add") {
            ML::Metal::ComputePipeline.new("simd_mv_q6k_f32_add", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv8_pipeline : ML::Metal::ComputePipeline
          @@mv8_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q8_0_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q8_0_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv8_add_pipeline : ML::Metal::ComputePipeline
          @@mv8_add_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q8_0_f32_add") {
            ML::Metal::ComputePipeline.new("simd_mv_q8_0_f32_add", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv8_dual_pipeline : ML::Metal::ComputePipeline
          @@mv8_dual_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q8_0_dual_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q8_0_dual_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv_iq4_nl_pipeline : ML::Metal::ComputePipeline
          @@mv_iq4_nl_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_iq4_nl_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_iq4_nl_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv_f32_pipeline : ML::Metal::ComputePipeline
          @@mv_f32_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_f32_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_f32_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.bf16_gemv_pipeline : ML::Metal::ComputePipeline
          @@bf16_gemv_pipeline ||= ML::Metal::PipelineCache.get("qwen35_bf16_gemv_f32") {
            ML::Metal::ComputePipeline.new("qwen35_bf16_gemv_f32", MTP_SOURCE)
          }
        end

        private def self.bf16_q_gate_gemv_pipeline : ML::Metal::ComputePipeline
          @@bf16_q_gate_gemv_pipeline ||= ML::Metal::PipelineCache.get("qwen35_bf16_q_gate_gemv_f32") {
            ML::Metal::ComputePipeline.new("qwen35_bf16_q_gate_gemv_f32", MTP_SOURCE)
          }
        end

        private def self.gemv_pipeline_for(qw : QuantWeight) : ML::Metal::ComputePipeline?
          case qw.type
          when .q4_k? then mv_pipeline
          when .q5_k? then mv5_pipeline
          when .q6_k? then mv6_pipeline
          when .q8_0? then mv8_pipeline
          when .iq4_nl? then mv_iq4_nl_pipeline
          when .f32? then mv_f32_pipeline
          else             nil
          end
        end

        private def self.gemv_add_pipeline_for(qw : QuantWeight) : ML::Metal::ComputePipeline?
          case qw.type
          when .q4_k? then mv_add_pipeline
          when .q6_k? then mv6_add_pipeline
          when .q8_0? then mv8_add_pipeline
          else             nil
          end
        end

        private def self.gemv_rows_per_tg_for(pipeline : ML::Metal::ComputePipeline) : Int32
          case pipeline
          when .same?(mv5_pipeline)
            MV_Q5_NSG * MV_Q5_NR0
          when .same?(mv6_pipeline), .same?(mv6_add_pipeline)
            MV_Q6_NSG * MV_Q6_NR0
          when .same?(mv8_pipeline), .same?(mv8_add_pipeline)
            MV_Q8_NSG * MV_Q8_NR0
          when .same?(mv_iq4_nl_pipeline)
            MV_IQ4_NL_NSG * MV_IQ4_NL_NR0
          when .same?(mv_f32_pipeline)
            MV_F32_NSG * MV_F32_NR0
          else
            MV_Q4_NSG * MV_Q4_NR0
          end
        end

        private def self.gemv_threads_per_tg_for(pipeline : ML::Metal::ComputePipeline) : Int32
          case pipeline
          when .same?(mv8_pipeline), .same?(mv8_add_pipeline), .same?(mv8_top1_tiles_pipeline), .same?(mv8_top2_tiles_pipeline)
            MV_Q8_NSG * 32
          when .same?(mv_iq4_nl_pipeline)
            MV_IQ4_NL_NSG * 32
          when .same?(mv_f32_pipeline)
            MV_F32_NSG * 32
          else
            64
          end
        end

        private def self.gemv_profile_quant(pipeline : ML::Metal::ComputePipeline) : {String, Int32, Int32}
          case pipeline
          when .same?(mv5_pipeline)
            {"Q5_K", Q5K_BLOCK_BYTES, QK_K}
          when .same?(mv6_pipeline), .same?(mv6_add_pipeline), .same?(mv6_top1_tiles_pipeline), .same?(mv6_top2_tiles_pipeline)
            {"Q6_K", Q6K_BLOCK_BYTES, QK_K}
          when .same?(mv8_pipeline), .same?(mv8_add_pipeline), .same?(mv8_top1_tiles_pipeline), .same?(mv8_top2_tiles_pipeline)
            {"Q8_0", Q8_0_BLOCK_BYTES, Q8_0_QK}
          when .same?(mv_iq4_nl_pipeline)
            {"IQ4_NL", IQ4_NL_BLOCK_BYTES, IQ4_NL_QK}
          when .same?(mv_f32_pipeline)
            {"F32", 4, 1}
          else
            {"Q4_K", Q4K_BLOCK_BYTES, QK_K}
          end
        end

        private def self.encode_gemv(enc : ML::Metal::ComputeEncoder | ML::Metal::GraphEncoder,
                                     pipeline : ML::Metal::ComputePipeline,
                                     x_buf : ML::MetalBuffer,
                                     out_buf : ML::MetalBuffer,
                                     w_buf : ML::MetalBuffer,
                                     w_offset : Int64,
                                     in_dim : Int32,
                                     out_dim : Int32,
                                     batch : Int32 = 1,
                                     profile_shape : Bool = true,
                                     route_qw : QuantWeight? = nil) : Nil
          if profile_shape
            quant_name, block_bytes, block_elems = gemv_profile_quant(pipeline)
            blocks_per_row = (in_dim + block_elems - 1) // block_elems
            weight_bytes = out_dim.to_i64 * blocks_per_row.to_i64 * block_bytes.to_i64
            Profile.bump_matmul_shape("gemv #{quant_name} #{in_dim}x#{out_dim} b#{batch}#{profile_route_tag(route_qw)}", weight_bytes)
          end
          actual_pipeline = pipeline
          rows_per_tg = gemv_rows_per_tg_for(pipeline)
          threads_per_tg = gemv_threads_per_tg_for(pipeline)
          if q4_gemv_x16_enabled?(route_qw) && batch <= GEMM_BATCH_THRESHOLD && pipeline.same?(mv_pipeline) && (in_dim % QK_K == 0)
            actual_pipeline = mv_q4_x16_pipeline
            rows_per_tg = MV_Q4_NSG * 2
            threads_per_tg = MV_Q4_NSG * 32
          elsif q4_gemv_shape_layout_enabled?(route_qw) && batch <= GEMM_BATCH_THRESHOLD && pipeline.same?(mv_pipeline)
            if layout = q4_gemv_shape_layout(in_dim, out_dim)
              nsg, nr0 = layout
              actual_pipeline = mv_q4_layout_pipeline(nsg, nr0)
              rows_per_tg = nsg * nr0
              threads_per_tg = nsg * 32
            end
          elsif q6_gemv_shape_layout_enabled? && batch <= GEMM_BATCH_THRESHOLD && pipeline.same?(mv6_pipeline)
            if layout = q6_gemv_shape_layout(in_dim, out_dim)
              nsg, nr0 = layout
              actual_pipeline = mv_q6_layout_pipeline(nsg, nr0)
              rows_per_tg = nsg * nr0
              threads_per_tg = nsg * 32
            end
          end

          enc.set_pipeline(actual_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32,  3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32,   5)
          grid = {(out_dim + rows_per_tg - 1) // rows_per_tg, batch, 1}
          enc.dispatch_threadgroups(grid, {threads_per_tg, 1, 1})
        end

        private def self.encode_gemv_layout(enc : ML::Metal::ComputeEncoder,
                                            pipeline : ML::Metal::ComputePipeline,
                                            x_buf : ML::MetalBuffer,
                                            out_buf : ML::MetalBuffer,
                                            w_buf : ML::MetalBuffer,
                                            w_offset : Int64,
                                            in_dim : Int32,
                                            out_dim : Int32,
                                            batch : Int32,
                                            rows_per_tg : Int32,
                                            threads_per_tg : Int32) : Nil
          enc.set_pipeline(pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32, 5)
          grid = {(out_dim + rows_per_tg - 1) // rows_per_tg, batch, 1}
          enc.dispatch_threadgroups(grid, {threads_per_tg, 1, 1})
        end

        private def self.encode_gemv_add(enc : ML::Metal::ComputeEncoder,
                                         pipeline : ML::Metal::ComputePipeline,
                                         x_buf : ML::MetalBuffer,
                                         residual_buf : ML::MetalBuffer,
                                         out_buf : ML::MetalBuffer,
                                         w_buf : ML::MetalBuffer,
                                         w_offset : Int64,
                                         in_dim : Int32,
                                         out_dim : Int32,
                                         batch : Int32 = 1,
                                         profile_shape : Bool = true) : Nil
          if profile_shape
            quant_name, block_bytes, block_elems = gemv_profile_quant(pipeline)
            blocks_per_row = (in_dim + block_elems - 1) // block_elems
            weight_bytes = out_dim.to_i64 * blocks_per_row.to_i64 * block_bytes.to_i64
            Profile.bump_matmul_shape("gemv_add #{quant_name} #{in_dim}x#{out_dim} b#{batch}", weight_bytes)
          end
          enc.set_pipeline(pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32, 5)
          enc.set_buffer(residual_buf, 6)
          rows_per_tg = gemv_rows_per_tg_for(pipeline)
          grid = {(out_dim + rows_per_tg - 1) // rows_per_tg, batch, 1}
          enc.dispatch_threadgroups(grid, {gemv_threads_per_tg_for(pipeline), 1, 1})
        end

        private def self.encode_gemv_q8_dual(enc : ML::Metal::ComputeEncoder,
                                             x_buf : ML::MetalBuffer,
                                             gate_out_buf : ML::MetalBuffer,
                                             up_out_buf : ML::MetalBuffer,
                                             gate_w_buf : ML::MetalBuffer,
                                             gate_w_offset : Int64,
                                             up_w_buf : ML::MetalBuffer,
                                             up_w_offset : Int64,
                                             in_dim : Int32,
                                             out_dim : Int32,
                                             batch : Int32 = 1,
                                             profile_shape : Bool = true) : Nil
          if profile_shape
            blocks_per_row = (in_dim + Q8_0_QK - 1) // Q8_0_QK
            weight_bytes = 2_i64 * out_dim.to_i64 * blocks_per_row.to_i64 * Q8_0_BLOCK_BYTES.to_i64
            Profile.bump_matmul_shape("gemv Q8_0 dual #{in_dim}x#{out_dim} b#{batch}", weight_bytes)
          end
          enc.set_pipeline(mv8_dual_pipeline)
          enc.set_buffer(gate_w_buf, 0, ML::Metal::BufferAccess::Read, offset: gate_w_offset)
          enc.set_buffer(up_w_buf, 1, ML::Metal::BufferAccess::Read, offset: up_w_offset)
          enc.set_buffer(x_buf, 2)
          enc.set_buffer(gate_out_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_buffer(up_out_buf, 4, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32,  5)
          enc.set_value(out_dim.to_u32, 6)
          enc.set_value(batch.to_u32,   7)
          rows_per_tg = MV_Q8_NSG * MV_Q8_NR0
          grid = {(out_dim + rows_per_tg - 1) // rows_per_tg, batch, 1}
          enc.dispatch_threadgroups(grid, {MV_Q8_NSG * 32, 1, 1})
        end

        private def self.encode_gemv_q4_dual(enc : ML::Metal::ComputeEncoder,
                                             x_buf : ML::MetalBuffer,
                                             gate_out_buf : ML::MetalBuffer,
                                             up_out_buf : ML::MetalBuffer,
                                             gate_w_buf : ML::MetalBuffer,
                                             gate_w_offset : Int64,
                                             up_w_buf : ML::MetalBuffer,
                                             up_w_offset : Int64,
                                             in_dim : Int32,
                                             out_dim : Int32,
                                             batch : Int32 = 1,
                                             profile_shape : Bool = true) : Nil
          if profile_shape
            blocks_per_row = (in_dim + QK_K - 1) // QK_K
            weight_bytes = 2_i64 * out_dim.to_i64 * blocks_per_row.to_i64 * Q4K_BLOCK_BYTES.to_i64
            Profile.bump_matmul_shape("gemv Q4_K dual #{in_dim}x#{out_dim} b#{batch}", weight_bytes)
          end
          enc.set_pipeline(mv_q4_dual_pipeline)
          enc.set_buffer(gate_w_buf, 0, ML::Metal::BufferAccess::Read, offset: gate_w_offset)
          enc.set_buffer(up_w_buf, 1, ML::Metal::BufferAccess::Read, offset: up_w_offset)
          enc.set_buffer(x_buf, 2)
          enc.set_buffer(gate_out_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_buffer(up_out_buf, 4, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 5)
          enc.set_value(out_dim.to_u32, 6)
          enc.set_value(batch.to_u32, 7)
          rows_per_tg = MV_Q4_NSG * MV_Q4_NR0
          grid = {(out_dim + rows_per_tg - 1) // rows_per_tg, batch, 1}
          enc.dispatch_threadgroups(grid, {MV_Q4_NSG * 32, 1, 1})
        end

        private def self.encode_gemv_input_offset(enc : ML::Metal::ComputeEncoder,
                                                  pipeline : ML::Metal::ComputePipeline,
                                                  x_buf : ML::MetalBuffer,
                                                  x_offset : Int64,
                                                  out_buf : ML::MetalBuffer,
                                                  w_buf : ML::MetalBuffer,
                                                  w_offset : Int64,
                                                  in_dim : Int32,
                                                  out_dim : Int32) : Nil
          enc.set_pipeline(pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x_buf, 1, ML::Metal::BufferAccess::Read, offset: x_offset)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32,  3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(1_u32,          5)
          rows_per_tg = gemv_rows_per_tg_for(pipeline)
          enc.dispatch_threadgroups({(out_dim + rows_per_tg - 1) // rows_per_tg, 1, 1}, {gemv_threads_per_tg_for(pipeline), 1, 1})
        end

        private def self.encode_q4k_gemm(enc : ML::Metal::ComputeEncoder,
                                         x_buf : ML::MetalBuffer,
                                         out_buf : ML::MetalBuffer,
                                         w_buf : ML::MetalBuffer,
                                         w_offset : Int64,
                                         in_dim : Int32,
                                         out_dim : Int32,
                                         batch : Int32) : Nil
          enc.set_pipeline(mm_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32,  3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32,   5)
          enc.set_threadgroup_memory(MM_SHMEM, 0)
          grid = {
            (batch   + MM_NR1 - 1) // MM_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }
          enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
        end

        private def self.encode_q4k_gemm_h16(enc : ML::Metal::ComputeEncoder,
                                             x_buf : ML::MetalBuffer,
                                             out_buf : ML::MetalBuffer,
                                             w_buf : ML::MetalBuffer,
                                             w_offset : Int64,
                                             in_dim : Int32,
                                             out_dim : Int32,
                                             batch : Int32) : Nil
          x16_buf = Scratch.get(:mm4_x16, (batch * in_dim).to_i64 * 2_i64)

          Profile.bump_conversion("f32_to_f16 q4_gemm_input #{in_dim} b#{batch}", (batch * in_dim).to_i64 * 6_i64)
          enc.set_pipeline(f32_to_f16_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(x16_buf, 1, ML::Metal::BufferAccess::Write)
          enc.set_value((batch * in_dim).to_u32, 2)
          enc.dispatch_1d(batch * in_dim, 256)

          enc.set_pipeline(mm_h16_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x16_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32, 5)
          enc.set_threadgroup_memory(MM_SHMEM, 0)
          grid = {
            (batch   + MM_NR1 - 1) // MM_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }
          enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
        end

        private def self.encode_q4k_gemm_h16_from_h16(enc : ML::Metal::ComputeEncoder,
                                                      x16_buf : ML::MetalBuffer,
                                                      out_buf : ML::MetalBuffer,
                                                      w_buf : ML::MetalBuffer,
                                                      w_offset : Int64,
                                                      in_dim : Int32,
                                                      out_dim : Int32,
                                                      batch : Int32) : Nil
          if q4_tensor_mm_enabled? && q4_tensor_ffn_candidate?(out_dim) && batch >= Q4_TENSOR_NR1
            enc.set_pipeline(mm_q4_tensor_f32out_pipeline)
            enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
            enc.set_buffer(x16_buf, 1)
            enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
            enc.set_value(in_dim.to_u32, 3)
            enc.set_value(out_dim.to_u32, 4)
            enc.set_value(batch.to_u32, 5)
            enc.set_threadgroup_memory(Q4_TENSOR_SHMEM, 0)
            enc.dispatch_threadgroups({
              (batch + Q4_TENSOR_NR1 - 1) // Q4_TENSOR_NR1,
              (out_dim + MM_NR0 - 1) // MM_NR0,
              1,
            }, {Q4_TENSOR_TG, 1, 1})
            return
          end

          if q4_h16_b48_gemm_enabled? && batch == MM48_NR1
            enc.set_pipeline(mm_h16_b48_pipeline)
            enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
            enc.set_buffer(x16_buf, 1)
            enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
            enc.set_value(in_dim.to_u32, 3)
            enc.set_value(out_dim.to_u32, 4)
            enc.set_value(batch.to_u32, 5)
            enc.set_threadgroup_memory(MM48_SHMEM, 0)
            enc.dispatch_threadgroups({
              1,
              (out_dim + MM_NR0 - 1) // MM_NR0,
              1,
            }, {MM48_TG, 1, 1})
            return
          end

          if q4_h16_b80_gemm_enabled? && batch == MM80_NR1
            enc.set_pipeline(mm_h16_b80_pipeline)
            enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
            enc.set_buffer(x16_buf, 1)
            enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
            enc.set_value(in_dim.to_u32, 3)
            enc.set_value(out_dim.to_u32, 4)
            enc.set_value(batch.to_u32, 5)
            enc.set_threadgroup_memory(MM80_SHMEM, 0)
            enc.dispatch_threadgroups({
              1,
              (out_dim + MM_NR0 - 1) // MM_NR0,
              1,
            }, {MM80_TG, 1, 1})
            return
          end

          if q4_h16_b96_gemm_enabled? && batch == MM96_NR1
            enc.set_pipeline(mm_h16_b96_pipeline)
            enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
            enc.set_buffer(x16_buf, 1)
            enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
            enc.set_value(in_dim.to_u32, 3)
            enc.set_value(out_dim.to_u32, 4)
            enc.set_value(batch.to_u32, 5)
            enc.set_threadgroup_memory(MM96_SHMEM, 0)
            enc.dispatch_threadgroups({
              (batch + MM96_NR1 - 1) // MM96_NR1,
              (out_dim + MM_NR0 - 1) // MM_NR0,
              1,
            }, {MM96_TG, 1, 1})
            return
          end

          if q4_h16_b112_gemm_enabled? && batch == MM112_NR1
            enc.set_pipeline(mm_h16_b112_pipeline)
            enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
            enc.set_buffer(x16_buf, 1)
            enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
            enc.set_value(in_dim.to_u32, 3)
            enc.set_value(out_dim.to_u32, 4)
            enc.set_value(batch.to_u32, 5)
            enc.set_threadgroup_memory(MM112_SHMEM, 0)
            enc.dispatch_threadgroups({
              1,
              (out_dim + MM_NR0 - 1) // MM_NR0,
              1,
            }, {MM112_TG, 1, 1})
            return
          end
          if q4_h16_b128_sg8_policy?(batch, in_dim, out_dim, ML::Metal::Device.instance.name,
               ENV["QWEN35_Q4K_H16_B128_SG8"]?)
            Profile.bump_route_marker("q4_h16_b128_sg8")
            enc.set_pipeline(mm_h16_b128_sg8_pipeline)
            enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
            enc.set_buffer(x16_buf, 1)
            enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
            enc.set_value(in_dim.to_u32, 3)
            enc.set_value(out_dim.to_u32, 4)
            enc.set_value(batch.to_u32, 5)
            enc.set_threadgroup_memory(MM128_SG8_SHMEM, 0)
            enc.dispatch_threadgroups({
              batch // MM128_SG8_NR1,
              out_dim // MM_NR0,
              1,
            }, {MM128_SG8_TG, 1, 1})
            return
          end

          use_b64_tail = q4_h16_b64_tail_candidate?(batch)
          if q4_h16_b64_gemm_enabled? && batch >= MM64_NR1 && ((batch % MM64_NR1) == 0 || use_b64_tail)
            enc.set_pipeline(mm_h16_b64_pipeline)
            enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
            enc.set_buffer(x16_buf, 1)
            enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
            enc.set_value(in_dim.to_u32, 3)
            enc.set_value(out_dim.to_u32, 4)
            enc.set_value(batch.to_u32, 5)
            enc.set_threadgroup_memory(MM64_SHMEM, 0)
            enc.dispatch_threadgroups({
              (batch + MM64_NR1 - 1) // MM64_NR1,
              (out_dim + MM_NR0 - 1) // MM_NR0,
              1,
            }, {MM64_TG, 1, 1})
            return
          end

          enc.set_pipeline(mm_h16_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x16_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32, 5)
          enc.set_threadgroup_memory(MM_SHMEM, 0)
          grid = {
            (batch   + MM_NR1 - 1) // MM_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }
          enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
        end

        private def self.encode_q4k_gemm_h16_pair(enc : ML::Metal::ComputeEncoder,
                                                  x_buf : ML::MetalBuffer,
                                                  out_a_buf : ML::MetalBuffer,
                                                  out_b_buf : ML::MetalBuffer,
                                                  w_a_buf : ML::MetalBuffer,
                                                  w_a_offset : Int64,
                                                  w_b_buf : ML::MetalBuffer,
                                                  w_b_offset : Int64,
                                                  in_dim : Int32,
                                                  out_dim : Int32,
                                                  batch : Int32) : Nil
          x16_buf = Scratch.get(:mm4_pair_x16, (batch * in_dim).to_i64 * 2_i64)

          Profile.bump_conversion("f32_to_f16 q4_pair_input #{in_dim} b#{batch}", (batch * in_dim).to_i64 * 6_i64)
          enc.set_pipeline(f32_to_f16_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(x16_buf, 1, ML::Metal::BufferAccess::Write)
          enc.set_value((batch * in_dim).to_u32, 2)
          enc.dispatch_1d(batch * in_dim, 256)

          encode_q4k_gemm_h16_from_h16(enc, x16_buf, out_a_buf, w_a_buf, w_a_offset, in_dim, out_dim, batch)
          encode_q4k_gemm_h16_from_h16(enc, x16_buf, out_b_buf, w_b_buf, w_b_offset, in_dim, out_dim, batch)
        end

        private def self.encode_q4k_gemm_h16_b64_swiglu_from_h16(enc : ML::Metal::ComputeEncoder,
                                                                 x16_buf : ML::MetalBuffer,
                                                                 gate_buf : ML::MetalBuffer,
                                                                 out_buf : ML::MetalBuffer,
                                                                 w_buf : ML::MetalBuffer,
                                                                 w_offset : Int64,
                                                                 in_dim : Int32,
                                                                 out_dim : Int32,
                                                                 batch : Int32) : Nil
          enc.set_pipeline(mm_h16_b64_swiglu_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x16_buf, 1)
          enc.set_buffer(gate_buf, 2)
          enc.set_buffer(out_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 4)
          enc.set_value(out_dim.to_u32, 5)
          enc.set_value(batch.to_u32, 6)
          enc.set_threadgroup_memory(MM64_SHMEM, 0)
          enc.dispatch_threadgroups({
            (batch + MM64_NR1 - 1) // MM64_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }, {MM64_TG, 1, 1})
        end

        private def self.encode_q4k_gemm_h16_b64_gelu_mul_from_h16(enc : ML::Metal::ComputeEncoder,
                                                                   x16_buf : ML::MetalBuffer,
                                                                   gate_buf : ML::MetalBuffer,
                                                                   out_buf : ML::MetalBuffer,
                                                                   w_buf : ML::MetalBuffer,
                                                                   w_offset : Int64,
                                                                   in_dim : Int32,
                                                                   out_dim : Int32,
                                                                   batch : Int32) : Nil
          enc.set_pipeline(mm_h16_b64_gelu_mul_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x16_buf, 1)
          enc.set_buffer(gate_buf, 2)
          enc.set_buffer(out_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 4)
          enc.set_value(out_dim.to_u32, 5)
          enc.set_value(batch.to_u32, 6)
          enc.set_threadgroup_memory(MM64_SHMEM, 0)
          enc.dispatch_threadgroups({
            (batch + MM64_NR1 - 1) // MM64_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }, {MM64_TG, 1, 1})
        end

        private def self.encode_q4k_gemm_h16_b64_swiglu_h16_from_h16(enc : ML::Metal::ComputeEncoder,
                                                                     x16_buf : ML::MetalBuffer,
                                                                     gate_buf : ML::MetalBuffer,
                                                                     out_h16_buf : ML::MetalBuffer,
                                                                     w_buf : ML::MetalBuffer,
                                                                     w_offset : Int64,
                                                                     in_dim : Int32,
                                                                     out_dim : Int32,
                                                                     batch : Int32) : Nil
          use_b128_sg8 = q4_h16_b128_sg8_policy?(batch, in_dim, out_dim, ML::Metal::Device.instance.name,
            ENV["QWEN35_Q4K_H16_B128_SG8"]?)
          Profile.bump_route_marker("q4_h16_b128_sg8_swiglu_h16") if use_b128_sg8
          enc.set_pipeline(use_b128_sg8 ? mm_h16_b128_sg8_swiglu_h16_pipeline : mm_h16_b64_swiglu_h16_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x16_buf, 1)
          enc.set_buffer(gate_buf, 2)
          enc.set_buffer(out_h16_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 4)
          enc.set_value(out_dim.to_u32, 5)
          enc.set_value(batch.to_u32, 6)
          if use_b128_sg8
            enc.set_threadgroup_memory(MM128_SG8_SHMEM, 0)
            enc.dispatch_threadgroups({
              batch // MM128_SG8_NR1,
              out_dim // MM_NR0,
              1,
            }, {MM128_SG8_TG, 1, 1})
          else
            enc.set_threadgroup_memory(MM64_SHMEM, 0)
            enc.dispatch_threadgroups({
              (batch + MM64_NR1 - 1) // MM64_NR1,
              (out_dim + MM_NR0 - 1) // MM_NR0,
              1,
            }, {MM64_TG, 1, 1})
          end
        end

        private def self.encode_q4k_gemm_h16_pair_b64_swiglu(enc : ML::Metal::ComputeEncoder,
                                                             x_buf : ML::MetalBuffer,
                                                             gate_buf : ML::MetalBuffer,
                                                             act_buf : ML::MetalBuffer,
                                                             gate_w_buf : ML::MetalBuffer,
                                                             gate_w_offset : Int64,
                                                             up_w_buf : ML::MetalBuffer,
                                                             up_w_offset : Int64,
                                                             in_dim : Int32,
                                                             out_dim : Int32,
                                                             batch : Int32) : Nil
          x16_buf = Scratch.get(:mm4_pair_swiglu_x16, (batch * in_dim).to_i64 * 2_i64)

          Profile.bump_conversion("f32_to_f16 q4_pair_swiglu_input #{in_dim} b#{batch}", (batch * in_dim).to_i64 * 6_i64)
          enc.set_pipeline(f32_to_f16_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(x16_buf, 1, ML::Metal::BufferAccess::Write)
          enc.set_value((batch * in_dim).to_u32, 2)
          enc.dispatch_1d(batch * in_dim, 256)

          encode_q4k_gemm_h16_from_h16(enc, x16_buf, gate_buf, gate_w_buf, gate_w_offset, in_dim, out_dim, batch)
          encode_q4k_gemm_h16_b64_swiglu_from_h16(enc, x16_buf, gate_buf, act_buf, up_w_buf, up_w_offset, in_dim, out_dim, batch)
        end

        private def self.encode_q4k_gemm_h16_pair_b64_swiglu_h16(enc : ML::Metal::ComputeEncoder,
                                                                 x_buf : ML::MetalBuffer,
                                                                 gate_buf : ML::MetalBuffer,
                                                                 act_h16_buf : ML::MetalBuffer,
                                                                 gate_w_buf : ML::MetalBuffer,
                                                                 gate_w_offset : Int64,
                                                                 up_w_buf : ML::MetalBuffer,
                                                                 up_w_offset : Int64,
                                                                 in_dim : Int32,
                                                                 out_dim : Int32,
                                                                 batch : Int32) : Nil
          x16_buf = Scratch.get(:mm4_pair_swiglu_h16_x16, (batch * in_dim).to_i64 * 2_i64)

          Profile.bump_conversion("f32_to_f16 q4_pair_swiglu_h16_input #{in_dim} b#{batch}", (batch * in_dim).to_i64 * 6_i64)
          enc.set_pipeline(f32_to_f16_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(x16_buf, 1, ML::Metal::BufferAccess::Write)
          enc.set_value((batch * in_dim).to_u32, 2)
          enc.dispatch_1d(batch * in_dim, 256)

          encode_q4k_gemm_h16_from_h16(enc, x16_buf, gate_buf, gate_w_buf, gate_w_offset, in_dim, out_dim, batch)
          encode_q4k_gemm_h16_b64_swiglu_h16_from_h16(enc, x16_buf, gate_buf, act_h16_buf, up_w_buf, up_w_offset, in_dim, out_dim, batch)
        end

        private def self.encode_q56k_gemm_f32(enc : ML::Metal::ComputeEncoder,
                                              pipeline : ML::Metal::ComputePipeline,
                                              x_buf : ML::MetalBuffer,
                                              out_buf : ML::MetalBuffer,
                                              w_buf : ML::MetalBuffer,
                                              w_offset : Int64,
                                              in_dim : Int32,
                                              out_dim : Int32,
                                              batch : Int32) : Nil
          x16_buf = Scratch.get(:mm56_x16, (batch * in_dim).to_i64 * 2_i64)

          Profile.bump_conversion("f32_to_f16 q56_gemm_input #{in_dim} b#{batch}", (batch * in_dim).to_i64 * 6_i64)
          enc.set_pipeline(f32_to_f16_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(x16_buf, 1, ML::Metal::BufferAccess::Write)
          enc.set_value((batch * in_dim).to_u32, 2)
          enc.dispatch_1d(batch * in_dim, 256)

          enc.set_pipeline(pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x16_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32, 5)
          enc.set_threadgroup_memory(MM_SHMEM, 0)
          grid = {
            (batch   + MM_NR1 - 1) // MM_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }
          enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
        end

        private def self.encode_q56k_gemm_f32_add(enc : ML::Metal::ComputeEncoder,
                                                  pipeline : ML::Metal::ComputePipeline,
                                                  x_buf : ML::MetalBuffer,
                                                  residual_buf : ML::MetalBuffer,
                                                  out_buf : ML::MetalBuffer,
                                                  w_buf : ML::MetalBuffer,
                                                  w_offset : Int64,
                                                  in_dim : Int32,
                                                  out_dim : Int32,
                                                  batch : Int32) : Nil
          x16_buf = Scratch.get(:mm56_x16, (batch * in_dim).to_i64 * 2_i64)

          Profile.bump_conversion("f32_to_f16 q56_gemm_input #{in_dim} b#{batch}", (batch * in_dim).to_i64 * 6_i64)
          enc.set_pipeline(f32_to_f16_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(x16_buf, 1, ML::Metal::BufferAccess::Write)
          enc.set_value((batch * in_dim).to_u32, 2)
          enc.dispatch_1d(batch * in_dim, 256)

          enc.set_pipeline(pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x16_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32, 5)
          enc.set_buffer(residual_buf, 6)
          enc.set_threadgroup_memory(MM_SHMEM, 0)
          grid = {
            (batch   + MM_NR1 - 1) // MM_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }
          enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
        end

        private def self.encode_q56k_gemm_f32_from_h16(enc : ML::Metal::ComputeEncoder,
                                                       pipeline : ML::Metal::ComputePipeline,
                                                       x16_buf : ML::MetalBuffer,
                                                       out_buf : ML::MetalBuffer,
                                                       w_buf : ML::MetalBuffer,
                                                       w_offset : Int64,
                                                       in_dim : Int32,
                                                       out_dim : Int32,
                                                       batch : Int32) : Nil
          enc.set_pipeline(pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x16_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32, 5)
          enc.set_threadgroup_memory(MM_SHMEM, 0)
          grid = {
            (batch   + MM_NR1 - 1) // MM_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }
          enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
        end

        private def self.encode_q56k_gemm_f32_add_from_h16(enc : ML::Metal::ComputeEncoder,
                                                           pipeline : ML::Metal::ComputePipeline,
                                                           x16_buf : ML::MetalBuffer,
                                                           residual_buf : ML::MetalBuffer,
                                                           out_buf : ML::MetalBuffer,
                                                           w_buf : ML::MetalBuffer,
                                                           w_offset : Int64,
                                                           in_dim : Int32,
                                                           out_dim : Int32,
                                                           batch : Int32) : Nil
          enc.set_pipeline(pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x16_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32, 5)
          enc.set_buffer(residual_buf, 6)
          enc.set_threadgroup_memory(MM_SHMEM, 0)
          grid = {
            (batch   + MM_NR1 - 1) // MM_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }
          enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
        end

        private def self.encode_q56k_gemm_h16(enc : ML::Metal::ComputeEncoder,
                                              pipeline : ML::Metal::ComputePipeline,
                                              x_buf : ML::MetalBuffer,
                                              out16_buf : ML::MetalBuffer,
                                              w_buf : ML::MetalBuffer,
                                              w_offset : Int64,
                                              in_dim : Int32,
                                              out_dim : Int32,
                                              batch : Int32) : Nil
          x16_buf = Scratch.get(:mm56_x16, (batch * in_dim).to_i64 * 2_i64)

          Profile.bump_conversion("f32_to_f16 q56_h16_input #{in_dim} b#{batch}", (batch * in_dim).to_i64 * 6_i64)
          enc.set_pipeline(f32_to_f16_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(x16_buf, 1, ML::Metal::BufferAccess::Write)
          enc.set_value((batch * in_dim).to_u32, 2)
          enc.dispatch_1d(batch * in_dim, 256)

          encode_q56k_gemm_h16_from_h16(enc, pipeline, x16_buf, out16_buf, w_buf, w_offset, in_dim, out_dim, batch)
        end

        private def self.encode_q56k_gemm_h16_from_h16(enc : ML::Metal::ComputeEncoder,
                                                       pipeline : ML::Metal::ComputePipeline,
                                                       x16_buf : ML::MetalBuffer,
                                                       out16_buf : ML::MetalBuffer,
                                                       w_buf : ML::MetalBuffer,
                                                       w_offset : Int64,
                                                       in_dim : Int32,
                                                       out_dim : Int32,
                                                       batch : Int32) : Nil
          bias_buf = Scratch.get("mm56_bias_#{out_dim}", out_dim.to_i64 * sizeof(Float32))
          ConstCache.write_zero_f32_once("mm56_bias_#{out_dim}", bias_buf, out_dim)

          enc.set_pipeline(pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x16_buf, 1)
          enc.set_buffer(bias_buf, 2)
          enc.set_buffer(out16_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 4)
          enc.set_value(out_dim.to_u32, 5)
          enc.set_value(batch.to_u32, 6)
          enc.set_value(0_u32, 7)
          enc.set_threadgroup_memory(MM_SHMEM, 0)
          grid = {
            (batch   + MM_NR1 - 1) // MM_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }
          enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
        end

        private def self.encode_matmul(enc : ML::Metal::ComputeEncoder,
                                       gemv_pipeline : ML::Metal::ComputePipeline,
                                       qw : QuantWeight,
                                       x_buf : ML::MetalBuffer,
                                       out_buf : ML::MetalBuffer,
                                       w_buf : ML::MetalBuffer,
                                       w_offset : Int64,
                                       in_dim : Int32,
                                       out_dim : Int32,
                                       batch : Int32) : Nil
          force_small_q4_gemv = small_q4_gemv_enabled? && qw.type.q4_k? && out_dim <= 64
          use_q4_h16 = q4_h16_gemm_enabled? && qw.type.q4_k? && batch > GEMM_BATCH_THRESHOLD && !force_small_q4_gemv
          route = if use_q4_h16
                    "q4_h16_gemm"
                  elsif qw.type.q4_k? && batch > GEMM_BATCH_THRESHOLD && !force_small_q4_gemv
                    "q4_gemm"
                  elsif q56_batch_gemm_enabled? && qw.type.q5_k? && batch > GEMM_BATCH_THRESHOLD
                    "q5_gemm"
                  elsif q56_batch_gemm_enabled? && qw.type.q6_k? && batch > GEMM_BATCH_THRESHOLD
                    "q6_gemm"
                  else
                    "gemv"
                  end
          Profile.bump_matmul_shape("#{route} #{qw.type.name} #{in_dim}x#{out_dim} b#{batch}#{profile_route_tag(qw)}", qw.raw.size.to_i64)

          if use_q4_h16
            encode_q4k_gemm_h16(enc, x_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch)
          elsif qw.type.q4_k? && batch > GEMM_BATCH_THRESHOLD && !force_small_q4_gemv
            encode_q4k_gemm(enc, x_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch)
          elsif q56_batch_gemm_enabled? && qw.type.q5_k? && batch > GEMM_BATCH_THRESHOLD
            encode_q56k_gemm_f32(enc, mm5_f32out_pipeline, x_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch)
          elsif q56_batch_gemm_enabled? && qw.type.q6_k? && batch > GEMM_BATCH_THRESHOLD
            encode_q56k_gemm_f32(enc, mm6_f32out_pipeline, x_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch)
          else
            encode_gemv(enc, gemv_pipeline, x_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch, profile_shape: false, route_qw: qw)
          end
        end

        private def self.encode_matmul_add(enc : ML::Metal::ComputeEncoder,
                                           gemv_pipeline : ML::Metal::ComputePipeline,
                                           qw : QuantWeight,
                                           x_buf : ML::MetalBuffer,
                                           residual_buf : ML::MetalBuffer,
                                           out_buf : ML::MetalBuffer,
                                           w_buf : ML::MetalBuffer,
                                           w_offset : Int64,
                                           in_dim : Int32,
                                           out_dim : Int32,
                                           batch : Int32) : Bool
          route = if q56_batch_gemm_enabled? && qw.type.q6_k? && batch > GEMM_BATCH_THRESHOLD
                    "q6_gemm_add"
                  elsif batch <= GEMM_BATCH_THRESHOLD && (add_pipe = gemv_add_pipeline_for(qw))
                    "gemv_add"
                  else
                    return false
                  end
          Profile.bump_matmul_shape("#{route} #{qw.type.name} #{in_dim}x#{out_dim} b#{batch}#{profile_route_tag(qw)}", qw.raw.size.to_i64)

          if q56_batch_gemm_enabled? && qw.type.q6_k? && batch > GEMM_BATCH_THRESHOLD
            Profile.bump_route_marker("q6_gemm_add")
            encode_q56k_gemm_f32_add(enc, mm6_f32out_add_pipeline, x_buf, residual_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch)
          elsif batch <= GEMM_BATCH_THRESHOLD && (add_pipe = gemv_add_pipeline_for(qw))
            encode_gemv_add(enc, add_pipe, x_buf, residual_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch, profile_shape: false)
          else
            return false
          end
          true
        end

        private def self.matmul_many_shared_h16_enabled?(batch : Int32) : Bool
          return false if ENV["QWEN35_MATMUL_MANY_SHARED_H16_OFF"]? == "1"
          return true if ENV["QWEN35_MATMUL_MANY_SHARED_H16"]? == "1"
          min_batch = (ENV["QWEN35_MATMUL_MANY_SHARED_H16_MIN_BATCH"]? || "256").to_i32
          batch >= min_batch
        end

        private def self.h16_batch_gemm_candidate?(qw : QuantWeight, batch : Int32) : Bool
          return false unless batch > GEMM_BATCH_THRESHOLD
          force_small_q4_gemv = small_q4_gemv_enabled? && qw.type.q4_k? && qw.out_dim <= 64
          (q4_h16_gemm_enabled? && qw.type.q4_k? && !force_small_q4_gemv) ||
            (q56_batch_gemm_enabled? && (qw.type.q5_k? || qw.type.q6_k?))
        end

        private def self.prefill_shared_projection_h16_candidate?(qws : Array(QuantWeight),
                                                                  batch : Int32,
                                                                  checkpoint_requested : Bool) : Bool
          prefill_shared_projection_h16_policy?(
            qws, batch, checkpoint_requested, ML::Metal::Device.instance.name,
            ENV["QWEN35_PREFILL_SHARED_PROJECTION_H16"]?
          ) && qws.all? { |qw| h16_batch_gemm_candidate?(qw, batch) }
        end

        private def self.encode_shared_projection_h16(enc : ML::Metal::ComputeEncoder,
                                                       qws : Array(QuantWeight),
                                                       x_buf : ML::MetalBuffer,
                                                       out_bufs : Array(ML::MetalBuffer),
                                                       batch : Int32,
                                                       scratch_tag : String) : Nil
          raise ArgumentError.new("shared projection arity mismatch") unless qws.size == out_bufs.size && qws.size >= 2
          in_dim = qws[0].in_dim
          x16_buf = Scratch.get(scratch_tag, batch.to_i64 * in_dim * 2_i64)
          Profile.bump_conversion("f32_to_f16 shared_projection_input #{in_dim} b#{batch}", (batch * in_dim).to_i64 * 6_i64)
          enc.set_pipeline(f32_to_f16_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(x16_buf, 1, ML::Metal::BufferAccess::Write)
          enc.set_value((batch * in_dim).to_u32, 2)
          enc.dispatch_1d(batch * in_dim, 256)

          qws.each_with_index do |qw, i|
            w_buf, w_offset = weight_slot(qw)
            unless encode_matmul_from_h16(enc, qw, x16_buf, out_bufs[i], w_buf, w_offset,
                                         qw.in_dim, qw.out_dim, batch)
              raise "unsupported shared H16 projection #{qw.type.name} #{qw.in_dim}x#{qw.out_dim} b#{batch}"
            end
          end
        end

        private def self.encode_matmul_from_h16(enc : ML::Metal::ComputeEncoder,
                                                qw : QuantWeight,
                                                x16_buf : ML::MetalBuffer,
                                                out_buf : ML::MetalBuffer,
                                                w_buf : ML::MetalBuffer,
                                                w_offset : Int64,
                                                in_dim : Int32,
                                                out_dim : Int32,
                                                batch : Int32) : Bool
          force_small_q4_gemv = small_q4_gemv_enabled? && qw.type.q4_k? && out_dim <= 64
          if q4_h16_gemm_enabled? && qw.type.q4_k? && batch > GEMM_BATCH_THRESHOLD && !force_small_q4_gemv
            Profile.bump_matmul_shape("q4_h16_gemm #{qw.type.name} #{in_dim}x#{out_dim} b#{batch}", qw.raw.size.to_i64)
            encode_q4k_gemm_h16_from_h16(enc, x16_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch)
          elsif q56_batch_gemm_enabled? && qw.type.q5_k? && batch > GEMM_BATCH_THRESHOLD
            Profile.bump_matmul_shape("q5_gemm #{qw.type.name} #{in_dim}x#{out_dim} b#{batch}", qw.raw.size.to_i64)
            encode_q56k_gemm_f32_from_h16(enc, mm5_f32out_pipeline, x16_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch)
          elsif q56_batch_gemm_enabled? && qw.type.q6_k? && batch > GEMM_BATCH_THRESHOLD
            Profile.bump_matmul_shape("q6_gemm #{qw.type.name} #{in_dim}x#{out_dim} b#{batch}", qw.raw.size.to_i64)
            encode_q56k_gemm_f32_from_h16(enc, mm6_f32out_pipeline, x16_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch)
          else
            return false
          end
          true
        end

        private def self.encode_matmul_add_from_h16(enc : ML::Metal::ComputeEncoder,
                                                    qw : QuantWeight,
                                                    x16_buf : ML::MetalBuffer,
                                                    residual_buf : ML::MetalBuffer,
                                                    out_buf : ML::MetalBuffer,
                                                    w_buf : ML::MetalBuffer,
                                                    w_offset : Int64,
                                                    in_dim : Int32,
                                                    out_dim : Int32,
                                                    batch : Int32) : Bool
          return false unless q56_batch_gemm_enabled? && qw.type.q6_k? && batch > GEMM_BATCH_THRESHOLD

          Profile.bump_matmul_shape("q6_gemm_add #{qw.type.name} #{in_dim}x#{out_dim} b#{batch}", qw.raw.size.to_i64)
          Profile.bump_route_marker("q6_gemm_add")
          encode_q56k_gemm_f32_add_from_h16(enc, mm6_f32out_add_pipeline, x16_buf, residual_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch)
          true
        end

        private def self.q56_batch_gemm_enabled? : Bool
          ENV["QWEN35_Q56K_BATCH_GEMM_OFF"]? != "1"
        end

        private def self.small_q4_gemv_enabled? : Bool
          ENV["QWEN35_SMALL_Q4_GEMV_OFF"]? != "1"
        end

        private def self.q4_h16_gemm_enabled? : Bool
          ENV["QWEN35_Q4K_H16_GEMM_OFF"]? != "1"
        end

        private def self.q4_tensor_mm_enabled? : Bool
          ENV["QWEN35_Q4K_TENSOR_MM"]? == "1" &&
            ENV["GEMMA4_ROW_PREFILL_Q4_GELU_FUSE"]? != "1"
        end

        private def self.q4_h16_b64_gemm_enabled? : Bool
          ENV["QWEN35_Q4K_H16_B64_OFF"]? != "1"
        end

        private def self.q4_h16_b64_tail_candidate?(batch : Int32) : Bool
          q4_h16_b64_tail_policy?(
            batch,
            ML::Metal::Device.instance.name,
            ENV["QWEN35_Q4K_H16_B64_TAIL_MIN"]?,
          )
        end

        private def self.q4_h16_exact_rowpack_candidate?(batch : Int32) : Bool
          (q4_h16_b48_gemm_enabled? && batch == MM48_NR1) ||
            (q4_h16_b80_gemm_enabled? && batch == MM80_NR1) ||
            (q4_h16_b96_gemm_enabled? && batch == MM96_NR1) ||
            (q4_h16_b112_gemm_enabled? && batch == MM112_NR1)
        end

        private def self.q4_h16_b48_gemm_enabled? : Bool
          ENV["QWEN35_Q4K_H16_B48_OFF"]? != "1"
        end

        private def self.q4_h16_b80_gemm_enabled? : Bool
          ENV["QWEN35_Q4K_H16_B80_OFF"]? != "1"
        end

        private def self.q4_h16_b96_gemm_enabled? : Bool
          ENV["QWEN35_Q4K_H16_B96_OFF"]? != "1"
        end

        private def self.q4_h16_b112_gemm_enabled? : Bool
          ENV["QWEN35_Q4K_H16_B112_OFF"]? != "1"
        end

        private def self.q4_pair_h16_gemm_enabled? : Bool
          ENV["QWEN35_Q4K_PAIR_H16_GEMM_OFF"]? != "1"
        end

        private def self.q4_pair_h16_min_batch : Int32
          (ENV["QWEN35_Q4K_PAIR_H16_MIN_BATCH"]? || Q4_PAIR_H16_MIN_BATCH.to_s).to_i32
        end

        private def self.q8_dual_gemv_enabled? : Bool
          ENV["QWEN35_Q8_DUAL_GEMV_OFF"]? != "1"
        end

        private def self.q8_dual_gemv_candidate?(gate_qw : QuantWeight,
                                                 up_qw : QuantWeight,
                                                 batch : Int32 = 1) : Bool
          q8_dual_gemv_enabled? && batch == 1 &&
            gate_qw.type.q8_0? && up_qw.type.q8_0? &&
            gate_qw.in_dim == up_qw.in_dim &&
            gate_qw.out_dim == up_qw.out_dim
        end

        private def self.q4_dual_gemv_candidate?(gate_qw : QuantWeight,
                                                 up_qw : QuantWeight,
                                                 batch : Int32 = 1) : Bool
          batch == 1 &&
            gate_qw.type.q4_k? && up_qw.type.q4_k? &&
            gate_qw.in_dim == up_qw.in_dim &&
            gate_qw.out_dim == up_qw.out_dim &&
            gate_qw.in_dim % QK_K == 0
        end

        def self.encode_q4k_gemv_dual_to_buffers(enc : ML::Metal::ComputeEncoder,
                                                  gate_qw : QuantWeight,
                                                  up_qw : QuantWeight,
                                                  x_buf : ML::MetalBuffer,
                                                  gate_buf : ML::MetalBuffer,
                                                  up_buf : ML::MetalBuffer,
                                                  batch : Int32 = 1) : Bool
          return false unless q4_dual_gemv_candidate?(gate_qw, up_qw, batch)
          return false if x_buf.size < batch.to_i64 * gate_qw.in_dim * sizeof(Float32)
          return false if gate_buf.size < batch.to_i64 * gate_qw.out_dim * sizeof(Float32)
          return false if up_buf.size < batch.to_i64 * up_qw.out_dim * sizeof(Float32)

          gate_w_buf, gate_w_off = weight_slot(gate_qw)
          up_w_buf, up_w_off = weight_slot(up_qw)
          encode_gemv_q4_dual(enc, x_buf, gate_buf, up_buf,
            gate_w_buf, gate_w_off, up_w_buf, up_w_off,
            gate_qw.in_dim, gate_qw.out_dim, batch)
          true
        end

        private def self.q4_pair_h16_gemm_candidate?(gate_qw : QuantWeight,
                                                     up_qw : QuantWeight,
                                                     batch : Int32) : Bool
          q4_pair_h16_gemm_enabled? && q4_h16_gemm_enabled? &&
            batch >= q4_pair_h16_min_batch &&
            gate_qw.type.q4_k? && up_qw.type.q4_k? &&
            gate_qw.in_dim == up_qw.in_dim &&
            gate_qw.out_dim == up_qw.out_dim
        end

        private def self.gemv_route_pipeline_supported?(qw : QuantWeight,
                                                        batch : Int32,
                                                        route_qw : QuantWeight?) : Bool
          pipeline = gemv_pipeline_for(qw)
          return false unless pipeline

          if q4_gemv_x16_enabled?(route_qw) && batch <= GEMM_BATCH_THRESHOLD &&
             pipeline.same?(mv_pipeline) && (qw.in_dim % QK_K == 0)
            mv_q4_x16_pipeline
          elsif q4_gemv_shape_layout_enabled?(route_qw) && batch <= GEMM_BATCH_THRESHOLD &&
                pipeline.same?(mv_pipeline)
            if layout = q4_gemv_shape_layout(qw.in_dim, qw.out_dim)
              mv_q4_layout_pipeline(layout[0], layout[1])
            end
          elsif q6_gemv_shape_layout_enabled? && batch <= GEMM_BATCH_THRESHOLD &&
                pipeline.same?(mv6_pipeline)
            if layout = q6_gemv_shape_layout(qw.in_dim, qw.out_dim)
              mv_q6_layout_pipeline(layout[0], layout[1])
            end
          end
          true
        end

        private def self.matmul_pipeline_supported?(qw : QuantWeight, batch : Int32) : Bool
          return false unless gemv_pipeline_for(qw)

          force_small_q4_gemv = small_q4_gemv_enabled? && qw.type.q4_k? && qw.out_dim <= 64
          if qw.type.q4_k? && batch > GEMM_BATCH_THRESHOLD && !force_small_q4_gemv
            if q4_h16_gemm_enabled?
              f32_to_f16_pipeline
              mm_h16_pipeline
            else
              mm_pipeline
            end
          elsif q56_batch_gemm_enabled? && qw.type.q5_k? && batch > GEMM_BATCH_THRESHOLD
            f32_to_f16_pipeline
            mm5_f32out_pipeline
          elsif q56_batch_gemm_enabled? && qw.type.q6_k? && batch > GEMM_BATCH_THRESHOLD
            f32_to_f16_pipeline
            mm6_f32out_pipeline
          else
            return gemv_route_pipeline_supported?(qw, batch, qw)
          end
          true
        end

        # Shared GEMV machinery: takes pre-allocated weight buffer and a
        # byte offset into it (for zero-copy whole-mmap dispatch), uploads
        # x, dispatches the kernel, returns the downloaded output.
        private def self.matmul_gemv_buf(pipeline : ML::Metal::ComputePipeline,
                                         x : Array(Float32),
                                         w_buf : ML::MetalBuffer,
                                         w_offset : Int64,
                                         in_dim : Int32,
                                         out_dim : Int32,
                                         batch : Int32) : Array(Float32)
          t0 = Time.instant if Profile.enabled?
          x_buf   = Scratch.get(:mv_x,   x.size.to_i64 * sizeof(Float32))
          x_buf.write(x)

          out_buf = Scratch.get(:mv_out, (batch * out_dim).to_i64 * sizeof(Float32))

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(enc, pipeline, x_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch)
          enc.end_encoding
          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, batch * out_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        private def self.matmul_q4_layout_wait_ms(qw : QuantWeight,
                                                  x : Array(Float32),
                                                  batch : Int32,
                                                  nsg : Int32,
                                                  nr0 : Int32,
                                                  read_output : Bool = false) : {Float64, Array(Float32)}
          raise ArgumentError.new("Q4 layout bench requires Q4_K weight") unless qw.type.q4_k?
          raise ArgumentError.new("Q4 layout bench x size mismatch") unless x.size == qw.in_dim * batch

          ML::Metal::Device.init!
          w_buf, w_off = if slot = mmap_slot_for(qw.raw)
                           slot
                         else
                           {qw.fallback_metal_buffer, 0_i64}
                         end
          pipeline = mv_q4_layout_pipeline(nsg, nr0)
          rows_per_tg = nsg * nr0
          threads_per_tg = nsg * 32

          x_buf = Scratch.get(:bench_q4_layout_x, x.size.to_i64 * sizeof(Float32))
          x_buf.write(x)
          out_buf = Scratch.get(:bench_q4_layout_out, (batch * qw.out_dim).to_i64 * sizeof(Float32))

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv_layout(enc, pipeline, x_buf, out_buf, w_buf, w_off,
            qw.in_dim, qw.out_dim, batch, rows_per_tg, threads_per_tg)
          enc.end_encoding

          t0 = Time.instant
          cmd.commit
          cmd.wait
          elapsed = (Time.instant - t0).total_milliseconds
          result = read_output ? read_shared_f32(out_buf, batch * qw.out_dim) : [] of Float32
          {elapsed, result}
        end

        private def self.matmul_q6_layout_wait_ms(qw : QuantWeight,
                                                  x : Array(Float32),
                                                  batch : Int32,
                                                  nsg : Int32,
                                                  nr0 : Int32,
                                                  read_output : Bool = false) : {Float64, Array(Float32)}
          raise ArgumentError.new("Q6 layout bench requires Q6_K weight") unless qw.type.q6_k?
          raise ArgumentError.new("Q6 layout bench x size mismatch") unless x.size == qw.in_dim * batch

          ML::Metal::Device.init!
          w_buf, w_off = if slot = mmap_slot_for(qw.raw)
                           slot
                         else
                           {qw.fallback_metal_buffer, 0_i64}
                         end
          pipeline = mv_q6_layout_pipeline(nsg, nr0)
          rows_per_tg = nsg * nr0
          threads_per_tg = nsg * 32

          x_buf = Scratch.get(:bench_q6_layout_x, x.size.to_i64 * sizeof(Float32))
          x_buf.write(x)
          out_buf = Scratch.get(:bench_q6_layout_out, (batch * qw.out_dim).to_i64 * sizeof(Float32))

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv_layout(enc, pipeline, x_buf, out_buf, w_buf, w_off,
            qw.in_dim, qw.out_dim, batch, rows_per_tg, threads_per_tg)
          enc.end_encoding

          t0 = Time.instant
          cmd.commit
          cmd.wait
          elapsed = (Time.instant - t0).total_milliseconds
          result = read_output ? read_shared_f32(out_buf, batch * qw.out_dim) : [] of Float32
          {elapsed, result}
        end

        # Q4_K-specific GEMM path (prefill batch > threshold). Takes
        # pre-allocated weight buffer and byte offset.
        private def self.matmul_q4k_gemm_buf(x : Array(Float32),
                                             w_buf : ML::MetalBuffer,
                                             w_offset : Int64,
                                             in_dim : Int32,
                                             out_dim : Int32,
                                             batch : Int32) : Array(Float32)
          x_buf   = Scratch.get(:mm_x,   x.size.to_i64 * sizeof(Float32))
          x_buf.write(x)

          out_buf = Scratch.get(:mm_out, (batch * out_dim).to_i64 * sizeof(Float32))

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(mm_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32,  3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32,   5)
          enc.set_threadgroup_memory(MM_SHMEM, 0)
          grid = {
            (batch   + MM_NR1 - 1) // MM_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }
          enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
          enc.end_encoding
          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, batch * out_dim)
          if Profile.enabled?
            Profile.bump_gemm(
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        # Q5_K/Q6_K simdgroup-matrix GEMM path. The prefill f32-output
        # kernels round through half internally to preserve the previous
        # F16-output numeric contract without a separate conversion dispatch.
        private def self.matmul_q56k_gemm_buf(pipeline : ML::Metal::ComputePipeline,
                                              x : Array(Float32),
                                              w_buf : ML::MetalBuffer,
                                              w_offset : Int64,
                                              in_dim : Int32,
                                              out_dim : Int32,
                                              batch : Int32) : Array(Float32)
          x_buf = Scratch.get(:mm56_x, x.size.to_i64 * 2_i64)
          write_shared_f16(x_buf, x)

          out_buf = Scratch.get(:mm56_out, (batch * out_dim).to_i64 * sizeof(Float32))

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32,  3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32,   5)
          enc.set_threadgroup_memory(MM_SHMEM, 0)
          grid = {
            (batch   + MM_NR1 - 1) // MM_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }
          enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
          enc.end_encoding
          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          if Profile.enabled?
            Profile.bump_gemm(
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
            )
          end
          read_shared_f32(out_buf, batch * out_dim)
        end

        # Upload w_raw into a fresh MetalBuffer (test + one-shot paths).
        private def self.upload_weights(w_raw : Bytes) : ML::MetalBuffer
          buf = ML::MetalBuffer.new(w_raw.size.to_i64)
          buf.write_bytes(w_raw.to_unsafe, w_raw.size)
          buf
        end

        # BF16 MTP sidecar tensors live in a separate safetensors mmap, outside
        # the registered GGUF mmap. Cache one MetalBuffer per raw slice so the
        # correctness probe does not re-upload 100-170 MiB weights per matvec.
        private def self.bf16_weight_slot(w_raw : Bytes) : {ML::MetalBuffer, Int64}
          if slot = mmap_slot_for(w_raw)
            return slot
          end

          key = "#{w_raw.to_unsafe.address}:#{w_raw.size}"
          @@bf16_weight_mutex.synchronize do
            if buf = @@bf16_weight_buffers[key]?
              {buf, 0_i64}
            else
              buf = upload_weights(w_raw)
              @@bf16_weight_buffers[key] = buf
              {buf, 0_i64}
            end
          end
        end

        def self.clear_bf16_weight_cache : Nil
          @@bf16_weight_mutex.synchronize do
            @@bf16_weight_buffers.clear
          end
        end

        private def self.encode_bf16_gemv(enc : ML::Metal::ComputeEncoder,
                                          w_buf : ML::MetalBuffer,
                                          w_off : Int64,
                                          x_buf : ML::MetalBuffer,
                                          out_buf : ML::MetalBuffer,
                                          in_dim : Int32,
                                          out_dim : Int32) : Nil
          enc.set_pipeline(bf16_gemv_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_off)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(out_dim.to_u32, 4)
          rows_per_tg = 2
          enc.dispatch_threadgroups({(out_dim + rows_per_tg - 1) // rows_per_tg, 1, 1}, {64, 1, 1})
        end

        private def self.encode_bf16_q_gate_gemv(enc : ML::Metal::ComputeEncoder,
                                                 w_buf : ML::MetalBuffer,
                                                 w_off : Int64,
                                                 x_buf : ML::MetalBuffer,
                                                 out_buf : ML::MetalBuffer,
                                                 in_dim : Int32,
                                                 q_dim : Int32,
                                                 head_dim : Int32) : Nil
          enc.set_pipeline(bf16_q_gate_gemv_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_off)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(q_dim.to_u32, 4)
          enc.set_value(head_dim.to_u32, 5)
          rows_per_tg = 2
          enc.dispatch_threadgroups({(q_dim + rows_per_tg - 1) // rows_per_tg, 1, 1}, {64, 1, 1})
        end

        def self.bf16_gemv(w_raw : Bytes,
                           in_dim : Int32,
                           out_dim : Int32,
                           x : Array(Float32)) : Array(Float32)
          raise "bf16_gemv input mismatch: expected #{in_dim}, got #{x.size}" unless x.size == in_dim
          expected_w = out_dim.to_i64 * in_dim.to_i64 * 2_i64
          raise "bf16_gemv weight size mismatch: expected #{expected_w}, got #{w_raw.size}" unless w_raw.size.to_i64 == expected_w

          ML::Metal::Device.init!
          w_buf, w_off = bf16_weight_slot(w_raw)
          x_buf = Scratch.get(:mtp_bf16_x, in_dim.to_i64 * sizeof(Float32))
          out_buf = Scratch.get("mtp_bf16_out_#{out_dim}", out_dim.to_i64 * sizeof(Float32))
          x_buf.write(x)

          Profile.bump_matmul_shape("gemv BF16 #{in_dim}x#{out_dim} b1", expected_w)
          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(bf16_gemv_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_off)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(out_dim.to_u32, 4)
          rows_per_tg = 2
          enc.dispatch_threadgroups({(out_dim + rows_per_tg - 1) // rows_per_tg, 1, 1}, {64, 1, 1})
          enc.end_encoding
          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, out_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        def self.bf16_q_gate_gemv(w_raw : Bytes,
                                  in_dim : Int32,
                                  q_dim : Int32,
                                  head_dim : Int32,
                                  x : Array(Float32)) : Array(Float32)
          raise "bf16_q_gate_gemv input mismatch: expected #{in_dim}, got #{x.size}" unless x.size == in_dim
          raise "bf16_q_gate_gemv q_dim must be divisible by head_dim" unless head_dim > 0 && q_dim % head_dim == 0
          expected_w = q_dim.to_i64 * 2_i64 * in_dim.to_i64 * 2_i64
          raise "bf16_q_gate_gemv weight size mismatch: expected #{expected_w}, got #{w_raw.size}" unless w_raw.size.to_i64 == expected_w

          ML::Metal::Device.init!
          w_buf, w_off = bf16_weight_slot(w_raw)
          x_buf = Scratch.get(:mtp_bf16_x, in_dim.to_i64 * sizeof(Float32))
          out_buf = Scratch.get("mtp_bf16_q_gate_out_#{q_dim}", q_dim.to_i64 * sizeof(Float32))
          x_buf.write(x)

          Profile.bump_matmul_shape("gemv BF16 q_gate #{in_dim}x#{q_dim} b1", expected_w // 2_i64)
          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(bf16_q_gate_gemv_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_off)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(q_dim.to_u32, 4)
          enc.set_value(head_dim.to_u32, 5)
          rows_per_tg = 2
          enc.dispatch_threadgroups({(q_dim + rows_per_tg - 1) // rows_per_tg, 1, 1}, {64, 1, 1})
          enc.end_encoding
          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, q_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        def self.matmul_q5k(x : Array(Float32),
                            w_raw : Bytes,
                            in_dim : Int32,
                            out_dim : Int32,
                            batch : Int32) : Array(Float32)
          raise "in_dim must be multiple of #{QK_K}: got #{in_dim}" unless in_dim % QK_K == 0
          raise "x size mismatch: expected #{batch * in_dim}, got #{x.size}" unless x.size == batch * in_dim
          expected_w = (in_dim // QK_K) * Q5K_BLOCK_BYTES * out_dim
          raise "w_raw size mismatch: expected #{expected_w}, got #{w_raw.size}" unless w_raw.size == expected_w
          ML::Metal::Device.init!
          buf, off = weight_slot(w_raw)
          if q56_batch_gemm_enabled? && batch > GEMM_BATCH_THRESHOLD
            matmul_q56k_gemm_buf(mm5_f32out_pipeline, x, buf, off, in_dim, out_dim, batch)
          else
            matmul_gemv_buf(mv5_pipeline, x, buf, off, in_dim, out_dim, batch)
          end
        end

        def self.matmul_q6k(x : Array(Float32),
                            w_raw : Bytes,
                            in_dim : Int32,
                            out_dim : Int32,
                            batch : Int32) : Array(Float32)
          raise "in_dim must be multiple of #{QK_K}: got #{in_dim}" unless in_dim % QK_K == 0
          raise "x size mismatch: expected #{batch * in_dim}, got #{x.size}" unless x.size == batch * in_dim
          expected_w = (in_dim // QK_K) * Q6K_BLOCK_BYTES * out_dim
          raise "w_raw size mismatch: expected #{expected_w}, got #{w_raw.size}" unless w_raw.size == expected_w
          ML::Metal::Device.init!
          buf, off = weight_slot(w_raw)
          if q56_batch_gemm_enabled? && batch > GEMM_BATCH_THRESHOLD
            matmul_q56k_gemm_buf(mm6_f32out_pipeline, x, buf, off, in_dim, out_dim, batch)
          else
            matmul_gemv_buf(mv6_pipeline, x, buf, off, in_dim, out_dim, batch)
          end
        end

        def self.matmul_q8_0(x : Array(Float32),
                             w_raw : Bytes,
                             in_dim : Int32,
                             out_dim : Int32,
                             batch : Int32) : Array(Float32)
          raise "in_dim must be multiple of #{Q8_0_QK}: got #{in_dim}" unless in_dim % Q8_0_QK == 0
          raise "x size mismatch: expected #{batch * in_dim}, got #{x.size}" unless x.size == batch * in_dim
          expected_w = (in_dim // Q8_0_QK) * Q8_0_BLOCK_BYTES * out_dim
          raise "w_raw size mismatch: expected #{expected_w}, got #{w_raw.size}" unless w_raw.size == expected_w
          ML::Metal::Device.init!
          buf, off = weight_slot(w_raw)
          matmul_gemv_buf(mv8_pipeline, x, buf, off, in_dim, out_dim, batch)
        end

        def self.matmul_iq4_nl(x : Array(Float32),
                               w_raw : Bytes,
                               in_dim : Int32,
                               out_dim : Int32,
                               batch : Int32) : Array(Float32)
          raise "in_dim must be multiple of #{IQ4_NL_QK}: got #{in_dim}" unless in_dim % IQ4_NL_QK == 0
          raise "x size mismatch: expected #{batch * in_dim}, got #{x.size}" unless x.size == batch * in_dim
          expected_w = (in_dim // IQ4_NL_QK) * IQ4_NL_BLOCK_BYTES * out_dim
          raise "w_raw size mismatch: expected #{expected_w}, got #{w_raw.size}" unless w_raw.size == expected_w
          ML::Metal::Device.init!
          buf, off = weight_slot(w_raw)
          matmul_gemv_buf(mv_iq4_nl_pipeline, x, buf, off, in_dim, out_dim, batch)
        end

        # Full-upload Q4_K matmul. Output row-major [batch, out_dim].
        # result[b, o] = Σ_k x[b, k] * W_dequant[o, k]
        #
        # w_raw: quantized weights [out_dim rows × in_dim cols],
        #        each row packed as (in_dim/256) Q4_K blocks of 144 bytes.
        def self.matmul_q4k(x : Array(Float32),
                            w_raw : Bytes,
                            in_dim : Int32,
                            out_dim : Int32,
                            batch : Int32) : Array(Float32)
          raise "in_dim must be multiple of #{QK_K}: got #{in_dim}" unless in_dim % QK_K == 0
          raise "x size mismatch: expected #{batch * in_dim}, got #{x.size}" unless x.size == batch * in_dim
          expected_w = (in_dim // QK_K) * Q4K_BLOCK_BYTES * out_dim
          raise "w_raw size mismatch: expected #{expected_w}, got #{w_raw.size}" unless w_raw.size == expected_w

          ML::Metal::Device.init!
          w_buf, w_off = weight_slot(w_raw)

          if batch > GEMM_BATCH_THRESHOLD
            matmul_q4k_gemm_buf(x, w_buf, w_off, in_dim, out_dim, batch)
          else
            matmul_gemv_buf(mv_pipeline, x, w_buf, w_off, in_dim, out_dim, batch)
          end
        end

        # Resolve `raw` into a (buffer, byte-offset) pair, preferring the
        # whole-mmap NoCopy buffer. Falls back to a one-shot upload for
        # bytes outside the registered mmap region.
        private def self.weight_slot(w_raw : Bytes) : {ML::MetalBuffer, Int64}
          if slot = mmap_slot_for(w_raw)
            slot
          else
            {upload_weights(w_raw), 0_i64}
          end
        end

        # Batched single-input GEMV: upload x once, encode a GEMV per
        # qw on the same compute encoder, commit+wait once, read all
        # outputs. Returns `nil` if any qw is a type we don't GPU-route
        # (caller falls back to per-qw `matmul`).
        #
        # All qws must share `in_dim == x.size`. Output shape is the list
        # of [out_dim] arrays in the same order as `qws`.
        def self.matmul_many(qws : Array(QuantWeight), x : Array(Float32)) : Array(Array(Float32))?
          return [] of Array(Float32) if qws.empty?
          ML::Metal::Device.init!

          # Resolve pipeline + weight buf for each qw upfront.
          # Bail out if any qw isn't Metal-routable.
          resolved = Array({ML::Metal::ComputePipeline, ML::MetalBuffer, Int64, Int32, Int32}).new(qws.size)
          qws.each do |qw|
            pipeline = case qw.type
                       when .q4_k? then mv_pipeline
                       when .q5_k? then mv5_pipeline
                       when .q6_k? then mv6_pipeline
                       when .q8_0? then mv8_pipeline
                       when .iq4_nl? then mv_iq4_nl_pipeline
                       when .f32? then mv_f32_pipeline
                       else
                         return nil
                       end
            buf, off = if slot = mmap_slot_for(qw.raw)
                         slot
                       else
                         {qw.fallback_metal_buffer, 0_i64}
                       end
            resolved << {pipeline, buf, off, qw.in_dim, qw.out_dim}
          end

          t0 = Time.instant if Profile.enabled?
          x_buf = Scratch.get(:mv_many_x, x.size.to_i64 * sizeof(Float32))
          x_buf.write(x)

          # Per-slot tags so concurrently-alive outputs don't alias within
          # one encoder. Up to MANY_SLOT_TAGS.size simultaneous outputs.
          out_bufs = Array(ML::MetalBuffer).new(qws.size) do |i|
            Scratch.get(MANY_SLOT_TAGS[i], qws[i].out_dim.to_i64 * sizeof(Float32))
          end

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          resolved.each_with_index do |(pipeline, w_buf, w_off, in_dim, out_dim), i|
            enc.set_pipeline(pipeline)
            enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_off)
            enc.set_buffer(x_buf, 1)
            enc.set_buffer(out_bufs[i], 2, ML::Metal::BufferAccess::Write)
            enc.set_value(in_dim.to_u32,  3)
            enc.set_value(out_dim.to_u32, 4)
            enc.set_value(1_u32,          5)
            rows_per_tg = gemv_rows_per_tg_for(pipeline)
            grid = {(out_dim + rows_per_tg - 1) // rows_per_tg, 1, 1}
            enc.dispatch_threadgroups(grid, {gemv_threads_per_tg_for(pipeline), 1, 1})
          end
          enc.end_encoding
          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          results = Array(Array(Float32)).new(qws.size) { |i| read_shared_f32(out_bufs[i], qws[i].out_dim) }
          if Profile.enabled?
            t_read = Time.instant
            # One sync for N dispatches — count as ONE gemv call so
            # `total metal syncs` reflects actual barriers, not work.
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          results
        end

        # Persistent-buffer path: dispatch by QuantWeight type, using the
        # whole-mmap buffer when available (zero-copy) or falling back to
        # a per-weight upload held by the QuantWeight itself. Returns nil
        # when the type is not GPU-supported (caller falls back to CPU).
        def self.bench_q4_layout_wait_ms(qw : QuantWeight,
                                         x : Array(Float32),
                                         batch : Int32,
                                         nsg : Int32,
                                         nr0 : Int32,
                                         validate : Bool = false) : Float64
          if validate
            _, candidate = matmul_q4_layout_wait_ms(qw, x, batch, nsg, nr0, read_output: true)
            baseline = matmul(qw, x, batch)
            raise "default Q4 baseline returned nil" if baseline.nil?
            max_diff = 0.0_f32
            candidate.each_with_index do |value, i|
              diff = (value - baseline.not_nil![i]).abs
              max_diff = diff if diff > max_diff
            end
            raise "Q4 layout nsg=#{nsg} nr0=#{nr0} max diff #{max_diff}" if max_diff > 1.0e-3_f32
          end

          elapsed, _ = matmul_q4_layout_wait_ms(qw, x, batch, nsg, nr0)
          elapsed
        end

        def self.bench_q6_layout_wait_ms(qw : QuantWeight,
                                         x : Array(Float32),
                                         batch : Int32,
                                         nsg : Int32,
                                         nr0 : Int32,
                                         validate : Bool = false) : Float64
          if validate
            _, candidate = matmul_q6_layout_wait_ms(qw, x, batch, nsg, nr0, read_output: true)
            baseline = matmul(qw, x, batch)
            raise "default Q6 baseline returned nil" if baseline.nil?
            max_diff = 0.0_f32
            candidate.each_with_index do |value, i|
              diff = (value - baseline.not_nil![i]).abs
              max_diff = diff if diff > max_diff
            end
            raise "Q6 layout nsg=#{nsg} nr0=#{nr0} max diff #{max_diff}" if max_diff > 1.0e-3_f32
          end

          elapsed, _ = matmul_q6_layout_wait_ms(qw, x, batch, nsg, nr0)
          elapsed
        end

        def self.bench_q4_h16_pair_wait_ms(gate_qw : QuantWeight,
                                           up_qw : QuantWeight,
                                           x : Array(Float32),
                                           batch : Int32,
                                           validate : Bool = false) : Float64
          raise ArgumentError.new("Q4 H16 pair bench requires Q4_K weights") unless gate_qw.type.q4_k? && up_qw.type.q4_k?
          raise ArgumentError.new("Q4 H16 pair bench requires matching dimensions") unless gate_qw.in_dim == up_qw.in_dim && gate_qw.out_dim == up_qw.out_dim
          raise ArgumentError.new("Q4 H16 pair bench x size mismatch") unless x.size == gate_qw.in_dim * batch

          ML::Metal::Device.init!
          gate_buf, gate_off = if slot = mmap_slot_for(gate_qw.raw)
                                 slot
                               else
                                 {gate_qw.fallback_metal_buffer, 0_i64}
                               end
          up_buf, up_off = if slot = mmap_slot_for(up_qw.raw)
                             slot
                           else
                             {up_qw.fallback_metal_buffer, 0_i64}
                           end

          x_buf = Scratch.get(:bench_q4_pair_x, x.size.to_i64 * sizeof(Float32))
          x_buf.write(x)
          gate_out = Scratch.get(:bench_q4_pair_gate, (batch * gate_qw.out_dim).to_i64 * sizeof(Float32))
          up_out = Scratch.get(:bench_q4_pair_up, (batch * up_qw.out_dim).to_i64 * sizeof(Float32))
          if validate
            gate_out.contents.as(Pointer(Float32))[0] = Float32::NAN
            up_out.contents.as(Pointer(Float32))[0] = Float32::NAN
          end

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_q4k_gemm_h16_pair(enc, x_buf, gate_out, up_out,
            gate_buf, gate_off, up_buf, up_off,
            gate_qw.in_dim, gate_qw.out_dim, batch)
          enc.end_encoding

          t0 = Time.instant
          cmd.commit
          cmd.wait
          elapsed = (Time.instant - t0).total_milliseconds
          if validate
            gate0 = gate_out.contents.as(Pointer(Float32))[0]
            up0 = up_out.contents.as(Pointer(Float32))[0]
            unless gate0.finite? && up0.finite?
              raise "Q4_H16 pair bench validation failed: non-finite sample gate0=#{gate0} up0=#{up0}"
            end
          end
          elapsed
        end

        # Measure the default production B64 FFN corridor: one shared F32->F16 input
        # conversion, a Float32 gate projection, and a fused H16 up+SwiGLU
        # projection. Fail closed when environment policy selects another
        # route. This is a serialized profiling seam; it does not alter policy.
        def self.bench_q4_h16_fused_swiglu_timing_ms(gate_qw : QuantWeight,
                                                      up_qw : QuantWeight,
                                                      down_qw : QuantWeight,
                                                      x : Array(Float32),
                                                      batch : Int32,
                                                      validate : Bool = false) : NamedTuple(submit_wait_ms: Float64, gpu_ms: Float64)
          @@q4_fused_bench_mutex.synchronize do
            bench_q4_h16_fused_swiglu_timing_serialized_ms(
              gate_qw, up_qw, down_qw, x, batch, validate)
          end
        end

        private def self.bench_q4_h16_fused_swiglu_timing_serialized_ms(gate_qw : QuantWeight,
                                                                        up_qw : QuantWeight,
                                                                        down_qw : QuantWeight,
                                                                        x : Array(Float32),
                                                                        batch : Int32,
                                                                        validate : Bool) : NamedTuple(submit_wait_ms: Float64, gpu_ms: Float64)
          unless batch > 0 && gate_qw.in_dim > 0 && gate_qw.out_dim > 0 &&
                 gate_qw.in_dim == up_qw.in_dim && gate_qw.out_dim == up_qw.out_dim &&
                 (gate_qw.in_dim % QK_K) == 0
            raise ArgumentError.new("fused Q4 H16 SwiGLU bench requires positive, matching Q4_K block dimensions")
          end
          unless gate_qw.type.q4_k? && up_qw.type.q4_k?
            raise ArgumentError.new("fused Q4 H16 SwiGLU bench requires Q4_K gate and up weights")
          end
          expected_weight_bytes = gate_qw.out_dim.to_i64 * (gate_qw.in_dim // QK_K).to_i64 * Q4K_BLOCK_BYTES.to_i64
          unless gate_qw.raw.size.to_i64 == expected_weight_bytes && up_qw.raw.size.to_i64 == expected_weight_bytes
            raise ArgumentError.new("fused Q4 H16 SwiGLU bench weight size mismatch")
          end
          unless (down_qw.type.q4_k? || down_qw.type.q5_k? || down_qw.type.q6_k?) &&
                 down_qw.in_dim == gate_qw.out_dim && down_qw.out_dim == gate_qw.in_dim &&
                 (down_qw.in_dim % QK_K) == 0
            raise ArgumentError.new("fused Q4 H16 SwiGLU bench requires a reversed-shape Q4_K, Q5_K, or Q6_K down weight")
          end
          down_block_bytes = if down_qw.type.q4_k?
                               Q4K_BLOCK_BYTES
                             elsif down_qw.type.q5_k?
                               Q5K_BLOCK_BYTES
                             else
                               Q6K_BLOCK_BYTES
                             end
          expected_down_bytes = down_qw.out_dim.to_i64 * (down_qw.in_dim // QK_K).to_i64 * down_block_bytes.to_i64
          unless down_qw.raw.size.to_i64 == expected_down_bytes
            raise ArgumentError.new("fused Q4 H16 SwiGLU bench down weight size mismatch")
          end
          raise ArgumentError.new("fused Q4 H16 SwiGLU bench requires pooled scratch") if ENV["QWEN35_SCRATCH_OFF"]? == "1"
          down_h16 = prefill_swiglu_h16_down_candidate?(down_qw, batch) ||
            q4_b64_up_swiglu_h16_down_candidate?(gate_qw, up_qw, down_qw, batch)
          unless q4_b64_up_swiglu_h16_candidate?(gate_qw, up_qw, batch, down_h16)
            raise ArgumentError.new("fused Q4 H16 SwiGLU bench is not selected by current gate/up/down policy")
          end
          if prefill_addnorm_h16_ffn_enabled? ||
             (q4_tensor_mm_enabled? && q4_tensor_ffn_candidate?(gate_qw.out_dim) && batch >= Q4_TENSOR_NR1) ||
             q4_h16_exact_rowpack_candidate?(batch)
            raise ArgumentError.new("fused Q4 H16 SwiGLU bench requires the default F32-input B64 gate route")
          end
          expected_input_count = gate_qw.in_dim.to_i64 * batch.to_i64
          unless expected_input_count > 0 && expected_input_count <= Int32::MAX
            raise ArgumentError.new("fused Q4 H16 SwiGLU bench input size overflow")
          end
          raise ArgumentError.new("fused Q4 H16 SwiGLU bench x size mismatch") unless x.size.to_i64 == expected_input_count
          output_count_i64 = batch.to_i64 * gate_qw.out_dim.to_i64
          unless output_count_i64 > 0 && output_count_i64 <= Int32::MAX
            raise ArgumentError.new("fused Q4 H16 SwiGLU bench output size overflow")
          end

          ML::Metal::Device.init!
          gate_w_buf, gate_w_off = weight_slot(gate_qw)
          up_w_buf, up_w_off = weight_slot(up_qw)
          output_count = output_count_i64.to_i32
          x_buf = Scratch.get(:bench_q4_fused_swiglu_x, x.size.to_i64 * sizeof(Float32))
          gate_out = Scratch.get(:bench_q4_fused_swiglu_gate, output_count.to_i64 * sizeof(Float32))
          act_h16_out = Scratch.get(:bench_q4_fused_swiglu_act_h16, output_count.to_i64 * sizeof(UInt16))
          x_buf.write(x)

          if validate
            gate_ptr = gate_out.contents.as(Pointer(Float32))
            act_ptr = act_h16_out.contents.as(Pointer(UInt16))
            gate_ptr[0] = Float32::NAN
            gate_ptr[output_count - 1] = Float32::NAN
            act_ptr[0] = 0x7e00_u16
            act_ptr[output_count - 1] = 0x7e00_u16
          end

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_q4k_gemm_h16_pair_b64_swiglu_h16(enc,
            x_buf, gate_out, act_h16_out,
            gate_w_buf, gate_w_off, up_w_buf, up_w_off,
            gate_qw.in_dim, gate_qw.out_dim, batch)
          enc.end_encoding

          t0 = Time.instant
          gpu_ms = cmd.commit_and_wait_gpu_elapsed_seconds * 1000.0
          submit_wait_ms = (Time.instant - t0).total_milliseconds

          if validate
            gate_ptr = gate_out.contents.as(Pointer(Float32))
            act_ptr = act_h16_out.contents.as(Pointer(UInt16))
            gate_first = gate_ptr[0]
            gate_last = gate_ptr[output_count - 1]
            act_first = Dequant.fp16_to_f32(act_ptr[0])
            act_last = Dequant.fp16_to_f32(act_ptr[output_count - 1])
            unless gate_first.finite? && gate_last.finite? && act_first.finite? && act_last.finite?
              raise "fused Q4 H16 SwiGLU bench validation failed: non-finite boundary output"
            end

            ref_gate_out = Scratch.get(:bench_q4_fused_swiglu_ref_gate, output_count.to_i64 * sizeof(Float32))
            ref_up_out = Scratch.get(:bench_q4_fused_swiglu_ref_up, output_count.to_i64 * sizeof(Float32))
            ref_act_h16_out = Scratch.get(:bench_q4_fused_swiglu_ref_act_h16, output_count.to_i64 * sizeof(UInt16))
            ref_cmd = ML::Metal::CommandBuffer.new
            ref_enc = ML::Metal::ComputeEncoder.new(ref_cmd)
            encode_q4k_gemm_h16_pair(ref_enc,
              x_buf, ref_gate_out, ref_up_out,
              gate_w_buf, gate_w_off, up_w_buf, up_w_off,
              gate_qw.in_dim, gate_qw.out_dim, batch)
            ref_enc.set_pipeline(ffn_swiglu_h16_pipeline)
            ref_enc.set_buffer(ref_gate_out, 0)
            ref_enc.set_buffer(ref_up_out, 1)
            ref_enc.set_buffer(ref_act_h16_out, 2, ML::Metal::BufferAccess::Write)
            ref_enc.set_value(output_count.to_u32, 3)
            ref_enc.dispatch_1d(output_count, 256)
            ref_enc.end_encoding
            ref_cmd.commit_and_wait

            ref_gate_ptr = ref_gate_out.contents.as(Pointer(Float32))
            ref_act_ptr = ref_act_h16_out.contents.as(Pointer(UInt16))
            output_count.times do |i|
              unless gate_ptr[i].unsafe_as(UInt32) == ref_gate_ptr[i].unsafe_as(UInt32)
                raise "fused Q4 H16 SwiGLU gate parity failed at index #{i}"
              end
              unless act_ptr[i] == ref_act_ptr[i]
                raise "fused Q4 H16 SwiGLU activation parity failed at index #{i}"
              end
            end
          end

          {submit_wait_ms: submit_wait_ms, gpu_ms: gpu_ms}
        end

        # Pair-only Q4_K FFN projection: share one F32->F16 input conversion
        # across gate/up while preserving the caller's activation path.
        def self.encode_q4k_gemm_h16_pair_to_buffers(enc : ML::Metal::ComputeEncoder,
                                                      gate_qw : QuantWeight,
                                                      up_qw : QuantWeight,
                                                      x_buf : ML::MetalBuffer,
                                                      gate_buf : ML::MetalBuffer,
                                                      up_buf : ML::MetalBuffer,
                                                      batch : Int32) : Bool
          return false unless q4_pair_h16_gemm_candidate?(gate_qw, up_qw, batch)
          return false if x_buf.size < batch.to_i64 * gate_qw.in_dim * sizeof(Float32)
          return false if gate_buf.size < batch.to_i64 * gate_qw.out_dim * sizeof(Float32)
          return false if up_buf.size < batch.to_i64 * up_qw.out_dim * sizeof(Float32)

          gate_w_buf, gate_w_off = weight_slot(gate_qw)
          up_w_buf, up_w_off = weight_slot(up_qw)
          encode_q4k_gemm_h16_pair(enc, x_buf, gate_buf, up_buf,
            gate_w_buf, gate_w_off, up_w_buf, up_w_off,
            gate_qw.in_dim, gate_qw.out_dim, batch)
          true
        end

        # Gemma-style FFN producer-consumer fusion: gate is materialized as F32,
        # up is consumed tile-local, and the output is GELU(gate) * up.
        def self.encode_q4k_gemm_h16_pair_b64_gelu_mul(enc : ML::Metal::ComputeEncoder,
                                                        gate_qw : QuantWeight,
                                                        up_qw : QuantWeight,
                                                        x_buf : ML::MetalBuffer,
                                                        gate_buf : ML::MetalBuffer,
                                                        act_buf : ML::MetalBuffer,
                                                        batch : Int32) : Bool
          return false unless q4_pair_h16_gemm_candidate?(gate_qw, up_qw, batch)
          return false unless q4_h16_b64_gemm_enabled?
          return false unless q4_h16_b64_swiglu_batch_candidate?(batch)
          return false unless (up_qw.out_dim % MM_NR0) == 0
          return false if x_buf.size < batch.to_i64 * gate_qw.in_dim * sizeof(Float32)
          return false if gate_buf.size < batch.to_i64 * gate_qw.out_dim * sizeof(Float32)
          return false if act_buf.size < batch.to_i64 * gate_qw.out_dim * sizeof(Float32)

          gate_w_buf, gate_w_off = weight_slot(gate_qw)
          up_w_buf, up_w_off = weight_slot(up_qw)
          x16_buf = Scratch.get(:mm4_pair_gelu_x16, (batch * gate_qw.in_dim).to_i64 * 2_i64)

          Profile.bump_matmul_shape("q4_h16_gemm #{gate_qw.type.name} #{gate_qw.in_dim}x#{gate_qw.out_dim} b#{batch}", gate_qw.raw.size.to_i64)
          Profile.bump_matmul_shape("q4_h16_gemm_gelu_mul #{up_qw.type.name} #{up_qw.in_dim}x#{up_qw.out_dim} b#{batch}", up_qw.raw.size.to_i64)
          Profile.bump_conversion("f32_to_f16 q4_pair_gelu_input #{gate_qw.in_dim} b#{batch}", (batch * gate_qw.in_dim).to_i64 * 6_i64)
          enc.set_pipeline(f32_to_f16_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(x16_buf, 1, ML::Metal::BufferAccess::Write)
          enc.set_value((batch * gate_qw.in_dim).to_u32, 2)
          enc.dispatch_1d(batch * gate_qw.in_dim, 256)

          encode_q4k_gemm_h16_from_h16(enc, x16_buf, gate_buf, gate_w_buf, gate_w_off, gate_qw.in_dim, gate_qw.out_dim, batch)
          encode_q4k_gemm_h16_b64_gelu_mul_from_h16(enc, x16_buf, gate_buf, act_buf, up_w_buf, up_w_off, up_qw.in_dim, up_qw.out_dim, batch)
          true
        end

        def self.matmul(qw : QuantWeight, x : Array(Float32), batch : Int32) : Array(Float32)?
          ML::Metal::Device.init!

          buf, off = if slot = mmap_slot_for(qw.raw)
                       slot
                     else
                       {qw.fallback_metal_buffer, 0_i64}
                     end

          case qw.type
          when .q4_k?
            if batch > GEMM_BATCH_THRESHOLD
              matmul_q4k_gemm_buf(x, buf, off, qw.in_dim, qw.out_dim, batch)
            else
              matmul_gemv_buf(mv_pipeline, x, buf, off, qw.in_dim, qw.out_dim, batch)
            end
          when .q5_k?
            if q56_batch_gemm_enabled? && batch > GEMM_BATCH_THRESHOLD
              matmul_q56k_gemm_buf(mm5_pipeline, x, buf, off, qw.in_dim, qw.out_dim, batch)
            else
              matmul_gemv_buf(mv5_pipeline, x, buf, off, qw.in_dim, qw.out_dim, batch)
            end
          when .q6_k?
            if q56_batch_gemm_enabled? && batch > GEMM_BATCH_THRESHOLD
              matmul_q56k_gemm_buf(mm6_pipeline, x, buf, off, qw.in_dim, qw.out_dim, batch)
            else
              matmul_gemv_buf(mv6_pipeline, x, buf, off, qw.in_dim, qw.out_dim, batch)
            end
          when .q8_0?
            matmul_gemv_buf(mv8_pipeline, x, buf, off, qw.in_dim, qw.out_dim, batch)
          when .iq4_nl?
            if batch > GEMM_BATCH_THRESHOLD
              nil
            else
              matmul_gemv_buf(mv_iq4_nl_pipeline, x, buf, off, qw.in_dim, qw.out_dim, batch)
            end
          when .f32?
            if batch > GEMM_BATCH_THRESHOLD
              nil
            else
              matmul_gemv_buf(mv_f32_pipeline, x, buf, off, qw.in_dim, qw.out_dim, batch)
            end
          else
            nil
          end
        end

        # Resident-buffer matmul transport for callers that already keep the
        # input vector on Metal. This intentionally reuses the same encoder and
        # routing as `matmul` so new resident paths cannot silently diverge from
        # the established Q4/Q5/Q6/Q8/IQ4/F32 kernels.
        def self.matmul_to_buffer(qw : QuantWeight,
                                  x_buf : ML::MetalBuffer,
                                  out_buf : ML::MetalBuffer,
                                  batch : Int32 = 1) : Bool
          return false if batch <= 0
          return false if x_buf.size < batch.to_i64 * qw.in_dim * sizeof(Float32)
          return false if out_buf.size < batch.to_i64 * qw.out_dim * sizeof(Float32)

          ML::Metal::Device.init!
          pipeline = gemv_pipeline_for(qw)
          return false if pipeline.nil?
          w_buf, w_off = weight_slot(qw)

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_matmul(enc, pipeline, qw, x_buf, out_buf, w_buf, w_off, qw.in_dim, qw.out_dim, batch)
          enc.end_encoding
          cmd.commit
          cmd.wait
          true
        end

        def self.encode_matmul_to_buffer(enc : ML::Metal::ComputeEncoder,
                                         qw : QuantWeight,
                                         x_buf : ML::MetalBuffer,
                                         out_buf : ML::MetalBuffer,
                                         batch : Int32 = 1) : Bool
          return false if batch <= 0
          return false if x_buf.size < batch.to_i64 * qw.in_dim * sizeof(Float32)
          return false if out_buf.size < batch.to_i64 * qw.out_dim * sizeof(Float32)

          ML::Metal::Device.init!
          pipeline = gemv_pipeline_for(qw)
          return false if pipeline.nil?
          w_buf, w_off = weight_slot(qw)

          encode_matmul(enc, pipeline, qw, x_buf, out_buf, w_buf, w_off, qw.in_dim, qw.out_dim, batch)
          true
        end

        def self.encode_matmul_to_buffer(enc : ML::Metal::GraphEncoder,
                                         qw : QuantWeight,
                                         x_buf : ML::MetalBuffer,
                                         out_buf : ML::MetalBuffer,
                                         batch : Int32 = 1) : Bool
          return false if batch <= 0 || batch > GEMM_BATCH_THRESHOLD
          return false if x_buf.size < batch.to_i64 * qw.in_dim * sizeof(Float32)
          return false if out_buf.size < batch.to_i64 * qw.out_dim * sizeof(Float32)

          ML::Metal::Device.init!
          pipeline = gemv_pipeline_for(qw)
          return false if pipeline.nil?
          w_buf, w_off = weight_slot(qw)

          encode_gemv(enc, pipeline, x_buf, out_buf, w_buf, w_off, qw.in_dim, qw.out_dim, batch,
            profile_shape: true, route_qw: qw)
          true
        end

        def self.encode_matmul_from_h16_to_buffer(enc : ML::Metal::ComputeEncoder,
                                                  qw : QuantWeight,
                                                  x16_buf : ML::MetalBuffer,
                                                  out_buf : ML::MetalBuffer,
                                                  batch : Int32 = 1) : Bool
          return false if batch <= 0
          return false if x16_buf.size < batch.to_i64 * qw.in_dim * 2_i64
          return false if out_buf.size < batch.to_i64 * qw.out_dim * sizeof(Float32)

          ML::Metal::Device.init!
          w_buf, w_off = weight_slot(qw)
          encode_matmul_from_h16(enc, qw, x16_buf, out_buf, w_buf, w_off, qw.in_dim, qw.out_dim, batch)
        end

        def self.matmul_many_to_buffers(qws : Array(QuantWeight),
                                        x_buf : ML::MetalBuffer,
                                        out_bufs : Array(ML::MetalBuffer),
                                        batch : Int32 = 1) : Bool
          return false if batch <= 0
          return false if qws.empty? || qws.size != out_bufs.size
          in_dim = qws[0].in_dim
          return false if x_buf.size < batch.to_i64 * in_dim * sizeof(Float32)

          slots = [] of {ML::Metal::ComputePipeline, ML::MetalBuffer, Int64, QuantWeight, ML::MetalBuffer}
          qws.each_with_index do |qw, i|
            return false unless qw.in_dim == in_dim
            out_buf = out_bufs[i]
            return false if out_buf.size < batch.to_i64 * qw.out_dim * sizeof(Float32)
            pipeline = gemv_pipeline_for(qw)
            return false if pipeline.nil?
            w_buf, w_off = weight_slot(qw)
            slots << {pipeline, w_buf, w_off, qw, out_buf}
          end

          ML::Metal::Device.init!
          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          slots.each do |pipeline, w_buf, w_off, qw, out_buf|
            encode_matmul(enc, pipeline, qw, x_buf, out_buf, w_buf, w_off, qw.in_dim, qw.out_dim, batch)
          end
          enc.end_encoding
          cmd.commit
          cmd.wait
          true
        end

        def self.encode_matmul_many_to_buffers(enc : ML::Metal::ComputeEncoder,
                                               qws : Array(QuantWeight),
                                               x_buf : ML::MetalBuffer,
                                               out_bufs : Array(ML::MetalBuffer),
                                               batch : Int32 = 1) : Bool
          return false if batch <= 0
          return false if qws.empty? || qws.size != out_bufs.size
          in_dim = qws[0].in_dim
          return false if x_buf.size < batch.to_i64 * in_dim * sizeof(Float32)

          slots = [] of {ML::Metal::ComputePipeline, ML::MetalBuffer, Int64, QuantWeight, ML::MetalBuffer}
          qws.each_with_index do |qw, i|
            return false unless qw.in_dim == in_dim
            out_buf = out_bufs[i]
            return false if out_buf.size < batch.to_i64 * qw.out_dim * sizeof(Float32)
            pipeline = gemv_pipeline_for(qw)
            return false if pipeline.nil?
            w_buf, w_off = weight_slot(qw)
            slots << {pipeline, w_buf, w_off, qw, out_buf}
          end

          ML::Metal::Device.init!
          if matmul_many_shared_h16_enabled?(batch) && batch > GEMM_BATCH_THRESHOLD && qws.all? { |qw| h16_batch_gemm_candidate?(qw, batch) }
            x16_buf = Scratch.get(:matmul_many_x16, batch.to_i64 * in_dim * 2_i64)
            Profile.bump_conversion("f32_to_f16 matmul_many_shared_input #{in_dim} b#{batch}", (batch * in_dim).to_i64 * 6_i64)
            enc.set_pipeline(f32_to_f16_pipeline)
            enc.set_buffer(x_buf, 0)
            enc.set_buffer(x16_buf, 1, ML::Metal::BufferAccess::Write)
            enc.set_value((batch * in_dim).to_u32, 2)
            enc.dispatch_1d(batch * in_dim, 256)

            slots.each do |_pipeline, w_buf, w_off, qw, out_buf|
              return false unless encode_matmul_from_h16(enc, qw, x16_buf, out_buf, w_buf, w_off, qw.in_dim, qw.out_dim, batch)
            end
            return true
          end

          slots.each do |pipeline, w_buf, w_off, qw, out_buf|
            encode_matmul(enc, pipeline, qw, x_buf, out_buf, w_buf, w_off, qw.in_dim, qw.out_dim, batch)
          end
          true
        end

        def self.encode_matmul_many_to_buffers(enc : ML::Metal::GraphEncoder,
                                               qws : Array(QuantWeight),
                                               x_buf : ML::MetalBuffer,
                                               out_bufs : Array(ML::MetalBuffer),
                                               batch : Int32 = 1) : Bool
          return false if batch <= 0 || batch > GEMM_BATCH_THRESHOLD
          return false if qws.empty? || qws.size != out_bufs.size
          in_dim = qws[0].in_dim
          return false if x_buf.size < batch.to_i64 * in_dim * sizeof(Float32)

          qws.each_with_index do |qw, i|
            return false unless qw.in_dim == in_dim
            out_buf = out_bufs[i]
            return false if out_buf.size < batch.to_i64 * qw.out_dim * sizeof(Float32)
            pipeline = gemv_pipeline_for(qw)
            return false if pipeline.nil?
            w_buf, w_off = weight_slot(qw)
            encode_gemv(enc, pipeline, x_buf, out_buf, w_buf, w_off, qw.in_dim, qw.out_dim, batch,
              profile_shape: true, route_qw: qw)
          end
          true
        end
      {% end %}
    end
  end
end
