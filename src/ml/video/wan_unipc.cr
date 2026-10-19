require "./metal_ops"

module ML::Video
  # The upstream default only: flow prediction, UniPC order 2, bh2, x0,
  # lower_order_final, no thresholding or disabled correctors. Scalar schedule
  # preparation is setup work; samples and all solver history stay on the GPU.
  class WanUniPC
    SOURCE = {{ read_file("#{__DIR__}/kernels/wan_unipc.metal") }}
    @sigmas : Array(Float32)
    @timesteps : Array(Int32)
    getter steps : Int32
    getter count : Int32
    getter index : Int32 = 0

    record Coefficients, ratio : Float32, alpha_b : Float32, rk : Float32,
      rho0 : Float32, rho1 : Float32, order : Int32

    @predict : Array(Coefficients)
    @correct : Array(Coefficients)

    def initialize(@count : Int32, @steps : Int32 = 40, shift : Float64 = 3.0)
      raise ArgumentError.new("Invalid Wan UniPC schedule") unless count > 0 && steps > 0 && steps <= 1000 && shift.finite? && shift > 0
      sigma_max = 0.999_f32.to_f64
      original = Array(Float64).new(steps) do |i|
        s = sigma_max * (1.0 - i.to_f64 / steps)
        shift * s / (1.0 + (shift - 1.0) * s)
      end
      @sigmas = original.map(&.to_f32) << 0_f32
      unless @sigmas[0, steps].all? { |s| s.finite? && s > 0 && s < 1 } &&
             @sigmas.each_cons_pair.all? { |a, b| a > b }
        raise ArgumentError.new("Wan UniPC requires finite strictly decreasing F32 sigmas below one")
      end
      @timesteps = original.map { |s| (s * 1000).to_i32 }
      unless @timesteps.uniq.size == steps
        raise ArgumentError.new("Wan UniPC requires distinct integer timesteps")
      end
      @predict = Array(Coefficients).new(steps) do |i|
        order = i == 0 || i == steps - 1 ? 1 : 2
        coefficients(i, i + 1, order)
      end
      @correct = Array(Coefficients).new(steps) do |i|
        i == 0 ? Coefficients.new(0_f32, 0_f32, 1_f32, 0_f32, 0_f32, 0) : coefficients(i - 1, i, i == 1 ? 1 : 2)
      end
    end

    def sigmas : Array(Float32)
      @sigmas.dup
    end

    def timesteps : Array(Int32)
      @timesteps.dup
    end

    # These are the same F32 scalar operations as the upstream CPU schedule.
    # They neither inspect activations nor synchronize GPU work.
    private def lambda_at(i : Int32) : Float32
      s = @sigmas[i]
      Math.log(1_f32 - s).to_f32 - Math.log(s).to_f32
    end

    private def coefficients(source : Int32, target : Int32, order : Int32) : Coefficients
      s = @sigmas[source]
      t = @sigmas[target]
      alpha = 1_f32 - t
      if t == 0
        return Coefficients.new(0_f32, -1_f32, 1_f32, 0_f32, 0_f32, 1)
      end
      h = lambda_at(target) - lambda_at(source)
      hh = -h
      b = Math.expm1(hh).to_f32
      rk = order == 2 ? (lambda_at(source - 1) - lambda_at(source)) / h : 1_f32
      # Predictor order 2 always uses rho=1/2; corrector order 2 solves
      # [[1,1],[rk,1]] rho = [phi2/B, 2*phi3/B].
      phi2 = b / hh - 1_f32
      b1 = phi2 / b
      b2 = ((phi2 / hh - 0.5_f32) * 2_f32) / b
      rho0 = order == 2 ? (b2 - b1) / (rk - 1_f32) : 0_f32
      rho1 = order == 2 ? b1 - rho0 : 0.5_f32
      Coefficients.new(t / s, alpha * b, rk, rho0, rho1, order)
    end

    def self.pipeline(name : String) : ML::Metal::ComputePipeline
      ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
    end

    def prepare_scratch!(a : Arena, prefix : String = "unipc.") : Nil
      {"wan_unipc_convert", "wan_unipc_correct", "wan_unipc_predict", "wan_unipc_guidance"}.each { |n| self.class.pipeline(n) }
      {"m0", "m1", "current", "last", "corrected"}.each { |n| a.alloc(prefix + n, count) }
    end

    def reset! : Nil
      @index = 0
    end

    def scratch_bytes : Int64
      5_i64 * count * 4
    end

    # CFG follows upstream uncond + scale * (cond - uncond), without FMA.
    def guidance(e : ML::Metal::ComputeEncoder, conditional : ML::MetalBuffer,
                 unconditional : ML::MetalBuffer, output : ML::MetalBuffer,
                 scale : Float32 = 4.5_f32) : Nil
      raise ArgumentError.new("Invalid guidance scale") unless scale.finite?
      {conditional, unconditional, output}.each { |b| MetalOps.check(b, count.to_i64) }
      MetalOps.inplace_or_distinct(conditional, output)
      MetalOps.inplace_or_distinct(unconditional, output)
      e.set_pipeline(self.class.pipeline("wan_unipc_guidance"))
      e.set_buffer(conditional, 0).set_buffer(unconditional, 1).set_buffer(output, 2, ML::Metal::BufferAccess::Write)
      e.set_value(count.to_u32, 3).set_value(scale, 4).dispatch_1d(count)
    end

    # All steps can be encoded into one caller-owned serial encoder. Reset only
    # after the preceding request has been fully encoded, preserving command
    # order and the lifetime of every bound buffer until terminal completion.
    def encode_step(e : ML::Metal::ComputeEncoder, a : Arena, model : ML::MetalBuffer,
                    sample : ML::MetalBuffer, output : ML::MetalBuffer,
                    prefix : String = "unipc.") : Nil
      raise ArgumentError.new("Wan UniPC schedule exhausted") unless index < steps
      buffers = {"m0", "m1", "current", "last", "corrected"}.map { |n| a[prefix + n] }
      buffers.each_with_index do |b, i|
        MetalOps.check(b, count.to_i64)
        MetalOps.distinct(b, model, sample, output)
        i.times { |j| MetalOps.distinct(b, buffers[j]) }
      end
      {model, sample, output}.each { |b| MetalOps.check(b, count.to_i64) }
      MetalOps.distinct(output, model)
      MetalOps.inplace_or_distinct(sample, output)
      m0, m1, current, last, corrected = buffers
      e.set_pipeline(self.class.pipeline("wan_unipc_convert"))
      e.set_buffer(sample, 0).set_buffer(model, 1).set_buffer(current, 2, ML::Metal::BufferAccess::Write)
      e.set_value(count.to_u32, 3).set_value(@sigmas[index], 4).dispatch_1d(count)
      if index > 0
        c = @correct[index]
        e.set_pipeline(self.class.pipeline("wan_unipc_correct"))
        e.set_buffer(last, 0).set_buffer(m0, 1).set_buffer(m1, 2).set_buffer(current, 3)
        e.set_buffer(corrected, 4, ML::Metal::BufferAccess::Write)
        e.set_value(count.to_u32, 5).set_value(c.ratio, 6).set_value(c.alpha_b, 7)
        e.set_value(c.rk, 8).set_value(c.rho0, 9).set_value(c.rho1, 10).set_value(c.order.to_u32, 11)
        e.dispatch_1d(count)
      else
        MetalOps.copy(e, sample, corrected, count)
      end
      # Shift history after the corrector has consumed both preceding outputs.
      MetalOps.copy(e, m0, m1, count) if index > 0
      MetalOps.copy(e, current, m0, count)
      MetalOps.copy(e, corrected, last, count)
      c = @predict[index]
      e.set_pipeline(self.class.pipeline("wan_unipc_predict"))
      e.set_buffer(corrected, 0).set_buffer(m0, 1).set_buffer(m1, 2)
      e.set_buffer(output, 3, ML::Metal::BufferAccess::Write)
      e.set_value(count.to_u32, 4).set_value(c.ratio, 5).set_value(c.alpha_b, 6)
      e.set_value(c.rk, 7).set_value(c.order.to_u32, 8).dispatch_1d(count)
      @index += 1
    end
  end
end
