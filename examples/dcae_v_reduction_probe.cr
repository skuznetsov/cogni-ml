require "option_parser"
require "json"
require "./support/hy_nomas_probe_instrumentation"
require "../src/ml/metal/device"
require "../src/ml/video/bf16_weight_cache"
require "../src/ml/video/dcae_v_mps"
require "./support/dcae_v_activation_prefix"

@[Link(ldflags: "-framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph")]
lib DCAEReductionFFI
  fun template_create = gs_dcae_reduction_template_create(device : Pointer(Void), mode : Int32, cache_input : Int32, height : Int32, width : Int32) : Pointer(Void)
  fun plan_create = gs_dcae_reduction_plan_create(template : Pointer(Void), input : Pointer(Void), cache : Pointer(Void), weight : Pointer(Void), bias : Pointer(Void), out0 : Pointer(Void), out1 : Pointer(Void)) : Pointer(Void)
  fun encode = gs_dcae_reduction_encode(plan : Pointer(Void), command : Pointer(Void)) : Int32
  fun plan_release = gs_dcae_reduction_plan_release(plan : Pointer(Void)) : Void
  fun template_release = gs_dcae_reduction_template_release(template : Pointer(Void)) : Void
end

# Offline diagnostic only. All model arithmetic and coordinate selection stay on
# one existing command chain, with exactly one terminal drain before exports.
module DCAEReductionProbe
  extend self
  DEFAULT_WEIGHTS = "/opt/mldata_ssd/models/hy-nomas/dcae-v-native-decoder-ebbee483/decoder-bf16.safetensors"
  DEFAULT_SHA     = "f53c94b06685027b8131b7f52cab519edee1e51287d93d8fdac6a17506f8881b"
  WEIGHT          = "decoder.stages.2.op_list.1.main.conv1.conv.weight"
  BIAS            = "decoder.stages.2.op_list.1.main.conv1.conv.bias"
  CHANNELS        =  512
  FRAMES          =   40
  SAMPLES         = 8192
  UNIFORM         = 4096
  VECTOR_K        =   32
  VECTOR_COUNT    =  256
  SOURCE          = {{ read_file("#{__DIR__}/support/dcae_v_reduction_probe.metal") }} + "\n" + {{ read_file("#{__DIR__}/support/dcae_v_reduction_native.metal") }}

  private record Args, weights : String, sha : String, output : String, vectors : String,
    small : Bool, reference : String, reference_sha : String

  private def parse_args : Args
    weights, sha, output, vectors, small, admitted = DEFAULT_WEIGHTS, DEFAULT_SHA, "", "", false, false
    reference, reference_sha = "", ""
    OptionParser.parse do |p|
      p.banner = "Usage: dcae_v_reduction_probe --admitted --output DIR --vectors DIR [--small-fixture]"
      p.on("--weights PATH", "Pinned BF16 checkpoint") { |v| weights = v }
      p.on("--weights-sha256 SHA", "Checkpoint hash") { |v| sha = v }
      p.on("--output DIR", "Fresh absolute output directory") { |v| output = v }
      p.on("--vectors DIR", "Exact-reference validation vectors") { |v| vectors = v }
      p.on("--small-fixture", "Retained H3/W5 ordered-FMA positive control") { small = true }
      p.on("--activation-prefix PATH", "Authenticated two-tile decoder reference for real stage-2 inputs") { |v| reference = v }
      p.on("--activation-prefix-sha256 SHA", "Two-tile reference hash") { |v| reference_sha = v }
      p.on("--admitted", "Required external guard admission") { admitted = true }
      p.on("-h", "--help", "Show help") { puts p; exit }
    end
    raise ArgumentError.new("Guard admission is required") unless admitted
    raise ArgumentError.new("Absolute paths and a lowercase checkpoint hash are required") unless {weights, output, vectors}.all? { |v| Path[v].absolute? } && sha.matches?(/\A[0-9a-f]{64}\z/)
    raise ArgumentError.new("Output must be fresh; vectors must exist") if File.exists?(output) || !File.directory?(vectors)
    if !reference.empty? || !reference_sha.empty?
      unless !small && Path[reference].absolute? && reference_sha.matches?(/\A[0-9a-f]{64}\z/)
        raise ArgumentError.new("Real activations require a pinned absolute reference and full geometry")
      end
    end
    Args.new(weights, sha, output, vectors, small, reference, reference_sha)
  end

  private def words(values : Array(UInt32)) : ML::MetalBuffer
    b = ML::MetalBuffer.new(values.size.to_i64 * 4)
    b.write_bytes(values.to_unsafe.as(UInt8*), values.size * 4)
    b
  end

  private def raw_file(path : String, bytes : Int32) : ML::MetalBuffer
    raw = File.read(path).to_slice
    raise ArgumentError.new("Wrong vector bytes: #{path}") unless raw.size == bytes
    b = ML::MetalBuffer.new(bytes.to_i64)
    b.write_bytes(raw.to_unsafe, raw.size)
    b
  end

  private def export_raw(path : String, buffer : ML::MetalBuffer) : Nil
    raise ArgumentError.new("Refusing to overwrite #{path}") if File.exists?(path)
    File.open(path, "wb") { |io| io.write(Bytes.new(buffer.contents.as(UInt8*), buffer.size.to_i32)) }
  end

  private record Candidate, name : String, params : ML::MetalBuffer, raw : ML::MetalBuffer, pre : ML::MetalBuffer, post : ML::MetalBuffer
  private record CaseBuffers, name : String, valid : Bool, indices : ML::MetalBuffer, stats : ML::MetalBuffer, original : ML::MetalBuffer, pre : ML::MetalBuffer, post : ML::MetalBuffer, raw : ML::MetalBuffer, candidates : Array(Candidate)

  class ReductionTemplate
    @handle = Pointer(Void).null
    @released = false
    getter mode : Int32
    getter cache_input : Bool

    def initialize(@mode : Int32, @cache_input : Bool, height : Int32, width : Int32)
      @handle = DCAEReductionFFI.template_create(ML::Metal::Device.instance.device_handle,
        mode, cache_input ? 1 : 0, height, width)
      raise "Failed to compile arithmetic-boundary MPSGraph template mode=#{mode} cache=#{cache_input}" if @handle.null?
    end

    def bind(input : ML::MetalBuffer, cache : ML::MetalBuffer?, weight : ML::MetalBuffer,
             bias : ML::MetalBuffer?, output0 : ML::MetalBuffer,
             output1 : ML::MetalBuffer? = nil) : ReductionPlan
      raise "Boundary template released" if @released || @handle.null?
      handle = DCAEReductionFFI.plan_create(@handle, input.handle,
        cache.try(&.handle) || Pointer(Void).null, weight.handle,
        bias.try(&.handle) || Pointer(Void).null, output0.handle,
        output1.try(&.handle) || Pointer(Void).null)
      raise "Failed to bind arithmetic-boundary graph outputs mode=#{@mode}" if handle.null?
      buffers = [input, weight, output0] of ML::MetalBuffer
      buffers << cache if cache
      buffers << bias if bias
      buffers << output1 if output1
      ReductionPlan.new(handle, buffers)
    end

    def release : Nil
      return if @released
      DCAEReductionFFI.template_release(@handle) unless @handle.null?
      @handle = Pointer(Void).null
      @released = true
    end
  end

  class ReductionPlan
    @handle : Pointer(Void)
    @buffers : Array(ML::MetalBuffer)
    @chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def initialize(@handle : Pointer(Void), @buffers : Array(ML::MetalBuffer))
      @buffers = @buffers.uniq { |buffer| buffer.object_id }
    end

    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      raise "Boundary plan released" if @released || @handle.null?
      if previous = @chain
        raise ArgumentError.new("Boundary plan cannot span an in-flight chain") unless previous.same?(chain) || previous.completed?
      end
      @chain = chain
      chain.pause!
      status = -1
      begin
        status = DCAEReductionFFI.encode(@handle, chain.handle)
      ensure
        chain.resume!
      end
      raise "Boundary MPSGraph encode rejected (#{status})" unless status == 0
    end

    def release(chain : ML::Metal::MPSCommandChain? = nil) : Nil
      return if @released
      if previous = @chain
        raise ArgumentError.new("Drain boundary graph chain before release") unless previous.completed?
        raise ArgumentError.new("Boundary plan release chain mismatch") if chain && !previous.same?(chain.not_nil!)
      end
      DCAEReductionFFI.plan_release(@handle) unless @handle.null?
      @handle = Pointer(Void).null
      @chain = nil
      @buffers.clear
      @released = true
    end
  end

  private def native(chain, pipeline, input, cache, weight, bias, item : Candidate, indices) : Nil
    e = chain.encoder
    e.set_pipeline(pipeline)
    {input, cache, weight, indices, item.raw, item.pre, item.post, bias, item.params}.each_with_index do |b, i|
      e.set_buffer(b, i, {4, 5, 6}.includes?(i) ? ML::Metal::BufferAccess::Write : ML::Metal::BufferAccess::Read)
    end
    e.dispatch_1d(SAMPLES, 64)
  end

  def run : Nil
    args = parse_args
    h, w = args.small ? {3, 5} : {72, 128}
    count = CHANNELS * FRAMES * h * w
    cache_count = CHANNELS * h * w
    Dir.mkdir(args.output)
    checkpoint = ML::Video::BF16WeightCache.authenticate(args.weights, args.sha)
    reference = nil.as(ML::Video::CheckpointFile?)
    prefix = nil.as(DCAEVActivationPrefix?)
    arena = ML::Video::Arena.new
    prepared = nil.as(ML::Video::PreparedWeights?)
    buffers = [] of ML::MetalBuffer
    baseline_templates = [] of ML::Video::DCAEVConvMPS::Template
    baseline_plans = [] of ML::Video::DCAEVConvMPS::Plan
    templates = [] of ReductionTemplate
    plans = [] of ReductionPlan
    chain = nil.as(ML::Metal::MPSCommandChain?)
    begin
      unless args.reference.empty?
        reference = ML::Video::BF16WeightCache.authenticate(args.reference, args.reference_sha)
        prefix = DCAEVActivationPrefix.new(checkpoint, reference.not_nil!)
      end
      weight_shape = [512_i64, 512_i64, 3_i64, 3_i64, 3_i64]
      bias_shape = [512_i64]
      bindings = [
        ML::Video::WeightBinding.new(WEIGHT, WEIGHT, weight_shape, ML::GGUF::SafeTensorDType::BF16),
        ML::Video::WeightBinding.new(BIAS, BIAS, bias_shape, ML::GGUF::SafeTensorDType::BF16),
      ]
      reserve_bytes = count.to_i64 * 14 + cache_count.to_i64 * 2 + 64_i64 * 1024 * 1024
      weight_plan = ML::Video::WeightPlan.new([checkpoint], bindings, reserve_bytes + 32_i64 * 1024 * 1024,
        reserve_bytes: reserve_bytes,
        resident_dtype: ML::Video::WeightDType::BF16)
      raise "Expected direct resident BF16 weights" unless weight_plan.staging_bytes == 0
      prepared = ML::Video::PreparedWeights.new(weight_plan, arena)
      weight, bias = arena[WEIGHT], arena[BIAS]
      alloc = ->(bytes : Int64) do
        b = ML::MetalBuffer.new(bytes)
        buffers << b
        b
      end
      params = ->(v : Array(UInt32)) do
        b = words(v)
        buffers << b
        b
      end
      if actual = prefix
        inputs = [actual.activation(0), actual.activation(1)]
        caches = [actual.old_cache(0), actual.old_cache(1)]
      else
        input, cache = alloc.call(count.to_i64 * 2), alloc.call(cache_count.to_i64 * 2)
        inputs, caches = [input, input], [cache, cache]
      end
      original, pre, post = alloc.call(count.to_i64 * 2), alloc.call(count.to_i64 * 2), alloc.call(count.to_i64 * 2)
      f32 = alloc.call(count.to_i64 * 4)
      fill_input = params.call([count.to_u32, 0x2026_u32, args.small ? 0_u32 : 1_u32])
      fill_cache = params.call([cache_count.to_u32, 0x0512_u32, args.small ? 0_u32 : 1_u32])
      uniform_params = params.call([CHANNELS.to_u32, FRAMES.to_u32, h.to_u32, w.to_u32, UNIFORM.to_u32])
      midpoint_params = params.call([count.to_u32, (SAMPLES - UNIFORM).to_u32, UNIFORM.to_u32])
      gather_params = params.call([SAMPLES.to_u32, count.to_u32])
      cases = [false, true].map do |valid|
        name = valid ? "valid" : "invalid"
        candidates = [] of Candidate
        [0_u32, 1_u32].each do |order|
          [1_u32, 4_u32, 8_u32, 16_u32, 32_u32].each do |lanes|
            (lanes == 1 ? [0_u32] : [0_u32, 1_u32]).each do |tree|
              p = params.call([CHANNELS.to_u32, FRAMES.to_u32, h.to_u32, w.to_u32,
                               SAMPLES.to_u32, valid ? 1_u32 : 0_u32, order, lanes, tree])
              candidates << Candidate.new("order#{order}-lanes#{lanes}-tree#{tree}", p,
                alloc.call(SAMPLES.to_i64 * 4), alloc.call(SAMPLES.to_i64 * 2), alloc.call(SAMPLES.to_i64 * 2))
            end
          end
        end
        p = params.call([CHANNELS.to_u32, FRAMES.to_u32, h.to_u32, w.to_u32,
                         SAMPLES.to_u32, valid ? 1_u32 : 0_u32, 0_u32, 1_u32, 0_u32])
        candidates << Candidate.new("exact-integer", p, alloc.call(SAMPLES.to_i64 * 4),
          alloc.call(SAMPLES.to_i64 * 2), alloc.call(SAMPLES.to_i64 * 2))
        stats = params.call(Array(UInt32).new(6, 0_u32))
        CaseBuffers.new(name, valid, alloc.call(SAMPLES.to_i64 * 4), stats,
          alloc.call(SAMPLES.to_i64 * 2), alloc.call(SAMPLES.to_i64 * 2),
          alloc.call(SAMPLES.to_i64 * 2), alloc.call(SAMPLES.to_i64 * 4), candidates)
      end

      cases.each_with_index do |item, index|
        input, cache = inputs[index], caches[index]
        t = ML::Video::DCAEVConvMPS::Template.new(1, CHANNELS, CHANNELS, CHANNELS,
          FRAMES, FRAMES, h, w, 0, false, true, item.valid, true, 1, 1, false)
        baseline_templates << t
        baseline_plans << t.bind(input, item.valid ? cache : nil, weight, bias, original)
        [0, 1].each do |mode|
          rt = ReductionTemplate.new(mode, item.valid, h, w)
          templates << rt
          plans << rt.bind(input, item.valid ? cache : nil, weight, mode == 0 ? bias : nil,
            mode == 0 ? pre : f32, mode == 0 ? post : nil)
        end
      end

      vector_x = raw_file(File.join(args.vectors, "x.bf16"), VECTOR_K * VECTOR_COUNT * 2)
      vector_w = raw_file(File.join(args.vectors, "w.bf16"), VECTOR_K * VECTOR_COUNT * 2)
      buffers << vector_x << vector_w
      vector_raw, vector_pre = alloc.call(VECTOR_COUNT.to_i64 * 4), alloc.call(VECTOR_COUNT.to_i64 * 2)
      vector_params = params.call([VECTOR_K.to_u32, VECTOR_COUNT.to_u32])
      kernels = {} of String => ML::Metal::ComputePipeline
      {"fill", "uniform", "midpoints", "compare_full", "gather", "native", "exact", "exact_vectors"}.each do |suffix|
        name = "dcae_reduce_#{suffix}"
        kernels[suffix] = ML::Metal::ComputePipeline.new(name, SOURCE, name)
      end
      named_bytes = buffers.sum(&.size) + arena.resident_bytes + (prefix.try(&.named_bytes) || 0_i64)
      raise "Diagnostic named residency exceeds the admitted 6 GiB plan" if named_bytes > 6_i64 * 1024 * 1024 * 1024
      active = chain = ML::Metal::MPSCommandChain.new
      allocations_before = ML::MetalBuffer.hy_text_probe_allocations
      readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks
      prepared.encode(active.encoder)
      e = active.encoder
      unless prefix
        e.set_pipeline(kernels["fill"]).set_buffer(inputs[0], 0, ML::Metal::BufferAccess::Write).set_buffer(fill_input, 1).dispatch_1d(count)
        e.set_pipeline(kernels["fill"]).set_buffer(caches[0], 0, ML::Metal::BufferAccess::Write).set_buffer(fill_cache, 1).dispatch_1d(cache_count)
      end
      e.set_pipeline(kernels["exact_vectors"]).set_buffer(vector_x, 0).set_buffer(vector_w, 1)
      e.set_buffer(vector_raw, 2, ML::Metal::BufferAccess::Write).set_buffer(vector_pre, 3, ML::Metal::BufferAccess::Write)
      e.set_buffer(vector_params, 4).dispatch_1d(VECTOR_COUNT, 64)
      cases.each_with_index do |item, index|
        prefix.try(&.encode_tile(active, index))
        baseline_plans[index].encode(active)
        plans[index * 2].encode(active)
        plans[index * 2 + 1].encode(active)
        e = active.encoder
        e.set_pipeline(kernels["compare_full"])
        {original, pre, post, f32, item.stats}.each_with_index do |b, i|
          e.set_buffer(b, i, i == 4 ? ML::Metal::BufferAccess::ReadWrite : ML::Metal::BufferAccess::Read)
        end
        e.set_value(count.to_u32, 5).dispatch_1d(count)
        e.set_pipeline(kernels["uniform"]).set_buffer(item.indices, 0, ML::Metal::BufferAccess::Write)
        e.set_buffer(uniform_params, 1).dispatch_1d(UNIFORM)
        e.set_pipeline(kernels["midpoints"]).set_buffer(f32, 0).set_buffer(item.indices, 1, ML::Metal::BufferAccess::ReadWrite)
        e.set_buffer(midpoint_params, 2).dispatch_threadgroups({SAMPLES - UNIFORM, 1, 1}, {256, 1, 1})
        e.set_pipeline(kernels["gather"])
        {item.indices, original, pre, post, f32, item.original, item.pre, item.post, item.raw}.each_with_index do |b, i|
          e.set_buffer(b, i, i >= 5 ? ML::Metal::BufferAccess::Write : ML::Metal::BufferAccess::Read)
        end
        e.set_buffer(gather_params, 9).dispatch_1d(SAMPLES)
        item.candidates.each do |candidate|
          native(active, kernels[candidate.name == "exact-integer" ? "exact" : "native"],
            inputs[index], caches[index], weight, bias, candidate, item.indices)
        end
      end
      hot = {allocations:       ML::MetalBuffer.hy_text_probe_allocations - allocations_before,
             readbacks:         ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before,
             preterminal_waits: active.wait_count}
      raise "Unexpected hot allocation/readback/wait" unless hot.values.all?(&.zero?)
      active.commit_and_wait
      raise "Expected sole terminal drain" unless active.wait_count == 1 && active.drained?
      checkpoint.unchanged!
      reference.try(&.unchanged!)
      {"weight.bf16" => weight,
       "bias.bf16" => bias, "vectors.f32" => vector_raw, "vectors.bf16" => vector_pre}.each do |name, b|
        export_raw(File.join(args.output, name), b)
      end
      if prefix
        cases.each_with_index do |item, index|
          export_raw(File.join(args.output, "#{item.name}-input.bf16"), inputs[index])
          export_raw(File.join(args.output, "#{item.name}-cache.bf16"), caches[index])
        end
      else
        export_raw(File.join(args.output, "input.bf16"), inputs[0])
        export_raw(File.join(args.output, "cache.bf16"), caches[0])
      end
      cases.each do |item|
        {"indices.u32" => item.indices, "full-stats.u32" => item.stats, "original.bf16" => item.original,
         "mps-pre.bf16" => item.pre, "mps-post.bf16" => item.post, "mps-f32.f32" => item.raw}.each do |name, b|
          export_raw(File.join(args.output, "#{item.name}-#{name}"), b)
        end
        item.candidates.each do |c|
          {"f32" => c.raw, "pre.bf16" => c.pre, "post.bf16" => c.post}.each do |suffix, b|
            export_raw(File.join(args.output, "#{item.name}-#{c.name}.#{suffix}"), b)
          end
        end
      end
      manifest = {
        schema: "dcae_conv3d_reduction_probe/v1", weights_path: args.weights, weights_sha256: args.sha,
        weight_tensor: WEIGHT, bias_tensor: BIAS, shape: [1, CHANNELS, FRAMES, h, w],
        weight_shape: weight_shape, samples: SAMPLES, uniform_samples: UNIFORM, boundary_samples: 512,
        input_kind: prefix ? "Real BF16 stage-2 activations from two causal decoder-prefix tiles" : (args.small ? "legacy periodic synthetic BF16 positive control" : "GPU hash seeded finite BF16 synthetic real-geometry input"),
        activation_prefix: prefix ? {
          reference_path: args.reference, reference_sha256: args.reference_sha,
          target_conv: "decoder.stages.2.op_list.1.main.conv1.conv", tile_count: 2,
          activation_shape: [1, CHANNELS, FRAMES, h, w], cache_shape: [1, CHANNELS, 1, h, w],
        } : nil,
        input_seed: prefix ? nil : 0x2026, cache_seed: prefix ? nil : 0x0512, named_bytes: named_bytes,
        candidates: cases.first.candidates.map(&.name), hot_metrics: hot, terminal_waits: active.wait_count,
        graph_roots: active.segment_count, diagnostic_gpu_seconds: active.gpu_elapsed_seconds,
        vectors_directory: args.vectors, vector_k: VECTOR_K, vector_count: VECTOR_COUNT,
        full_stats: ["original-vs-multi-post-bit", "original-vs-multi-post-tolerance", "bf16-nonfinite",
                     "mps-f32-rounded-vs-pre-bit", "mps-f32-rounded-vs-pre-tolerance", "f32-nonfinite"],
        sampling: "512 boundary + 3584 stratified + 4096 GPU-selected near-BF16-midpoint coordinates; not unbiased error prevalence",
        scope: prefix ? "Bounded real decoder-prefix activations and full-shape MPS operators; sampled native reductions. Requires separate upstream prefix authentication. No hidden-MPS-accumulator, speedup or full-decoder parity claim." : "Full-shape MPS operators; sampled native reductions on synthetic input. No actual activation, hidden-MPS-accumulator, speedup or full-decoder parity claim.",
      }
      File.write(File.join(args.output, "manifest.json"), manifest.to_json + "\n")
      puts manifest.to_json
    ensure
      if active = chain
        active.commit_and_wait unless active.drained?
        plans.each(&.release(active))
        baseline_plans.each(&.release(active))
        prefix.try(&.release(active))
        active.release
      else
        plans.each(&.release)
        baseline_plans.each(&.release)
        prefix.try(&.release)
      end
      templates.each(&.release)
      baseline_templates.each(&.release)
      prepared.try(&.release)
      arena.release
      buffers.each(&.release)
      checkpoint.close
      reference.try(&.close)
    end
  end
end

DCAEReductionProbe.run
