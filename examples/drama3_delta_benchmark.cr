require "json"
require "../src/ml/audio/drama3/backbone_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  include ML::Audio::Drama3

  # Build with the project's normal Metal link flags; run for 2 warmups and 7x30-step matched pairs.
  VH          =  32
  VD          = 128
  KH          =  16
  KD          = 128
  OUT_COUNT   = VH * VD
  STATE_COUNT = VH * VD * KD
  PAIRS       =  7
  STEPS       = 30

  def pattern(n : Int32, m : Int32, mul : Int32, add : Int32, scale : Float32) : Array(Float32)
    Array(Float32).new(n) { |i| (((i * mul + add) % m - m // 2).to_f32 * scale) }
  end

  def timed(pipeline : ML::Metal::ComputePipeline, bindings : Array(Resident::Buffer),
            params : Array(UInt64), steps : Int32) : Tuple(Float64, Float64)
    started = Time.instant
    command = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(command)
    encoder.set_pipeline(pipeline)
    bindings.each_with_index { |b, i| encoder.set_buffer(b.buffer, i, offset: b.offset) }
    encoder.set_bytes(params.to_unsafe.as(Void*), (params.size * sizeof(UInt64)).to_i32, 9)
    steps.times { encoder.dispatch_1d(OUT_COUNT, 64) }
    encoder.end_encoding
    gpu = command.commit_and_wait_gpu_elapsed_seconds
    {gpu, (Time.instant - started).total_seconds}
  end

  abort "usage: no arguments; fixed 7 pairs x 30 steps" unless ARGV.empty?
  ctx = Resident::Context.new
  inputs = [ctx.upload(pattern(KH * KD, 101, 13, 7, 0.0002_f32)),
            ctx.upload(pattern(KH * KD, 97, 7, 11, 0.0002_f32)),
            ctx.upload(pattern(OUT_COUNT, 89, 5, 3, 0.002_f32)),
            ctx.upload(Array(Float32).new(VH, -3_f32)), ctx.upload(Array(Float32).new(VH, -2_f32)),
            ctx.upload(Array(Float32).new(VH, -1.5_f32)), ctx.upload(Array(Float32).new(VH, 0.02_f32))]
  logical = pattern(STATE_COUNT, 127, 17, 19, 0.001_f32)
  transposed = Array(Float32).new(STATE_COUNT, 0_f32)
  VH.times { |h| VD.times { |d| KD.times { |j| transposed[(h * KD + j) * VD + d] = logical[(h * VD + d) * KD + j] } } }
  legacy_state, transposed_state = ctx.upload(logical), ctx.upload(transposed)
  legacy_out, transposed_out = ctx.alloc(OUT_COUNT), ctx.alloc(OUT_COUNT)
  legacy_bindings, transposed_bindings = inputs + [legacy_state, legacy_out], inputs + [transposed_state, transposed_out]
  source = BackboneResident::SOURCE
  legacy = ML::Metal::PipelineCache.get_or_compile("dr3_delta_step", source)
  transposed_kernel = ML::Metal::PipelineCache.get_or_compile("dr3_delta_step_transposed", source)
  check_source = source + <<-METAL
kernel void dr3_delta_benchmark_compare(device const float* old_s [[buffer(0)]], device const float* new_s [[buffer(1)]],
 device const float* old_o [[buffer(2)]], device const float* new_o [[buffer(3)]], device uint* status [[buffer(4)]],
 constant ulong* p [[buffer(5)]], uint tid [[thread_position_in_grid]]) {
 if (tid) return; uint vh=uint(p[0]), vd=uint(p[1]), kd=uint(p[2]), n=uint(p[3]);
 for(uint i=0;i<n;i++){float a=old_o[i],b=new_o[i];if(!isfinite(a)||!isfinite(b)){status[0]=2;return;}if(as_type<uint>(a)!=as_type<uint>(b)){status[0]=1;return;}}
 for(uint h=0;h<vh;h++)for(uint d=0;d<vd;d++)for(uint j=0;j<kd;j++){float a=old_s[(h*vd+d)*kd+j],b=new_s[(h*kd+j)*vd+d];if(!isfinite(a)||!isfinite(b)){status[0]=2;return;}if(as_type<uint>(a)!=as_type<uint>(b)){status[0]=1;return;}}
 status[0]=0;
}
METAL
  check = ML::Metal::PipelineCache.get_or_compile("dr3_delta_benchmark_compare", check_source)
  params = [VH.to_u64, VD.to_u64, KH.to_u64, KD.to_u64]
  run = ->(arm : String) { arm == "legacy" ? timed(legacy, legacy_bindings, params, STEPS) : timed(transposed_kernel, transposed_bindings, params, STEPS) }
  2.times { |i| (i.even? ? ["legacy", "transposed"] : ["transposed", "legacy"]).each { |arm| run.call(arm) } }
  samples = [] of NamedTuple(pair: Int32, order: Int32, arm: String, gpu_seconds: Float64, wall_seconds: Float64)
  matched = [] of NamedTuple(pair: Int32, gpu_ratio: Float64, wall_ratio: Float64)
  PAIRS.times do |pair|
    order = pair.even? ? ["legacy", "transposed"] : ["transposed", "legacy"]
    results = Hash(String, Tuple(Float64, Float64)).new
    order.each_with_index { |arm, i| r = run.call(arm); results[arm] = r; samples << {pair: pair, order: i, arm: arm, gpu_seconds: r[0], wall_seconds: r[1]} }
    a, b = results["legacy"], results["transposed"]
    matched << {pair: pair, gpu_ratio: b[0] / a[0], wall_ratio: b[1] / a[1]}
  end
  status = ctx.upload([0_f32])
  cmd = ML::Metal::CommandBuffer.new
  enc = ML::Metal::ComputeEncoder.new(cmd)
  enc.set_pipeline(check)
  [legacy_state, transposed_state, legacy_out, transposed_out, status].each_with_index { |b, i| enc.set_buffer(b.buffer, i, offset: b.offset) }
  cp = [VH.to_u64, VD.to_u64, KD.to_u64, OUT_COUNT.to_u64]
  enc.set_bytes(cp.to_unsafe.as(Void*), (cp.size * sizeof(UInt64)).to_i32, 5); enc.dispatch_1d(1, 1); enc.end_encoding; cmd.commit_and_wait
  code = status.buffer.contents.as(UInt32*).value
  abort "exact finite state/output comparison failed: #{code}" unless code == 0
  ratios = matched.map(&.[:gpu_ratio]).sort
  median = ratios[ratios.size // 2]
  puts({benchmark: "drama3_delta_state_layout", shape: {vh: VH, vd: VD, kh: KH, kd: KD}, warmups_per_arm: 2,
        pairs: PAIRS, steps_per_command: STEPS, exact_finite_state_and_output_check: true,
        matched: matched, median_gpu_ratio: median, samples: samples}.to_json)
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
