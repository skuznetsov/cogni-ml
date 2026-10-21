# Standalone native F32 text cross-attention dispatch-cadence benchmark.
# The reference is a source-pinned MPS SDPA adapter fixture, not an official
# Wan CUDA execution or a full-block/video-quality result.
require "json"
require "option_parser"
require "../src/ml/video/metal_attention"
require "../src/ml/video/metal_mps_command_chain"
require "../src/ml/video/metal_ops"
require "../spec/support/video_allocation_counter"
require "./support/wan_cross_cadence_probe"

private def wan_cross_emit(record) : Nil
  STDOUT.puts(record.to_json)
  STDOUT.flush
end

private def wan_cross_pipeline(name : String) : ML::Metal::ComputePipeline
  ML::Metal::ComputePipeline.new(name, WanCrossCadenceProbe::METAL_SOURCE)
end

private def wan_cross_dispatch_fill(encoder : ML::Metal::ComputeEncoder,
                                    pipeline : ML::Metal::ComputePipeline,
                                    buffer : ML::MetalBuffer, count : Int32,
                                    flat_base : UInt32, salt : UInt32) : Nil
  encoder.set_pipeline(pipeline).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
  encoder.set_value(count.to_u32, 1).set_value(flat_base, 2).set_value(salt, 3)
  encoder.dispatch_1d(count)
end

private def wan_cross_nan_fill(encoder : ML::Metal::ComputeEncoder,
                               pipeline : ML::Metal::ComputePipeline,
                               output : ML::MetalBuffer, count : Int32) : Nil
  encoder.set_pipeline(pipeline).set_buffer(output, 0, ML::Metal::BufferAccess::Write)
  encoder.set_value(count.to_u32, 1).dispatch_1d(count)
end

private def wan_cross_copy(encoder : ML::Metal::ComputeEncoder,
                           pipeline : ML::Metal::ComputePipeline,
                           source : ML::MetalBuffer, destination : ML::MetalBuffer,
                           count : Int32) : Nil
  encoder.set_pipeline(pipeline)
    .set_buffer(source, 0)
    .set_buffer(destination, 1, ML::Metal::BufferAccess::Write)
    .set_value(count.to_u32, 2)
    .dispatch_1d(count)
end

private def wan_cross_encode_compare(encoder : ML::Metal::ComputeEncoder,
                                     zero_pipeline : ML::Metal::ComputePipeline,
                                     compare_pipeline : ML::Metal::ComputePipeline,
                                     reference : ML::MetalBuffer,
                                     actual : ML::MetalBuffer,
                                     metrics : ML::MetalBuffer,
                                     value_count : Int32) : Nil
  raise ArgumentError.new("Comparison count must fill complete 256-lane groups") unless value_count % 256 == 0
  encoder.set_pipeline(zero_pipeline)
    .set_buffer(metrics, 0, ML::Metal::BufferAccess::Write)
    .dispatch_1d(5)
  encoder.set_pipeline(compare_pipeline)
    .set_buffer(reference, 0)
    .set_buffer(actual, 1)
    .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
    .set_value(value_count.to_u32, 3)
    .dispatch_threadgroups({value_count // 256, 1, 1}, {256, 1, 1})
end

private def wan_cross_wait_setup(command : ML::Metal::CommandBuffer,
                                 encoder : ML::Metal::ComputeEncoder) : Nil
  encoder.end_encoding
  command.commit_and_wait
end

private def wan_cross_fill_inputs(q : ML::MetalBuffer, k : ML::MetalBuffer,
                                  v : ML::MetalBuffer, q_values : Int32,
                                  kv_values : Int32,
                                  fill_pipeline : ML::Metal::ComputePipeline) : Float64
  started = Time.monotonic
  command = ML::Metal::CommandBuffer.new
  encoder = ML::Metal::ComputeEncoder.new(command)
  wan_cross_dispatch_fill(encoder, fill_pipeline, q, q_values, 0_u32, 1_u32)
  wan_cross_dispatch_fill(encoder, fill_pipeline, k, kv_values, 0_u32, 2_u32)
  wan_cross_dispatch_fill(encoder, fill_pipeline, v, kv_values, 0_u32, 3_u32)
  wan_cross_wait_setup(command, encoder)
  (Time.monotonic - started).total_seconds
end

private def wan_cross_prefill_nan(output : ML::MetalBuffer, value_count : Int32,
                                  pipeline : ML::Metal::ComputePipeline) : Float64
  started = Time.monotonic
  command = ML::Metal::CommandBuffer.new
  encoder = ML::Metal::ComputeEncoder.new(command)
  wan_cross_nan_fill(encoder, pipeline, output, value_count)
  wan_cross_wait_setup(command, encoder)
  (Time.monotonic - started).total_seconds
end

private def wan_cross_diagnostic(reference : ML::MetalBuffer,
                                 actual : ML::MetalBuffer,
                                 metrics : ML::MetalBuffer,
                                 value_count : Int32,
                                 copy_pipeline : ML::Metal::ComputePipeline,
                                 zero_pipeline : ML::Metal::ComputePipeline,
                                 compare_pipeline : ML::Metal::ComputePipeline,
                                 copy_anchor : Bool = false) : NamedTuple(
  metrics: NamedTuple(bit_differences: UInt32,
    tolerance_violations: UInt32, nonfinite: UInt32,
    max_abs_error: Float32, max_tolerance_ratio: Float32),
  waits: Int32)
  chain = ML::Metal::MPSCommandChain.new
  begin
    encoder = chain.encoder
    wan_cross_encode_compare(encoder, zero_pipeline, compare_pipeline,
      reference, actual, metrics, value_count)
    # The initial oracle check must finish logically before the fixture buffer
    # becomes the bitwise native anchor. The same serial encoder preserves that
    # order and the single diagnostic terminal wait makes both effects visible.
    wan_cross_copy(encoder, copy_pipeline, actual, reference, value_count) if copy_anchor
    chain.commit_and_wait
    raise "diagnostic chain did not use one terminal wait" unless chain.wait_count == 1
    {metrics: WanCrossCadenceProbe.read_metrics(metrics), waits: chain.wait_count}
  ensure
    begin
      chain.commit_and_wait unless chain.completed?
    ensure
      chain.release if chain.completed?
    end
  end
end

private def wan_cross_run_arm(cadence : Int32,
                              query_rows : Int32,
                              q : ML::MetalBuffer,
                              k : ML::MetalBuffer,
                              v : ML::MetalBuffer,
                              output : ML::MetalBuffer) : NamedTuple(
  encode_interval_seconds: Float64,
  chain_creation_seconds: Float64,
  terminal_wait_seconds: Float64,
  total_completed_wall_seconds: Float64,
  summed_command_root_gpu_seconds: Float64,
  segment_count: Int32,
  wait_count: Int32,
  tracked_metalbuffer_allocations: Int64,
  dispatch_count: Int32,
  expected_roots: Int32)
  total_groups = ((query_rows + 7) // 8) * WanCrossCadenceProbe::HEADS
  dispatch_count = (total_groups + WanCrossCadenceProbe::GROUPS_PER_DISPATCH - 1) //
                   WanCrossCadenceProbe::GROUPS_PER_DISPATCH
  expected_roots = (dispatch_count + cadence - 1) // cadence
  allocation_start = ML::MetalBuffer.video_spec_allocations
  total_started = Time.monotonic
  chain = ML::Metal::MPSCommandChain.new
  encode_started = Time.monotonic
  begin
    first_group = 0
    dispatches_in_root = 0
    while first_group < total_groups
      count = Math.min(WanCrossCadenceProbe::GROUPS_PER_DISPATCH, total_groups - first_group)
      ML::Video::MetalOps.attention_tiled_range(chain.encoder, q, k, v, output,
        query_rows, WanCrossCadenceProbe::KEY_ROWS, WanCrossCadenceProbe::HEADS,
        WanCrossCadenceProbe::HEAD_DIM, first_group, count)
      first_group += count
      dispatches_in_root += 1
      if dispatches_in_root == cadence && first_group < total_groups
        chain.pause!
        chain.continue_async!
        chain.resume!
        dispatches_in_root = 0
      end
    end
    chain.pause!
    encode_ended = Time.monotonic
    wait_started = encode_ended
    chain.commit_and_wait
    wait_ended = Time.monotonic
    allocations = ML::MetalBuffer.video_spec_allocations - allocation_start
    raise "attention arm allocated a tracked MetalBuffer" unless allocations == 0
    unless chain.wait_count == 1 && chain.segment_count == expected_roots
      raise "attention arm root/wait count mismatch (expected #{expected_roots}, got #{chain.segment_count}/#{chain.wait_count})"
    end
    {
      encode_interval_seconds:         (encode_ended - encode_started).total_seconds,
      chain_creation_seconds:          (encode_started - total_started).total_seconds,
      terminal_wait_seconds:           (wait_ended - wait_started).total_seconds,
      total_completed_wall_seconds:    (wait_ended - total_started).total_seconds,
      summed_command_root_gpu_seconds: chain.gpu_elapsed_seconds,
      segment_count:                   chain.segment_count,
      wait_count:                      chain.wait_count,
      tracked_metalbuffer_allocations: allocations,
      dispatch_count:                  dispatch_count,
      expected_roots:                  expected_roots,
    }
  ensure
    begin
      chain.commit_and_wait unless chain.completed?
    ensure
      chain.release if chain.completed?
    end
  end
end

private def wan_cross_negative_control(kind : String,
                                       anchor : ML::MetalBuffer,
                                       working : ML::MetalBuffer,
                                       metrics : ML::MetalBuffer,
                                       copy_pipeline : ML::Metal::ComputePipeline,
                                       edit_pipeline : ML::Metal::ComputePipeline,
                                       zero_pipeline : ML::Metal::ComputePipeline,
                                       compare_pipeline : ML::Metal::ComputePipeline)
  control_value_count = 256
  chain = ML::Metal::MPSCommandChain.new
  begin
    encoder = chain.encoder
    wan_cross_copy(encoder, copy_pipeline, anchor, working, control_value_count)
    encoder.set_pipeline(edit_pipeline)
      .set_buffer(working, 0, ML::Metal::BufferAccess::ReadWrite)
      .dispatch_1d(1)
    wan_cross_encode_compare(encoder, zero_pipeline, compare_pipeline,
      anchor, working, metrics, control_value_count)
    chain.commit_and_wait
    raise "negative control did not use one terminal wait" unless chain.wait_count == 1
    result = WanCrossCadenceProbe.read_metrics(metrics)
    passed = if kind == "plus_one"
               result[:bit_differences] == 1 && result[:tolerance_violations] == 1 && result[:nonfinite] == 0
             else
               result[:bit_differences] == 1 && result[:nonfinite] == 1 && result[:tolerance_violations] == 0
             end
    wan_cross_emit({
      schema:                      "wan_s2v_text_cross_cadence_negative_control_v1",
      control:                     kind,
      corrupted_values:            1,
      bit_differences:             result[:bit_differences],
      strict_tolerance_violations: result[:tolerance_violations],
      nonfinite:                   result[:nonfinite],
      rejected:                    passed,
      terminal_waits:              chain.wait_count,
    })
    raise "#{kind} negative control was not rejected" unless passed
  ensure
    begin
      chain.commit_and_wait unless chain.completed?
    ensure
      chain.release if chain.completed?
    end
  end
end

reference_dir = File.expand_path("../ev/cross-cadence-reference-v1", __DIR__)
pairs = 3
query_rows = WanCrossCadenceProbe::QUERY_ROWS
OptionParser.parse do |parser|
  parser.banner = "Usage: crystal run examples/wan_s2v_cross_cadence_benchmark.cr -- [options]"
  parser.on("--reference=DIR", "Pinned MPS SDPA reference fixture directory") { |value| reference_dir = value }
  parser.on("--pairs=N", "Measured cadence rounds, 1 through 3 (default: 3)") { |value| pairs = value.to_i }
  parser.on("--query-rows=N", "Full 36608 rows or prefix smoke 257 (default: full)") { |value| query_rows = value.to_i }
  parser.on("-h", "--help", "Show this help") { puts parser; exit }
end
raise ArgumentError.new("--pairs must be between 1 and 3") unless (1..3).includes?(pairs)
unless {WanCrossCadenceProbe::QUERY_ROWS, 257}.includes?(query_rows)
  raise ArgumentError.new("--query-rows must be 36608 or 257")
end

fixture = WanCrossCadenceProbe.validate_fixture!(reference_dir)
query_values = query_rows.to_i64 * WanCrossCadenceProbe::CHANNELS
kv_values = WanCrossCadenceProbe::KEY_ROWS.to_i64 * WanCrossCadenceProbe::CHANNELS
query_bytes = query_values * sizeof(Float32)
kv_bytes = kv_values * sizeof(Float32)
planned_resident_bytes = 3_i64 * query_bytes + 2_i64 * kv_bytes + 5_i64 * sizeof(UInt32)
unless planned_resident_bytes <= WanCrossCadenceProbe::MEMORY_CAP_BYTES
  raise ArgumentError.new("Explicit Metal buffer plan exceeds the 3 GiB cap")
end

arena = ML::Video::Arena.new
begin
  q = arena.alloc("cross_cadence_query", query_values.to_i32)
  anchor = arena.alloc("cross_cadence_reference_or_native_anchor", query_values.to_i32)
  working = arena.alloc("cross_cadence_working_output", query_values.to_i32)
  k = arena.alloc("cross_cadence_key", kv_values.to_i32)
  v = arena.alloc("cross_cadence_value", kv_values.to_i32)
  metrics = arena.alloc("cross_cadence_metrics", 5)

  actual_resident_bytes = arena.resident_bytes
  unless actual_resident_bytes == planned_resident_bytes &&
         actual_resident_bytes <= WanCrossCadenceProbe::MEMORY_CAP_BYTES
    raise ArgumentError.new("Actual Arena resident bytes differ from the admitted plan")
  end

  load_started = Time.monotonic
  expected_sha256 = WanCrossCadenceProbe.load_expected_prefix!(
    fixture[:expected_path], anchor, query_bytes)
  fixture_load_seconds = (Time.monotonic - load_started).total_seconds

  fill_pipeline = wan_cross_pipeline("wan_cross_cadence_input_fill")
  nan_pipeline = wan_cross_pipeline("wan_cross_cadence_nan_fill")
  zero_pipeline = wan_cross_pipeline("wan_cross_cadence_zero_metrics")
  compare_pipeline = wan_cross_pipeline("wan_cross_cadence_compare")
  compare_pipeline_max_threads = compare_pipeline.max_total_threads_per_threadgroup
  unless compare_pipeline_max_threads >= 256
    raise ArgumentError.new("Cadence comparator pipeline cannot launch its fixed 256-lane reduction")
  end
  plus_one_pipeline = wan_cross_pipeline("wan_cross_cadence_add_one")
  nan_control_pipeline = wan_cross_pipeline("wan_cross_cadence_make_nan")
  ML::Video::MetalOps.tiled_attention_pipeline(WanCrossCadenceProbe::HEAD_DIM)
  copy_pipeline = ML::Video::MetalOps.pipeline("video_copy_f32")

  input_fill_seconds = wan_cross_fill_inputs(q, k, v,
    query_values.to_i32, kv_values.to_i32, fill_pipeline)
  initial_nan_seconds = wan_cross_prefill_nan(working, query_values.to_i32, nan_pipeline)

  setup = {
    schema:                                         "wan_s2v_text_cross_cadence_setup_v1",
    reference_kind:                                 "pinned_f32_mps_sdpa_adapter",
    official_wan_execution:                         false,
    query_rows:                                     query_rows,
    key_rows:                                       WanCrossCadenceProbe::KEY_ROWS,
    heads:                                          WanCrossCadenceProbe::HEADS,
    head_dim:                                       WanCrossCadenceProbe::HEAD_DIM,
    native_layout:                                  "B,Q,H,D",
    dispatch_groups:                                ((query_rows + 7) // 8) * WanCrossCadenceProbe::HEADS,
    groups_per_dispatch:                            WanCrossCadenceProbe::GROUPS_PER_DISPATCH,
    values:                                         query_values,
    expected_file_bytes:                            File.size(fixture[:expected_path]),
    expected_file_sha256:                           expected_sha256,
    manifest_path:                                  fixture[:manifest_path],
    manifest_sha256:                                fixture[:manifest_sha256],
    oracle_query_chunk_rows:                        fixture[:query_chunk_rows],
    comparator_max_threads_per_threadgroup:         compare_pipeline_max_threads,
    arena_resident_bytes:                           actual_resident_bytes,
    arena_memory_cap_bytes:                         WanCrossCadenceProbe::MEMORY_CAP_BYTES,
    fixture_prefix_load_host_seconds:               fixture_load_seconds,
    input_gpu_fill_host_seconds:                    input_fill_seconds,
    initial_nan_prefill_host_seconds:               initial_nan_seconds,
    cadence_dispatches_per_command_root:            [1, 2, 4],
    measured_pair_rounds:                           pairs,
    measured_order_schedule:                        [[1, 2, 4], [2, 4, 1], [4, 1, 2]].first(pairs),
    timing_includes_chain_creation:                 true,
    oracle_qualification_serves_as_cadence1_warmup: true,
    measured_arms_have_one_terminal_wait:           true,
    all_attention_math_on_gpu:                      true,
    scope:                                          "standalone native text cross-attention cadence only; no full block, generation, or video claim",
  }
  wan_cross_emit(setup)

  # First qualify cadence=1 against every selected oracle value. Only after the
  # strict comparison completes is the fixture buffer overwritten by the
  # native output to become the bitwise anchor for subsequent arms.
  qualification_nan_seconds = wan_cross_prefill_nan(working, query_values.to_i32, nan_pipeline)
  qualification = wan_cross_run_arm(1, query_rows, q, k, v, working)
  oracle_check = wan_cross_diagnostic(anchor, working, metrics,
    query_values.to_i32, copy_pipeline, zero_pipeline, compare_pipeline, true)
  oracle_metrics = oracle_check[:metrics]
  oracle_passed = oracle_metrics[:tolerance_violations] == 0 && oracle_metrics[:nonfinite] == 0
  wan_cross_emit({
    schema:                      "wan_s2v_text_cross_cadence_validation_v1",
    stage:                       "native_cadence1_against_pinned_mps_sdpa_adapter",
    compared_values:             query_values,
    bit_differences:             oracle_metrics[:bit_differences],
    strict_tolerance_violations: oracle_metrics[:tolerance_violations],
    nonfinite:                   oracle_metrics[:nonfinite],
    max_abs_error:               oracle_metrics[:max_abs_error],
    max_tolerance_ratio:         oracle_metrics[:max_tolerance_ratio],
    comparison_tolerance:        "2e-5 + 2e-5 * abs(reference)",
    diagnostic_terminal_waits:   oracle_check[:waits],
    qualification_arm:           qualification,
    prefill_host_seconds:        qualification_nan_seconds,
    native_anchor_copy:          "GPU copy working output to reference buffer after comparison dispatch",
    passed:                      oracle_passed,
  })
  raise "native cadence 1 failed the pinned MPS SDPA adapter comparison" unless oracle_passed

  # The untimed cadence=1 oracle-qualification arm already warms this path.
  warm_order = [2, 4]
  warm_order.each do |cadence|
    prefill_seconds = wan_cross_prefill_nan(working, query_values.to_i32, nan_pipeline)
    result = wan_cross_run_arm(cadence, query_rows, q, k, v, working)
    diagnostic = wan_cross_diagnostic(anchor, working, metrics,
      query_values.to_i32, copy_pipeline, zero_pipeline, compare_pipeline)
    warm_metrics = diagnostic[:metrics]
    passed = warm_metrics[:bit_differences] == 0 &&
             warm_metrics[:tolerance_violations] == 0 && warm_metrics[:nonfinite] == 0
    wan_cross_emit({
      schema:                              "wan_s2v_text_cross_cadence_arm_v1",
      phase:                               "warmup",
      round:                               0,
      cadence_dispatches_per_command_root: cadence,
      query_rows:                          query_rows,
      dispatch_count:                      result[:dispatch_count],
      expected_roots:                      result[:expected_roots],
      total_completed_wall_seconds:        result[:total_completed_wall_seconds],
      chain_creation_seconds:              result[:chain_creation_seconds],
      encode_interval_seconds:             result[:encode_interval_seconds],
      terminal_wait_seconds:               result[:terminal_wait_seconds],
      summed_command_root_gpu_seconds:     result[:summed_command_root_gpu_seconds],
      segment_count:                       result[:segment_count],
      wait_count:                          result[:wait_count],
      tracked_metalbuffer_allocations:     result[:tracked_metalbuffer_allocations],
      output_prefill_host_seconds:         prefill_seconds,
      diagnostic_terminal_waits:           diagnostic[:waits],
      bit_differences:                     warm_metrics[:bit_differences],
      strict_tolerance_violations:         warm_metrics[:tolerance_violations],
      nonfinite:                           warm_metrics[:nonfinite],
      max_abs_error:                       warm_metrics[:max_abs_error],
      max_tolerance_ratio:                 warm_metrics[:max_tolerance_ratio],
      passed:                              passed,
    })
    raise "warmup cadence #{cadence} failed bitwise native-anchor validation" unless passed
  end

  order_schedule = [[1, 2, 4], [2, 4, 1], [4, 1, 2]].first(pairs)
  measured = [] of NamedTuple(round: Int32, cadence: Int32, wall_seconds: Float64)
  order_schedule.each_with_index do |order, round_index|
    order.each do |cadence|
      prefill_seconds = wan_cross_prefill_nan(working, query_values.to_i32, nan_pipeline)
      result = wan_cross_run_arm(cadence, query_rows, q, k, v, working)
      diagnostic = wan_cross_diagnostic(anchor, working, metrics,
        query_values.to_i32, copy_pipeline, zero_pipeline, compare_pipeline)
      arm_metrics = diagnostic[:metrics]
      passed = arm_metrics[:bit_differences] == 0 &&
               arm_metrics[:tolerance_violations] == 0 && arm_metrics[:nonfinite] == 0
      wan_cross_emit({
        schema:                              "wan_s2v_text_cross_cadence_arm_v1",
        phase:                               "measured",
        round:                               round_index + 1,
        order:                               order,
        cadence_dispatches_per_command_root: cadence,
        query_rows:                          query_rows,
        dispatch_count:                      result[:dispatch_count],
        expected_roots:                      result[:expected_roots],
        total_completed_wall_seconds:        result[:total_completed_wall_seconds],
        chain_creation_seconds:              result[:chain_creation_seconds],
        encode_interval_seconds:             result[:encode_interval_seconds],
        terminal_wait_seconds:               result[:terminal_wait_seconds],
        summed_command_root_gpu_seconds:     result[:summed_command_root_gpu_seconds],
        segment_count:                       result[:segment_count],
        wait_count:                          result[:wait_count],
        tracked_metalbuffer_allocations:     result[:tracked_metalbuffer_allocations],
        output_prefill_host_seconds:         prefill_seconds,
        diagnostic_terminal_waits:           diagnostic[:waits],
        bit_differences:                     arm_metrics[:bit_differences],
        strict_tolerance_violations:         arm_metrics[:tolerance_violations],
        nonfinite:                           arm_metrics[:nonfinite],
        max_abs_error:                       arm_metrics[:max_abs_error],
        max_tolerance_ratio:                 arm_metrics[:max_tolerance_ratio],
        passed:                              passed,
      })
      raise "measured cadence #{cadence} failed bitwise native-anchor validation" unless passed
      measured << {round: round_index + 1, cadence: cadence,
                   wall_seconds: result[:total_completed_wall_seconds]}
    end
  end

  wan_cross_negative_control("plus_one", anchor, working, metrics,
    copy_pipeline, plus_one_pipeline,
    zero_pipeline, compare_pipeline)
  wan_cross_negative_control("nan", anchor, working, metrics,
    copy_pipeline, nan_control_pipeline,
    zero_pipeline, compare_pipeline)

  timings = {1 => [] of Float64, 2 => [] of Float64, 4 => [] of Float64}
  measured.each { |sample| timings[sample[:cadence]] << sample[:wall_seconds] }
  medians = {} of Int32 => Float64
  timings.each do |cadence, samples|
    sorted = samples.sort
    middle = sorted.size // 2
    medians[cadence] = if sorted.size.odd?
                         sorted[middle]
                       else
                         (sorted[middle - 1] + sorted[middle]) / 2.0
                       end
  end
  wan_cross_emit({
    schema:                      "wan_s2v_text_cross_cadence_summary_v1",
    query_rows:                  query_rows,
    dispatches_per_command_root: [1, 2, 4],
    command_roots_by_cadence:    {"1" => ((query_rows + 7) // 8 * 40 + 319) // 320,
                               "2" => ((((query_rows + 7) // 8 * 40 + 319) // 320) + 1) // 2,
                               "4" => ((((query_rows + 7) // 8 * 40 + 319) // 320) + 3) // 4},
    measured_rounds:                                  pairs,
    measured_order_schedule:                          order_schedule,
    arm_wall_samples_seconds:                         measured,
    median_total_completed_wall_seconds:              medians,
    median_timing_scope:                              "attention arm total wall including chain creation, encoding, and one terminal wait",
    all_measured_arms_bitwise_equal_to_native_anchor: true,
    all_arm_output_values_finite:                     true,
    negative_controls_rejected:                       true,
    full_generation_run:                              false,
    official_wan_execution:                           false,
    passed:                                           true,
  })
ensure
  arena.release
end
