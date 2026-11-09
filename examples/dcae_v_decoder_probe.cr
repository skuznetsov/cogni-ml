require "option_parser"
require "digest/sha256"
require "json"
require "./support/hy_nomas_probe_instrumentation"
require "../src/ml/metal/device"
require "../src/ml/video/bf16_weight_cache"
require "../src/ml/video/dcae_v_decoder"

# A diagnostic/benchmark boundary, not a video-generation pipeline. All runs
# share weights, scratch and graph bindings. Export follows the final drain.
module DCAEVDecoderProbe
  extend self
  @@in_flight = [] of Proc(Nil)

  def validate_output(path : String) : Nil
    unless Path[path].absolute? && !File.exists?(path) && File.directory?(File.dirname(path)) &&
           File.realpath(File.dirname(path)).starts_with?(File.realpath("/opt/mldata_ssd") + "/")
      raise ArgumentError.new("Decoder output must be a fresh absolute file on the external scratch SSD")
    end
  end

  def write_artifact(path : String, tensors : Hash(String, Tuple(ML::MetalBuffer, Array(Int64))), layout : String) : Nil
    offset = 0_i64
    header = JSON.build do |json|
      json.object do
        json.field "__metadata__" do
          json.object do
            json.field "conv_layout", layout
          end
        end
        tensors.each do |name, entry|
          buffer, shape = entry
          bytes = shape.product * 2
          raise "DCAE-V decoder export extent mismatch" unless buffer.size == bytes
          json.field name do
            json.object do
              json.field "dtype", "BF16"
              json.field "shape", shape
              json.field "data_offsets", [offset, offset + bytes]
            end
          end
          offset += bytes
        end
      end
    end
    header += " " * ((8 - header.bytesize % 8) % 8)
    fd = LibC.open(path, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
    raise IO::Error.from_errno("create fresh decoder artifact") if fd < 0
    io = IO::FileDescriptor.new(fd)
    begin
      io.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
      io.write(header.to_slice)
      tensors.each_value do |entry|
        buffer = entry[0]
        io.write(Bytes.new(buffer.contents.as(UInt8*), buffer.size.to_i32))
      end
    ensure
      io.close
    end
  end

  def write_profile(profile : ML::Video::DCAEVDecoderProfile,
                    request : Int32, layout : String) : Nil
    puts(JSON.build do |json|
      json.object do
        json.field "event", "dcae_decoder_profile"
        json.field "request", request
        json.field "conv_layout", layout
        json.field "timing_boundary", "operation host encode wall time; existing MPS root intervals read after the request terminal wait"
        json.field "gpu_attribution_semantics", "Root candidate sums may double-count shared intervals; definite sums include roots with a single operation or stage/category candidate. These are interval attribution bounds, not exclusive kernel times."
        json.field "gpu_root_interval_summary" do
          json.object do
            json.field "root_count", profile.root_profile.root_count
            json.field "root_duration_sum_seconds", profile.root_profile.root_duration_seconds
            json.field "union_seconds", profile.root_profile.union_seconds
            json.field "envelope_seconds", profile.root_profile.envelope_seconds
            json.field "overlap_seconds", profile.root_profile.overlap_seconds
          end
        end
        json.field "operations" do
          json.array do
            profile.operations.each do |operation|
              json.object do
                json.field "name", operation.name
                json.field "stage", operation.stage
                json.field "category", operation.category
                json.field "encode_wall_seconds", operation.encode_wall_seconds
                json.field "start_root_count", operation.start_root_count
                json.field "end_root_count", operation.end_root_count
                json.field "gpu_candidate_root_duration_sum_seconds",
                  profile.operation_candidate_root_duration_sum_seconds[operation.name]
                json.field "gpu_definite_root_duration_sum_seconds",
                  profile.operation_definite_root_duration_sum_seconds[operation.name]
              end
            end
          end
        end
        json.field "gpu_candidate_root_duration_sum_seconds_by_stage_category",
          profile.group_candidate_root_duration_sum_seconds
        json.field "gpu_definite_root_duration_sum_seconds_by_stage_category",
          profile.group_definite_root_duration_sum_seconds
        json.field "gpu_root_candidates" do
          json.array do
            profile.root_candidates.each_with_index do |candidates, index|
              root = profile.roots[index]
              json.object do
                json.field "sequence", root.sequence
                json.field "gpu_start_seconds", root.gpu_start_seconds
                json.field "gpu_end_seconds", root.gpu_end_seconds
                json.field "candidates", candidates
              end
            end
          end
        end
        json.field "mps_api_profile_phase0_unassigned" do
          if phase = profile.api_profile_unassigned
            json.object do
              json.field "name", phase.name
              json.field "operations" do
                json.object do
                  phase.operations.each do |name, stats|
                    json.field name do
                      json.object do
                        json.field "count", stats.count
                        json.field "wall_total_ns", stats.wall_total_ns
                        json.field "wall_max_ns", stats.wall_max_ns
                        json.field "thread_cpu_total_ns", stats.thread_cpu_total_ns
                        json.field "thread_cpu_max_ns", stats.thread_cpu_max_ns
                        json.field "nested_count", stats.nested_count
                        json.field "nested_wall_total_ns", stats.nested_wall_total_ns
                        json.field "nested_thread_cpu_total_ns", stats.nested_thread_cpu_total_ns
                        json.field "exclusive_wall_total_ns", stats.exclusive_wall_total_ns
                        json.field "exclusive_thread_cpu_total_ns", stats.exclusive_thread_cpu_total_ns
                      end
                    end
                  end
                end
              end
            end
          else
            json.null
          end
        end
      end
    end)
  end

  # A root can straddle a causal tile boundary. Retain the shared root in the
  # upper bound only. GPU interval sums do not measure exclusive kernel time.
  def write_cache_profile(profile : ML::Video::DCAEVDecoderProfile, request : Int32,
                          tiles : Int32, layout : String) : Nil
    root_tiles = profile.root_candidates.map do |operations|
      operations.map do |name|
        match = name.match(/\Atile(\d+)\/|\.tile(\d+)\z/) || raise "Unlabelled DCAE-V tile operation: #{name}"
        (match[1]? || match[2]).to_i32
      end.uniq
    end
    tiles.times do |tile|
      possible = [] of ML::Video::MPSCommandRootTiming
      definite_seconds = 0.0
      root_tiles.each_with_index do |candidates, index|
        next unless candidates.includes?(tile)
        root = profile.roots[index]
        possible << root
        definite_seconds += root.gpu_end_seconds - root.gpu_start_seconds if candidates.size == 1
      end
      raise "No completed DCAE-V roots for tile #{tile}" if possible.empty?
      puts({event: "dcae_decoder_cache_profile", request: request, conv_layout: layout,
            tile: tile, cache: tile == 0 ? "cold" : "live",
            possible_root_count: possible.size,
            shared_root_count: root_tiles.count { |candidates| candidates.size > 1 && candidates.includes?(tile) },
            first_possible_gpu_start_seconds: possible.min_of(&.gpu_start_seconds),
            last_possible_gpu_end_seconds: possible.max_of(&.gpu_end_seconds),
            definite_gpu_root_seconds: definite_seconds,
            possible_gpu_root_seconds: possible.sum { |root| root.gpu_end_seconds - root.gpu_start_seconds }}.to_json)
    end
  end

  def run : Nil
    setup_start = Time.instant
    weights_path = weights_sha = reference_path = reference_sha = output_path = ""
    budget_mib = 8192_i64
    runs = 1_i32
    plan_only = false
    capture_stages = false
    profile_enabled = false
    cache_profile = false
    packed_c8 = false
    OptionParser.parse do |p|
      p.banner = "Native original BF16 DCAE-V decoder; run under wan_memory_guard.py"
      p.on("--weights PATH", "Original decoder-only BF16 safetensors") { |v| weights_path = v }
      p.on("--weights-sha SHA", "Trusted weight SHA-256") { |v| weights_sha = v }
      p.on("--reference PATH", "Upstream BF16 input/output capture") { |v| reference_path = v }
      p.on("--reference-sha SHA", "Trusted reference SHA-256") { |v| reference_sha = v }
      p.on("--output PATH", "Fresh native BF16 capture on external SSD") { |v| output_path = v }
      p.on("--budget-mib N", "Caller-owned residency cap") { |v| budget_mib = v.to_i64 }
      p.on("--runs N", "Sequential requests reusing buffers and plans") { |v| runs = v.to_i32 }
      p.on("--capture-stages", "Capture every tile's stage outputs on GPU") { capture_stages = true }
      p.on("--profile", "Record per-operation host encode time and existing MPS root intervals") { profile_enabled = true }
      p.on("--cache-profile", "Summarize existing GPU root intervals by causal cache tile") { cache_profile = true }
      p.on("--c8", "Use packed C8 for the verified stage-2 residual Conv3D geometry") { packed_c8 = true }
      p.on("--plan-only", "Authenticate and plan without Metal") { plan_only = true }
    end
    layout = packed_c8 ? "c8_stage2" : "native"
    unless [weights_path, weights_sha, reference_path, reference_sha].none?(&.empty?) &&
           (3072..32768).includes?(budget_mib) && (1..4).includes?(runs)
      raise ArgumentError.new("Pinned weights/reference and budget 3..32 GiB, runs 1..4 are required")
    end
    weights = ML::Video::BF16WeightCache.authenticate(weights_path, weights_sha)
    begin
      reference = ML::Video::BF16WeightCache.authenticate(reference_path, reference_sha)
    rescue ex
      weights.close
      raise ex
    end
    weight_arena = workspace_arena = io_arena = nil.as(ML::Video::Arena?)
    prepared = nil.as(ML::Video::PreparedWeights?)
    plan = nil.as(ML::Video::DCAEVDecoderPlan?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    begin
      input_info = reference.tensors["input.latent"]
      shape = input_info.shape
      unless shape.size == 5 && shape[0] == 1 && shape[1] == 32 && shape.all? { |d| 0 < d <= Int32::MAX }
        raise ArgumentError.new("DCAE-V decoder reference must be BF16 [1,32,T,H,W]")
      end
      cfg = ML::Video::DCAEVDecoderConfig.new(shape[3].to_i32, shape[4].to_i32, shape[2].to_i32)
      expected = {"input.latent"  => shape,
                  "output.pixels" => [1_i64, 3_i64, cfg.output_frames.to_i64, cfg.output_height.to_i64, cfg.output_width.to_i64]}
      ML::Video::DCAEVDecoder.trace_shapes(cfg).each { |name, dims| expected[name] = dims } if capture_stages
      unless reference.tensors.transform_values(&.shape) == expected
        raise ArgumentError.new("DCAE-V decoder reference keys or shapes differ from declared capture")
      end
      io_bytes = expected.sum { |_, dims| dims.product * 2 }
      workspace_bytes = ML::Video::DCAEVDecoder.workspace_required_bytes(cfg, packed_c8: packed_c8)
      budget = budget_mib * 1024 * 1024
      weight_plan = ML::Video::WeightPlan.new([weights], ML::Video::DCAEVDecoder.checkpoint_bindings(cfg),
        budget, reserve_bytes: io_bytes + workspace_bytes,
        resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
      raise "Original BF16 decoder weights must not require GPU staging" unless weight_plan.staging_bytes.zero?
      named_bytes = weight_plan.destination_bytes + workspace_bytes + io_bytes
      # Graph-private pack extent, not a named allocation or an upper bound on
      # all MPS scratch. The process guard remains the admission boundary.
      puts({event: "dcae_decoder_memory_plan", input_shape: shape,
            weights_sha256: weights_sha, reference_sha256: reference_sha,
            weight_bytes: weight_plan.destination_bytes, workspace_bytes: workspace_bytes,
            io_trace_bytes: io_bytes, named_bytes: named_bytes, budget_bytes: budget,
            temporal_tiles: cfg.tile_count, opaque_mpsgraph_bytes_known: false,
            conv_layout: layout,
            full_video_generation: false}.to_json)
      return if plan_only
      validate_output(output_path)
      weight_arena = wa = ML::Video::Arena.new
      workspace_arena = ws = ML::Video::Arena.new
      io_arena = ia = ML::Video::Arena.new
      prepared = pw = ML::Video::PreparedWeights.new(weight_plan, wa)
      workspace = ML::Video::DCAEVDecoder.prepare!(ws, cfg, budget, weight_plan.destination_bytes + io_bytes,
        packed_c8: packed_c8)
      tensors = {} of String => Tuple(ML::MetalBuffer, Array(Int64))
      expected.each do |name, dims|
        tensors[name] = {ia.alloc_weight(name, dims.product.to_i32, ML::Video::WeightDType::BF16), dims}
      end
      input = tensors["input.latent"][0]
      reference.copy_raw(input_info, Bytes.new(input.contents.as(UInt8*), input.size.to_i32))
      traces = {} of String => ML::MetalBuffer
      tensors.each { |name, entry| traces[name] = entry[0] if name.starts_with?("trace.") }
      plan_prepare_start = Time.instant
      plan = active_plan = ML::Video::DCAEVDecoder.prepare_mps!(input, tensors["output.pixels"][0], wa,
        workspace, cfg, traces)
      if packed_c8 && active_plan.c8_convolution_count != cfg.tile_count * 2 * ML::Video::DCAEVDecoder::DEPTHS[2]
        raise "DCAE-V C8 plan does not cover all stage-2 residual convolutions"
      end
      plan_prepare_seconds = (Time.instant - plan_prepare_start).total_seconds
      raise "DCAE-V decoder memory plan/allocation mismatch" unless wa.resident_bytes + ws.resident_bytes + ia.resident_bytes == named_bytes
      puts({event: "dcae_decoder_prepared", setup_seconds: (Time.instant - setup_start).total_seconds,
            operations: active_plan.operation_count, convolution_templates: active_plan.template_count,
            runs: runs, measured_requests: runs,
            plan_prepare_seconds: plan_prepare_seconds,
            capture_stages: capture_stages, profile: profile_enabled,
            cache_profile: cache_profile, c8_convolutions: active_plan.c8_convolution_count,
            weight_residency: "BF16 bytes copied to resident buffers during setup",
            weight_staging_bytes: weight_plan.staging_bytes,
            conv_layout: layout}.to_json)
      runs.times do |index|
        chain = active = ML::Metal::MPSCommandChain.new
        profile_capture = profile_enabled || cache_profile ? ML::Video::DCAEVDecoderProfileRecorder.new : nil
        active.enable_api_profile!(phase_roots: false) if profile_enabled
        allocations = ML::MetalBuffer.hy_text_probe_allocations
        reads = ML::MetalBuffer.hy_text_probe_readbacks
        start = Time.instant
        # Required BF16 sources were copied directly during setup. This call
        # validates their lifetimes; it dispatches no upload or GPU conversion.
        pw.encode(active.encoder) if index == 0
        active_plan.encode(active, profile: profile_capture)
        encoded_seconds = (Time.instant - start).total_seconds
        metrics = {allocations:       ML::MetalBuffer.hy_text_probe_allocations - allocations,
                   readbacks:         ML::MetalBuffer.hy_text_probe_readbacks - reads,
                   preterminal_waits: active.wait_count}
        raise "DCAE-V decode allocated/read back/waited before terminal" unless metrics.values.all?(&.zero?)
        wait_start = Time.instant
        active.commit_and_wait
        wait_seconds = (Time.instant - wait_start).total_seconds
        wall_seconds = (Time.instant - start).total_seconds
        puts({event: "dcae_decoder_native_request", request: index, weights_upload: false,
              prepared_weight_validation: index == 0,
              sample_kind: index == 0 ? "first_request" : "reused_request_diagnostic",
              wall_seconds: wall_seconds, encode_seconds: encoded_seconds,
              terminal_wait_seconds: wait_seconds, gpu_seconds: active.gpu_elapsed_seconds, segments: active.segment_count,
              waits: active.wait_count, metrics: metrics, named_bytes: named_bytes,
              conv_layout: layout, profile: profile_enabled, cache_profile: cache_profile}.to_json)
        if recorder = profile_capture
          profile = recorder.build(active)
          write_profile(profile, index.to_i32, layout) if profile_enabled
          write_cache_profile(profile, index.to_i32, cfg.tile_count, layout) if cache_profile
        end
        active.release
        # The plan retains the completed chain as a lifetime certificate. The
        # wrapper may be released; completed? remains true after release.
      end
      write_artifact(output_path, tensors, layout)
      puts({event: "dcae_decoder_native_completed", output: output_path,
            output_sha256: File.open(output_path) { |io| Digest::SHA256.hexdigest(io) },
            named_bytes: named_bytes, full_video_generation: false,
            conv_layout: layout,
            upstream_comparison_required: true}.to_json)
    ensure
      can_release = true
      chain.try do |active|
        begin
          active.commit_and_wait unless active.drained?
        ensure
          can_release = active.drained?
          unless can_release
            @@in_flight << -> { weights.object_id; reference.object_id; plan.try(&.object_id); prepared.try(&.object_id); weight_arena.try(&.object_id); workspace_arena.try(&.object_id); io_arena.try(&.object_id); active.object_id; nil }
          end
        end
      end
      if can_release
        plan.try(&.release)
        chain.try(&.release)
        prepared.try(&.release)
        io_arena.try(&.release)
        workspace_arena.try(&.release)
        weight_arena.try(&.release)
        reference.close
        weights.close
      end
    end
  end
end

DCAEVDecoderProbe.run
