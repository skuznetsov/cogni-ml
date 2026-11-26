require "option_parser"
require "digest/sha256"
require "json"
require "./support/hy_nomas_probe_instrumentation"
require "../src/ml/metal/device"
require "../src/ml/video/bf16_weight_cache"
require "../src/ml/video/dcae_v_encoder"

# Diagnostic captures are exported only after the encoder's terminal drain.
module DCAEVEncoderProbe
  extend self
  @@in_flight = [] of Proc(Nil)

  def write_artifact(path : String, tensors : Hash(String, Tuple(ML::MetalBuffer, Array(Int64)))) : Nil
    offset = 0_i64
    header = JSON.build do |json|
      json.object do
        tensors.each do |name, entry|
          buffer, shape = entry
          expected = shape.reduce(1_i64) { |n, d| n * d } * 2
          raise "BF16 export extent mismatch" unless buffer.size == expected
          json.field name do
            json.object do
              json.field "dtype", "BF16"
              json.field "shape", shape
              json.field "data_offsets", [offset, offset + expected]
            end
          end
          offset += expected
        end
      end
    end
    header += " " * ((8 - header.bytesize % 8) % 8)
    fd = LibC.open(path, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
    raise IO::Error.from_errno("create fresh artifact #{path}") if fd < 0
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

  def run : Nil
    weights_path = weights_sha = input_path = input_sha = output_path = ""
    rgb8_path = rgb8_sha = ""
    project_in_tile = nil.as(Int32?)
    budget_mib = 4096_i64
    plan_only = false
    OptionParser.parse do |p|
      p.banner = "Native BF16 DCAE-V diagnostic encoder; run under wan_memory_guard.py"
      p.on("--weights PATH", "Encoder-only BF16 safetensors") { |v| weights_path = v }
      p.on("--weights-sha SHA", "Trusted encoder SHA-256") { |v| weights_sha = v }
      p.on("--reference PATH", "Upstream diagnostic BF16 safetensors") { |v| input_path = v }
      p.on("--reference-sha SHA", "Trusted diagnostic SHA-256") { |v| input_sha = v }
      p.on("--output PATH", "Fresh native diagnostic safetensors") { |v| output_path = v }
      p.on("--rgb8 PATH", "Fitted raw RGB8 HWC frame to normalize and repeat on GPU") { |v| rgb8_path = v }
      p.on("--rgb8-sha SHA", "Trusted RGB8 frame SHA-256") { |v| rgb8_sha = v }
      p.on("--trace-project-in-tile N", "Capture only this post-unshuffle project_in tile to a sidecar") { |v| project_in_tile = v.to_i32 }
      p.on("--budget-mib N", "Named encoder residency budget") { |v| budget_mib = v.to_i64 }
      p.on("--plan-only", "Authenticate shapes and memory without Metal") { plan_only = true }
    end
    has_rgb8 = !rgb8_path.empty? && !rgb8_sha.empty?
    unless [weights_path, weights_sha, input_path, input_sha].none?(&.empty?) &&
           (rgb8_path.empty? == rgb8_sha.empty?) && (2048..32768).includes?(budget_mib)
      raise ArgumentError.new("weights/reference paths and trusted SHA-256 pins are required; budget 2..32 GiB")
    end
    weights = ML::Video::BF16WeightCache.authenticate(weights_path, weights_sha)
    reference = ML::Video::BF16WeightCache.authenticate(input_path, input_sha)
    weight_arena = workspace_arena = io_arena = nil.as(ML::Video::Arena?)
    rgb_buffer = nil.as(ML::MetalBuffer?)
    rgb_payload = nil.as(Bytes?)
    prepared = nil.as(ML::Video::PreparedWeights?)
    mps_plans = nil.as(ML::Video::DCAEVEncoderMPSPlanSet?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    begin
      input_info = reference.tensors["input.clip"]
      shape = input_info.shape
      unless shape.size == 5 && shape[0] == 1 && shape[1] == 3
        raise ArgumentError.new("reference input must be BF16 [1,3,T,H,W]")
      end
      cfg = ML::Video::DCAEVEncoderConfig.new(shape[3].to_i32, shape[4].to_i32, shape[2].to_i32)
      project_in_trace = nil.as(Tuple(String, Array(Int64))?)
      if tile = project_in_tile
        tile_frames = cfg.tile_frame_count(tile)
        project_in_trace = {
          "trace.project_in.tile#{tile}",
          [cfg.batches.to_i64, 256_i64, tile_frames.to_i64,
           (cfg.height // 2).to_i64, (cfg.width // 2).to_i64],
        }
      end
      expected = {"input.clip" => shape, "latent" => [1_i64, 32_i64, cfg.latent_frames.to_i64,
                                                      cfg.latent_height.to_i64, cfg.latent_width.to_i64]}
      cfg.tile_count.times do |tile|
        expected["trace.stages.5.tile#{tile}"] = [1_i64, 1024_i64, (cfg.tile_frame_count(tile) // 4).to_i64,
                                                  cfg.latent_height.to_i64, cfg.latent_width.to_i64]
      end
      unless reference.tensors.transform_values(&.shape) == expected
        raise ArgumentError.new("reference capture keys or shapes differ from DCAE-V contract")
      end
      budget = budget_mib * 1024 * 1024
      output_io_bytes = expected.sum { |_, s| s.reduce(1_i64) { |n, d| n * d } * 2 }
      project_in_trace_bytes = project_in_trace.try { |_, s| s.reduce(1_i64) { |n, d| n * d } * 2 } || 0_i64
      rgb_bytes = has_rgb8 ? cfg.rgb_bytes : 0_i64
      io_bytes = output_io_bytes + project_in_trace_bytes + rgb_bytes
      scratch_bytes = ML::Video::DCAEVEncoder.workspace_required_bytes(cfg)
      bindings = ML::Video::DCAEVEncoder.checkpoint_bindings(cfg)
      plan = ML::Video::WeightPlan.new([weights], bindings, budget,
        reserve_bytes: io_bytes + scratch_bytes, resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
      named_bytes = plan.destination_bytes + io_bytes + scratch_bytes
      if has_rgb8
        rgb_path = Path.new(rgb8_path)
        unless File.file?(rgb_path) && !File.symlink?(rgb_path) && File.size(rgb_path) == cfg.rgb_bytes
          raise ArgumentError.new("RGB8 input must be a regular, non-symlink file of exactly cfg.rgb_bytes")
        end
        payload = Bytes.new(cfg.rgb_bytes.to_i32)
        File.open(rgb_path) { |io| io.read_fully(payload) }
        unless Digest::SHA256.hexdigest(payload) == rgb8_sha
          raise ArgumentError.new("RGB8 input SHA-256 differs from the caller-pinned digest")
        end
        rgb_payload = payload
      end
      puts({event: "dcae_memory_plan", shape: shape, weight_bytes: plan.destination_bytes,
            workspace_bytes: scratch_bytes, output_io_bytes: output_io_bytes,
            project_in_trace_bytes: project_in_trace_bytes, rgb_upload_bytes: rgb_bytes,
            io_trace_bytes: io_bytes, named_bytes: named_bytes,
            budget_bytes: budget, full_video_generation: false}.to_json)
      return if plan_only
      unless Path[output_path].absolute? && !File.exists?(output_path) &&
             File.directory?(File.dirname(output_path)) &&
             File.realpath(File.dirname(output_path)).starts_with?(File.realpath("/opt/mldata_ssd") + "/")
        raise ArgumentError.new("output must be a fresh absolute file on the scratch SSD")
      end
      project_in_output_path = project_in_trace ? "#{output_path}.project-in-tile#{project_in_tile}.safetensors" : nil
      if path = project_in_output_path
        raise ArgumentError.new("project_in trace sidecar must be fresh") if File.exists?(path)
      end
      weight_arena = wa = ML::Video::Arena.new
      workspace_arena = ws = ML::Video::Arena.new
      io_arena = ia = ML::Video::Arena.new
      prepared = pw = ML::Video::PreparedWeights.new(plan, wa)
      workspace = ML::Video::DCAEVEncoder.prepare!(ws, cfg, budget,
        plan.destination_bytes + io_bytes)
      tensors = {} of String => Tuple(ML::MetalBuffer, Array(Int64))
      expected.each do |name, s|
        count = s.reduce(1_i64) { |n, d| n * d }.to_i32
        tensors[name] = {ia.alloc_weight(name, count, ML::Video::WeightDType::BF16), s}
      end
      project_in_tensors = {} of String => Tuple(ML::MetalBuffer, Array(Int64))
      if spec = project_in_trace
        name, s = spec
        count = s.reduce(1_i64) { |n, d| n * d }.to_i32
        project_in_tensors[name] = {ia.alloc_weight(name, count, ML::Video::WeightDType::BF16), s}
      end
      input_buffer = tensors["input.clip"][0]
      latent_buffer = tensors["latent"][0]
      if payload = rgb_payload
        rgb_buffer = ML::MetalBuffer.new(cfg.rgb_bytes)
        rgb_buffer.not_nil!.write_bytes(payload.to_unsafe, payload.size)
      else
        reference.copy_raw(input_info, Bytes.new(input_buffer.contents.as(UInt8*), input_buffer.size.to_i32))
      end
      trace_buffers = {} of String => ML::MetalBuffer
      tensors.each do |name, entry|
        trace_buffers[name] = entry[0] unless name == "input.clip" || name == "latent"
      end
      project_in_tensors.each { |name, entry| trace_buffers[name] = entry[0] }
      trace = ML::Video::DCAEVEncoderTrace.new(trace_buffers)
      mps_plans = ML::Video::DCAEVEncoder.prepare_mps!(input_buffer, latent_buffer,
        wa, workspace, cfg)
      actual_rgb_bytes = rgb_buffer.try(&.size) || 0_i64
      unless wa.resident_bytes + ws.resident_bytes + ia.resident_bytes + actual_rgb_bytes == named_bytes
        raise "DCAE encoder memory plan/allocation mismatch"
      end
      chain = active = ML::Metal::MPSCommandChain.new
      allocations = ML::MetalBuffer.hy_text_probe_allocations
      reads = ML::MetalBuffer.hy_text_probe_readbacks
      start = Time.instant
      pw.encode(active.encoder)
      if rgb = rgb_buffer
        ML::Video::DCAEVEncoder.encode_image_pixels!(active.encoder, rgb,
          input_buffer, cfg)
      end
      ML::Video::DCAEVEncoder.encode(active.encoder, input_buffer, tensors["latent"][0],
        wa, workspace, cfg, active, trace, mps_plans)
      metrics = {allocations:       ML::MetalBuffer.hy_text_probe_allocations - allocations,
                 readbacks:         ML::MetalBuffer.hy_text_probe_readbacks - reads,
                 preterminal_waits: active.wait_count}
      raise "DCAE encode used tracked buffer allocation, readback, or preterminal wait" unless metrics.values.all?(&.zero?)
      active.commit_and_wait
      wall = (Time.instant - start).total_seconds
      write_artifact(output_path, tensors)
      if path = project_in_output_path
        write_artifact(path, project_in_tensors)
      end
      puts({event: "dcae_native_completed", output: output_path,
            output_sha256: File.open(output_path) { |io| Digest::SHA256.hexdigest(io) },
            wall_seconds: wall, gpu_seconds: active.gpu_elapsed_seconds, metrics: metrics,
            waits: active.wait_count, named_bytes: named_bytes,
            mpsgraph_private_bytes_known: mps_plans.not_nil!.graph_private_bytes_known,
            tracked_metalbuffer_allocations: metrics[:allocations],
            rgb_upload_bytes: actual_rgb_bytes, gpu_pixel_normalization: has_rgb8,
            project_in_trace: project_in_output_path,
            comparison_required: true, full_video_generation: false}.to_json)
    ensure
      can_release = true
      chain.try do |active|
        begin
          active.commit_and_wait unless active.drained?
        ensure
          can_release = active.drained?
          unless can_release
            @@in_flight << -> { weights.object_id; reference.object_id; rgb_buffer.try(&.object_id); mps_plans.try(&.object_id); weight_arena.try(&.object_id); workspace_arena.try(&.object_id); io_arena.try(&.object_id); prepared.try(&.object_id); active.object_id; nil }
          end
        end
      end
      if can_release
        chain.try(&.release)
        mps_plans.try { |plans| plans.release(chain) }
        prepared.try(&.release)
        io_arena.try(&.release)
        rgb_buffer.try(&.release)
        workspace_arena.try(&.release)
        weight_arena.try(&.release)
        reference.close
        weights.close
      end
    end
  end
end

DCAEVEncoderProbe.run
