require "digest/sha256"
require "json"
require "set"
require "../../src/ml/video/hy_nomas_serial_mps"
require "./hy_nomas_text_probe"
require "./hy_nomas_text_session"
require "./hy_nomas_probe_instrumentation"
require "./qwen25_prefix_probe"

# Bounded same-input replay for two distinct pretrained Hy-Nomas DiT blocks.
# This module only owns probe setup, execution, comparison, and cleanup; all
# neural operations remain on the existing native MPS block path.
module HyNomasPretrainedSerialProbe
  extend self

  SCHEMA                            = "hy-nomas-pretrained-serial-reference-v1"
  REPORT_SCHEMA                     = "cogni_ml.hy_nomas_pretrained_serial_probe"
  PINNED_UPSTREAM_REVISION          = "911cea6458a89c2663513ff7099561ec190dd491"
  PINNED_CONDITIONING_SHA           = "167dd498c75efa396ca1f8ae33b68fa34368d2e4c96feb5d4de2d89c2666992e"
  PINNED_CONDITIONING_TENSORS_SHA   = "1668ef1699da4d66cc8d88005fb7c3e3a288feba53999d3019a81af5dfb7b849"
  PINNED_CONDITIONING_TENSORS_BYTES = 15_605_764_i64
  MAX_REFERENCE_JSON_BYTES          = 2_i64 * 1024 * 1024
  MAX_SIDECAR_BYTES                 = 64_i64 * 1024 * 1024
  SAMPLED_CAP_BYTES                 = 4_i64 * 1024 * 1024 * 1024
  PROCESS_HEADROOM_BYTES            = SAMPLED_CAP_BYTES // 10
  NAMED_BUDGET_LIMIT_BYTES          = SAMPLED_CAP_BYTES - PROCESS_HEADROOM_BYTES
  ATOL                              = 2.0e-5_f64
  RTOL                              = 2.0e-5_f64
  CONFIG                            = ML::Video::HyNomasConfig.new(
    batches: 1, video_tokens: 6, text_tokens: 23, dim: 2048, heads: 16,
    head_dim: 128, ffn_dim: 8192,
    attention_precision: ML::Video::HyNomasAttentionPrecision::F32Control
  )

  INPUT_SHAPES = {
    "input.img"      => [1, 6, 2048],
    "input.txt"      => [1, 23, 2048],
    "input.vec"      => [1, 2048],
    "input.vec_txt"  => [1, 2048],
    "input.rope_cos" => [6, 128],
    "input.rope_sin" => [6, 128],
  }

  CAPTURE_NAMES = [
    "joint.attention", "joint.key", "joint.query", "joint.value",
    "img.after_attention", "img.attn.k", "img.attn.k_norm", "img.attn.k_rope",
    "img.attn.modulated", "img.attn.norm", "img.attn.projected", "img.attn.q",
    "img.attn.q_norm", "img.attn.q_rope", "img.attn.raw", "img.attn.v",
    "img.ff.activated", "img.ff.hidden", "img.ff.modulated", "img.ff.norm",
    "img.ff.projected", "img.modulation", "img.output", "txt.after_attention",
    "txt.attn.k", "txt.attn.k_norm", "txt.attn.modulated", "txt.attn.norm",
    "txt.attn.projected", "txt.attn.q", "txt.attn.q_norm", "txt.attn.raw",
    "txt.attn.v", "txt.ff.activated", "txt.ff.hidden", "txt.ff.modulated",
    "txt.ff.norm", "txt.ff.projected", "txt.modulation", "txt.output",
  ]

  record BlockPin, index : Int32, file_bytes : Int64, file_sha256 : String,
    manifest_sha256 : String, payload_sha256 : String, header_sha256 : String?,
    range_start : Int64, range_end : Int64

  BLOCK0_PIN = BlockPin.new(0, 604_231_456_i64,
    "6810bf7cd7f9a8ac4d66240eee514637a7ccaa47dc9de02e8bc26ae4543530d1",
    "c7afec8da5ac22a37d8e38b42c9f5ecdabab7929b5c40da9260e7c30c9c73016",
    "4711e53c3b9ab29abe552f25b95d3cea6f9d75df74e098d7a6acc880befae28b",
    nil, 968_270_320_i64, 1_572_497_903_i64)
  BLOCK1_PIN = BlockPin.new(1, 604_231_488_i64,
    "026568ecff75c5c4a4d771d313469a9245eb115c34f9117af7ac21290369651c",
    "2903edf38d98e4006414ab58d39bfd36ced879fe8d41d218430bd8336bbd3d22",
    "d23df0c9abdcd65c4a858d6f1df7b6f07741e8c5253512c32756aeb047fd2e94",
    "cbfa83dc81af315b425fd783e0ce42d8ff09df772952ef2a8aff48493e21b200",
    1_572_497_904_i64, 2_176_725_487_i64)

  alias TensorResult = NamedTuple(name: String, elements: Int32,
    failures: Int32, max_abs: Float64, raw_bit_differences: Int32)
  alias SharedTensorResult = NamedTuple(name: String, elements: Int32,
    failures: Int32, max_abs: Float64, raw_bit_differences: Int32,
    exact_separate_match: Bool, exact_first_repeat_match: Bool?)
  alias RequestCounters = NamedTuple(request: Int32,
    allocations_during_encode: Int64, readbacks_before_terminal_wait: Int64,
    diagnostic_readbacks_after_terminal_wait: Int64, terminal_wait_count: Int32,
    poison_dispatches: Int32, snapshot_copy_dispatches: Int32,
    snapshot_count: Int32, snapshot_bytes: Int64,
    owner_rooted_before_dispatch: Bool, chain_lease_started_before_dispatch: Bool,
    owner_retained_through_terminal: Bool)

  def sha256_file(path : String) : String
    HyNomasTextProbe.sha256(path)
  end

  def expected_capture_shape(name : String) : Array(Int32)
    case name
    when "joint.attention"
      [1, 29, 2048]
    when "joint.query", "joint.key", "joint.value"
      [1, 29, 16, 128]
    when "img.modulation", "txt.modulation"
      [1, 12_288]
    when "img.ff.hidden", "img.ff.activated"
      [1, 6, 8192]
    when "txt.ff.hidden", "txt.ff.activated"
      [1, 23, 8192]
    else
      if name.starts_with?("img.")
        if {"img.attn.q", "img.attn.k", "img.attn.v", "img.attn.q_norm",
            "img.attn.k_norm", "img.attn.q_rope", "img.attn.k_rope"}.includes?(name)
          [1, 6, 16, 128]
        else
          [1, 6, 2048]
        end
      elsif name.starts_with?("txt.")
        if {"txt.attn.q", "txt.attn.k", "txt.attn.v", "txt.attn.q_norm",
            "txt.attn.k_norm"}.includes?(name)
          [1, 23, 16, 128]
        else
          [1, 23, 2048]
        end
      else
        raise ArgumentError.new("unknown Hy-Nomas checkpoint name: #{name}")
      end
    end
  end

  def expected_tensor_shapes : Hash(String, Array(Int32))
    result = {} of String => Array(Int32)
    INPUT_SHAPES.each { |name, shape| result[name] = shape }
    {"block0", "block1"}.each do |block|
      CAPTURE_NAMES.each do |name|
        result["#{block}.#{name}"] = expected_capture_shape(name)
      end
    end
    result
  end

  def checked_element_count(shape : Array(Int32)) : Int32
    raise ArgumentError.new("invalid reference tensor rank") unless 0 < shape.size <= 4
    count = 1_i64
    shape.each do |dimension|
      raise ArgumentError.new("invalid reference tensor dimension") unless dimension > 0 && count <= Int32::MAX // dimension
      count *= dimension
    end
    count.to_i32
  end

  # Shared pure admission gate: intentionally callable by CPU-only falsifiers.
  def admit_named_peak!(phase : String, peak_bytes : Int64, budget_bytes : Int64) : Nil
    unless peak_bytes >= 0 && budget_bytes > 0 && peak_bytes <= budget_bytes
      raise ArgumentError.new("#{phase} named-byte peak #{peak_bytes} exceeds budget #{budget_bytes}")
    end
  end

  # The base text fixture SHA is pinned. Its Qwen document digest is read only
  # to satisfy the fixture API, then the constructor rechecks the entire
  # authenticated document against the supplied base SHA.
  def conditioning_qwen_sha256(path : String, expected_sha256 : String) : String
    raise ArgumentError.new("conditioning fixture must be a regular non-symlink file") unless File.file?(path) && !File.symlink?(path)
    raise ArgumentError.new("conditioning fixture exceeds its bounded JSON size") unless 0 < File.size(path) <= MAX_REFERENCE_JSON_BYTES
    resolved = File.realpath(path)
    raw = File.read(resolved)
    raise ArgumentError.new("conditioning fixture digest mismatch") unless Digest::SHA256.hexdigest(raw) == expected_sha256
    document = JSON.parse(raw)
    document["qwen_source"]["reference_sha256"].as_s
  end

  class ReferenceFixture
    getter path : String
    getter sha256 : String
    getter document : JSON::Any
    getter sidecar_bytes : Int64
    @sidecar : Bytes
    @block_paths = {} of String => String
    @manifest_paths = {} of String => String

    def initialize(path : String, expected_sha256 : String)
      raise ArgumentError.new("reference must be a regular non-symlink file") unless File.file?(path) && !File.symlink?(path)
      raise ArgumentError.new("reference exceeds its bounded JSON size") unless 0 < File.size(path) <= MAX_REFERENCE_JSON_BYTES
      @path = File.realpath(path)
      raw = File.read(@path)
      @sha256 = Digest::SHA256.hexdigest(raw)
      unless expected_sha256.matches?(/\A[0-9a-f]{64}\z/) && @sha256 == expected_sha256
        raise ArgumentError.new("trusted two-block reference digest mismatch")
      end
      @document = JSON.parse(raw)
      validate_document!
      tensor_file = @document["tensor_file"]
      @sidecar_bytes = tensor_file["nbytes"].as_i64
      name = tensor_file["name"].as_s
      unless name.matches?(/\A[A-Za-z0-9_.-]+\.tensors\.f32le\z/) && name != "." && name != ".."
        raise ArgumentError.new("reference tensor sidecar must be a sibling basename")
      end
      sidecar_path = File.join(File.dirname(@path), name)
      unless File.file?(sidecar_path) && !File.symlink?(sidecar_path) && File.size(sidecar_path) == @sidecar_bytes
        raise ArgumentError.new("reference tensor sidecar extent mismatch")
      end
      @sidecar = File.read(sidecar_path).to_slice.dup
      unless Digest::SHA256.hexdigest(@sidecar) == tensor_file["sha256"].as_s
        raise ArgumentError.new("reference tensor sidecar digest mismatch")
      end
    end

    def tensor_bytes(name : String) : Int64
      @document["tensors"][name]["nbytes"].as_i64
    end

    def tensor_shape(name : String) : Array(Int32)
      @document["tensors"][name]["shape"].as_a.map(&.as_i.to_i32)
    end

    def values(name : String) : Array(Float32)
      metadata = @document["tensors"][name]
      count = HyNomasPretrainedSerialProbe.checked_element_count(tensor_shape(name))
      offset = metadata["offset_bytes"].as_i64.to_i32
      Array(Float32).new(count) do |index|
        IO::ByteFormat::LittleEndian.decode(Float32, @sidecar[offset + index * 4, 4])
      end
    end

    def block_path(name : String) : String
      @block_paths[name]
    end

    def manifest_path(name : String) : String
      @manifest_paths[name]
    end

    # The reference's path and digest fields are claims. Before CheckpointFile
    # parses or plans anything, authenticate both local checkpoint bytes and
    # the sibling provenance manifest against the fixed artifact pins.
    def authenticate_weight_artifacts!(conditioning : HyNomasTextProbe::Fixture) : Nil
      {"block0" => BLOCK0_PIN, "block1" => BLOCK1_PIN}.each do |name, pin|
        path = block_path(name)
        manifest = manifest_path(name)
        unless File.file?(path) && !File.symlink?(path) && File.size(path) == pin.file_bytes
          raise ArgumentError.new("#{name} checkpoint extent/type mismatch")
        end
        unless HyNomasTextProbe.sha256(path) == pin.file_sha256
          raise ArgumentError.new("#{name} checkpoint bytes differ from the authenticated pin")
        end
        unless File.file?(manifest) && !File.symlink?(manifest) && File.size(manifest) <= 64 * 1024
          raise ArgumentError.new("#{name} provenance manifest extent/type mismatch")
        end
        raw_manifest = File.read(manifest)
        unless Digest::SHA256.hexdigest(raw_manifest) == pin.manifest_sha256
          raise ArgumentError.new("#{name} provenance manifest bytes differ from the authenticated pin")
        end
        validate_manifest!(name, JSON.parse(raw_manifest), path, pin)
      end
      unless File.realpath(block_path("block0")) == conditioning.block_weights
        raise ArgumentError.new("block0 checkpoint is not the block authenticated by saved conditioning")
      end
      unless File.realpath(block_path("block0")) != File.realpath(block_path("block1"))
        raise ArgumentError.new("block0 and block1 checkpoint files alias")
      end
    end

    def input_bytes : Int64
      INPUT_SHAPES.keys.sum { |name| tensor_bytes(name) }
    end

    def snapshot_bytes : Int64
      {"block0", "block1"}.sum do |block|
        CAPTURE_NAMES.sum { |name| tensor_bytes("#{block}.#{name}") }
      end
    end

    def validate_conditioning_inputs!(conditioning : HyNomasTextProbe::Fixture) : Nil
      bindings = {
        "input.img"      => "input.video",
        "input.txt"      => "refiner.projected_text",
        "input.vec"      => "refiner.vec",
        "input.vec_txt"  => "refiner.vec",
        "input.rope_cos" => "input.rope_cos",
        "input.rope_sin" => "input.rope_sin",
      }
      bindings.each do |actual_name, source_name|
        raise ArgumentError.new("saved conditioning tensor is missing: #{source_name}") unless conditioning.document["tensors"].as_h.has_key?(source_name)
        unless tensor_shape(actual_name) == conditioning.shape(source_name)
          raise ArgumentError.new("inherited input shape differs from saved conditioning: #{actual_name}")
        end
        actual = values(actual_name)
        expected = conditioning.values(source_name)
        unless actual.size == expected.size && actual.each_with_index.all? do |value, index|
                 value.finite? && expected[index].finite? && value.unsafe_as(UInt32) == expected[index].unsafe_as(UInt32)
               end
          raise ArgumentError.new("inherited input differs from authenticated saved conditioning: #{actual_name}")
        end
      end
      vec = values("input.vec")
      vec_txt = values("input.vec_txt")
      unless vec.each_with_index.all? { |value, index| value.unsafe_as(UInt32) == vec_txt[index].unsafe_as(UInt32) }
        raise ArgumentError.new("the upstream saved fixture requires vec_txt to duplicate vec")
      end
    end

    private def validate_document! : Nil
      raise ArgumentError.new("two-block reference schema mismatch") unless @document["schema"].as_s == SCHEMA && @document["schema_version"].as_i == 1
      raise ArgumentError.new("two-block upstream revision mismatch") unless @document["upstream_revision"].as_s == PINNED_UPSTREAM_REVISION
      validate_geometry!
      validate_precision!
      validate_producer!
      validate_source!
      validate_inventory!
      validate_layout!
    end

    private def validate_geometry! : Nil
      geometry = @document["geometry"]
      expected = {
        "batches" => 1, "text_tokens" => 23, "dim" => 2048, "heads" => 16,
        "head_dim" => 128, "ffn_dim" => 8192, "video_tokens" => 6,
        "segments" => 1, "chunk_size_tokens" => 0, "text_seg_pos_emb_num" => 0,
      }
      expected.each do |key, value|
        raise ArgumentError.new("two-block geometry mismatch: #{key}") unless geometry[key].as_i == value
      end
      raise ArgumentError.new("two-block geometry has unexpected fields") unless geometry.as_h.keys.to_set == expected.keys.to_set
    end

    private def validate_precision! : Nil
      precision = @document["precision"]
      expected_strings = {
        "source_weight_dtype" => "F32",
        "parameter_values"    => "BF16-rounded values",
        "parameter_storage"   => "F32",
        "compute"             => "F32",
        "attention"           => "PyTorch MPS SDPA F32",
        "weight_rounding"     => "source F32 values converted F32->BF16->F32 on MPS",
      }
      expected_strings.each do |key, value|
        raise ArgumentError.new("two-block precision mismatch: #{key}") unless precision[key].as_s == value
      end
      {"native_bf16_parameter_storage", "native_bf16_compute", "cuda_bf16_parity"}.each do |key|
        raise ArgumentError.new("two-block precision scope mismatch: #{key}") unless precision[key].as_bool == false
      end
      raise ArgumentError.new("two-block precision has unexpected fields") unless precision.as_h.keys.to_set == (expected_strings.keys + ["native_bf16_parameter_storage", "native_bf16_compute", "cuda_bf16_parity"]).to_set
      tolerance = @document["tolerance"]
      raise ArgumentError.new("two-block tolerance mismatch") unless tolerance["atol"].as_f == ATOL && tolerance["rtol"].as_f == RTOL
    end

    private def validate_producer! : Nil
      producer = @document["producer"]
      raise ArgumentError.new("two-block producer receipt mismatch") unless producer["device"].as_s == "mps" &&
                                                                            producer["activation_readback"].as_s == "one packed final F32 tensor sidecar readback" &&
                                                                            producer["attention_adapter_calls"].as_i == 2 &&
                                                                            producer["attention_adapter_calls_by_block"].as_a.map(&.as_i) == [1, 1]
    end

    private def validate_source! : Nil
      source = @document["source"]
      conditioning = source["conditioning"]
      unless conditioning["reference_sha256"].as_s == PINNED_CONDITIONING_SHA &&
             conditioning["tensor_file_name"].as_s == "text-conditioning-v2.tensors.f32le" &&
             conditioning["tensor_sha256"].as_s == PINNED_CONDITIONING_TENSORS_SHA &&
             conditioning["tensor_nbytes"].as_i64 == PINNED_CONDITIONING_TENSORS_BYTES &&
             conditioning["source_schema"].as_s == "hy-nomas-text-conditioning-reference-v1" &&
             conditioning["upstream_revision"].as_s == PINNED_UPSTREAM_REVISION
        raise ArgumentError.new("inherited conditioning source pin mismatch")
      end
      blocks = source["blocks"]
      raise ArgumentError.new("two-block source inventory mismatch") unless blocks.as_h.keys.to_set == {"block0", "block1"}.to_set
      validate_block_record!("block0", blocks["block0"], BLOCK0_PIN)
      validate_block_record!("block1", blocks["block1"], BLOCK1_PIN)
      block0 = blocks["block0"]
      block1 = blocks["block1"]
      if block0["file_sha256"].as_s == block1["file_sha256"].as_s ||
         block0["manifest_sha256"].as_s == block1["manifest_sha256"].as_s ||
         block0["source_payload_sha256"].as_s == block1["source_payload_sha256"].as_s
        raise ArgumentError.new("block identity alias rejected: block1 must be distinct")
      end
    end

    private def validate_block_record!(name : String, record : JSON::Any, pin : BlockPin) : Nil
      expected = {
        "block_index"           => pin.index,
        "runtime_layer_name"    => "double_block_#{pin.index + 1}",
        "file_bytes"            => pin.file_bytes,
        "file_sha256"           => pin.file_sha256,
        "manifest_sha256"       => pin.manifest_sha256,
        "source_repository"     => "tencent/Hy-Nomas",
        "source_revision"       => "bac719a80c8d02f91c485fca804bb8221e149b70",
        "source_git_commit"     => PINNED_UPSTREAM_REVISION,
        "checkpoint_path"       => "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors",
        "checkpoint_lfs_sha256" => "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70",
        "source_payload_sha256" => pin.payload_sha256,
        "range_start_inclusive" => pin.range_start,
        "range_end_inclusive"   => pin.range_end,
        "tensor_prefix"         => "double_blocks.#{pin.index}.",
        "tensor_count"          => 32,
      }
      expected.each do |key, value|
        actual = record[key]
        matches = case value
                  when String then actual.as_s == value
                  when Int32  then actual.as_i == value
                  when Int64  then actual.as_i64 == value
                  else             false
                  end
        raise ArgumentError.new("#{name} pinned source metadata mismatch: #{key}") unless matches
      end
      header = record["source_header_sha256"]
      if pin.header_sha256
        raise ArgumentError.new("#{name} checkpoint header pin mismatch") unless header.as_s == pin.header_sha256
      else
        raise ArgumentError.new("#{name} checkpoint header pin mismatch") unless header.raw.nil?
      end
      path = record["path"].as_s
      manifest = record["manifest_path"].as_s
      unless path.starts_with?("/") && path.ends_with?(".safetensors") &&
             manifest.starts_with?("/") && manifest.ends_with?(".safetensors.manifest.json")
        raise ArgumentError.new("#{name} source paths are not absolute safetensors files")
      end
      @block_paths[name] = path
      @manifest_paths[name] = manifest
    end

    private def validate_manifest!(name : String, manifest : JSON::Any,
                                   path : String, pin : BlockPin) : Nil
      raise ArgumentError.new("#{name} provenance schema mismatch") unless manifest["format"].as_s == "hy-nomas-block-subset-provenance-v1"
      output = manifest["output"]
      unless output["filename"].as_s == File.basename(path) &&
             output["sha256"].as_s == pin.file_sha256 &&
             output["size_bytes"].as_i64 == pin.file_bytes
        raise ArgumentError.new("#{name} provenance output does not bind the checkpoint bytes")
      end
      source = manifest["source"]
      unless source["repository"].as_s == "tencent/Hy-Nomas" &&
             source["revision"].as_s == "bac719a80c8d02f91c485fca804bb8221e149b70" &&
             source["source_git_commit"].as_s == PINNED_UPSTREAM_REVISION &&
             source["checkpoint_path"].as_s == "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors" &&
             source["checkpoint_lfs_sha256"].as_s == "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70" &&
             source["payload_sha256_observed"].as_s == pin.payload_sha256 &&
             source["range_start_inclusive"].as_i64 == pin.range_start &&
             source["range_end_inclusive"].as_i64 == pin.range_end
        raise ArgumentError.new("#{name} provenance source range mismatch")
      end
      header = source["header_sha256_observed"]?
      if pin.header_sha256
        raise ArgumentError.new("#{name} provenance header mismatch") unless header && header.as_s == pin.header_sha256
      elsif header
        raise ArgumentError.new("#{name} block0 provenance unexpectedly has a header digest")
      end
      subset = manifest["subset"]
      unless subset["dtype"].as_s == "F32" &&
             subset["payload_bytes"].as_i64 == pin.range_end - pin.range_start + 1 &&
             subset["tensor_count"].as_i == 32 &&
             subset["tensor_prefix"].as_s == "double_blocks.#{pin.index}." &&
             (name == "block0" || subset["block_index"].as_i == 1)
        raise ArgumentError.new("#{name} provenance subset metadata mismatch")
      end
    end

    private def validate_inventory! : Nil
      expected = HyNomasPretrainedSerialProbe.expected_tensor_shapes
      actual = @document["tensors"].as_h
      raise ArgumentError.new("two-block tensor inventory mismatch") unless actual.keys.to_set == expected.keys.to_set && expected.size == 86
      expected.each do |name, shape|
        metadata = actual[name]
        raise ArgumentError.new("two-block tensor must be F32: #{name}") unless metadata["dtype"].as_s == "F32"
        unless metadata["shape"].as_a.map(&.as_i.to_i32) == shape
          raise ArgumentError.new("two-block tensor shape mismatch: #{name}")
        end
        count = HyNomasPretrainedSerialProbe.checked_element_count(shape)
        raise ArgumentError.new("two-block tensor size mismatch: #{name}") unless metadata["nbytes"].as_i64 == count.to_i64 * 4
      end
      tensor_file = @document["tensor_file"]
      raise ArgumentError.new("two-block sidecar must be bounded to 64 MiB") unless 0 < tensor_file["nbytes"].as_i64 <= MAX_SIDECAR_BYTES
      raise ArgumentError.new("two-block sidecar digest malformed") unless tensor_file["sha256"].as_s.matches?(/\A[0-9a-f]{64}\z/)
    end

    private def validate_layout! : Nil
      tensors = @document["tensors"].as_h
      cursor = 0_i64
      tensors.values.sort_by { |metadata| metadata["offset_bytes"].as_i64 }.each do |metadata|
        raise ArgumentError.new("two-block tensors must have contiguous sidecar offsets") unless metadata["offset_bytes"].as_i64 == cursor
        size = metadata["nbytes"].as_i64
        raise ArgumentError.new("two-block tensor exceeds sidecar extent") if size > @document["tensor_file"]["nbytes"].as_i64 - cursor
        cursor += size
      end
      raise ArgumentError.new("two-block sidecar has trailing or missing bytes") unless cursor == @document["tensor_file"]["nbytes"].as_i64
    end
  end

  class RunOwner
    @@lifetime_mutex = Mutex.new
    @@live_owners = [] of RunOwner

    getter reference : ReferenceFixture
    getter named_byte_budget : Int64
    @conditioning : HyNomasTextProbe::Fixture
    @chain_lease = HyNomasTextProbe::SessionChainLease(ML::Metal::MPSCommandChain).new
    @active_chain : ML::Metal::MPSCommandChain? = nil
    @released = false
    @rooted = false
    @control_phase_released = false
    @input_arena : ML::Video::Arena? = nil
    @weight_arenas = [] of ML::Video::Arena
    @checkpoint_files = [] of ML::Video::CheckpointFile
    @weight_plans = [] of ML::Video::WeightPlan
    @prepared_weights = [] of ML::Video::PreparedWeights
    @control_scratch = [] of ML::Video::Arena
    @control_blocks = [] of ML::Video::HyNomasDualStreamBlock
    @control_workspaces = [] of ML::Video::HyNomasMPSWorkspace
    @serial_workspace : ML::Video::HyNomasMPSSerialWorkspace? = nil
    @snapshot_arena : ML::Video::Arena? = nil
    @snapshot_buffers = {} of String => ML::MetalBuffer
    @poison_pipeline : ML::Metal::ComputePipeline? = nil
    @control_values = {} of String => Array(Float32)
    @control_results = [] of TensorResult
    @first_shared_values = {} of String => Array(Float32)
    @control_peak_named_bytes = 0_i64
    @shared_peak_named_bytes = 0_i64
    @control_workspace_bytes = 0_i64
    @control_scratch_bytes = 0_i64
    @shared_workspace_bytes = 0_i64
    @shared_scratch_bytes = 0_i64
    @snapshot_bytes = 0_i64
    @weight_destination_bytes = 0_i64
    @weight_raw_staging_bytes = 0_i64
    @request_metrics = [] of RequestCounters
    @shared_results = [] of Array(SharedTensorResult)

    def initialize(@reference : ReferenceFixture,
                   @conditioning : HyNomasTextProbe::Fixture,
                   max_named_bytes : Int64)
      unless max_named_bytes > 0 && max_named_bytes <= NAMED_BUDGET_LIMIT_BYTES
        raise ArgumentError.new("max named byte limit must be positive and no larger than the admitted 90% budget")
      end
      @named_byte_budget = max_named_bytes
      begin
        plan_weights!
        admit_control_peak!
        admit_shared_peak!
        prepare_weight_destinations!
        prepare_inputs!
        @poison_pipeline = ML::Metal::ComputePipeline.new("hy_nomas_serial_poison_f32", <<-METAL)
        #include <metal_stdlib>
        using namespace metal;
        kernel void hy_nomas_serial_poison_f32(device float *dst [[buffer(0)]],
                                                constant uint &count [[buffer(1)]],
                                                uint i [[thread_position_in_grid]]) {
          if (i < count) dst[i] = as_type<float>(0x7fc00000u);
        }
        METAL
        retain_lifetime_root!
      rescue ex
        release_setup_before_submission!
        raise ex
      end
    end

    def rooted? : Bool
      @rooted
    end

    def released? : Bool
      @released
    end

    def control_phase_released? : Bool
      @control_phase_released
    end

    def named_bytes_report
      {
        sampled_cap_bytes:                SAMPLED_CAP_BYTES,
        process_headroom_bytes:           PROCESS_HEADROOM_BYTES,
        process_estimate_scope:           "named Metal buffers plus bounded host reference and diagnostic arrays; opaque MPS/runtime allocations excluded",
        admitted_named_limit_bytes:       @named_byte_budget,
        input_bytes:                      @reference.input_bytes,
        weight_destination_bytes:         @weight_destination_bytes,
        weight_raw_staging_bytes:         @weight_raw_staging_bytes,
        control_scratch_bytes:            @control_scratch_bytes,
        control_workspace_bytes:          @control_workspace_bytes,
        separate_scratch_workspace_bytes: @control_scratch_bytes + @control_workspace_bytes,
        control_peak_named_bytes:         @control_peak_named_bytes,
        shared_scratch_bytes:             @shared_scratch_bytes,
        shared_workspace_bytes:           @shared_workspace_bytes,
        shared_scratch_workspace_bytes:   @shared_scratch_bytes + @shared_workspace_bytes,
        scratch_workspace_reuse_bytes:    (@control_scratch_bytes + @control_workspace_bytes) -
          (@shared_scratch_bytes + @shared_workspace_bytes),
        diagnostic_snapshot_bytes:      @snapshot_bytes,
        shared_peak_named_bytes:        @shared_peak_named_bytes,
        host_reference_bytes:           @reference.sidecar_bytes + PINNED_CONDITIONING_TENSORS_BYTES,
        host_capture_bytes:             @reference.snapshot_bytes * 4,
        named_plus_host_estimate_bytes: Math.max(@control_peak_named_bytes, @shared_peak_named_bytes) +
          @reference.sidecar_bytes + PINNED_CONDITIONING_TENSORS_BYTES + @reference.snapshot_bytes * 4,
      }
    end

    def control_results : Array(TensorResult)
      @control_results
    end

    def request_metrics : Array(RequestCounters)
      @request_metrics
    end

    def shared_results : Array(Array(SharedTensorResult))
      @shared_results
    end

    def run_control! : Array(TensorResult)
      prepare_control!
      chain = ML::Metal::MPSCommandChain.new
      begin_request!(chain)
      allocations_before = ML::MetalBuffer.hy_text_probe_allocations
      readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks
      @prepared_weights.each { |weights| weights.encode(chain.encoder) }
      block0, block1 = @control_blocks
      img, txt = block0.encode_mps(chain,
        input("input.img"), input("input.txt"), input("input.vec"), input("input.vec_txt"),
        input("input.rope_cos"), input("input.rope_sin"), @control_workspaces[0])
      block1.encode_mps(chain, img, txt, input("input.vec"), input("input.vec_txt"),
        input("input.rope_cos"), input("input.rope_sin"), @control_workspaces[1])
      allocations_during = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
      readbacks_before_wait = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
      ensure_preterminal_counters!(chain, allocations_during, readbacks_before_wait)
      chain.commit_and_wait
      wait_count = chain.wait_count
      raise "two-plan control did not use one terminal wait" unless chain.completed? && wait_count == 1
      finish_chain_after_terminal!(chain)
      expected_reads = 0_i64
      @control_blocks.each_with_index do |block, block_index|
        block.checkpoints.each do |name, buffer|
          key = "block#{block_index}.#{name}"
          values = buffer.read((buffer.size // 4).to_i32)
          @control_values[key] = values
          result = compare_to_reference(key, values)
          @control_results << result
          expected_reads += 1
        end
      end
      diagnostic_reads = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
      raise "two-plan control checkpoint readback count mismatch" unless diagnostic_reads == expected_reads && expected_reads == 80
      @control_results.sort_by!(&.[:name])
      @request_metrics << {
        request: 0, allocations_during_encode: allocations_during,
        readbacks_before_terminal_wait: readbacks_before_wait,
        diagnostic_readbacks_after_terminal_wait: diagnostic_reads,
        terminal_wait_count: wait_count, poison_dispatches: 0,
        snapshot_copy_dispatches: 0, snapshot_count: 80,
        snapshot_bytes: @reference.snapshot_bytes,
        owner_rooted_before_dispatch: @rooted,
        chain_lease_started_before_dispatch: true,
        owner_retained_through_terminal: @rooted,
      }
      @control_values.size == 80 ? @control_results : raise "two-plan control did not capture all 80 checkpoints"
    rescue ex
      drain_active_chain_after_error!
      raise ex
    end

    # Free the independent-plan scratch/workspaces and conversion staging only
    # after the control's terminal completion. Resident BF16 weight destinations
    # and the six uploaded inputs remain rooted for the shared probe.
    def release_control_phase! : Nil
      ensure_no_pending_chain!
      @control_workspaces.reverse_each(&.release)
      @control_workspaces.clear
      @control_blocks.clear
      @control_scratch.reverse_each(&.release)
      @control_scratch.clear
      @prepared_weights.each(&.release)
      @prepared_weights.clear
      @checkpoint_files.each(&.close)
      @checkpoint_files.clear
      @weight_plans.clear
      @control_phase_released = true
    end

    def prepare_serial!(successor_block0_control : Bool = false) : Nil
      raise ArgumentError.new("control phase must be released before shared setup") unless @control_workspaces.empty? && @prepared_weights.empty?
      unless @shared_peak_named_bytes <= @named_byte_budget
        raise ArgumentError.new("shared probe named-byte peak exceeds the admitted 90% budget")
      end
      weights = successor_block0_control ? [@weight_arenas[0], @weight_arenas[0]] : @weight_arenas
      @serial_workspace = ML::Video::HyNomasMPSSerialWorkspace.new(CONFIG, weights,
        input("input.img"), input("input.txt"), input("input.vec"), input("input.vec_txt"),
        input("input.rope_cos"), input("input.rope_sin"))
      serial_blocks = @serial_workspace.not_nil!.blocks
      validate_runtime_capture_inventory!(serial_blocks[0], serial_blocks[1])
      @snapshot_arena = ML::Video::Arena.new
      serial_blocks.each_with_index do |block, block_index|
        CAPTURE_NAMES.each do |name|
          key = "block#{block_index}.#{name}"
          source = block.checkpoints[name]
          bytes = @reference.tensor_bytes(key)
          unless source.size == bytes && bytes % 4 == 0
            raise ArgumentError.new("checkpoint snapshot extent mismatch: #{key}")
          end
          @snapshot_buffers[key] = @snapshot_arena.not_nil!.alloc(key, (bytes // 4).to_i32)
        end
      end
      @snapshot_bytes = @snapshot_arena.not_nil!.resident_bytes
      raise ArgumentError.new("diagnostic snapshot allocation does not match authenticated inventory") unless @snapshot_bytes == @reference.snapshot_bytes
      unless @serial_workspace.not_nil!.scratch_bytes == @shared_scratch_bytes &&
             @serial_workspace.not_nil!.mps_workspace_bytes == @shared_workspace_bytes
        raise ArgumentError.new("serial shared-plan bytes differ from the pre-allocation estimate")
      end
    rescue ex
      raise ex
    end

    def run_shared_request!(request : Int32, capture_first_repeat : Bool,
                            negative_mode : Bool) : Array(SharedTensorResult)
      serial = @serial_workspace || raise ArgumentError.new("shared workspace is unavailable")
      chain = ML::Metal::MPSCommandChain.new
      begin_request!(chain)
      allocations_before = ML::MetalBuffer.hy_text_probe_allocations
      readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks
      poison_dispatches = poison_shared_scratch!(chain)
      copy_dispatches = 0_i32
      callback = ->(block_index : Int32, block : ML::Video::HyNomasDualStreamBlock) do
        raise ArgumentError.new("serial callback block index escaped [0,1]") unless 0 <= block_index <= 1
        raise ArgumentError.new("serial callback checkpoint inventory mismatch") unless block.checkpoints.size == 40
        block.checkpoints.each do |name, source|
          key = "block#{block_index}.#{name}"
          target = @snapshot_buffers[key]?
          raise ArgumentError.new("missing preallocated diagnostic snapshot: #{key}") unless target
          raise ArgumentError.new("diagnostic snapshot aliases checkpoint source: #{key}") if source.same?(target)
          ML::Video::MetalOps.copy(chain.encoder, source, target.not_nil!, (source.size // 4).to_i32)
          copy_dispatches += 1
        end
        nil
      end
      serial.encode(chain, after_block: callback)
      allocations_during = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
      readbacks_before_wait = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
      ensure_preterminal_counters!(chain, allocations_during, readbacks_before_wait)
      raise "serial snapshot copy dispatch count mismatch" unless copy_dispatches == 80
      chain.commit_and_wait
      wait_count = chain.wait_count
      raise "shared workspace request did not use one terminal wait" unless chain.completed? && wait_count == 1
      finish_chain_after_terminal!(chain)

      results = [] of SharedTensorResult
      current_values = {} of String => Array(Float32) if capture_first_repeat
      expected_reads = 0_i64
      {"block0", "block1"}.each do |block_name|
        CAPTURE_NAMES.each do |capture_name|
          key = "#{block_name}.#{capture_name}"
          buffer = @snapshot_buffers[key]
          actual = buffer.read((buffer.size // 4).to_i32)
          reference_result = compare_to_reference(key, actual)
          control = @control_values[key]?
          raise ArgumentError.new("missing independent control snapshot: #{key}") unless control
          exact_control = raw_bit_differences(actual, control.not_nil!) == 0
          exact_first = if capture_first_repeat
                          current_values.not_nil![key] = actual
                          true
                        else
                          previous = @first_shared_values[key]?
                          previous && raw_bit_differences(actual, previous) == 0
                        end
          results << {
            name: key, elements: reference_result[:elements],
            failures: reference_result[:failures], max_abs: reference_result[:max_abs],
            raw_bit_differences: reference_result[:raw_bit_differences],
            exact_separate_match: exact_control, exact_first_repeat_match: exact_first,
          }
          expected_reads += 1
        end
      end
      if capture_first_repeat
        @first_shared_values = current_values.not_nil!
      end
      diagnostic_reads = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
      raise "shared diagnostic readback count mismatch" unless diagnostic_reads == expected_reads && expected_reads == 80
      @request_metrics << {
        request: request, allocations_during_encode: allocations_during,
        readbacks_before_terminal_wait: readbacks_before_wait,
        diagnostic_readbacks_after_terminal_wait: diagnostic_reads,
        terminal_wait_count: wait_count, poison_dispatches: poison_dispatches,
        snapshot_copy_dispatches: copy_dispatches, snapshot_count: 80,
        snapshot_bytes: @snapshot_bytes,
        owner_rooted_before_dispatch: @rooted,
        chain_lease_started_before_dispatch: true,
        owner_retained_through_terminal: @rooted,
      }
      @shared_results << results
      results
    rescue ex
      drain_active_chain_after_error!
      raise ex
    end

    def control_passed? : Bool
      @control_results.size == 80 && @control_results.all? { |row| row[:failures] == 0 }
    end

    def shared_passed?(negative_mode : Bool = false) : Bool
      return false if @shared_results.empty?
      if negative_mode
        rows = @shared_results.last
        block0_ok = rows.select(&.[:name].starts_with?("block0.")).all? do |row|
          row[:failures] == 0 && row[:exact_separate_match]
        end
        output_rows = rows.select { |row| row[:name] == "block1.img.output" || row[:name] == "block1.txt.output" }
        block1_output_rejected = output_rows.any? do |row|
          row[:failures] > 0 && row[:raw_bit_differences] > 0
        end
        return block0_ok && block1_output_rejected
      end
      @shared_results.size == 2 && @shared_results.all? do |rows|
        rows.size == 80 && rows.all? do |row|
          row[:failures] == 0 && row[:exact_separate_match] &&
            (row[:exact_first_repeat_match] == true)
        end
      end
    end

    def cleanup_after_terminal! : Nil
      return if @released
      ensure_no_pending_chain!
      @serial_workspace.try(&.release)
      @serial_workspace = nil
      @snapshot_arena.try(&.release)
      @snapshot_arena = nil
      @snapshot_buffers.clear
      @control_workspaces.reverse_each(&.release)
      @control_workspaces.clear
      @control_blocks.clear
      @control_scratch.reverse_each(&.release)
      @control_scratch.clear
      @prepared_weights.each(&.release)
      @prepared_weights.clear
      @checkpoint_files.each(&.close)
      @checkpoint_files.clear
      @weight_plans.clear
      @input_arena.try(&.release)
      @input_arena = nil
      @weight_arenas.reverse_each(&.release)
      @weight_arenas.clear
      @released = true
      release_lifetime_root!
    end

    # Used on exceptional paths. If the terminal drain cannot prove completion,
    # keep the complete owner rooted and retain every borrowed resource.
    def drain_active_chain_after_error! : Nil
      return unless chain = @active_chain
      if chain.completed?
        finish_chain_after_terminal!(chain)
      else
        Qwen25PrefixProbe.cleanup_after_drain(chain) do
          finish_chain_after_terminal!(chain)
        end
      end
    end

    # Header admission and the complete named-byte estimate run before any
    # MetalBuffer is created. WeightPlan and CheckpointFile are CPU-only.
    private def plan_weights! : Nil
      ["block0", "block1"].each do |name|
        file = ML::Video::CheckpointFile.new(@reference.block_path(name))
        @checkpoint_files << file
        index = name == "block0" ? 0 : 1
        plan = ML::Video::HyNomasDualStreamBlock.weight_plan([file], CONFIG,
          NAMED_BUDGET_LIMIT_BYTES, resident_dtype: ML::Video::WeightDType::BF16,
          block_index: index, require_all: true, allow_f32_to_bf16: true)
        raise ArgumentError.new("pretrained #{name} must contain exactly 32 checkpoint tensors") unless plan.entries.size == 32
        @weight_destination_bytes += plan.destination_bytes
        @weight_raw_staging_bytes += plan.staging_bytes
        @weight_plans << plan
      end
    end

    private def prepare_weight_destinations! : Nil
      @weight_plans.each do |plan|
        arena = ML::Video::Arena.new
        @weight_arenas << arena
        @prepared_weights << ML::Video::PreparedWeights.new(plan, arena)
      end
      unless @weight_arenas[0].resident_bytes + @weight_arenas[1].resident_bytes == @weight_destination_bytes
        raise ArgumentError.new("prepared BF16 destination bytes differ from the admitted weight plan")
      end
    end

    private def prepare_inputs! : Nil
      arena = ML::Video::Arena.new
      @input_arena = arena
      {"input.img", "input.txt", "input.vec", "input.vec_txt", "input.rope_cos", "input.rope_sin"}.each do |name|
        values = @reference.values(name)
        raise ArgumentError.new("non-finite saved input: #{name}") unless values.all?(&.finite?)
        arena.upload(name, values)
      end
      raise ArgumentError.new("uploaded input bytes differ from authenticated inventory") unless arena.resident_bytes == @reference.input_bytes
    end

    private def admit_control_peak! : Nil
      scratch_bytes = ML::Video::HyNomasDualStreamBlock.scratch_plan(CONFIG).sum do |_, count|
        count.to_i64 * sizeof(Float32)
      end
      workspace_bytes = workspace_named_bytes
      @control_scratch_bytes = scratch_bytes * 2
      @control_workspace_bytes = workspace_bytes * 2
      @control_peak_named_bytes = @weight_destination_bytes + @weight_raw_staging_bytes +
                                  @reference.input_bytes + @control_scratch_bytes + @control_workspace_bytes
      HyNomasPretrainedSerialProbe.admit_named_peak!("independent two-plan control",
        @control_peak_named_bytes, @named_byte_budget)
    end

    private def admit_shared_peak! : Nil
      scratch_bytes = ML::Video::HyNomasDualStreamBlock.scratch_plan(CONFIG).sum do |_, count|
        count.to_i64 * sizeof(Float32)
      end
      second_output_bytes = (CONFIG.batches.to_i64 * CONFIG.video_tokens * CONFIG.dim +
                             CONFIG.batches.to_i64 * CONFIG.text_tokens * CONFIG.dim) * sizeof(Float32)
      @shared_scratch_bytes = scratch_bytes + second_output_bytes
      @shared_workspace_bytes = workspace_named_bytes
      @snapshot_bytes = @reference.snapshot_bytes
      @shared_peak_named_bytes = @weight_destination_bytes + @reference.input_bytes +
                                 @shared_scratch_bytes + @shared_workspace_bytes + @snapshot_bytes
      HyNomasPretrainedSerialProbe.admit_named_peak!("shared workspace",
        @shared_peak_named_bytes, @named_byte_budget)
    end

    private def workspace_named_bytes : Int64
      config = CONFIG
      weight_stage = ML::Video::HyNomasMPSWorkspace.required_weight_stage_count(config).to_i64 * 4
      attention = 4_i64 * config.joint_rows * config.dim * 4
      rms = 2_i64 * config.joint_rows * (config.dim + config.heads) * 4
      weight_stage + attention + rms
    end

    private def prepare_control! : Nil
      raise ArgumentError.new("control plans already prepared") unless @control_workspaces.empty?
      input = @input_arena || raise ArgumentError.new("probe inputs are unavailable")
      2.times do
        scratch = ML::Video::Arena.new
        @control_scratch << scratch
        ML::Video::HyNomasDualStreamBlock.prepare_scratch!(scratch, CONFIG)
      end
      @control_blocks << ML::Video::HyNomasDualStreamBlock.new(CONFIG, @weight_arenas[0], @control_scratch[0])
      @control_blocks << ML::Video::HyNomasDualStreamBlock.new(CONFIG, @weight_arenas[1], @control_scratch[1])
      @control_workspaces << ML::Video::HyNomasMPSWorkspace.new(@control_blocks[0],
        input["input.img"], input["input.txt"], input["input.vec"], input["input.vec_txt"],
        input["input.rope_cos"], input["input.rope_sin"])
      @control_workspaces << ML::Video::HyNomasMPSWorkspace.new(@control_blocks[1],
        @control_blocks[0].image_output, @control_blocks[0].text_output,
        input["input.vec"], input["input.vec_txt"], input["input.rope_cos"], input["input.rope_sin"])
      validate_runtime_capture_inventory!(@control_blocks[0], @control_blocks[1])
      actual_scratch = @control_scratch.sum(&.resident_bytes)
      actual_workspace = @control_workspaces.sum(&.resident_bytes)
      unless actual_scratch == @control_scratch_bytes && actual_workspace == @control_workspace_bytes
        raise ArgumentError.new("two-plan named-byte estimate differs from allocated plans")
      end
    end

    private def validate_runtime_capture_inventory!(block0 : ML::Video::HyNomasDualStreamBlock,
                                                    block1 : ML::Video::HyNomasDualStreamBlock) : Nil
      expected = CAPTURE_NAMES.to_set
      unless block0.checkpoints.keys.to_set == expected && block1.checkpoints.keys.to_set == expected
        raise ArgumentError.new("native block checkpoint map differs from authenticated 40-capture inventory")
      end
      {block0, block1}.each_with_index do |block, index|
        CAPTURE_NAMES.each do |name|
          key = "block#{index}.#{name}"
          buffer = block.checkpoints[name]
          unless buffer.size == @reference.tensor_bytes(key) && @reference.tensor_shape(key) == HyNomasPretrainedSerialProbe.expected_capture_shape(name)
            raise ArgumentError.new("native checkpoint shape/extent differs from oracle: #{key}")
          end
        end
      end
    end

    private def poison_shared_scratch!(chain : ML::Metal::MPSCommandChain) : Int32
      serial = @serial_workspace || raise ArgumentError.new("shared workspace is unavailable")
      scratch_plan = ML::Video::HyNomasDualStreamBlock.scratch_plan(CONFIG)
      dispatches = 0_i32
      buffers = [] of ML::MetalBuffer
      serial.blocks.each do |block|
        scratch, _weights = block.mps_workspace_buffers
        raise ArgumentError.new("native scratch map differs from poison inventory") unless scratch.size == scratch_plan.size
        scratch.each_with_index do |buffer, index|
          name, count = scratch_plan[index]
          next if name == "hy_nomas.zero_head_bias"
          raise ArgumentError.new("scratch poison extent mismatch: #{name}") unless buffer.size == count.to_i64 * 4
          next if buffers.any? { |other| other.same?(buffer) }
          encoder = chain.encoder
          encoder.set_pipeline(@poison_pipeline || raise ArgumentError.new("poison pipeline is unavailable"))
            .set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
            .set_value(count.to_u32, 1)
            .dispatch_1d(count)
          buffers << buffer
          dispatches += 1
        end
      end
      dispatches
    end

    private def compare_to_reference(name : String, values : Array(Float32)) : TensorResult
      expected = @reference.values(name)
      raise ArgumentError.new("comparison extent mismatch: #{name}") unless values.size == expected.size
      result = HyNomasTextProbe.compare_values(values, expected)
      {
        name: name, elements: values.size.to_i32, failures: result[:failures],
        max_abs: result[:max_abs], raw_bit_differences: result[:raw_bit_differences],
      }
    end

    private def raw_bit_differences(left : Array(Float32), right : Array(Float32)) : Int32
      raise ArgumentError.new("exact snapshot extent mismatch") unless left.size == right.size
      differences = 0_i32
      left.each_with_index do |value, index|
        differences += 1 if value.unsafe_as(UInt32) != right[index].unsafe_as(UInt32)
      end
      differences
    end

    private def ensure_preterminal_counters!(chain : ML::Metal::MPSCommandChain,
                                             allocations : Int64, readbacks : Int64) : Nil
      raise "GPU encoding allocated named Metal buffers" unless allocations == 0
      raise "GPU encoding performed activation readbacks before terminal wait" unless readbacks == 0
      raise "GPU encoding submitted an early terminal wait" unless chain.wait_count == 0
    end

    private def input(name : String) : ML::MetalBuffer
      arena = @input_arena || raise ArgumentError.new("probe input arena is unavailable")
      arena[name]
    end

    private def begin_request!(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("previous request has not reached terminal completion") if @active_chain
      @active_chain = chain
      @chain_lease.begin_encode!(chain)
    end

    private def finish_chain_after_terminal!(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("terminal chain differs from active owner chain") unless @active_chain && @active_chain.not_nil!.same?(chain)
      @chain_lease.ensure_terminal!
      raise ArgumentError.new("cannot release incomplete command chain") unless chain.completed?
      chain.release
      @active_chain = nil
    end

    private def ensure_no_pending_chain! : Nil
      if chain = @active_chain
        raise ArgumentError.new("probe owner retains resources until its terminal wait") unless chain.completed?
        finish_chain_after_terminal!(chain)
      end
    end

    private def release_setup_before_submission! : Nil
      return if @released
      return if @active_chain && !@active_chain.not_nil!.completed?
      ensure_no_pending_chain!
      @serial_workspace.try(&.release)
      @serial_workspace = nil
      @snapshot_arena.try(&.release)
      @snapshot_arena = nil
      @control_workspaces.reverse_each(&.release)
      @control_workspaces.clear
      @control_blocks.clear
      @control_scratch.reverse_each(&.release)
      @control_scratch.clear
      @prepared_weights.each(&.release)
      @prepared_weights.clear
      @checkpoint_files.each(&.close)
      @checkpoint_files.clear
      @weight_plans.clear
      @input_arena.try(&.release)
      @input_arena = nil
      @weight_arenas.reverse_each(&.release)
      @weight_arenas.clear
      @released = true
      release_lifetime_root!
    end

    private def retain_lifetime_root! : Nil
      @@lifetime_mutex.synchronize do
        unless @@live_owners.any? { |owner| owner.same?(self) }
          @@live_owners << self
        end
      end
      @rooted = true
    end

    private def release_lifetime_root! : Nil
      @@lifetime_mutex.synchronize do
        @@live_owners.reject! { |owner| owner.same?(self) }
      end
      @rooted = false
    end
  end

  def run(reference_path : String, reference_sha256 : String,
          conditioning_path : String, conditioning_sha256 : String,
          report_path : String, max_named_bytes : Int64,
          successor_block0_control : Bool = false) : Bool
    owner = nil.as(RunOwner?)
    raise ArgumentError.new("refusing to replace probe report") if File.exists?(report_path) || File.symlink?(report_path)
    parent = File.dirname(File.expand_path(report_path))
    raise ArgumentError.new("report parent directory does not exist") unless Dir.exists?(parent)
    reference = ReferenceFixture.new(reference_path, reference_sha256)
    unless conditioning_sha256 == PINNED_CONDITIONING_SHA
      raise ArgumentError.new("conditioning fixture digest argument is not the pinned saved fixture")
    end
    qwen_sha = conditioning_qwen_sha256(conditioning_path, conditioning_sha256)
    conditioning = HyNomasTextProbe::Fixture.new(conditioning_path, conditioning_sha256, qwen_sha)
    reference.validate_conditioning_inputs!(conditioning)
    reference.authenticate_weight_artifacts!(conditioning)
    owner = RunOwner.new(reference, conditioning, max_named_bytes)
    active_owner = owner.not_nil!
    mode = successor_block0_control ? "successor_block0_negative_control" : "shared_workspace_repeat"
    control = active_owner.run_control!
    active_owner.release_control_phase!
    active_owner.prepare_serial!(successor_block0_control: successor_block0_control)
    if successor_block0_control
      active_owner.run_shared_request!(1, capture_first_repeat: false, negative_mode: true)
    else
      active_owner.run_shared_request!(1, capture_first_repeat: true, negative_mode: false)
      active_owner.run_shared_request!(2, capture_first_repeat: false, negative_mode: false)
    end
    control_passed = active_owner.control_passed?
    shared_passed = active_owner.shared_passed?(successor_block0_control)
    passed = control_passed && shared_passed
    active_owner.cleanup_after_terminal!
    write_report(report_path, reference, conditioning, mode, active_owner, control_passed, shared_passed, passed)
    puts passed ? "PASS: #{mode}" : "FAIL: #{mode}; see #{report_path}"
    passed
  rescue ex
    if active_owner = owner
      active_owner.drain_active_chain_after_error!
      active_owner.cleanup_after_terminal!
    end
    raise ex
  end

  private def write_report(path : String, reference : ReferenceFixture,
                           conditioning : HyNomasTextProbe::Fixture,
                           mode : String, owner : RunOwner,
                           control_passed : Bool, shared_passed : Bool,
                           passed : Bool) : Nil
    report = {
      schema:                        REPORT_SCHEMA,
      schema_version:                1,
      mode:                          mode,
      reference:                     reference.path,
      reference_sha256:              reference.sha256,
      conditioning_reference:        conditioning.path,
      conditioning_reference_sha256: conditioning.sha256,
      input_scope:                   "authenticated saved text-refiner output and seeded hidden video state; no live Qwen/refiner rerun",
      precision_scope:               "two real pretrained blocks; BF16-resident weights converted from pinned F32 subsets; F32 MPS/attention",
      tolerance:                     {atol: ATOL, rtol: RTOL},
      resource_budget:               owner.named_bytes_report,
      owner_lifecycle:               {
        root_registered_before_first_dispatch:                   owner.request_metrics.first?.try(&.[:owner_rooted_before_dispatch]) == true,
        lease_registered_before_each_request_dispatch:           !owner.request_metrics.empty? && owner.request_metrics.all?(&.[:chain_lease_started_before_dispatch]),
        prepared_weight_staging_released_after_control_terminal: owner.control_phase_released?,
        owner_retained_through_each_terminal:                    owner.request_metrics.all?(&.[:owner_retained_through_terminal]),
        final_owner_cleanup_completed:                           owner.released?,
        lifetime_root_released_after_cleanup:                    !owner.rooted?,
        incomplete_terminal_drain_policy:                        "retain rooted owner and every borrowed resource; no release callback runs",
      },
      control_comparisons:       owner.control_results,
      shared_comparisons:        owner.shared_results,
      request_counters:          owner.request_metrics,
      control_passed:            control_passed,
      shared_passed:             shared_passed,
      negative_control_detected: mode == "successor_block0_negative_control" && shared_passed,
      passed:                    passed,
    }
    File.write(path, report.to_json + "\n")
  end
end
