require "digest/sha256"
require "json"
require "set"
require "../../src/ml/metal/device"
require "../../src/ml/core/buffer"
require "../../src/ml/video/hy_nomas_block"
require "../../src/ml/video/hy_nomas_attention_mps"
require "../../src/ml/video/hy_nomas_vector_attention"
require "../../spec/support/hy_nomas_mps_reductions"

# Count the wrapper's named MetalBuffer allocations and read APIs around the
# encode region. Shared-storage capture uses contents only after the terminal
# wait and is intentionally outside these counters.
{% unless flag?(:cpu_only) %}
  class ML::MetalBuffer
    @@hy_nomas_refiner_attention_probe_allocations = 0_i64
    @@hy_nomas_refiner_attention_probe_readbacks = 0_i64

    def self.hy_nomas_refiner_attention_probe_allocations : Int64
      @@hy_nomas_refiner_attention_probe_allocations
    end

    def self.hy_nomas_refiner_attention_probe_readbacks : Int64
      @@hy_nomas_refiner_attention_probe_readbacks
    end

    private def track_alloc(bytes : Int64) : Nil
      @@hy_nomas_refiner_attention_probe_allocations += 1
      previous_def(bytes)
    end

    def read(count : Int32) : Array(Float32)
      @@hy_nomas_refiner_attention_probe_readbacks += 1
      previous_def(count)
    end

    def read_to_slice(dest : Slice(Float32)) : Nil
      @@hy_nomas_refiner_attention_probe_readbacks += 1
      previous_def(dest)
    end

    def read_bytes(ptr : Pointer(UInt8), byte_size : Int) : Nil
      @@hy_nomas_refiner_attention_probe_readbacks += 1
      previous_def(ptr, byte_size)
    end
  end
{% end %}

module HyNomasRefinerAttentionProbe
  extend self

  SCHEMA          = "cogni_ml.hy_nomas_refiner_attention_replay/v1"
  CAPTURE_SCHEMA  = "cogni_ml.hy_nomas_refiner_attention_probe_capture/v1"
  MAX_NAMED_BYTES = 512_i64 * 1024 * 1024
  BATCHES         =   1
  QUERY_ROWS      = 263
  KEY_ROWS        = 263
  HEADS           =  16
  HEAD_DIM        = 128
  Q_SHAPE         = [BATCHES, QUERY_ROWS, HEADS, HEAD_DIM]
  SCORE_SHAPE     = [BATCHES, QUERY_ROWS, HEADS, KEY_ROWS]
  Q_ELEMENTS      = BATCHES.to_i64 * QUERY_ROWS * HEADS * HEAD_DIM
  SCORE_ELEMENTS  = BATCHES.to_i64 * QUERY_ROWS * HEADS * KEY_ROWS
  Q_BYTES         = Q_ELEMENTS * sizeof(Float32)
  SCORE_BYTES     = SCORE_ELEMENTS * sizeof(Float32)
  TAIL_ROWS       = 7
  TAIL_SHAPE      = [BATCHES, TAIL_ROWS, HEADS, HEAD_DIM]
  TAIL_ELEMENTS   = BATCHES.to_i64 * TAIL_ROWS * HEADS * HEAD_DIM
  TAIL_BYTES      = TAIL_ELEMENTS * sizeof(Float32)
  INPUT_NAMES     = ["q", "k", "v", "source_probabilities"]
  CAPTURE_ORDER   = [
    "input.q", "input.k", "input.v", "input.source_probabilities",
    "qk_raw", "qk_scaled", "softmax", "pv", "canonical_attention",
    "pv_source_probabilities",
  ]
  BF16_CAPTURE_ORDER = CAPTURE_ORDER + [
    "bf16_boundary_attention", "bf16_boundary_attention_with_probabilities",
    "bf16_boundary_probabilities",
  ]
  VECTOR_CAPTURE_ORDER = BF16_CAPTURE_ORDER + ["vector_tail_attention"]
  SOURCE_FILES         = [
    "examples/hy_nomas_refiner_attention_probe.cr",
    "examples/support/hy_nomas_refiner_attention_probe.cr",
    "spec/support/hy_nomas_mps_reductions.cr",
    "src/ml/video/hy_nomas_mps_reductions_bridge.mm",
    "src/ml/video/hy_nomas_attention_mps.cr",
    "src/ml/video/hy_nomas_vector_attention.cr",
    "src/ml/metal/kernels/hy_nomas/hy_nomas_vector_attention.metal",
    "src/ml/video/metal_mps_command_chain.cr",
    "src/ml/video/metal_ops.cr",
    "src/ml/core/buffer.cr",
    "src/ml/metal/device.cr",
    "src/ml/video/hy_nomas_block.cr",
  ]
  COMPILED_SOURCE = {
    "examples/hy_nomas_refiner_attention_probe.cr"         => {{ read_file("#{__DIR__}/../hy_nomas_refiner_attention_probe.cr") }},
    "examples/support/hy_nomas_refiner_attention_probe.cr" => {{ read_file("#{__DIR__}/hy_nomas_refiner_attention_probe.cr") }},
    "spec/support/hy_nomas_mps_reductions.cr"              => {{ read_file("#{__DIR__}/../../spec/support/hy_nomas_mps_reductions.cr") }},
    "src/ml/video/hy_nomas_mps_reductions_bridge.mm"       => {{ read_file("#{__DIR__}/../../src/ml/video/hy_nomas_mps_reductions_bridge.mm") }},
    "src/ml/video/hy_nomas_attention_mps.cr"               => {{ read_file("#{__DIR__}/../../src/ml/video/hy_nomas_attention_mps.cr") }},
    "src/ml/video/hy_nomas_vector_attention.cr"            => {{ read_file("#{__DIR__}/../../src/ml/video/hy_nomas_vector_attention.cr") }},
    "src/ml/metal/kernels/hy_nomas/hy_nomas_vector_attention.metal" => {{ read_file("#{__DIR__}/../../src/ml/metal/kernels/hy_nomas/hy_nomas_vector_attention.metal") }},
    "src/ml/video/metal_mps_command_chain.cr"              => {{ read_file("#{__DIR__}/../../src/ml/video/metal_mps_command_chain.cr") }},
    "src/ml/video/metal_ops.cr"                            => {{ read_file("#{__DIR__}/../../src/ml/video/metal_ops.cr") }},
    "src/ml/core/buffer.cr"                                => {{ read_file("#{__DIR__}/../../src/ml/core/buffer.cr") }},
    "src/ml/metal/device.cr"                               => {{ read_file("#{__DIR__}/../../src/ml/metal/device.cr") }},
    "src/ml/video/hy_nomas_block.cr"                       => {{ read_file("#{__DIR__}/../../src/ml/video/hy_nomas_block.cr") }},
  }

  record TensorRecord, dtype : String, shape : Array(Int32), offset : Int64,
    nbytes : Int64, sha256 : String

  record Packet, path : String, sha256 : String, payload_path : String,
    payload_bytes : Int64, payload_sha256 : String, tensors : Hash(String, TensorRecord),
    tensor_order : Array(String)

  def valid_sha256?(value : String) : Bool
    /\A[0-9a-f]{64}\z/.matches?(value)
  end

  def require_sha256!(value : String, label : String) : Nil
    raise ArgumentError.new("#{label} must be a lowercase SHA-256") unless valid_sha256?(value)
  end

  def shape_count(shape : Array(Int32)) : Int64
    raise ArgumentError.new("attention replay tensor shape must be nonempty") if shape.empty?
    shape.reduce(1_i64) do |count, extent|
      raise ArgumentError.new("attention replay tensor dimensions must be positive") unless extent > 0
      raise ArgumentError.new("attention replay tensor extent overflows") if count > Int64::MAX // extent
      count * extent
    end
  end

  def validate_controls!(bf16_boundaries_control : Bool, vector_tail_control : Bool) : Nil
    if vector_tail_control && !bf16_boundaries_control
      raise ArgumentError.new("Vector tail control requires the BF16 boundaries control")
    end
  end

  def validate_budget!(max_named_bytes : Int64, bf16_boundaries_control : Bool = false,
                       vector_tail_control : Bool = false) : Int64
    validate_controls!(bf16_boundaries_control, vector_tail_control)
    unless 0 < max_named_bytes <= MAX_NAMED_BYTES
      raise ArgumentError.new("named-buffer budget must be positive and no greater than 512 MiB")
    end
    required = required_named_bytes(bf16_boundaries_control, vector_tail_control)
    raise ArgumentError.new("attention probe named buffers exceed caller budget") if required > max_named_bytes
    required
  end

  def allocation_counter : Int64
    {% if flag?(:cpu_only) %}
      0_i64
    {% else %}
      ML::MetalBuffer.hy_nomas_refiner_attention_probe_allocations
    {% end %}
  end

  def readback_counter : Int64
    {% if flag?(:cpu_only) %}
      0_i64
    {% else %}
      ML::MetalBuffer.hy_nomas_refiner_attention_probe_readbacks
    {% end %}
  end

  # Includes the four resident inputs, six distinct outputs, and all packed
  # input/output buffers allocated by MPSReductionControls during setup.
  def required_named_bytes(bf16_boundaries_control : Bool = false,
                           vector_tail_control : Bool = false) : Int64
    validate_controls!(bf16_boundaries_control, vector_tail_control)
    input_elements = 3_i64 * Q_ELEMENTS + SCORE_ELEMENTS
    output_elements = 3_i64 * Q_ELEMENTS + 3_i64 * SCORE_ELEMENTS
    packed_elements = 12_i64 * Q_ELEMENTS + 6_i64 * SCORE_ELEMENTS
    # Two opt-in plans: context-only and instrumented P. Each owns four
    # Q-sized stages; the latter also owns one score-sized capture stage.
    # Caller outputs add two Q-sized buffers and one score-sized buffer.
    extra_elements = bf16_boundaries_control ? 10_i64 * Q_ELEMENTS + 2_i64 * SCORE_ELEMENTS : 0_i64
    tail_elements = vector_tail_control ? TAIL_ELEMENTS : 0_i64
    (input_elements + output_elements + packed_elements + extra_elements + tail_elements) * sizeof(Float32)
  end

  def validate_manifest_document!(document : JSON::Any, expected_sha256 : String,
                                  payload_bytes : Int64? = nil) : Packet
    require_sha256!(expected_sha256, "trusted replay manifest digest")
    fields = document.as_h
    raise ArgumentError.new("unsupported native attention replay schema") unless fields["schema"]?.try(&.as_s?) == SCHEMA && fields["schema_version"]?.try(&.as_i?) == 1

    geometry = fields["geometry"]?.try(&.as_h?) || raise ArgumentError.new("missing replay geometry")
    expected_geometry = {
      "batches" => BATCHES, "query_rows" => QUERY_ROWS, "key_rows" => KEY_ROWS,
      "heads" => HEADS, "head_dim" => HEAD_DIM,
    }
    valid_geometry = geometry.keys.sort == expected_geometry.keys.sort &&
                     expected_geometry.all? { |name, expected| geometry[name]?.try(&.as_i?) == expected }
    unless valid_geometry
      raise ArgumentError.new("native attention replay geometry mismatch")
    end
    raise ArgumentError.new("Q/K/V source must be explicitly source-exact") unless fields["source_exact_qkv"]?.try(&.as_bool?) == true

    artifact = fields["tensor_artifact"]?.try(&.as_h?) || raise ArgumentError.new("missing replay tensor artifact")
    basename = artifact["basename"]?.try(&.as_s?) || raise ArgumentError.new("missing replay tensor basename")
    unless basename.matches?(/\A[A-Za-z0-9_.-]+\.tensors\.bin\z/) && !basename.starts_with?(".")
      raise ArgumentError.new("replay tensor artifact basename is unsafe")
    end
    artifact_bytes = artifact["bytes"]?.try(&.as_i64?) || raise ArgumentError.new("missing replay tensor byte count")
    artifact_sha256 = artifact["sha256"]?.try(&.as_s?) || raise ArgumentError.new("missing replay tensor digest")
    require_sha256!(artifact_sha256, "replay payload digest")
    if payload_bytes && artifact_bytes != payload_bytes
      raise ArgumentError.new("replay tensor artifact byte count mismatch")
    end
    unless 0 < artifact_bytes <= MAX_NAMED_BYTES
      raise ArgumentError.new("replay tensor artifact must be positive and no larger than 512 MiB")
    end

    order_any = fields["tensor_order"]?.try(&.as_a?) || raise ArgumentError.new("missing replay tensor order")
    tensor_order = order_any.map do |value|
      value.as_s? || raise ArgumentError.new("replay tensor order entries must be strings")
    end
    tensors = fields["tensors"]?.try(&.as_h?) || raise ArgumentError.new("missing replay tensor records")
    unless tensor_order.size == tensors.size && tensor_order.uniq == tensor_order &&
           tensor_order.to_set == tensors.keys.to_set
      raise ArgumentError.new("replay tensor order must cover every record exactly once")
    end
    INPUT_NAMES.each do |name|
      raise ArgumentError.new("missing required replay tensor #{name}") unless tensors.has_key?(name)
    end

    records = {} of String => TensorRecord
    offset = 0_i64
    tensor_order.each do |name|
      value = tensors[name].as_h
      dtype = value["dtype"]?.try(&.as_s?) || raise ArgumentError.new("missing tensor dtype: #{name}")
      raise ArgumentError.new("attention replay tensors must be F32: #{name}") unless dtype == "F32"
      shape_any = value["shape"]?.try(&.as_a?) || raise ArgumentError.new("missing tensor shape: #{name}")
      shape = shape_any.map do |extent_any|
        extent = extent_any.as_i? || raise ArgumentError.new("tensor shape entries must be integers: #{name}")
        raise ArgumentError.new("tensor shape extent is outside Int32: #{name}") unless 0 < extent <= Int32::MAX
        extent.to_i32
      end
      count = shape_count(shape)
      raise ArgumentError.new("tensor byte extent overflows: #{name}") if count > Int64::MAX // sizeof(Float32)
      nbytes = value["nbytes"]?.try(&.as_i64?) || raise ArgumentError.new("missing tensor nbytes: #{name}")
      record_offset = value["offset"]?.try(&.as_i64?) || raise ArgumentError.new("missing tensor offset: #{name}")
      sha256 = value["sha256"]?.try(&.as_s?) || raise ArgumentError.new("missing tensor digest: #{name}")
      require_sha256!(sha256, "tensor digest #{name}")
      expected_bytes = count * sizeof(Float32)
      unless record_offset == offset && nbytes == expected_bytes && nbytes <= artifact_bytes - offset
        raise ArgumentError.new("attention replay tensor layout is not contiguous or exact: #{name}")
      end
      records[name] = TensorRecord.new(dtype, shape, record_offset, nbytes, sha256)
      offset += nbytes
    end
    raise ArgumentError.new("replay tensor records do not cover the declared payload") unless offset == artifact_bytes

    {"q", "k", "v"}.each do |name|
      unless records[name].shape == Q_SHAPE
        raise ArgumentError.new("native attention replay #{name} must use BQHD [1,263,16,128]")
      end
    end
    unless records["source_probabilities"].shape == SCORE_SHAPE
      raise ArgumentError.new("native attention source probabilities must use BQHK [1,263,16,263]")
    end

    Packet.new("", expected_sha256, basename, artifact_bytes, artifact_sha256,
      records, tensor_order)
  end

  def checked_input_path!(manifest_path : String) : String
    unless Path[manifest_path].absolute? && File.file?(manifest_path) && !File.symlink?(manifest_path) &&
           File.realpath(manifest_path) == manifest_path && File.size(manifest_path) <= 4 * 1024 * 1024
      raise ArgumentError.new("replay manifest must be a canonical absolute regular file no larger than 4 MiB")
    end
    manifest_path
  end

  def checked_output_paths!(manifest_path : String) : Tuple(String, String)
    unless Path[manifest_path].absolute? && manifest_path.ends_with?(".json")
      raise ArgumentError.new("attention probe output must be an absolute .json path")
    end
    parent = File.realpath(File.dirname(manifest_path))
    canonical = File.join(parent, File.basename(manifest_path))
    {canonical, canonical.sub(/\.json\z/, ".tensors.bin")}
  end

  def ensure_fresh_outputs!(manifest_path : String, payload_path : String) : Nil
    {manifest_path, payload_path}.each do |path|
      raise ArgumentError.new("refusing to replace attention probe output: #{path}") if File.exists?(path) || File.symlink?(path)
    end
  end

  def open_fresh_output!(path : String) : IO::FileDescriptor
    fd = LibC.open(path, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
    raise IO::Error.new("attention probe output must be fresh and writable: #{path}") if fd < 0
    IO::FileDescriptor.new(fd)
  end

  def sha_file(path : String) : String
    digest = Digest::SHA256.new
    chunk = Bytes.new(1024 * 1024)
    File.open(path) do |io|
      while (count = io.read(chunk)) > 0
        digest.update(chunk[0, count])
      end
    end
    digest.hexfinal
  end

  class ReplayInputs
    getter packet : Packet
    @input_bytes = {} of String => Bytes

    def initialize(manifest_path : String, expected_sha256 : String)
      canonical = HyNomasRefinerAttentionProbe.checked_input_path!(manifest_path)
      raw = File.read(canonical)
      actual_sha256 = Digest::SHA256.hexdigest(raw)
      unless HyNomasRefinerAttentionProbe.valid_sha256?(expected_sha256) && actual_sha256 == expected_sha256
        raise ArgumentError.new("attention replay manifest digest mismatch")
      end
      document = JSON.parse(raw)
      artifact = document["tensor_artifact"]?.try(&.as_h?) ||
                 raise ArgumentError.new("missing replay tensor artifact")
      basename = artifact["basename"]?.try(&.as_s?) ||
                 raise ArgumentError.new("missing replay tensor artifact basename")
      unless basename.matches?(/\A[A-Za-z0-9_.-]+\.tensors\.bin\z/) && !basename.starts_with?(".")
        raise ArgumentError.new("replay tensor artifact basename is unsafe")
      end
      payload_bytes = File.join(File.dirname(canonical), basename)
      unless Path[payload_bytes].absolute? && File.file?(payload_bytes) && !File.symlink?(payload_bytes) &&
             File.realpath(payload_bytes) == payload_bytes
        raise ArgumentError.new("replay tensor artifact must be an adjacent canonical regular file")
      end
      @packet = HyNomasRefinerAttentionProbe.validate_manifest_document!(document,
        actual_sha256, File.size(payload_bytes))
      @packet = Packet.new(canonical, @packet.sha256, payload_bytes, @packet.payload_bytes,
        @packet.payload_sha256, @packet.tensors, @packet.tensor_order)
      verify_payload!
      load_required_inputs!
    end

    # Verify the whole artifact and every ordered range in one bounded-memory
    # pass, before Metal is initialized or any buffer is allocated.
    def verify_payload! : Nil
      payload_digest = Digest::SHA256.new
      scratch = Bytes.new(1024 * 1024)
      File.open(@packet.payload_path) do |io|
        @packet.tensor_order.each do |name|
          record = @packet.tensors[name]
          io.seek(record.offset)
          record_digest = Digest::SHA256.new
          remaining = record.nbytes
          while remaining > 0
            count = Math.min(remaining, scratch.size.to_i64).to_i32
            chunk = scratch[0, count]
            io.read_fully(chunk)
            payload_digest.update(chunk)
            record_digest.update(chunk)
            remaining -= count
          end
          unless record_digest.hexfinal == record.sha256
            raise ArgumentError.new("replay tensor digest mismatch: #{name}")
          end
        end
      end
      unless payload_digest.hexfinal == @packet.payload_sha256
        raise ArgumentError.new("replay tensor artifact digest mismatch")
      end
    end

    # Keep only the four consumed inputs resident on CPU after verification.
    # Upload then uses these exact validated bytes, avoiding a second
    # unverified read of a payload file that could change during setup.
    private def load_required_inputs! : Nil
      @packet.tensors.select { |name, _| INPUT_NAMES.includes?(name) }.each do |name, record|
        bytes = Bytes.new(record.nbytes.to_i32)
        File.open(@packet.payload_path) do |io|
          io.seek(record.offset)
          io.read_fully(bytes)
        end
        unless Digest::SHA256.hexdigest(bytes) == record.sha256
          raise ArgumentError.new("replay input changed after payload verification: #{name}")
        end
        @input_bytes[name] = bytes
      end
      unless @input_bytes.keys.to_set == INPUT_NAMES.to_set
        raise ArgumentError.new("verified replay input inventory is incomplete")
      end
    end

    def upload!(arena : ML::Video::Arena) : Hash(String, ML::MetalBuffer)
      buffers = {} of String => ML::MetalBuffer
      {"q", "k", "v", "source_probabilities"}.each do |name|
        record = @packet.tensors[name]
        key = "input.#{name}"
        buffer = arena.alloc(key, (record.nbytes // sizeof(Float32)).to_i32)
        bytes = @input_bytes[name]
        buffer.write_bytes(bytes.to_unsafe, bytes.size)
        buffers[name] = buffer
      end
      buffers
    end
  end

  # MPSReductionControls relies on this shared attention scale validator.
  # Mode 0 accepts 1.0 (raw scores) or canonical 1/sqrt(D); mode 3 is canonical.
  def scale(op : JSON::Any, config : ML::Video::HyNomasConfig) : Float32
    canonical = 1.0_f32 / Math.sqrt(config.head_dim.to_f32)
    value = op["scale"]?.try(&.as_f) || canonical.to_f64
    unless value.finite? && value > 0.0 && value <= 1.0 &&
           (value.to_f32 == canonical || value == 1.0)
      raise ArgumentError.new("Unsupported native attention scale")
    end
    value.to_f32
  end

  def operation_documents : Array(JSON::Any)
    specs = [
      {name: "qk_raw", kind: "qk", inputs: {"q" => Q_SHAPE, "k" => Q_SHAPE}, expected: SCORE_SHAPE, scale: 1.0_f64},
      {name: "qk_scaled", kind: "qk", inputs: {"q" => Q_SHAPE, "k" => Q_SHAPE}, expected: SCORE_SHAPE, scale: 1.0_f64 / Math.sqrt(HEAD_DIM)},
      {name: "softmax", kind: "softmax", inputs: {"scores" => SCORE_SHAPE}, expected: SCORE_SHAPE, scale: 1.0_f64},
      {name: "pv", kind: "pv", inputs: {"probabilities" => SCORE_SHAPE, "v" => Q_SHAPE}, expected: Q_SHAPE, scale: 1.0_f64},
      {name: "canonical_attention", kind: "attention", inputs: {"q" => Q_SHAPE, "k" => Q_SHAPE, "v" => Q_SHAPE}, expected: Q_SHAPE, scale: 1.0_f64 / Math.sqrt(HEAD_DIM)},
      {name: "pv_source_probabilities", kind: "pv", inputs: {"probabilities" => SCORE_SHAPE, "v" => Q_SHAPE}, expected: Q_SHAPE, scale: 1.0_f64},
    ]
    specs.map do |spec|
      JSON.parse(JSON.build do |json|
        json.object do
          json.field "name", spec[:name]
          json.field "kind", spec[:kind]
          json.field "scale", spec[:scale]
          json.field "inputs" do
            json.object do
              spec[:inputs].each do |name, shape|
                json.field name do
                  json.object { json.field "shape", shape }
                end
              end
            end
          end
          json.field "expected" do
            json.object { json.field "shape", spec[:expected] }
          end
        end
      end)
    end
  end

  def bind_inputs_and_allocate_outputs!(arena : ML::Video::Arena,
                                        input_buffers : Hash(String, ML::MetalBuffer)) : Nil
    q_shape = Q_ELEMENTS.to_i32
    score_shape = SCORE_ELEMENTS.to_i32
    [score_shape, score_shape, score_shape, q_shape, q_shape, q_shape].each_with_index do |count, index|
      arena.alloc("#{index}.output", count)
    end
    ["q", "k", "v"].each do |name|
      {0, 1, 4}.each do |index|
        arena.alias_f32("#{index}.input.#{name}", "input.#{name}")
      end
    end
    # Complete the operation-specific aliases. Existing aliases above cover
    # Q/K/V for QK and canonical plans; the loop is deliberately explicit so
    # the two PV branches consume their own admitted probability tensors.
    arena.alias_f32("2.input.scores", "1.output")
    arena.alias_f32("3.input.probabilities", "2.output")
    arena.alias_f32("3.input.v", "input.v")
    arena.alias_f32("5.input.probabilities", "input.source_probabilities")
    arena.alias_f32("5.input.v", "input.v")
    raise "input binding cardinality changed" unless input_buffers.size == 4
  end

  def source_pins : Hash(String, String)
    SOURCE_FILES.to_h do |relative|
      {relative, Digest::SHA256.hexdigest(COMPILED_SOURCE[relative])}
    end
  end

  def capture_sources(arena : ML::Video::Arena, bf16_boundaries_control : Bool = false,
                      vector_tail_control : Bool = false) : Array(Tuple(String, ML::MetalBuffer, Array(Int32)))
    validate_controls!(bf16_boundaries_control, vector_tail_control)
    sources = [
      {"input.q", arena["input.q"], Q_SHAPE},
      {"input.k", arena["input.k"], Q_SHAPE},
      {"input.v", arena["input.v"], Q_SHAPE},
      {"input.source_probabilities", arena["input.source_probabilities"], SCORE_SHAPE},
      {"qk_raw", arena["0.output"], SCORE_SHAPE},
      {"qk_scaled", arena["1.output"], SCORE_SHAPE},
      {"softmax", arena["2.output"], SCORE_SHAPE},
      {"pv", arena["3.output"], Q_SHAPE},
      {"canonical_attention", arena["4.output"], Q_SHAPE},
      {"pv_source_probabilities", arena["5.output"], Q_SHAPE},
    ]
    if bf16_boundaries_control
      sources << {"bf16_boundary_attention", arena["bf16.output"], Q_SHAPE}
      sources << {"bf16_boundary_attention_with_probabilities", arena["bf16.instrumented_output"], Q_SHAPE}
      sources << {"bf16_boundary_probabilities", arena["bf16.probabilities"], SCORE_SHAPE}
    end
    sources << {"vector_tail_attention", arena["vector.output"], TAIL_SHAPE} if vector_tail_control
    sources
  end

  # Called only after the terminal wait. The artifact stores input and actual
  # native MPS stage bytes; no CPU neural arithmetic or metric calculation is
  # performed here.
  def write_capture!(manifest_path : String, payload_path : String, packet : Packet,
                     sources : Array(Tuple(String, ML::MetalBuffer, Array(Int32))),
                     setup_named_bytes : Int64, max_named_bytes : Int64,
                     encode_allocations : Int64, encode_readbacks : Int64,
                     preterminal_waits : Int32, terminal_waits : Int32,
                     bf16_boundaries_control : Bool = false,
                     vector_tail_control : Bool = false) : Nil
    validate_controls!(bf16_boundaries_control, vector_tail_control)
    capture_order = vector_tail_control ? VECTOR_CAPTURE_ORDER : (bf16_boundaries_control ? BF16_CAPTURE_ORDER : CAPTURE_ORDER)
    unless sources.map(&.[0]) == capture_order && encode_allocations == 0 &&
           encode_readbacks == 0 && preterminal_waits == 0 && terminal_waits == 1
      raise ArgumentError.new("native attention capture requires ordered stages and allocation/readback-free single-wait execution")
    end
    records = {} of String => NamedTuple(dtype: String, shape: Array(Int32), offset: Int64,
      nbytes: Int64, sha256: String)
    payload_digest = Digest::SHA256.new
    offset = 0_i64
    payload = open_fresh_output!(payload_path)
    begin
      sources.each do |name, buffer, shape|
        expected_bytes = shape_count(shape) * sizeof(Float32)
        unless buffer.valid? && buffer.storage_mode.shared? && buffer.size == expected_bytes
          raise ArgumentError.new("native attention capture buffer extent/storage mismatch: #{name}")
        end
        bytes = Bytes.new(buffer.contents.as(UInt8*), expected_bytes.to_i32)
        digest = Digest::SHA256.hexdigest(bytes)
        payload.write(bytes)
        payload_digest.update(bytes)
        records[name] = {dtype: "F32", shape: shape, offset: offset,
                         nbytes: expected_bytes, sha256: digest}
        offset += expected_bytes
      end
    ensure
      payload.close
    end
    payload_sha256 = payload_digest.hexfinal
    report = open_fresh_output!(manifest_path)
    begin
      JSON.build(report, indent: "  ") do |json|
        json.object do
          json.field "schema", CAPTURE_SCHEMA
          json.field "schema_version", 1
          json.field "input_manifest_sha256", packet.sha256
          json.field "input_payload_sha256", packet.payload_sha256
          json.field "geometry", {batches: BATCHES, query_rows: QUERY_ROWS,
                                  key_rows: KEY_ROWS, heads: HEADS, head_dim: HEAD_DIM}
          json.field "source_exact_qkv", true
          json.field "source_probabilities_scope",
            "caller-supplied frozen probabilities; pv_source_probabilities records P*V only and does not assert backend-internal probability identity or native vector-fast tail coverage"
          json.field "scope", "single block-0 Q/K/V attention diagnostic; no full generation"
          json.field "device", "mps"
          json.field "producer_dtype", "F32"
          json.field "bf16_boundaries_control", bf16_boundaries_control
          json.field "vector_tail_control", vector_tail_control
          json.field "vector_tail_scope",
            "default-off one-pass BF16 vector context, Q7/K263/D128; no probabilities, refiner wiring or native context assembly"
          json.field "probabilities_capture_scope",
            "opt-in instrumented general-263 graph only; context-only output is captured separately; source tail7 vector probabilities remain unobserved"
          json.field "setup_named_bytes", setup_named_bytes
          json.field "max_named_bytes", max_named_bytes
          json.field "encode_allocations", encode_allocations
          json.field "encode_readbacks", encode_readbacks
          json.field "preterminal_waits", preterminal_waits
          json.field "terminal_waits", terminal_waits
          json.field "compiled_source_sha256" do
            json.object { source_pins.each { |path, digest| json.field path, digest } }
          end
          json.field "tensor_order", capture_order
          json.field "tensor_artifact" do
            json.object do
              json.field "basename", File.basename(payload_path)
              json.field "bytes", offset
              json.field "sha256", payload_sha256
            end
          end
          json.field "tensors" do
            json.object do
              capture_order.each do |name|
                record = records[name]
                json.field name do
                  json.object do
                    json.field "dtype", record[:dtype]
                    json.field "shape", record[:shape]
                    json.field "offset", record[:offset]
                    json.field "nbytes", record[:nbytes]
                    json.field "sha256", record[:sha256]
                  end
                end
              end
            end
          end
        end
      end
    ensure
      report.close
    end
  end
end

module HyAttentionProbe
  # MPSReductionControls calls this module method; keep its shared scaling
  # contract separate from the legacy CPU fixture and comparison harness.
  def self.scale(op : JSON::Any, config : ML::Video::HyNomasConfig) : Float32
    HyNomasRefinerAttentionProbe.scale(op, config)
  end
end
