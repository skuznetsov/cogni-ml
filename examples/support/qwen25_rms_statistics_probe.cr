require "json"
require "digest/sha256"

# Authenticated, bounded source preparation for the standalone Qwen2.5-VL RMS
# statistics probe. This module does file validation and byte comparison only;
# all neural operations remain in the native MPSGraph plan.
module Qwen25RMSStatisticsProbe
  extend self

  STAGE_SCHEMA                        = "cogni_ml.qwen25_vl_pretrained_decoder_block_stage_reference"
  RMS_SCHEMA                          = "cogni_ml.qwen25_vl_pretrained_rms_reduction_reference"
  MODEL_ID                            = "Qwen/Qwen2.5-VL-7B-Instruct"
  REVISION                            = "eb6ba55b40ada64977147b52e50a4145ee89ac4f"
  RMS_SOURCE_SHA256                   = "08742a7c82a8713bc22f632f01c97b7eca3ef0e947180e7d6c9beec4897c4ff8"
  TORCH_SCALAR_LOWERING_SOURCE_SHA256 = "1a423fb13f2c91c84d4a721c649ca958066047fc48092af6316bc325d0279119"
  TORCH_SOURCE_COMMIT                 = "a1cb3cc05d46d198467bebbb6e8fba50a325d4e7"
  TRUSTED_STAGE_SHA256                = "d29c8a6bf457f0d848fa29c4dc734c806c54aee063db7c641032460811c331b9"
  RESIDUAL_STAGE_NAME                 = "residual_after_attention"
  NORM_WEIGHT_KEY                     = "model.layers.0.post_attention_layernorm.weight"
  RMS_PREFIX                          = "layers.0.post_attention_layernorm."
  RMS_SUFFIXES                        = %w(input_f32 square_f32 variance_f32 variance_plus_eps_f32 rsqrt_f32 normalized_f32 normalized_bf16 gained_output_bf16)
  F32_SUFFIXES                        = %w(input_f32 square_f32 variance_f32 variance_plus_eps_f32 rsqrt_f32 normalized_f32)
  BF16_SUFFIXES                       = %w(normalized_bf16 gained_output_bf16)
  MAX_SOURCE_SIDECAR_BYTES            = 512_i64 * 1024 * 1024
  MAX_REFERENCE_SIDECAR_BYTES         = 256_i64 * 1024 * 1024
  MAX_NAMED_BYTES                     = 256_i64 * 1024 * 1024
  ATOL                                = 2e-5
  RTOL                                = 2e-5
  OPTIMIZATION0_CONTROL               = {
    compilation_descriptor_type:                         "MPSGraphCompilationDescriptor",
    compilation_api:                                     "MPSGraph.compileWithDevice(..., compilationDescriptor:)",
    optimization_level:                                  0,
    reduction_axis:                                      [2],
    target:                                              "variance_plus_epsilon only",
    explicit_reshape:                                    true,
    encode_route:                                        "precompiled MPSGraphExecutable",
    executable_descriptor_type:                          "MPSGraphExecutableExecutionDescriptor",
    executable_wait_until_completed:                     false,
    differs_from_torch_graph_execution_descriptor_route: true,
  }
  EXPONENT_POWER_CONTROL = {
    dtype:                                          "float32",
    shape:                                          [1],
    value:                                          2.0,
    bits:                                           "40000000",
    operation:                                      "MPSGraph.powerWithPrimaryTensor:secondaryTensor:",
    torch_source_scalar_dtype:                      "int64",
    torch_source_scalar_shape:                      [1],
    torch_cast_before_power:                        "int64 to common float32",
    matches_torch_post_cast_operand:                true,
    reproduces_torch_scalar_feed_and_cast_topology: false,
  }
  TORCH_INT64_EXPONENT_CONTROL = {
    dtype:                                          "int64",
    shape:                                          [1],
    value:                                          2_i64,
    bytes:                                          8,
    cast_before_power:                              "int64 to common float32",
    operation:                                      "MPSGraph.powerWithPrimaryTensor:secondaryTensor:",
    reproduces_torch_scalar_feed_and_cast_topology: true,
  }

  alias RawSample = NamedTuple(index: Int64, row: Int64,
    actual: String, reference: String, actual_bits: String, reference_bits: String)
  alias FloatComparison = NamedTuple(name: String, elements: Int64,
    finite_failures: Int64, raw_unequal: Int64, max_abs: Float64,
    first_raw_unequal: Array(RawSample), first_finite_failures: Array(RawSample),
    actual_sha256: String, reference_sha256: String)
  alias StageBinding = NamedTuple(name: String, dtype: String,
    shape: Array(Int64), sidecar: String, offset_bytes: Int64, nbytes: Int64)

  def checked_sha256(value : String, label : String) : String
    raise ArgumentError.new("malformed #{label} SHA256") unless value.matches?(/\A[0-9a-f]{64}\z/)
    value
  end

  def contained_sibling(root : String, name : String) : String
    unless name.matches?(/\A[A-Za-z0-9_.-]+\z/) && name != "." && name != ".."
      raise ArgumentError.new("sidecar path must be a sibling basename")
    end
    path = File.join(root, name)
    unless File.file?(path) && !File.symlink?(path) && File.realpath(path) == path
      raise ArgumentError.new("sidecar must be a regular non-symlink sibling")
    end
    path
  end

  # Hash the exact file stream from which requested byte ranges are copied.
  # Callers retain these copies and never reopen the sidecar after validation.
  def authenticate_sidecar(path : String, expected_size : Int64,
                           expected_sha256 : String,
                           captures : Hash(String, Tuple(Int64, Int64))) : Tuple(String, Hash(String, Bytes))
    raise ArgumentError.new("sidecar must be a regular non-symlink file") unless File.file?(path) && !File.symlink?(path)
    real_path = File.realpath(path)
    raise ArgumentError.new("sidecar path changed during resolution") unless real_path == path
    expected = checked_sha256(expected_sha256, "sidecar")
    raise ArgumentError.new("sidecar size mismatch before authentication") unless File.size(real_path) == expected_size

    ranges = captures.map do |name, extent|
      offset, nbytes = extent
      raise ArgumentError.new("invalid captured sidecar extent") if name.empty? || offset < 0 || nbytes <= 0 || offset + nbytes > expected_size
      {name: name, start: offset, finish: offset + nbytes, bytes: Bytes.new(nbytes.to_i)}
    end.sort_by { |range| range[:start] }
    ranges.each_cons(2) do |pair|
      raise ArgumentError.new("captured sidecar ranges overlap") if pair[0][:finish] > pair[1][:start]
    end

    digest = Digest::SHA256.new
    cursor = 0_i64
    File.open(real_path) do |io|
      chunk = Bytes.new(1024 * 1024)
      while (count = io.read(chunk)) > 0
        view = chunk[0, count]
        digest.update(view)
        chunk_start = cursor
        chunk_finish = cursor + count
        ranges.each do |range|
          copy_start = Math.max(chunk_start, range[:start])
          copy_finish = Math.min(chunk_finish, range[:finish])
          next if copy_start >= copy_finish
          source_offset = (copy_start - chunk_start).to_i
          destination_offset = (copy_start - range[:start]).to_i
          copy_bytes = (copy_finish - copy_start).to_i
          range[:bytes][destination_offset, copy_bytes].copy_from(view[source_offset, copy_bytes])
        end
        cursor = chunk_finish
      end
    end
    raise ArgumentError.new("sidecar changed size while being authenticated") unless cursor == expected_size && File.size(real_path) == expected_size
    actual = digest.hexfinal
    raise ArgumentError.new("sidecar digest mismatch") unless actual == expected
    snapshots = {} of String => Bytes
    ranges.each { |range| snapshots[range[:name]] = range[:bytes] }
    {actual, snapshots}
  rescue ex : IO::Error
    raise ArgumentError.new("could not authenticate sidecar bytes: #{ex.message}")
  end

  def copy_extent(snapshot : Bytes, offset_bytes : Int64, nbytes : Int64) : Bytes
    raise ArgumentError.new("negative sidecar extent") if offset_bytes < 0 || nbytes <= 0 || offset_bytes + nbytes > snapshot.size
    copy = Bytes.new(nbytes.to_i)
    copy.copy_from(snapshot[offset_bytes.to_i, nbytes.to_i])
    copy
  end

  def file_manifest(path : String, trusted_sha256 : String, label : String) : Tuple(String, String, JSON::Any)
    raise ArgumentError.new("#{label} manifest must be a regular non-symlink file") unless File.file?(path) && !File.symlink?(path)
    real_path = File.realpath(path)
    trusted = checked_sha256(trusted_sha256, "trusted #{label} manifest")
    bytes = File.read(real_path)
    digest = Digest::SHA256.hexdigest(bytes)
    raise ArgumentError.new("trusted #{label} manifest digest mismatch") unless digest == trusted
    {real_path, digest, JSON.parse(bytes)}
  end

  def shape(record : JSON::Any, expected : Array(Int64), label : String) : Array(Int64)
    actual = record["shape"].as_a.map(&.as_i64)
    raise ArgumentError.new("#{label} shape mismatch") unless actual == expected
    actual
  end

  def required_named_bytes(tokens : Int32, dim : Int32) : Int64
    raise ArgumentError.new("positive RMS probe geometry required") unless tokens > 0 && dim > 0
    hidden = tokens.to_i64 * dim
    raise ArgumentError.new("RMS probe geometry exceeds native element bounds") if hidden > Int32::MAX
    input_bf16 = hidden * 2
    input_f32 = hidden * 4
    # F32 input, native square, authenticated upstream square, and three
    # exponent-power panels; six original plus six power-arm scalar outputs.
    scalar_outputs = tokens.to_i64 * 4 * 12
    exponent_buffers = 12_i64 # F32 [1] and Int64 [1] values, both equal 2
    input_bf16 + input_f32 * 6 + scalar_outputs + exponent_buffers
  end

  def exponent_two_f32le : Bytes
    Bytes[0x00_u8, 0x00_u8, 0x00_u8, 0x40_u8]
  end

  def exponent_two_i64le : Bytes
    Bytes[0x02_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8]
  end

  def admit_named_bytes!(tokens : Int32, dim : Int32, budget : Int64) : Int64
    required = required_named_bytes(tokens, dim)
    unless budget > 0 && budget <= MAX_NAMED_BYTES && required <= budget
      raise ArgumentError.new("RMS probe named-buffer budget rejected: requires #{required} bytes, cap #{budget}")
    end
    required
  end

  def descriptor_path_and_hash(document : JSON::Any, key : String,
                               root : String, encoding : String,
                               max_bytes : Int64) : Tuple(String, Int64, String)
    descriptor = document["sidecars"][key]
    raise ArgumentError.new("unexpected sidecar encoding for #{key}") unless descriptor["encoding"].as_s == encoding
    nbytes = descriptor["nbytes"].as_i64
    raise ArgumentError.new("sidecar size outside bounded range for #{key}") unless 0 < nbytes <= max_bytes
    if key == "f32"
      raise ArgumentError.new("F32 sidecar byte order mismatch") unless descriptor["byte_order"].as_s == "little" && descriptor["storage_dtype"].as_s == "float32"
    end
    path = contained_sibling(root, descriptor["path"].as_s)
    raise ArgumentError.new("sidecar file size mismatch for #{key}") unless File.size(path) == nbytes
    expected_hash = checked_sha256(descriptor["sha256"].as_s, "#{key} sidecar")
    {path, nbytes, expected_hash}
  end

  def parse_extent(record : JSON::Any, name : String,
                   expected_dtype : String, expected_shape : Array(Int64),
                   expected_sidecar : String) : StageBinding
    raise ArgumentError.new("RMS stage order/name mismatch") unless record["name"].as_s == name
    raise ArgumentError.new("RMS stage dtype mismatch: #{name}") unless record["dtype"].as_s == expected_dtype
    dims = shape(record, expected_shape, name)
    storage = record["storage"]
    sidecar = storage["sidecar"].as_s
    raise ArgumentError.new("RMS stage sidecar mismatch: #{name}") unless sidecar == expected_sidecar
    offset = storage["offset_bytes"].as_i64
    nbytes = storage["nbytes"].as_i64
    element_bytes = expected_dtype == "float32" ? 4_i64 : 2_i64
    elements = dims.product
    raise ArgumentError.new("RMS stage extent length mismatch: #{name}") unless nbytes == elements * element_bytes
    raise ArgumentError.new("negative RMS stage offset: #{name}") if offset < 0
    {name: name, dtype: expected_dtype, shape: dims,
     sidecar: sidecar, offset_bytes: offset, nbytes: nbytes}
  end

  def validate_contiguous_extents!(bindings : Array(StageBinding),
                                   sidecar_size : Int64,
                                   label : String) : Nil
    cursor = 0_i64
    bindings.each do |binding|
      raise ArgumentError.new("#{label} extents overlap or leave gaps") unless binding[:offset_bytes] == cursor
      cursor += binding[:nbytes]
      raise ArgumentError.new("#{label} extent exceeds sidecar bounds") if cursor > sidecar_size
    end
    raise ArgumentError.new("#{label} extent inventory does not cover sidecar") unless cursor == sidecar_size
  end

  def float_bits(bytes : Bytes, index : Int64) : UInt32
    offset = index.to_i * 4
    bytes[offset].to_u32 | (bytes[offset + 1].to_u32 << 8) |
      (bytes[offset + 2].to_u32 << 16) | (bytes[offset + 3].to_u32 << 24)
  end

  def float_value(bytes : Bytes, index : Int64) : Float32
    float_bits(bytes, index).unsafe_as(Float32)
  end

  private def bits_hex(bits : UInt32) : String
    bits.to_s(16).rjust(8, '0')
  end

  def compare_f32(name : String, reference : Bytes, actual : Bytes,
                  row_width : Int64, sample_limit : Int32 = 8) : FloatComparison
    raise ArgumentError.new("F32 comparison requires equal nonempty word-aligned buffers") unless reference.size == actual.size && reference.size > 0 && reference.size % 4 == 0
    raise ArgumentError.new("invalid F32 comparison row width") unless row_width > 0
    elements = (reference.size // 4).to_i64
    raw_unequal = 0_i64
    finite_failures = 0_i64
    maximum = 0.0
    raw_samples = [] of RawSample
    failure_samples = [] of RawSample
    elements.times do |index|
      reference_bits = float_bits(reference, index)
      actual_bits = float_bits(actual, index)
      reference_value = reference_bits.unsafe_as(Float32).to_f64
      actual_value = actual_bits.unsafe_as(Float32).to_f64
      raw_diff = reference_bits != actual_bits
      raw_unequal += 1 if raw_diff
      delta = (actual_value - reference_value).abs
      finite = reference_value.finite? && actual_value.finite?
      if finite
        maximum = Math.max(maximum, delta)
      else
        maximum = Float64::MAX
      end
      failed = !finite || delta > ATOL + RTOL * reference_value.abs
      finite_failures += 1 if failed
      if (raw_diff && raw_samples.size < sample_limit) || (failed && failure_samples.size < sample_limit)
        sample = {index: index, row: index // row_width,
                  actual: actual_value.to_s, reference: reference_value.to_s,
                  actual_bits: bits_hex(actual_bits), reference_bits: bits_hex(reference_bits)}
        raw_samples << sample if raw_diff && raw_samples.size < sample_limit
        failure_samples << sample if failed && failure_samples.size < sample_limit
      end
    end
    {name: name, elements: elements, finite_failures: finite_failures,
     raw_unequal: raw_unequal, max_abs: maximum,
     first_raw_unequal: raw_samples, first_finite_failures: failure_samples,
     actual_sha256: Digest::SHA256.hexdigest(actual),
     reference_sha256: Digest::SHA256.hexdigest(reference)}
  end

  class SourceFixture
    getter document : JSON::Any
    getter path : String
    getter manifest_sha256 : String
    getter tokens : Int32
    getter dim : Int32
    getter epsilon : Float32
    getter residual : Bytes
    getter residual_sha256 : String
    getter bf16_sidecar_sha256 : String
    getter f32_sidecar_sha256 : String
    getter norm_weight : JSON::Any
    getter residual_binding : StageBinding
    getter output_state_sha256 : String
    getter expected_output_name : String
    getter prefix_manifest_sha256 : String

    def initialize(path : String, trusted_sha256 : String,
                   expected_tokens : Int32 = 1280, expected_dim : Int32 = 3584)
      @path, @manifest_sha256, @document = Qwen25RMSStatisticsProbe.file_manifest(
        path, trusted_sha256, "source stage")
      raise ArgumentError.new("unexpected source stage schema") unless @document["schema"].as_s == Qwen25RMSStatisticsProbe::STAGE_SCHEMA && @document["schema_version"].as_i == 1
      checkpoint = @document["checkpoint"]
      unless checkpoint["repo_id"].as_s == Qwen25RMSStatisticsProbe::MODEL_ID &&
             checkpoint["revision"].as_s == Qwen25RMSStatisticsProbe::REVISION
        raise ArgumentError.new("source stage checkpoint identity mismatch")
      end
      config = @document["config"]
      @tokens = config["sequence_length"].as_i
      @dim = config["hidden_size"].as_i
      raise ArgumentError.new("source stage geometry differs from requested probe") unless @tokens == expected_tokens && @dim == expected_dim
      raise ArgumentError.new("source stage model geometry mismatch") unless config["num_hidden_layers"].as_i == 28 && config["intermediate_size"].as_i == 18944 && config["num_attention_heads"].as_i == 28 && config["num_key_value_heads"].as_i == 4 && config["head_dim"].as_i == 128
      @epsilon = config["rms_norm_eps"].as_f.to_f32
      raise ArgumentError.new("invalid source RMS epsilon") unless @epsilon.finite? && @epsilon > 0
      layer = @document["layer"]
      raise ArgumentError.new("source stage must identify decoder layer zero") unless layer["index"].as_i == 0
      prefix = @document["prefix_reference"]
      raise ArgumentError.new("source stage prefix state indexing mismatch") unless prefix["state_indexing"].as_s == "0=token_embedding; i+1=decoder_layer_i_output" && prefix["input_state_index"].as_i == 0 && prefix["expected_output_state_index"].as_i == 1
      @prefix_manifest_sha256 = Qwen25RMSStatisticsProbe.checked_sha256(prefix["manifest_sha256"].as_s, "source prefix manifest")

      weight = layer["weights"].as_a.find { |record| record["key"].as_s == Qwen25RMSStatisticsProbe::NORM_WEIGHT_KEY }
      raise ArgumentError.new("source stage is missing post-attention norm weight") unless weight
      raise ArgumentError.new("source norm weight metadata mismatch") unless weight["dtype"].as_s == "BF16" && weight["shape"].as_a.map(&.as_i) == [@dim] && weight["nbytes"].as_i64 == @dim.to_i64 * 2
      Qwen25RMSStatisticsProbe.checked_sha256(weight["tensor_sha256"].as_s, "source norm tensor")
      @norm_weight = weight

      bindings = @document["stages"].as_a
      record = bindings.find { |entry| entry["name"].as_s == Qwen25RMSStatisticsProbe::RESIDUAL_STAGE_NAME }
      raise ArgumentError.new("source stage lacks authenticated residual_after_attention") unless record
      @residual_binding = Qwen25RMSStatisticsProbe.parse_extent(
        record, Qwen25RMSStatisticsProbe::RESIDUAL_STAGE_NAME, "bfloat16",
        [1_i64, @tokens.to_i64, @dim.to_i64], "bf16")

      descriptors = @document["sidecars"]
      raise ArgumentError.new("unexpected source sidecar inventory") unless descriptors.as_h.keys.sort == ["bf16", "f32"]
      @bf16_sidecar_sha256 = ""
      @f32_sidecar_sha256 = ""
      bf16_path, bf16_size, bf16_expected_sha256 = Qwen25RMSStatisticsProbe.descriptor_path_and_hash(
        @document, "bf16", File.dirname(@path), "bfloat16-le",
        Qwen25RMSStatisticsProbe::MAX_SOURCE_SIDECAR_BYTES)
      f32_path, f32_size, f32_expected_sha256 = Qwen25RMSStatisticsProbe.descriptor_path_and_hash(
        @document, "f32", File.dirname(@path), "float32-le",
        Qwen25RMSStatisticsProbe::MAX_SOURCE_SIDECAR_BYTES)
      if @residual_binding[:offset_bytes] + @residual_binding[:nbytes] > bf16_size
        raise ArgumentError.new("source residual extent exceeds BF16 sidecar bounds")
      end
      old_output = bindings.find { |entry| entry["name"].as_s == "layers.0.post_attention_layernorm" }
      raise ArgumentError.new("source stage lacks post-attention RMS output") unless old_output
      output_binding = Qwen25RMSStatisticsProbe.parse_extent(
        old_output, "layers.0.post_attention_layernorm", "bfloat16",
        [1_i64, @tokens.to_i64, @dim.to_i64], "bf16")
      if output_binding[:offset_bytes] + output_binding[:nbytes] > bf16_size
        raise ArgumentError.new("source output extent exceeds BF16 sidecar bounds")
      end
      @expected_output_name = output_binding[:name]
      @bf16_sidecar_sha256, source_bf16_snapshots = Qwen25RMSStatisticsProbe.authenticate_sidecar(
        bf16_path, bf16_size, bf16_expected_sha256,
        {
          "residual" => {@residual_binding[:offset_bytes], @residual_binding[:nbytes]},
          "output"   => {output_binding[:offset_bytes], output_binding[:nbytes]},
        })
      @f32_sidecar_sha256, _ = Qwen25RMSStatisticsProbe.authenticate_sidecar(
        f32_path, f32_size, f32_expected_sha256, {} of String => Tuple(Int64, Int64))
      @residual = source_bf16_snapshots["residual"]
      @residual_sha256 = Digest::SHA256.hexdigest(@residual)
      unless @document["rms_trace"]["site"].as_s == "post_attention_layernorm" &&
             @document["rms_trace"]["input_state_sha256"].as_s == @residual_sha256
        raise ArgumentError.new("source stage RMS input is not the authenticated residual")
      end
      @output_state_sha256 = Digest::SHA256.hexdigest(source_bf16_snapshots["output"])
      raise ArgumentError.new("invalid source F32 sidecar size") unless f32_size > 0
    end
  end

  class ReferenceFixture
    getter document : JSON::Any
    getter path : String
    getter manifest_sha256 : String
    getter source : SourceFixture
    getter f32_sidecar_sha256 : String
    getter bf16_sidecar_sha256 : String
    getter bindings : Hash(String, StageBinding)
    getter input_sha256 : String

    @sidecar_sizes = {} of String => Int64
    @sidecar_snapshots = {} of String => Bytes

    def initialize(path : String, trusted_sha256 : String, @source : SourceFixture)
      @path, @manifest_sha256, @document = Qwen25RMSStatisticsProbe.file_manifest(
        path, trusted_sha256, "RMS reference")
      raise ArgumentError.new("unexpected RMS reference schema") unless @document["schema"].as_s == Qwen25RMSStatisticsProbe::RMS_SCHEMA && @document["schema_version"].as_i == 1
      rms = @document["rms_trace"]
      raise ArgumentError.new("RMS reference site mismatch") unless rms["site"].as_s == "post_attention_layernorm"
      source = @document["source_stage_manifest"]
      expected = {
        "path"                         => File.basename(@source.path),
        "manifest_sha256"              => @source.manifest_sha256,
        "input_stage_name"             => Qwen25RMSStatisticsProbe::RESIDUAL_STAGE_NAME,
        "input_state_sha256"           => @source.residual_sha256,
        "expected_output_stage_name"   => @source.expected_output_name,
        "expected_output_state_sha256" => @source.output_state_sha256,
      }
      expected.each do |key, value|
        raise ArgumentError.new("RMS reference source stage link mismatch: #{key}") unless source[key].as_s == value
      end
      raise ArgumentError.new("RMS reference source layer mismatch") unless source["layer_index"].as_i == 0
      raise ArgumentError.new("RMS reference prefix manifest link mismatch") unless source["prefix_reference_manifest_sha256"].as_s == @source.prefix_manifest_sha256
      source_sidecars = source["sidecar_sha256s"]
      raise ArgumentError.new("RMS reference source BF16 sidecar link mismatch") unless source_sidecars["bf16"].as_s == @source.bf16_sidecar_sha256
      raise ArgumentError.new("RMS reference source F32 sidecar link mismatch") unless source_sidecars["f32"].as_s == @source.f32_sidecar_sha256
      validate_norm_weight!(@document["layer"]["norm_weight"])

      raise ArgumentError.new("RMS input digest mismatch") unless rms["input_state_sha256"].as_s == @source.residual_sha256
      epsilon = rms["variance_epsilon"].as_f.to_f32
      raise ArgumentError.new("RMS reference epsilon mismatch") unless epsilon.finite? && epsilon == @source.epsilon
      validate_source!(rms["source"])
      validate_operation_sequence!(rms["operation_sequence"])

      records = rms["stages"].as_a
      raise ArgumentError.new("RMS reference stage count mismatch") unless records.size == Qwen25RMSStatisticsProbe::RMS_SUFFIXES.size
      @bindings = {} of String => StageBinding
      expected_hidden = [1_i64, @source.tokens.to_i64, @source.dim.to_i64]
      expected_scalar = [1_i64, @source.tokens.to_i64, 1_i64]
      Qwen25RMSStatisticsProbe::RMS_SUFFIXES.each_with_index do |suffix, index|
        name = "#{Qwen25RMSStatisticsProbe::RMS_PREFIX}#{suffix}"
        f32 = Qwen25RMSStatisticsProbe::F32_SUFFIXES.includes?(suffix)
        binding = Qwen25RMSStatisticsProbe.parse_extent(
          records[index], name, f32 ? "float32" : "bfloat16",
          suffix == "variance_f32" || suffix == "variance_plus_eps_f32" || suffix == "rsqrt_f32" ? expected_scalar : expected_hidden,
          f32 ? "f32" : "bf16")
        @bindings[name] = binding
      end

      descriptors = @document["sidecars"]
      raise ArgumentError.new("unexpected RMS reference sidecar inventory") unless descriptors.as_h.keys.sort == ["bf16", "f32"]
      f32_path, @sidecar_sizes["f32"], f32_expected_sha256 =
        Qwen25RMSStatisticsProbe.descriptor_path_and_hash(
          @document, "f32", File.dirname(@path), "float32-le",
          Qwen25RMSStatisticsProbe::MAX_REFERENCE_SIDECAR_BYTES)
      bf16_path, @sidecar_sizes["bf16"], bf16_expected_sha256 =
        Qwen25RMSStatisticsProbe.descriptor_path_and_hash(
          @document, "bf16", File.dirname(@path), "bfloat16-le",
          Qwen25RMSStatisticsProbe::MAX_REFERENCE_SIDECAR_BYTES)
      f32_bindings = Qwen25RMSStatisticsProbe::F32_SUFFIXES.map do |suffix|
        @bindings["#{Qwen25RMSStatisticsProbe::RMS_PREFIX}#{suffix}"]
      end
      bf16_bindings = Qwen25RMSStatisticsProbe::BF16_SUFFIXES.map do |suffix|
        @bindings["#{Qwen25RMSStatisticsProbe::RMS_PREFIX}#{suffix}"]
      end
      Qwen25RMSStatisticsProbe.validate_contiguous_extents!(f32_bindings,
        @sidecar_sizes["f32"], "RMS F32")
      Qwen25RMSStatisticsProbe.validate_contiguous_extents!(bf16_bindings,
        @sidecar_sizes["bf16"], "RMS BF16")
      @f32_sidecar_sha256, f32_snapshot = Qwen25RMSStatisticsProbe.authenticate_sidecar(
        f32_path, @sidecar_sizes["f32"], f32_expected_sha256,
        {"snapshot" => {0_i64, @sidecar_sizes["f32"]}})
      @bf16_sidecar_sha256, bf16_snapshot = Qwen25RMSStatisticsProbe.authenticate_sidecar(
        bf16_path, @sidecar_sizes["bf16"], bf16_expected_sha256,
        {"snapshot" => {0_i64, @sidecar_sizes["bf16"]}})
      @sidecar_snapshots["f32"] = f32_snapshot["snapshot"]
      @sidecar_snapshots["bf16"] = bf16_snapshot["snapshot"]
      @input_sha256 = @source.residual_sha256
    end

    def expected(suffix : String) : Bytes
      raise ArgumentError.new("unsupported RMS reference stage") unless Qwen25RMSStatisticsProbe::RMS_SUFFIXES.includes?(suffix)
      binding = @bindings["#{Qwen25RMSStatisticsProbe::RMS_PREFIX}#{suffix}"]
      Qwen25RMSStatisticsProbe.copy_extent(@sidecar_snapshots[binding[:sidecar]],
        binding[:offset_bytes], binding[:nbytes])
    end

    private def validate_norm_weight!(record : JSON::Any) : Nil
      source = @source.norm_weight
      %w(key shard dtype nbytes tensor_sha256).each do |key|
        raise ArgumentError.new("RMS reference norm-weight mismatch: #{key}") unless record[key].to_json == source[key].to_json
      end
      raise ArgumentError.new("RMS reference norm-weight shape mismatch") unless record["shape"].as_a.map(&.as_i) == source["shape"].as_a.map(&.as_i)
      unless record["key"].as_s == Qwen25RMSStatisticsProbe::NORM_WEIGHT_KEY &&
             record["dtype"].as_s == "BF16" && record["shape"].as_a.map(&.as_i) == [@source.dim] &&
             record["nbytes"].as_i64 == @source.dim.to_i64 * 2
        raise ArgumentError.new("RMS reference norm-weight geometry mismatch")
      end
    end

    private def validate_source!(record : JSON::Any) : Nil
      unless record["schema_version"].as_i == 1 &&
             record["source_file"].as_s == "models/qwen2/modeling_qwen2.py" &&
             record["source_sha256"].as_s == Qwen25RMSStatisticsProbe::RMS_SOURCE_SHA256
        raise ArgumentError.new("RMS reference source identity mismatch")
      end
    end

    private def validate_operation_sequence!(record : JSON::Any) : Nil
      expected = [
        {"stage" => "input_f32", "operator" => "aten::to", "overload" => "dtype"},
        {"stage" => "square_f32", "operator" => "aten::pow", "overload" => "Tensor_Scalar"},
        {"stage" => "variance_f32", "operator" => "aten::mean", "overload" => "dim"},
        {"stage" => "variance_plus_eps_f32", "operator" => "aten::add", "overload" => "Tensor"},
        {"stage" => "rsqrt_f32", "operator" => "aten::rsqrt", "overload" => ""},
        {"stage" => "normalized_f32", "operator" => "aten::mul", "overload" => "Tensor"},
        {"stage" => "normalized_bf16", "operator" => "aten::to", "overload" => "dtype"},
        {"stage" => "gained_output_bf16", "operator" => "aten::mul", "overload" => "Tensor"},
      ]
      actual = record.as_a.map do |step|
        {"stage" => step["stage"].as_s, "operator" => step["operator"].as_s,
         "overload" => step["overload"].as_s}
      end
      raise ArgumentError.new("RMS reference operation sequence mismatch") unless actual == expected
    end
  end
end
