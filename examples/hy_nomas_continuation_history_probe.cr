require "json"
require "option_parser"
require "digest/sha256"
require "../src/ml/video/hy_nomas_continuation_history"
require "./support/hy_nomas_probe_instrumentation"

{% unless flag?(:hy_nomas_history_probe) %}
  {% raise "Compile this probe with -Dhy_nomas_history_probe" %}
{% end %}

module HyNomasContinuationHistoryProbe
  LAYERS              = 54_i32
  SEGMENTS            =  3_i32
  VIDEO_TOKENS        =  2_i32
  VALID_TEXT_TOKENS   =  1_i32
  STORAGE_TEXT_TOKENS =  2_i32
  QUERY_ROWS          = VIDEO_TOKENS + VALID_TEXT_TOKENS
  STORAGE_ROWS        = VIDEO_TOKENS + STORAGE_TEXT_TOKENS
  HEADS               =      2_i32
  HEAD_DIM            =      2_i32
  FORWARD_COUNT       =     14_i32
  ATOL                =     2.0e-5
  SENTINEL            = -19.25_f32
  COPY_KERNEL         = "hn_continuation_probe_copy_f32"
  COPY_SOURCE         = ML::Video::MetalOps::SOURCE + "\n" + {{ read_file("#{__DIR__}/support/hy_nomas_continuation_history_probe.metal") }}

  class CopyPlan
    @pipeline : ML::Metal::ComputePipeline

    def initialize
      @pipeline = ML::Metal::PipelineCache.get(COPY_KERNEL) do
        ML::Metal::ComputePipeline.new(COPY_KERNEL, COPY_SOURCE)
      end
    end

    def encode(encoder : ML::Metal::ComputeEncoder,
               source : ML::MetalBuffer, destination : ML::MetalBuffer,
               source_offset : Int32, destination_offset : Int32,
               count : Int32) : Nil
      unless source_offset >= 0 && destination_offset >= 0 && count > 0 &&
             source_offset.to_i64 + count <= Int32::MAX &&
             destination_offset.to_i64 + count <= Int32::MAX
        raise ArgumentError.new("Hy-Nomas continuation probe copy extent exceeds the native limit")
      end
      ML::Video::MetalOps.check(source, source_offset.to_i64 + count)
      ML::Video::MetalOps.check(destination, destination_offset.to_i64 + count)
      unless source.storage_mode.shared? && destination.storage_mode.shared?
        raise ArgumentError.new("Hy-Nomas continuation probe copies require shared F32 buffers")
      end
      ML::Video::MetalOps.distinct(source, destination)
      encoder.set_pipeline(@pipeline)
        .set_buffer(source, 0)
        .set_buffer(destination, 1, ML::Metal::BufferAccess::Write)
        .set_value(source_offset.to_u32, 2)
        .set_value(destination_offset.to_u32, 3)
        .set_value(count.to_u32, 4)
        .dispatch_1d(count)
    end
  end

  record InputBanks,
    query_buffer : ML::MetalBuffer,
    key_buffer : ML::MetalBuffer,
    value_buffer : ML::MetalBuffer,
    query_values : Array(Float32),
    key_values : Array(Float32),
    value_values : Array(Float32)

  def self.storage_elements : Int32
    STORAGE_ROWS * HEADS * HEAD_DIM
  end

  def self.query_elements : Int32
    QUERY_ROWS * HEADS * HEAD_DIM
  end

  def self.cache_snapshot_elements : Int32
    LAYERS * ((SEGMENTS - 1) * VIDEO_TOKENS + VALID_TEXT_TOKENS) * HEADS * HEAD_DIM
  end

  def self.forward_elements : Int32
    LAYERS * query_elements
  end

  def self.input_offset(forward : Int32, layer : Int32) : Int32
    (forward * LAYERS + layer) * storage_elements
  end

  def self.index(forward : Int32, layer : Int32, row : Int32,
                 head : Int32, feature : Int32) : Int32
    ((((forward * LAYERS + layer) * STORAGE_ROWS + row) * HEADS + head) * HEAD_DIM + feature)
  end

  def self.build_input_banks : InputBanks
    count = FORWARD_COUNT * LAYERS * storage_elements
    query_values = Array(Float32).new(count, 0.0_f32)
    key_values = Array(Float32).new(count, 0.0_f32)
    value_values = Array(Float32).new(count, 0.0_f32)
    FORWARD_COUNT.times do |forward|
      LAYERS.times do |layer|
        STORAGE_ROWS.times do |row|
          HEADS.times do |head|
            HEAD_DIM.times do |feature|
              offset = index(forward, layer, row, head, feature)
              if row >= QUERY_ROWS
                query_values[offset] = 100.0_f32 + layer + feature
                key_values[offset] = 200.0_f32 + forward + layer
                value_values[offset] = 300.0_f32 + forward + feature
              else
                query_values[offset] = (0.025 + forward * 0.004 + layer * 0.0007 +
                                        row * 0.003 + head * 0.002 + feature * 0.001).to_f32
                key_values[offset] = (-0.04 + forward * 0.003 + layer * 0.0002 +
                                      row * 0.002 + head * 0.001 + feature * 0.0015).to_f32
                value_values[offset] = (-0.06 + forward * 0.005 + layer * 0.0003 +
                                        row * 0.0035 + head * 0.001 + feature * 0.002).to_f32
              end
            end
          end
        end
      end
    end
    InputBanks.new(
      ML::MetalBuffer.from_array(query_values),
      ML::MetalBuffer.from_array(key_values),
      ML::MetalBuffer.from_array(value_values),
      query_values, key_values, value_values)
  end

  def self.copy_current_inputs!(plan : CopyPlan, chain : ML::Metal::MPSCommandChain,
                                banks : InputBanks, forward : Int32, layer : Int32,
                                query : ML::MetalBuffer, key : ML::MetalBuffer,
                                value : ML::MetalBuffer) : Nil
    source_offset = input_offset(forward, layer)
    encoder = chain.encoder
    plan.encode(encoder, banks.query_buffer, query, source_offset, 0, storage_elements)
    plan.encode(encoder, banks.key_buffer, key, source_offset, 0, storage_elements)
    plan.encode(encoder, banks.value_buffer, value, source_offset, 0, storage_elements)
  end

  def self.run_forward!(provider : ML::Video::HyNomasContinuationHistory,
                        plan : CopyPlan, chain : ML::Metal::MPSCommandChain,
                        banks : InputBanks, forward : Int32,
                        query : ML::MetalBuffer, key : ML::MetalBuffer,
                        value : ML::MetalBuffer, output : ML::MetalBuffer,
                        capture_destination : ML::MetalBuffer? = nil,
                        capture_offset : Int32 = 0) : Nil
    LAYERS.times do |layer|
      copy_current_inputs!(plan, chain, banks, forward, layer, query, key, value)
      provider.encode(chain, layer)
      if destination = capture_destination
        plan.encode(chain.encoder, output, destination, 0,
          capture_offset + layer * query_elements, query_elements)
      end
    end
  end

  # Independent F64 reference over an explicitly ordered row list. Current
  # text is excluded only on later denoise calls; every writer attends it.
  def self.expected_output(banks : InputBanks, layer : Int32,
                           query_forward : Int32, segment : Int32, writer : Bool) : Array(Float32)
    sources = [] of Tuple(Int32, Int32)
    segment.times do |previous|
      rows = previous == 0 ? QUERY_ROWS : VIDEO_TOKENS
      rows.times { |row| sources << {previous * 5 + 4, row} }
    end
    current_rows = segment == 0 || writer ? QUERY_ROWS : VIDEO_TOKENS
    current_rows.times { |row| sources << {query_forward, row} }
    result = Array(Float32).new(query_elements, 0.0_f32)
    scale = 1.0 / Math.sqrt(HEAD_DIM.to_f64)
    QUERY_ROWS.times do |query_row|
      HEADS.times do |head|
        scores = sources.map do |source|
          source_forward, source_row = source
          dot = 0.0_f64
          HEAD_DIM.times do |feature|
            dot += banks.query_values[index(query_forward, layer, query_row, head, feature)].to_f64 *
                   banks.key_values[index(source_forward, layer, source_row, head, feature)].to_f64
          end
          dot * scale
        end
        maximum = scores.max
        weights = scores.map { |score| Math.exp(score - maximum) }
        denominator = weights.sum
        HEAD_DIM.times do |feature|
          total = 0.0_f64
          sources.each_with_index do |source, i|
            total += (weights[i] / denominator) *
                     banks.value_values[index(source[0], layer, source[1], head, feature)].to_f64
          end
          result[(query_row * HEADS + head) * HEAD_DIM + feature] = total.to_f32
        end
      end
    end
    result
  end

  def self.write_fresh_output(path : String, & : IO::FileDescriptor ->) : Nil
    fd = LibC.open(path, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
    raise IO::Error.new("probe output must be fresh and writable: #{path}") if fd < 0
    io = IO::FileDescriptor.new(fd)
    begin
      yield io
    ensure
      io.close
    end
  end

  def self.write_artifact(directory : String, name : String, values : Array(Float32), shape : Array(Int32))
    path = File.join(directory, name + ".f32le")
    write_fresh_output(path) { |io| io.write(Bytes.new(values.to_unsafe.as(UInt8*), values.size * sizeof(Float32))) }
    {path: File.basename(path), sha256: Digest::SHA256.hexdigest(File.read(path)), shape: shape}
  end

  def self.max_abs_error(actual : Array(Float32), expected : Array(Float32)) : Float64
    raise ArgumentError.new("probe comparison vector length mismatch") unless actual.size == expected.size
    error = 0.0_f64
    actual.each_with_index do |value, index|
      reference = expected[index]
      raise "non-finite continuation output" unless value.finite? && reference.finite?
      error = Math.max(error, (value.to_f64 - reference.to_f64).abs)
    end
    error
  end
end

alias H = HyNomasContinuationHistoryProbe

output_dir = ""
OptionParser.parse do |options|
  options.on("--output-dir PATH", "Existing empty external directory for the upstream replay banks") { |v| output_dir = v }
end
unless output_dir.empty?
  unless Path[output_dir].absolute? && File.directory?(output_dir) && Dir.empty?(output_dir) &&
         File.realpath(output_dir).starts_with?(File.realpath("/opt/mldata_ssd") + "/")
    raise ArgumentError.new("probe output directory must be empty and inside /opt/mldata_ssd")
  end
end
query = ML::MetalBuffer.new(H.storage_elements.to_i64 * sizeof(Float32))
key = ML::MetalBuffer.new(H.storage_elements.to_i64 * sizeof(Float32))
value = ML::MetalBuffer.new(H.storage_elements.to_i64 * sizeof(Float32))
output = ML::MetalBuffer.new(H.storage_elements.to_i64 * sizeof(Float32))
provider = ML::Video::HyNomasContinuationHistory.new(
  segment_count: H::SEGMENTS, video_tokens: H::VIDEO_TOKENS,
  valid_text_tokens: H::VALID_TEXT_TOKENS, storage_text_tokens: H::STORAGE_TEXT_TOKENS,
  layers: H::LAYERS, heads: H::HEADS, head_dim: H::HEAD_DIM,
  text_cache_policy: ML::Video::HyNomasContinuationHistory::TextCachePolicy::OncePerScene,
  query: query, key: key, value: value, output: output)
banks = H.build_input_banks
copy_plan = H::CopyPlan.new
pre_key_snapshot = ML::MetalBuffer.new(H.cache_snapshot_elements.to_i64 * sizeof(Float32))
pre_value_snapshot = ML::MetalBuffer.new(H.cache_snapshot_elements.to_i64 * sizeof(Float32))
post_key_snapshot = ML::MetalBuffer.new(H.cache_snapshot_elements.to_i64 * sizeof(Float32))
post_value_snapshot = ML::MetalBuffer.new(H.cache_snapshot_elements.to_i64 * sizeof(Float32))
final_key_snapshot = ML::MetalBuffer.new(H.cache_snapshot_elements.to_i64 * sizeof(Float32))
final_value_snapshot = ML::MetalBuffer.new(H.cache_snapshot_elements.to_i64 * sizeof(Float32))
output_capture = ML::MetalBuffer.new((H::FORWARD_COUNT * H.forward_elements).to_i64 * sizeof(Float32))
chain = ML::Metal::MPSCommandChain.new
allocation_start = ML::MetalBuffer.hy_text_probe_allocations
readback_start = ML::MetalBuffer.hy_text_probe_readbacks

begin
  provider.validate_bindings!(H::QUERY_ROWS, H::HEADS, H::HEAD_DIM, query, key, value, output)
  provider.encode_probe_fill_cache!(chain, H::SENTINEL)
  forward = 0
  H::SEGMENTS.times do |segment|
    4.times do
      provider.select_segment!(segment, writer: false)
      H.run_forward!(provider, copy_plan, chain, banks, forward, query, key, value,
        output, output_capture, forward * H.forward_elements)
      forward += 1
    end
    if segment == 0
      provider.encode_probe_snapshot_cache!(chain, pre_key_snapshot, pre_value_snapshot, :before_writer)
    end
    if segment < H::SEGMENTS - 1
      provider.select_segment!(segment, writer: true)
      H.run_forward!(provider, copy_plan, chain, banks, forward, query, key, value,
        output, output_capture, forward * H.forward_elements)
      forward += 1
      if segment == 0
        provider.encode_probe_snapshot_cache!(chain, post_key_snapshot, post_value_snapshot, :after_writer)
      end
    end
  end
  provider.encode_probe_snapshot_cache!(chain, final_key_snapshot, final_value_snapshot, :complete)
  raise "wrong forward count" unless forward == H::FORWARD_COUNT
  encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocation_start
  encode_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readback_start
  preterminal_waits = chain.wait_count
  unless encode_allocations == 0 && encode_readbacks == 0 && preterminal_waits == 0
    raise "continuation provider encode allocated, read back, or waited before terminal"
  end
  chain.commit_and_wait
  unless chain.completed? && chain.wait_count == 1
    raise "continuation provider probe requires one terminal wait"
  end

  pre_keys = pre_key_snapshot.read(H.cache_snapshot_elements)
  pre_values = pre_value_snapshot.read(H.cache_snapshot_elements)
  post_keys = post_key_snapshot.read(H.cache_snapshot_elements)
  post_values = post_value_snapshot.read(H.cache_snapshot_elements)
  final_keys = final_key_snapshot.read(H.cache_snapshot_elements)
  final_values = final_value_snapshot.read(H.cache_snapshot_elements)
  actual_outputs = output_capture.read(H::FORWARD_COUNT * H.forward_elements)
  pre_keys.each { |actual| raise "denoise appended K before writer" unless actual == H::SENTINEL }
  pre_values.each { |actual| raise "denoise appended V before writer" unless actual == H::SENTINEL }
  cache_rows = provider.shape.max_history_rows
  H::LAYERS.times do |layer|
    cache_rows.times do |row|
      H::HEADS.times do |head|
        H::HEAD_DIM.times do |feature|
          offset = ((layer * cache_rows + row) * H::HEADS + head) * H::HEAD_DIM + feature
          first = row < H::QUERY_ROWS
          source_forward = first ? 4 : 9
          source_row = first ? row : row - H::QUERY_ROWS
          source_offset = H.index(source_forward, layer, source_row, head, feature)
          expected_post_key = first ? banks.key_values[source_offset] : H::SENTINEL
          expected_post_value = first ? banks.value_values[source_offset] : H::SENTINEL
          unless post_keys[offset] == expected_post_key && post_values[offset] == expected_post_value &&
                 final_keys[offset] == banks.key_values[source_offset] &&
                 final_values[offset] == banks.value_values[source_offset]
            raise "writer cache append mismatch at layer #{layer}, row #{row}, head #{head}, feature #{feature}"
          end
        end
      end
    end
  end
  errors = Array(Float64).new(H::FORWARD_COUNT, 0.0_f64)
  H::FORWARD_COUNT.times do |f|
    segment = f // 5
    writer = f % 5 == 4
    H::LAYERS.times do |layer|
      actual = actual_outputs[f * H.forward_elements + layer * H.query_elements, H.query_elements]
      errors[f] = Math.max(errors[f], H.max_abs_error(actual, H.expected_output(banks, layer, f, segment, writer)))
    end
  end
  raise "provider output exceeded #{H::ATOL}: #{errors}" if errors.any? { |e| e > H::ATOL }
  diagnostic_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readback_start
  raise "unexpected continuation probe readback count" unless diagnostic_readbacks == 7
  artifacts = {} of String => NamedTuple(path: String, sha256: String, shape: Array(Int32))
  unless output_dir.empty?
    bank_shape = [H::FORWARD_COUNT, H::LAYERS, H::STORAGE_ROWS, H::HEADS, H::HEAD_DIM]
    artifacts["query"] = H.write_artifact(output_dir, "query", banks.query_values, bank_shape)
    artifacts["key"] = H.write_artifact(output_dir, "key", banks.key_values, bank_shape)
    artifacts["value"] = H.write_artifact(output_dir, "value", banks.value_values, bank_shape)
    artifacts["output"] = H.write_artifact(output_dir, "output", actual_outputs,
      [H::FORWARD_COUNT, H::LAYERS, H::QUERY_ROWS, H::HEADS, H::HEAD_DIM])
    cache_shape = [H::LAYERS, cache_rows, H::HEADS, H::HEAD_DIM]
    artifacts["cache_key"] = H.write_artifact(output_dir, "cache_key", final_keys, cache_shape)
    artifacts["cache_value"] = H.write_artifact(output_dir, "cache_value", final_values, cache_shape)
  end
  report = {
    schema: "cogni_ml.hy_nomas_continuation_history_probe", schema_version: 2,
    scope: "synthetic F32 three-chunk same-scene provider qualification; no model inference",
    text_cache_policy: "once_per_scene", segments: H::SEGMENTS, layers: H::LAYERS,
    denoise_forwards_per_segment: 4, writer_forwards: 2, forward_count: H::FORWARD_COUNT,
    video_tokens: H::VIDEO_TOKENS, valid_text_tokens: H::VALID_TEXT_TOKENS,
    active_query_rows: H::QUERY_ROWS, storage_rows: H::STORAGE_ROWS,
    heads: H::HEADS, head_dim: H::HEAD_DIM, final_cache_rows: cache_rows,
    cache_named_bytes: provider.shape.cache_bytes, workspace_named_bytes: provider.shape.workspace_bytes,
    named_bytes: provider.shape.named_bytes, forward_max_abs_errors: errors, tolerance_atol: H::ATOL,
    denoise_append_rejected_by_sentinel: true, writer_append_exact_f32: true,
    encode_buffer_allocations: encode_allocations, encode_readbacks: encode_readbacks,
    preterminal_waits: preterminal_waits, terminal_wait_count: chain.wait_count,
    postterminal_readbacks: diagnostic_readbacks, artifacts: artifacts, passed: true,
  }.to_pretty_json
  unless output_dir.empty?
    H.write_fresh_output(File.join(output_dir, "native.json")) { |f| f.puts(report) }
  end
  puts report
ensure
  unless chain.completed?
    begin
      chain.commit_and_wait
    rescue
    end
  end
  if chain.completed?
    provider.release
    chain.release
    {query, key, value, output, banks.query_buffer, banks.key_buffer,
     banks.value_buffer, pre_key_snapshot, pre_value_snapshot, post_key_snapshot,
     post_value_snapshot, final_key_snapshot, final_value_snapshot, output_capture}.each(&.release)
  end
end
