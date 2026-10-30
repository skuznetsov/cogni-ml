require "json"
require "../src/ml/video/hy_nomas_continuation_history"
require "./support/hy_nomas_probe_instrumentation"

{% unless flag?(:hy_nomas_history_probe) %}
  {% raise "Compile this probe with -Dhy_nomas_history_probe" %}
{% end %}

module HyNomasContinuationHistoryProbe
  LAYERS              = 54_i32
  SEGMENTS            =  2_i32
  VIDEO_TOKENS        =  2_i32
  VALID_TEXT_TOKENS   =  1_i32
  STORAGE_TEXT_TOKENS =  2_i32
  QUERY_ROWS          = VIDEO_TOKENS + VALID_TEXT_TOKENS
  STORAGE_ROWS        = VIDEO_TOKENS + STORAGE_TEXT_TOKENS
  HEADS               =      2_i32
  HEAD_DIM            =      2_i32
  FORWARD_COUNT       =      9_i32
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

  # Independent CPU reference for BQHD causal-free attention over either the
  # current segment alone or an explicit segment-major history prefix.
  def self.expected_output(banks : InputBanks, layer : Int32,
                           query_forward : Int32, history_forward : Int32?) : Array(Float32)
    result = Array(Float32).new(query_elements, 0.0_f32)
    key_rows = history_forward ? QUERY_ROWS * 2 : QUERY_ROWS
    scale = 1.0 / Math.sqrt(HEAD_DIM.to_f64)
    QUERY_ROWS.times do |query_row|
      HEADS.times do |head|
        scores = Array(Float64).new(key_rows, 0.0_f64)
        key_rows.times do |key_row|
          source_forward, source_row = if history_forward && key_row < QUERY_ROWS
                                         {history_forward.not_nil!, key_row}
                                       elsif history_forward
                                         {query_forward, key_row - QUERY_ROWS}
                                       else
                                         {query_forward, key_row}
                                       end
          dot = 0.0_f64
          HEAD_DIM.times do |feature|
            q_index = index(query_forward, layer, query_row, head, feature)
            k_index = index(source_forward, layer, source_row, head, feature)
            dot += banks.query_values[q_index].to_f64 * banks.key_values[k_index].to_f64
          end
          scores[key_row] = dot * scale
        end
        maximum = scores.max
        weights = scores.map { |score| Math.exp(score - maximum) }
        denominator = weights.sum
        HEAD_DIM.times do |feature|
          total = 0.0_f64
          key_rows.times do |key_row|
            source_forward, source_row = if history_forward && key_row < QUERY_ROWS
                                           {history_forward.not_nil!, key_row}
                                         elsif history_forward
                                           {query_forward, key_row - QUERY_ROWS}
                                         else
                                           {query_forward, key_row}
                                         end
            v_index = index(source_forward, layer, source_row, head, feature)
            total += (weights[key_row] / denominator) * banks.value_values[v_index].to_f64
          end
          result[(query_row * HEADS + head) * HEAD_DIM + feature] = total.to_f32
        end
      end
    end
    result
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

query = ML::MetalBuffer.new(H.storage_elements.to_i64 * sizeof(Float32))
key = ML::MetalBuffer.new(H.storage_elements.to_i64 * sizeof(Float32))
value = ML::MetalBuffer.new(H.storage_elements.to_i64 * sizeof(Float32))
output = ML::MetalBuffer.new(H.storage_elements.to_i64 * sizeof(Float32))
provider = ML::Video::HyNomasContinuationHistory.new(
  segment_count: H::SEGMENTS, video_tokens: H::VIDEO_TOKENS,
  valid_text_tokens: H::VALID_TEXT_TOKENS, storage_text_tokens: H::STORAGE_TEXT_TOKENS,
  layers: H::LAYERS, heads: H::HEADS, head_dim: H::HEAD_DIM,
  query: query, key: key, value: value, output: output)
banks = H.build_input_banks
copy_plan = H::CopyPlan.new
pre_key_snapshot = ML::MetalBuffer.new(H.cache_snapshot_elements.to_i64 * sizeof(Float32))
pre_value_snapshot = ML::MetalBuffer.new(H.cache_snapshot_elements.to_i64 * sizeof(Float32))
post_key_snapshot = ML::MetalBuffer.new(H.cache_snapshot_elements.to_i64 * sizeof(Float32))
post_value_snapshot = ML::MetalBuffer.new(H.cache_snapshot_elements.to_i64 * sizeof(Float32))
output_capture = ML::MetalBuffer.new((2 * H.cache_snapshot_elements).to_i64 * sizeof(Float32))
chain = ML::Metal::MPSCommandChain.new
allocation_start = ML::MetalBuffer.hy_text_probe_allocations
readback_start = ML::MetalBuffer.hy_text_probe_readbacks

begin
  provider.validate_bindings!(H::QUERY_ROWS, H::HEADS, H::HEAD_DIM, query, key, value, output)
  provider.encode_probe_fill_cache!(chain, H::SENTINEL)

  4.times do |pass|
    provider.select_segment!(0, writer: false)
    capture = pass == 3 ? output_capture : nil
    H.run_forward!(provider, copy_plan, chain, banks, pass, query, key, value,
      output, capture, 0)
  end

  provider.encode_probe_snapshot_cache!(chain, pre_key_snapshot,
    pre_value_snapshot, :before_writer)
  provider.select_segment!(0, writer: true)
  H.run_forward!(provider, copy_plan, chain, banks, 4, query, key, value, output)
  provider.encode_probe_snapshot_cache!(chain, post_key_snapshot,
    post_value_snapshot, :after_writer)

  4.times do |pass|
    provider.select_segment!(1, writer: false)
    capture_offset = H.cache_snapshot_elements
    capture = pass == 3 ? output_capture : nil
    H.run_forward!(provider, copy_plan, chain, banks, pass + 5, query, key,
      value, output, capture, capture_offset)
  end

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
  actual_outputs = output_capture.read(2 * H.cache_snapshot_elements)
  expected_sentinel = H::SENTINEL
  pre_keys.each { |actual| raise "denoise pass appended K before writer" unless actual == expected_sentinel }
  pre_values.each { |actual| raise "denoise pass appended V before writer" unless actual == expected_sentinel }

  H::LAYERS.times do |layer|
    H::QUERY_ROWS.times do |row|
      H::HEADS.times do |head|
        H::HEAD_DIM.times do |feature|
          offset = layer * H.query_elements + (row * H::HEADS + head) * H::HEAD_DIM + feature
          writer_offset = H.index(4, layer, row, head, feature)
          unless post_keys[offset] == banks.key_values[writer_offset] &&
                 post_values[offset] == banks.value_values[writer_offset]
            raise "writer cache append mismatch at layer #{layer}, row #{row}, head #{head}, feature #{feature}"
          end
        end
      end
    end
  end

  p0_error = 0.0_f64
  history_error = 0.0_f64
  H::LAYERS.times do |layer|
    p0_actual = actual_outputs[(layer * H.query_elements), H.query_elements]
    history_offset = H.cache_snapshot_elements + layer * H.query_elements
    history_actual = actual_outputs[history_offset, H.query_elements]
    p0_error = Math.max(p0_error, H.max_abs_error(p0_actual,
      H.expected_output(banks, layer, 3, nil)))
    history_error = Math.max(history_error, H.max_abs_error(history_actual,
      H.expected_output(banks, layer, 8, 4)))
  end
  raise "P=0 provider output exceeded #{H::ATOL}: #{p0_error}" if p0_error > H::ATOL
  raise "P>0 provider output exceeded #{H::ATOL}: #{history_error}" if history_error > H::ATOL
  diagnostic_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readback_start
  raise "unexpected continuation probe readback count" unless diagnostic_readbacks == 5

  report = JSON.build do |json|
    json.object do
      json.field "schema", "cogni_ml.hy_nomas_continuation_history_probe"
      json.field "schema_version", 1
      json.field "scope", "synthetic F32 provider append/read qualification; no model inference"
      json.field "segments", H::SEGMENTS
      json.field "layers", H::LAYERS
      json.field "denoise_forwards_per_segment", 4
      json.field "writer_forwards", 1
      json.field "active_query_rows", H::QUERY_ROWS
      json.field "storage_rows", H::STORAGE_ROWS
      json.field "heads", H::HEADS
      json.field "head_dim", H::HEAD_DIM
      json.field "cache_named_bytes", provider.shape.cache_bytes
      json.field "workspace_named_bytes", provider.shape.workspace_bytes
      json.field "named_bytes", provider.shape.named_bytes
      json.field "p0_max_abs_error", p0_error
      json.field "history_max_abs_error", history_error
      json.field "tolerance_atol", H::ATOL
      json.field "denoise_append_rejected_by_sentinel", true
      json.field "writer_append_exact_f32", true
      json.field "encode_buffer_allocations", encode_allocations
      json.field "encode_readbacks", encode_readbacks
      json.field "preterminal_waits", preterminal_waits
      json.field "terminal_wait_count", chain.wait_count
      json.field "postterminal_readbacks", diagnostic_readbacks
      json.field "passed", true
    end
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
     banks.value_buffer, pre_key_snapshot, pre_value_snapshot,
     post_key_snapshot, post_value_snapshot, output_capture}.each(&.release)
  end
end
