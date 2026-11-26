require "./support/hy_nomas_continuation_history_probe"

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
