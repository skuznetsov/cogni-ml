require "json"
require "option_parser"
require "./support/hy_nomas_prompt_history_probe"

alias H = HyNomasPromptHistoryProbe

output_dir = ""
OptionParser.parse do |options|
  options.on("--output-dir PATH", "Existing empty external directory for the upstream replay banks") { |value| output_dir = value }
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
  segment_text_tokens: H::SEGMENT_TEXT_TOKENS,
  segment_cache_text: H::SEGMENT_CACHE_TEXT,
  query: query, key: key, value: value, output: output)
banks = H.build_input_banks
copy_plan = HyNomasContinuationHistoryProbe::CopyPlan.new
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
  unless provider.shape.query_rows == H::QUERY_ROWS && provider.shape.storage_rows == H::STORAGE_ROWS &&
         provider.shape.max_history_rows == H::CACHE_ROWS &&
         provider.shape.workspace_geometries.map(&.current_rows) == [3, 4, 3]
    raise "prompt-history provider geometry differs from the admitted profile"
  end
  provider.encode_probe_fill_cache!(chain, H::SENTINEL)
  forward = 0
  H::SEGMENTS.times do |segment|
    H::DENOISE_STEPS.times do
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
    raise "prompt-history provider probe requires one terminal wait"
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
  H::LAYERS.times do |layer|
    H::CACHE_ROWS.times do |row|
      head_feature = H::HEADS * H::HEAD_DIM
      offset = (layer * H::CACHE_ROWS + row) * head_feature
      source = if row < H.active_rows(0)
                 {H.writer_index(0), row}
               elsif row < H.active_rows(0) + H.active_rows(1)
                 {H.writer_index(1), row - H.active_rows(0)}
               else
                 {-1, -1}
               end
      head_feature.times do |element|
        expected_post_key = row < H.active_rows(0) ? banks.key_values[H.index(source[0], layer, source[1], element // H::HEAD_DIM, element % H::HEAD_DIM)] : H::SENTINEL
        expected_post_value = row < H.active_rows(0) ? banks.value_values[H.index(source[0], layer, source[1], element // H::HEAD_DIM, element % H::HEAD_DIM)] : H::SENTINEL
        unless post_keys[offset + element] == expected_post_key && post_values[offset + element] == expected_post_value
          raise "first writer cache prefix mismatch at layer #{layer}, row #{row}"
        end
        expected_final_key = source[0] < 0 ? H::SENTINEL : banks.key_values[H.index(source[0], layer, source[1], element // H::HEAD_DIM, element % H::HEAD_DIM)]
        expected_final_value = source[0] < 0 ? H::SENTINEL : banks.value_values[H.index(source[0], layer, source[1], element // H::HEAD_DIM, element % H::HEAD_DIM)]
        unless final_keys[offset + element] == expected_final_key && final_values[offset + element] == expected_final_value
          raise "variable prompt writer cache mismatch at layer #{layer}, row #{row}"
        end
      end
    end
  end

  diagnostic_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readback_start
  raise "unexpected prompt-history probe readback count" unless diagnostic_readbacks == 7
  artifacts = {} of String => NamedTuple(path: String, sha256: String, shape: Array(Int32))
  unless output_dir.empty?
    bank_shape = [H::FORWARD_COUNT, H::LAYERS, H::STORAGE_ROWS, H::HEADS, H::HEAD_DIM]
    artifacts["query"] = H.write_artifact(output_dir, "query", banks.query_values, bank_shape)
    artifacts["key"] = H.write_artifact(output_dir, "key", banks.key_values, bank_shape)
    artifacts["value"] = H.write_artifact(output_dir, "value", banks.value_values, bank_shape)
    artifacts["output"] = H.write_artifact(output_dir, "output", actual_outputs,
      [H::FORWARD_COUNT, H::LAYERS, H::QUERY_ROWS, H::HEADS, H::HEAD_DIM])
    cache_shape = [H::LAYERS, H::CACHE_ROWS, H::HEADS, H::HEAD_DIM]
    artifacts["cache_key"] = H.write_artifact(output_dir, "cache_key", final_keys, cache_shape)
    artifacts["cache_value"] = H.write_artifact(output_dir, "cache_value", final_values, cache_shape)
  end
  report = {
    schema: "cogni_ml.hy_nomas_prompt_history_probe", schema_version: 1,
    scope: "synthetic F32 three-prompt continuation provider; no model inference",
    segments: H::SEGMENTS, layers: H::LAYERS, denoise_forwards_per_segment: H::DENOISE_STEPS,
    writer_forwards: H::WRITER_COUNT, forward_count: H::FORWARD_COUNT,
    video_tokens: H::VIDEO_TOKENS, valid_text_tokens: H::VALID_TEXT_TOKENS,
    storage_text_tokens: H::STORAGE_TEXT_TOKENS, segment_text_tokens: H::SEGMENT_TEXT_TOKENS,
    segment_cache_text: H::SEGMENT_CACHE_TEXT, active_query_rows: H::QUERY_ROWS,
    storage_rows: H::STORAGE_ROWS, heads: H::HEADS, head_dim: H::HEAD_DIM,
    final_cache_rows: H::CACHE_ROWS, cache_named_bytes: provider.shape.cache_bytes,
    workspace_named_bytes: provider.shape.workspace_bytes, named_bytes: provider.shape.named_bytes,
    encode_buffer_allocations: encode_allocations, encode_readbacks: encode_readbacks,
    preterminal_waits: preterminal_waits, terminal_wait_count: chain.wait_count,
    postterminal_readbacks: diagnostic_readbacks, artifacts: artifacts, passed: true,
  }.to_pretty_json
  unless output_dir.empty?
    H.write_fresh_output(File.join(output_dir, "native.json")) { |file| file.puts(report) }
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
