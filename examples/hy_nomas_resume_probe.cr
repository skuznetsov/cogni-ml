require "json"
require "option_parser"
require "digest/sha256"
require "../src/ml/video/hy_nomas_resume_bundle"
require "./support/hy_nomas_continuation_history_probe"

{% unless flag?(:hy_nomas_history_probe) %}
  {% raise "Compile this probe with -Dhy_nomas_history_probe" %}
{% end %}

module HyNomasResumeProbe
  alias H = HyNomasContinuationHistoryProbe
  alias Provider = ML::Video::HyNomasContinuationHistory
  alias Bundle = ML::Video::HyNomasResumeBundle

  record Owner,
    query : ML::MetalBuffer,
    key : ML::MetalBuffer,
    value : ML::MetalBuffer,
    output : ML::MetalBuffer,
    provider : Provider,
    banks : H::InputBanks,
    copy_plan : H::CopyPlan do
    def release : Nil
      provider.release
      {query, key, value, output, banks.query_buffer, banks.key_buffer,
       banks.value_buffer}.each(&.release)
    end
  end

  record ReferenceOutput, values : Array(Float32), sha256 : String

  def self.new_owner : Owner
    query = ML::MetalBuffer.new(H.storage_elements.to_i64 * sizeof(Float32))
    key = ML::MetalBuffer.new(H.storage_elements.to_i64 * sizeof(Float32))
    value = ML::MetalBuffer.new(H.storage_elements.to_i64 * sizeof(Float32))
    output = ML::MetalBuffer.new(H.storage_elements.to_i64 * sizeof(Float32))
    provider = Provider.new(
      segment_count: H::SEGMENTS, video_tokens: H::VIDEO_TOKENS,
      valid_text_tokens: H::VALID_TEXT_TOKENS, storage_text_tokens: H::STORAGE_TEXT_TOKENS,
      layers: H::LAYERS, heads: H::HEADS, head_dim: H::HEAD_DIM,
      text_cache_policy: Provider::TextCachePolicy::OncePerScene,
      query: query, key: key, value: value, output: output)
    Owner.new(query, key, value, output, provider, H.build_input_banks, H::CopyPlan.new)
  end

  def self.validate_output_dir!(directory : String) : Nil
    unless Path[directory].absolute? && File.directory?(directory) && Dir.empty?(directory)
      raise ArgumentError.new("resume probe output directory must be an existing empty absolute directory")
    end
    root = File.realpath("/opt/mldata_ssd")
    resolved = File.realpath(directory)
    unless !File.symlink?(directory) && resolved.starts_with?(root + "/")
      raise ArgumentError.new("resume probe output directory must be a real directory inside /opt/mldata_ssd")
    end
  end

  def self.validate_native_report!(report : JSON::Any, bank_sha256 : String) : JSON::Any
    fields = report.as_h
    unless fields["schema"].as_s == "cogni_ml.hy_nomas_continuation_history_probe" &&
           fields["schema_version"].as_i == 2 && fields["passed"].as_bool &&
           fields["input_bank_sha256"].as_s == bank_sha256 &&
           fields["text_cache_policy"].as_s == "once_per_scene" &&
           fields["segments"].as_i == H::SEGMENTS && fields["layers"].as_i == H::LAYERS &&
           fields["forward_count"].as_i == H::FORWARD_COUNT &&
           fields["video_tokens"].as_i == H::VIDEO_TOKENS &&
           fields["valid_text_tokens"].as_i == H::VALID_TEXT_TOKENS &&
           fields["active_query_rows"].as_i == H::QUERY_ROWS &&
           fields["storage_rows"].as_i == H::STORAGE_ROWS &&
           fields["heads"].as_i == H::HEADS && fields["head_dim"].as_i == H::HEAD_DIM
      raise ArgumentError.new("continuous control report does not match the synthetic resume geometry")
    end
    fields["artifacts"].as_h["output"]
  rescue ex : KeyError | TypeCastError
    raise ArgumentError.new("continuous control report is missing required resume evidence")
  end

  def self.load_control_output(path : String, bank_sha256 : String) : ReferenceOutput
    info = File.info?(path, false)
    unless Path[path].absolute? && info && info.file? && !info.symlink?
      raise ArgumentError.new("continuous control report must be a regular non-symlink file")
    end
    raw = File.read(path)
    raise ArgumentError.new("continuous control report exceeds 1 MiB") if raw.bytesize > 1024 * 1024
    report = JSON.parse(raw)
    artifact = validate_native_report!(report, bank_sha256).as_h
    unless artifact.keys.to_set == Set{"path", "sha256", "shape"}
      raise ArgumentError.new("continuous control output artifact fields are malformed")
    end
    name = artifact["path"].as_s
    unless !name.empty? && name == File.basename(name) && name != "." && name != ".."
      raise ArgumentError.new("continuous control output must be a sibling basename")
    end
    output_path = File.join(File.dirname(path), name)
    output_info = File.info?(output_path, false)
    unless output_info && output_info.file? && !output_info.symlink? &&
           output_info.size == H::FORWARD_COUNT.to_i64 * H.forward_elements * sizeof(Float32)
      raise ArgumentError.new("continuous control output extent or file type is invalid")
    end
    shape = artifact["shape"].as_a.map(&.as_i)
    expected_shape = [H::FORWARD_COUNT, H::LAYERS, H::QUERY_ROWS, H::HEADS, H::HEAD_DIM]
    unless shape.map(&.to_i) == expected_shape
      raise ArgumentError.new("continuous control output shape mismatch")
    end
    raw_output = File.read(output_path)
    bytes = raw_output.to_slice
    sha256 = Digest::SHA256.hexdigest(raw_output)
    unless sha256 == artifact["sha256"].as_s
      raise ArgumentError.new("continuous control output digest mismatch")
    end
    values = Array(Float32).new(H::FORWARD_COUNT * H.forward_elements, 0.0_f32)
    values.size.times do |index|
      offset = index * sizeof(Float32)
      values[index] = IO::ByteFormat::LittleEndian.decode(Float32, bytes[offset, sizeof(Float32)])
    end
    ReferenceOutput.new(values, sha256)
  end

  def self.compare_to_control(actual : Array(Float32), control : Array(Float32),
                              first_forward : Int32) : NamedTuple(max_abs_error: Float64, diverged_elements: Int64)
    expected_count = (H::FORWARD_COUNT - first_forward) * H.forward_elements
    raise ArgumentError.new("resumed output capture has the wrong extent") unless actual.size == expected_count
    max_error = 0.0_f64
    diverged = 0_i64
    actual.each_with_index do |value, offset|
      reference = control[first_forward * H.forward_elements + offset]
      unless value.finite? && reference.finite?
        raise ArgumentError.new("non-finite native continuation output")
      end
      error = (value.to_f64 - reference.to_f64).abs
      max_error = Math.max(max_error, error)
      diverged += 1 if error > H::ATOL + H::ATOL * reference.to_f64.abs
    end
    {max_abs_error: max_error, diverged_elements: diverged}
  end

  def self.validate_output_against_formula!(actual : Array(Float32), banks : H::InputBanks,
                                            first_forward : Int32) : Array(Float64)
    count = actual.size // H.forward_elements
    unless actual.size == count * H.forward_elements && first_forward + count <= H::FORWARD_COUNT
      raise ArgumentError.new("native output capture has an invalid forward extent")
    end
    errors = Array(Float64).new(count, 0.0_f64)
    count.times do |offset|
      forward = first_forward + offset
      segment = forward // 5
      writer = forward % 5 == 4
      H::LAYERS.times do |layer|
        actual_layer = actual[offset * H.forward_elements + layer * H.query_elements,
          H.query_elements]
        errors[offset] = Math.max(errors[offset], H.max_abs_error(actual_layer,
          H.expected_output(banks, layer, forward, segment, writer)))
      end
    end
    errors
  end

  def self.validate_pre_writer_snapshot!(keys : Array(Float32), values : Array(Float32)) : Nil
    unless keys.size == H.cache_snapshot_elements && values.size == H.cache_snapshot_elements &&
           keys.all? { |entry| entry == H::SENTINEL } && values.all? { |entry| entry == H::SENTINEL }
      raise "denoise forwards appended persistent K/V before the writer"
    end
  end

  def self.validate_post_first_writer_snapshot!(keys : Array(Float32), values : Array(Float32),
                                                banks : H::InputBanks) : Nil
    cache_rows = H::LAYERS == 0 ? 0 : keys.size // (H::LAYERS * H::HEADS * H::HEAD_DIM)
    unless keys.size == H.cache_snapshot_elements && values.size == H.cache_snapshot_elements
      raise ArgumentError.new("post-writer cache snapshot extent mismatch")
    end
    H::LAYERS.times do |layer|
      cache_rows.times do |row|
        H::HEADS.times do |head|
          H::HEAD_DIM.times do |feature|
            offset = ((layer * cache_rows + row) * H::HEADS + head) * H::HEAD_DIM + feature
            expected = if row < H::QUERY_ROWS
                         input = H.index(4, layer, row, head, feature)
                         {banks.key_values[input], banks.value_values[input]}
                       else
                         {H::SENTINEL, H::SENTINEL}
                       end
            unless keys[offset] == expected[0] && values[offset] == expected[1]
              raise "first writer cache snapshot mismatch at layer #{layer}, row #{row}"
            end
          end
        end
      end
    end
  end

  def self.run_control(output_dir : String) : Nil
    validate_output_dir!(output_dir)
    owner = new_owner
    pre_key = ML::MetalBuffer.new(H.cache_snapshot_elements.to_i64 * sizeof(Float32))
    pre_value = ML::MetalBuffer.new(H.cache_snapshot_elements.to_i64 * sizeof(Float32))
    post_key = ML::MetalBuffer.new(H.cache_snapshot_elements.to_i64 * sizeof(Float32))
    post_value = ML::MetalBuffer.new(H.cache_snapshot_elements.to_i64 * sizeof(Float32))
    final_key = ML::MetalBuffer.new(H.cache_snapshot_elements.to_i64 * sizeof(Float32))
    final_value = ML::MetalBuffer.new(H.cache_snapshot_elements.to_i64 * sizeof(Float32))
    capture = ML::MetalBuffer.new((H::FORWARD_COUNT * H.forward_elements).to_i64 * sizeof(Float32))
    chain = ML::Metal::MPSCommandChain.new
    begin
      owner.provider.validate_bindings!(H::QUERY_ROWS, H::HEADS, H::HEAD_DIM,
        owner.query, owner.key, owner.value, owner.output)
      allocation_start = ML::MetalBuffer.hy_text_probe_allocations
      readback_start = ML::MetalBuffer.hy_text_probe_readbacks
      owner.provider.encode_probe_fill_cache!(chain, H::SENTINEL)
      forward = 0
      H::SEGMENTS.times do |segment|
        4.times do
          owner.provider.select_segment!(segment, writer: false)
          H.run_forward!(owner.provider, owner.copy_plan, chain, owner.banks, forward,
            owner.query, owner.key, owner.value, owner.output, capture,
            forward * H.forward_elements)
          forward += 1
        end
        owner.provider.encode_probe_snapshot_cache!(chain, pre_key, pre_value, :before_writer) if segment == 0
        if segment < H::SEGMENTS - 1
          owner.provider.select_segment!(segment, writer: true)
          H.run_forward!(owner.provider, owner.copy_plan, chain, owner.banks, forward,
            owner.query, owner.key, owner.value, owner.output, capture,
            forward * H.forward_elements)
          forward += 1
          if segment == 0
            owner.provider.encode_probe_snapshot_cache!(chain, post_key, post_value, :after_writer)
          end
        end
      end
      owner.provider.encode_probe_snapshot_cache!(chain, final_key, final_value, :complete)
      raise "wrong control forward count" unless forward == H::FORWARD_COUNT
      allocations = ML::MetalBuffer.hy_text_probe_allocations - allocation_start
      early_reads = ML::MetalBuffer.hy_text_probe_readbacks - readback_start
      early_waits = chain.wait_count
      unless allocations == 0 && early_reads == 0 && early_waits == 0
        raise "control encode allocated, read back, or waited before its terminal drain"
      end
      chain.commit_and_wait
      unless chain.completed? && chain.wait_count == 1
        raise "control request requires one terminal wait"
      end

      pre_keys = pre_key.read(H.cache_snapshot_elements)
      pre_values = pre_value.read(H.cache_snapshot_elements)
      post_keys = post_key.read(H.cache_snapshot_elements)
      post_values = post_value.read(H.cache_snapshot_elements)
      final_keys = final_key.read(H.cache_snapshot_elements)
      final_values = final_value.read(H.cache_snapshot_elements)
      outputs = capture.read(H::FORWARD_COUNT * H.forward_elements)
      validate_pre_writer_snapshot!(pre_keys, pre_values)
      validate_post_first_writer_snapshot!(post_keys, post_values, owner.banks)

      cache_rows = owner.provider.shape.max_history_rows
      H::LAYERS.times do |layer|
        cache_rows.times do |row|
          H::HEADS.times do |head|
            H::HEAD_DIM.times do |feature|
              offset = ((layer * cache_rows + row) * H::HEADS + head) * H::HEAD_DIM + feature
              first = row < H::QUERY_ROWS
              source_forward = first ? 4 : 9
              source_row = first ? row : row - H::QUERY_ROWS
              input = H.index(source_forward, layer, source_row, head, feature)
              unless final_keys[offset] == owner.banks.key_values[input] &&
                     final_values[offset] == owner.banks.value_values[input]
                raise "final writer cache append mismatch at layer #{layer}, row #{row}"
              end
            end
          end
        end
      end
      errors = validate_output_against_formula!(outputs, owner.banks, 0)
      raise "provider output exceeded #{H::ATOL}: #{errors}" if errors.any? { |error| error > H::ATOL }
      reads = ML::MetalBuffer.hy_text_probe_readbacks - readback_start
      raise "unexpected control diagnostic readback count" unless reads == 7

      bank_shape = [H::FORWARD_COUNT, H::LAYERS, H::STORAGE_ROWS, H::HEADS, H::HEAD_DIM]
      artifacts = {} of String => NamedTuple(path: String, sha256: String, shape: Array(Int32))
      artifacts["query"] = H.write_artifact(output_dir, "query", owner.banks.query_values, bank_shape)
      artifacts["key"] = H.write_artifact(output_dir, "key", owner.banks.key_values, bank_shape)
      artifacts["value"] = H.write_artifact(output_dir, "value", owner.banks.value_values, bank_shape)
      artifacts["output"] = H.write_artifact(output_dir, "output", outputs,
        [H::FORWARD_COUNT, H::LAYERS, H::QUERY_ROWS, H::HEADS, H::HEAD_DIM])
      artifacts["cache_key"] = H.write_artifact(output_dir, "cache_key", final_keys,
        [H::LAYERS, cache_rows, H::HEADS, H::HEAD_DIM])
      artifacts["cache_value"] = H.write_artifact(output_dir, "cache_value", final_values,
        [H::LAYERS, cache_rows, H::HEADS, H::HEAD_DIM])
      bank_sha = H.input_bank_sha256(owner.banks)
      report = {
        schema: "cogni_ml.hy_nomas_continuation_history_probe", schema_version: 2,
        scope: "synthetic F32 three-chunk same-scene provider qualification; no model inference",
        text_cache_policy: "once_per_scene", segments: H::SEGMENTS, layers: H::LAYERS,
        denoise_forwards_per_segment: 4, writer_forwards: 2, forward_count: H::FORWARD_COUNT,
        video_tokens: H::VIDEO_TOKENS, valid_text_tokens: H::VALID_TEXT_TOKENS,
        active_query_rows: H::QUERY_ROWS, storage_rows: H::STORAGE_ROWS,
        heads: H::HEADS, head_dim: H::HEAD_DIM, final_cache_rows: cache_rows,
        cache_named_bytes: owner.provider.shape.cache_bytes,
        workspace_named_bytes: owner.provider.shape.workspace_bytes,
        named_bytes: owner.provider.shape.named_bytes, forward_max_abs_errors: errors,
        tolerance_atol: H::ATOL, denoise_append_rejected_by_sentinel: true,
        writer_append_exact_f32: true, input_bank_sha256: bank_sha,
        encode_buffer_allocations: allocations, encode_readbacks: early_reads,
        preterminal_waits: early_waits, terminal_wait_count: chain.wait_count,
        postterminal_readbacks: reads, artifacts: artifacts, passed: true,
      }.to_pretty_json
      H.write_fresh_output(File.join(output_dir, "native.json")) { |io| io.puts(report) }
      puts report
    ensure
      unless chain.completed?
        begin
          chain.commit_and_wait
        rescue
        end
      end
      if chain.completed?
        owner.release
        chain.release
        {pre_key, pre_value, post_key, post_value, final_key, final_value, capture}.each(&.release)
      end
    end
  end

  def self.run_save(output_dir : String, bundle_path : String) : Nil
    validate_output_dir!(output_dir)
    owner = new_owner
    pre_key = ML::MetalBuffer.new(H.cache_snapshot_elements.to_i64 * sizeof(Float32))
    pre_value = ML::MetalBuffer.new(H.cache_snapshot_elements.to_i64 * sizeof(Float32))
    post_key = ML::MetalBuffer.new(H.cache_snapshot_elements.to_i64 * sizeof(Float32))
    post_value = ML::MetalBuffer.new(H.cache_snapshot_elements.to_i64 * sizeof(Float32))
    capture = ML::MetalBuffer.new((5 * H.forward_elements).to_i64 * sizeof(Float32))
    chain = ML::Metal::MPSCommandChain.new
    begin
      owner.provider.validate_bindings!(H::QUERY_ROWS, H::HEADS, H::HEAD_DIM,
        owner.query, owner.key, owner.value, owner.output)
      allocation_start = ML::MetalBuffer.hy_text_probe_allocations
      readback_start = ML::MetalBuffer.hy_text_probe_readbacks
      owner.provider.encode_probe_fill_cache!(chain, H::SENTINEL)
      4.times do |forward|
        owner.provider.select_segment!(0, writer: false)
        H.run_forward!(owner.provider, owner.copy_plan, chain, owner.banks, forward,
          owner.query, owner.key, owner.value, owner.output, capture,
          forward * H.forward_elements)
      end
      owner.provider.encode_probe_snapshot_cache!(chain, pre_key, pre_value, :before_writer)
      owner.provider.select_segment!(0, writer: true)
      H.run_forward!(owner.provider, owner.copy_plan, chain, owner.banks, 4,
        owner.query, owner.key, owner.value, owner.output, capture, 4 * H.forward_elements)
      owner.provider.encode_probe_snapshot_cache!(chain, post_key, post_value, :after_writer)
      allocations = ML::MetalBuffer.hy_text_probe_allocations - allocation_start
      early_reads = ML::MetalBuffer.hy_text_probe_readbacks - readback_start
      early_waits = chain.wait_count
      unless allocations == 0 && early_reads == 0 && early_waits == 0
        raise "saveA encode allocated, read back, or waited before its terminal drain"
      end
      chain.commit_and_wait
      unless chain.completed? && chain.wait_count == 1
        raise "saveA request requires one terminal wait"
      end
      pre_keys = pre_key.read(H.cache_snapshot_elements)
      pre_values = pre_value.read(H.cache_snapshot_elements)
      post_keys = post_key.read(H.cache_snapshot_elements)
      post_values = post_value.read(H.cache_snapshot_elements)
      outputs = capture.read(5 * H.forward_elements)
      validate_pre_writer_snapshot!(pre_keys, pre_values)
      validate_post_first_writer_snapshot!(post_keys, post_values, owner.banks)
      errors = validate_output_against_formula!(outputs, owner.banks, 0)
      raise "saveA outputs exceeded #{H::ATOL}: #{errors}" if errors.any? { |error| error > H::ATOL }
      reads = ML::MetalBuffer.hy_text_probe_readbacks - readback_start
      raise "unexpected saveA diagnostic readback count" unless reads == 5

      bank_sha = H.input_bank_sha256(owner.banks)
      inventory = owner.provider.checkpoint_inventory(1)
      contract = resume_contract(bank_sha)
      cache_buffers = {} of String => Tuple(ML::MetalBuffer, Int32)
      owner.provider.each_checkpoint_cache(1) do |name, buffer, count|
        cache_buffers[name] = {buffer, count}
      end
      bundle_sha = Bundle.write(bundle_path, contract, inventory) do |name, io|
        buffer, count = cache_buffers[name]? || raise ArgumentError.new("checkpoint cache inventory is missing #{name}")
        unless count.to_i64 * sizeof(Float32) == inventory[name]
          raise ArgumentError.new("checkpoint cache callback differs from the admitted inventory")
        end
        io.write(Bytes.new(buffer.contents.as(UInt8*), count * sizeof(Float32)))
      end
      report = {
        schema: "cogni_ml.hy_nomas_resume_probe", schema_version: 1,
        mode: "save", scope: "synthetic F32 checkpoint after segment A writer; no model inference",
        input_bank_sha256: bank_sha, bundle_path: File.basename(bundle_path), bundle_sha256: bundle_sha,
        next_segment: 1, saved_forwards: 5, layers: H::LAYERS,
        encode_buffer_allocations: allocations, encode_readbacks: early_reads,
        preterminal_waits: early_waits, terminal_wait_count: chain.wait_count,
        postterminal_readbacks: reads, formula_max_abs_errors: errors, passed: true,
      }.to_pretty_json
      H.write_fresh_output(File.join(output_dir, "save.json")) { |io| io.puts(report) }
      puts report
    ensure
      unless chain.completed?
        begin
          chain.commit_and_wait
        rescue
        end
      end
      if chain.completed?
        owner.release
        chain.release
        {pre_key, pre_value, post_key, post_value, capture}.each(&.release)
      end
    end
  end

  def self.resume_contract(bank_sha256 : String) : String
    "hy-nomas-synthetic-f32-v1|segments=3|layers=54|tokens=2,1,2|heads=2|head_dim=2|next_segment=1|input_bank_sha256=#{bank_sha256}"
  end

  def self.apply_cache_variant!(name : String, buffer : ML::MetalBuffer,
                                count : Int32, variant : String) : Nil
    return if variant == "valid"
    values = buffer.contents.as(Float32*)
    case variant
    when "zero"
      count.times { |index| values[index] = 0.0_f32 }
    when "mutated"
      delta = name.ends_with?(".key") ? 0.75_f32 : 3.0_f32
      count.times { |index| values[index] += delta }
    else
      raise ArgumentError.new("unknown cache variant")
    end
  end

  def self.mutate_future_inputs!(banks : H::InputBanks) : Nil
    mutate_future_tensor!(banks.query_buffer, banks.query_values, 0.5_f32)
    mutate_future_tensor!(banks.key_buffer, banks.key_values, 0.75_f32)
    mutate_future_tensor!(banks.value_buffer, banks.value_values, 2.0_f32)
  end

  def self.mutate_future_tensor!(buffer : ML::MetalBuffer, values : Array(Float32),
                                 delta : Float32) : Nil
    target = buffer.contents.as(Float32*)
    H::QUERY_ROWS.times do |row|
      H::HEADS.times do |head|
        H::HEAD_DIM.times do |feature|
          offset = H.index(5, 0, row, head, feature)
          values[offset] += delta
          target[offset] = values[offset]
        end
      end
    end
  end

  def self.run_resume(output_dir : String, bundle_path : String, bundle_sha256 : String,
                      control_report : String, cache_variant : String,
                      input_variant : String) : Nil
    validate_output_dir!(output_dir)
    unless {"valid", "zero", "mutated"}.includes?(cache_variant) &&
           {"same", "mutated"}.includes?(input_variant) &&
           !(cache_variant != "valid" && input_variant != "same")
      raise ArgumentError.new("choose one resume mutation: cache zero/mutated or future QKV mutated")
    end
    owner = new_owner
    capture = ML::MetalBuffer.new(((H::FORWARD_COUNT - 5) * H.forward_elements).to_i64 * sizeof(Float32))
    chain : ML::Metal::MPSCommandChain? = nil
    reader : Bundle::Reader? = nil
    begin
      owner.provider.validate_bindings!(H::QUERY_ROWS, H::HEADS, H::HEAD_DIM,
        owner.query, owner.key, owner.value, owner.output)
      bank_sha = H.input_bank_sha256(owner.banks)
      reference = load_control_output(control_report, bank_sha)
      inventory = owner.provider.checkpoint_inventory(1)
      reader = Bundle.open(bundle_path, bundle_sha256, resume_contract(bank_sha), inventory)
      owner.provider.restore_boundary!(1) do |record_name, buffer, count|
        reader.not_nil!.copy_to!(record_name,
          Bytes.new(buffer.contents.as(UInt8*), count * sizeof(Float32)))
        apply_cache_variant!(record_name, buffer, count, cache_variant)
      end
      reader.not_nil!.close
      reader = nil

      mutated_input_sha = nil.as(String?)
      if input_variant == "mutated"
        mutate_future_inputs!(owner.banks)
        mutated_input_sha = H.input_bank_sha256(owner.banks)
      end

      chain = ML::Metal::MPSCommandChain.new
      allocation_start = ML::MetalBuffer.hy_text_probe_allocations
      readback_start = ML::MetalBuffer.hy_text_probe_readbacks
      forward = 5
      while forward < H::FORWARD_COUNT
        segment = forward // 5
        writer = forward % 5 == 4
        owner.provider.select_segment!(segment, writer: writer)
        H.run_forward!(owner.provider, owner.copy_plan, chain.not_nil!, owner.banks, forward,
          owner.query, owner.key, owner.value, owner.output, capture,
          (forward - 5) * H.forward_elements)
        forward += 1
      end
      allocations = ML::MetalBuffer.hy_text_probe_allocations - allocation_start
      early_reads = ML::MetalBuffer.hy_text_probe_readbacks - readback_start
      early_waits = chain.not_nil!.wait_count
      unless allocations == 0 && early_reads == 0 && early_waits == 0
        raise "resume B+C encode allocated, read back, or waited before its terminal drain"
      end
      chain.not_nil!.commit_and_wait
      unless chain.not_nil!.completed? && chain.not_nil!.wait_count == 1
        raise "resume B+C request requires one terminal wait"
      end
      outputs = capture.read((H::FORWARD_COUNT - 5) * H.forward_elements)
      reads = ML::MetalBuffer.hy_text_probe_readbacks - readback_start
      raise "unexpected resume diagnostic readback count" unless reads == 1
      comparison = compare_to_control(outputs, reference.values, 5)
      negative = cache_variant != "valid" || input_variant != "same"
      if negative
        raise "seeded resume mutation did not change any resumed output beyond tolerance" if comparison[:diverged_elements] == 0
      elsif comparison[:diverged_elements] != 0
        raise "restored B+C outputs differ from continuous control: #{comparison}"
      end
      artifact = H.write_artifact(output_dir, "resumed_output", outputs,
        [H::FORWARD_COUNT - 5, H::LAYERS, H::QUERY_ROWS, H::HEADS, H::HEAD_DIM])
      report = {
        schema: "cogni_ml.hy_nomas_resume_probe", schema_version: 1,
        mode: "resume", scope: "synthetic F32 native history restore at next_segment=1, then forwards 5..13; no model inference",
        input_bank_sha256: bank_sha, resumed_input_bank_sha256: mutated_input_sha || bank_sha,
        control_output_sha256: reference.sha256, bundle_sha256: bundle_sha256,
        next_segment: 1, resumed_forwards: 9, layers: H::LAYERS,
        cache_variant: cache_variant, input_variant: input_variant,
        max_abs_error_vs_control: comparison[:max_abs_error],
        diverged_elements_vs_control: comparison[:diverged_elements],
        comparison_tolerance: "abs(error) <= 2e-5 + 2e-5 * abs(control)",
        encode_buffer_allocations: allocations, encode_readbacks: early_reads,
        preterminal_waits: early_waits, terminal_wait_count: chain.not_nil!.wait_count,
        postterminal_readbacks: reads, artifact: artifact,
        expected_divergence_detected: negative, passed: true,
      }.to_pretty_json
      H.write_fresh_output(File.join(output_dir, "resume.json")) { |io| io.puts(report) }
      puts report
    ensure
      reader.try(&.close)
      if current_chain = chain
        unless current_chain.completed?
          begin
            current_chain.commit_and_wait
          rescue
          end
        end
        if current_chain.completed?
          owner.release
          current_chain.release
          capture.release
        end
      else
        owner.release
        capture.release
      end
    end
  end
end

mode = ""
output_dir = ""
bundle_path = ""
bundle_sha256 = ""
control_report = ""
cache_variant = "valid"
input_variant = "same"
OptionParser.parse do |options|
  options.on("--mode MODE", "control, save, or resume") { |value| mode = value }
  options.on("--output-dir PATH", "Existing empty run directory on /opt/mldata_ssd") { |value| output_dir = value }
  options.on("--bundle PATH", "Checkpoint file for save or resume") { |value| bundle_path = value }
  options.on("--bundle-sha256 SHA", "Authenticated checkpoint file digest for resume") { |value| bundle_sha256 = value }
  options.on("--control-report PATH", "Continuous native control report for resume comparison") { |value| control_report = value }
  options.on("--cache-variant KIND", "valid, zero, or mutated checkpoint K/V; resume only") { |value| cache_variant = value }
  options.on("--input-variant KIND", "same or mutated future Q/K/V; resume only") { |value| input_variant = value }
end

case mode
when "control"
  raise ArgumentError.new("--output-dir is required") if output_dir.empty?
  HyNomasResumeProbe.run_control(output_dir)
when "save"
  raise ArgumentError.new("--output-dir and --bundle are required") if output_dir.empty? || bundle_path.empty?
  HyNomasResumeProbe.run_save(output_dir, bundle_path)
when "resume"
  unless !output_dir.empty? && !bundle_path.empty? && !bundle_sha256.empty? && !control_report.empty?
    raise ArgumentError.new("resume requires --output-dir, --bundle, --bundle-sha256, and --control-report")
  end
  HyNomasResumeProbe.run_resume(output_dir, bundle_path, bundle_sha256,
    control_report, cache_variant, input_variant)
else
  raise ArgumentError.new("--mode must be control, save, or resume")
end
