require "spec"
require "../examples/support/hy_nomas_reference_block0_probe"
require "../examples/support/hy_nomas_reference_history_probe"

describe HyNomasReferenceHistoryProbe do
  it "admits only the BF16 graph history mode without pre-QKV companion outputs" do
    HyNomasReferenceBlock0Probe.validate_precision_options!(
      HyNomasReferenceHistoryProbe::PRECISION, "bf16-graph", false)
    HyNomasReferenceHistoryProbe.validate_options!(
      HyNomasReferenceHistoryProbe::PRECISION, "bf16-graph", false, "", "")

    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.validate_precision_options!(
        HyNomasReferenceHistoryProbe::PRECISION, "f32-accum", true)
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceHistoryProbe.validate_options!(
        HyNomasReferenceHistoryProbe::PRECISION, "bf16-graph", true, "", "")
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceHistoryProbe.validate_options!(
        HyNomasReferenceHistoryProbe::PRECISION, "bf16-graph", false,
        "/tmp/preqkv.safetensors", "/tmp/preqkv.json")
    end
  end

  it "uses active image-plus-valid-text rows for the physical BF16 and F32 buffers" do
    {1, 529, 1280}.each do |valid_text_tokens|
      active_rows = HyNomasReferenceHistoryProbe.active_rows(valid_text_tokens)
      active_rows.should eq(4_320 + valid_text_tokens)
      entries = HyNomasReferenceHistoryProbe.header_entries(valid_text_tokens)
      entries.map(&.name).should eq(HyNomasReferenceHistoryProbe::TENSOR_NAMES)
      entries.map(&.dtype).should eq(["BF16", "BF16", "F32", "F32"])
      entries.each do |entry|
        entry.shape.should eq([1, active_rows, 16, 128])
      end
      entries[0].byte_count.should eq(active_rows * 2_048 * 2)
      entries[1].byte_count.should eq(active_rows * 2_048 * 2)
      entries[2].byte_count.should eq(active_rows * 2_048 * 4)
      entries[3].byte_count.should eq(active_rows * 2_048 * 4)

      geometry = HyNomasReferenceHistoryProbe.geometry(5_600, active_rows)
      geometry[:persisted_bytes].should eq(active_rows.to_i64 * 2_048 * 4)
      geometry[:read_stage_bytes].should eq(active_rows.to_i64 * 2_048 * 8)
      HyNomasReferenceHistoryProbe.named_bytes(5_600, active_rows).should eq(
        active_rows.to_i64 * 2_048 * 12)
    end
  end

  it "builds a four-tensor safetensors header with exact mixed-dtype extents" do
    entries = HyNomasReferenceHistoryProbe.header_entries(529)
    header = HyNomasReferenceHistoryProbe.safetensors_header(entries)
    (header.bytesize % 8).should eq(0)
    parsed = JSON.parse(header).as_h
    offset = 0_i64
    entries.each do |entry|
      tensor = parsed[entry.name].as_h
      tensor["dtype"].as_s.should eq(entry.dtype)
      tensor["shape"].as_a.map(&.as_i.to_i32).should eq(entry.shape)
      tensor["data_offsets"].as_a.map(&.as_i64).should eq([offset, offset + entry.byte_count])
      offset += entry.byte_count
    end
    HyNomasReferenceHistoryProbe.safetensors_file_bytes(header, offset).should eq(
      sizeof(UInt64).to_i64 + header.bytesize + offset)

    wrong_dtype = entries.dup
    wrong_dtype[0] = HyNomasReferenceHistoryProbe::TensorEntry.new(
      entries[0].name, "F32", entries[0].shape, entries[0].byte_count)
    expect_raises(ArgumentError) { HyNomasReferenceHistoryProbe.safetensors_header(wrong_dtype) }
    wrong_geometry = entries.dup
    wrong_geometry[0] = HyNomasReferenceHistoryProbe::TensorEntry.new(
      entries[0].name, entries[0].dtype, [1, 4_320, 16, 128], entries[0].byte_count)
    expect_raises(ArgumentError) { HyNomasReferenceHistoryProbe.safetensors_header(wrong_geometry) }
  end

  it "emits the exact standalone history receipt contract and rejects weakened evidence" do
    valid_text_tokens = 529
    entries = HyNomasReferenceHistoryProbe.header_entries(valid_text_tokens)
    records = entries.map do |entry|
      HyNomasReferenceHistoryProbe::TensorArtifact.new(entry.name, entry.dtype,
        entry.shape, entry.byte_count, "a" * 64)
    end
    history_bytes = HyNomasReferenceHistoryProbe.named_bytes(5_600,
      HyNomasReferenceHistoryProbe.active_rows(valid_text_tokens))
    named_bytes = {
      "input_buffers"                  => 1_i64,
      "block_weight_residents"         => 2_i64,
      "block_scratch"                  => 3_i64,
      "mps_weight_stage"               => 4_i64,
      "compact_attention_buffers"      => 5_i64,
      "qk_rms_square_and_mean_buffers" => 6_i64,
      "reference_bf16_buffers"         => 7_i64,
      "reference_history_buffers"      => history_bytes,
      "total_named_bytes"              => history_bytes + 28_i64,
    }
    artifact_header = HyNomasReferenceHistoryProbe.safetensors_header(entries)
    artifact_bytes = HyNomasReferenceHistoryProbe.safetensors_file_bytes(artifact_header,
      entries.sum(0_i64) { |entry| entry.byte_count.to_i64 })
    receipt_json = HyNomasReferenceHistoryProbe.receipt_json(
      "a" * 40, "b" * 40, "c" * 64, "d" * 64, "e" * 64, "f" * 64,
      valid_text_tokens, "1" * 64, "history.safetensors", "2" * 64, artifact_bytes,
      records, named_bytes, 0_i64, 0_i64, 0, 1)
    receipt = JSON.parse(receipt_json).as_h
    receipt.keys.to_set.should eq(Set{
      "schema", "schema_version", "native_revision", "source_revision", "scope",
      "full_upstream_t0_preprocessing_parity", "history_cache_write_executed",
      "history_cache_read_executed", "attention_executed", "input_bundle_manifest_sha256",
      "input_bundle_payload_sha256", "reference_latent_artifact_sha256",
      "timeline_manifest_sha256", "valid_text_tokens", "text_storage_tokens",
      "weights_sha256", "precision_control", "bf16_linear_variant", "history",
      "tensor_artifact", "named_buffer_accounting", "execution_counters",
    })
    receipt["schema"].as_s.should eq(HyNomasReferenceHistoryProbe::SCHEMA)
    receipt["scope"].as_s.should eq(HyNomasReferenceHistoryProbe::SCOPE)
    receipt["full_upstream_t0_preprocessing_parity"].as_bool.should be_false
    receipt["history_cache_write_executed"].as_bool.should be_true
    receipt["history_cache_read_executed"].as_bool.should be_true
    receipt["attention_executed"].as_bool.should be_false
    receipt["precision_control"].as_s.should eq("reference-bf16-history")
    receipt["bf16_linear_variant"].as_s.should eq("bf16-graph")
    receipt["text_storage_tokens"].as_i.should eq(1_280)

    history = receipt["history"].as_h
    history.keys.to_set.should eq(Set{
      "storage_dtype", "read_dtype", "layout", "cache_valid_len", "cache_txt",
      "stored_tokens", "cache_writes", "cache_reads", "persisted_bytes", "read_stage_bytes",
    })
    history["storage_dtype"].as_s.should eq("BF16")
    history["read_dtype"].as_s.should eq("F32")
    history["layout"].as_s.should eq(HyNomasReferenceHistoryProbe::LAYOUT)
    history["cache_valid_len"].as_i.should eq(0)
    history["cache_txt"].as_bool.should be_true
    history["stored_tokens"].as_i.should eq(4_320 + valid_text_tokens)
    history["cache_writes"].as_i.should eq(1)
    history["cache_reads"].as_i.should eq(1)
    history["persisted_bytes"].as_i64.should eq((4_320 + valid_text_tokens).to_i64 * 2_048 * 4)
    history["read_stage_bytes"].as_i64.should eq((4_320 + valid_text_tokens).to_i64 * 2_048 * 8)

    artifact = receipt["tensor_artifact"].as_h
    artifact.keys.to_set.should eq(Set{"basename", "sha256", "bytes", "tensors"})
    artifact["tensors"].as_a.map { |tensor| tensor.as_h["name"].as_s }.should eq(
      HyNomasReferenceHistoryProbe::TENSOR_NAMES)
    artifact["tensors"].as_a.map { |tensor| tensor.as_h["dtype"].as_s }.should eq(
      ["BF16", "BF16", "F32", "F32"])
    artifact["tensors"].as_a.each do |tensor_any|
      tensor = tensor_any.as_h
      tensor.keys.to_set.should eq(Set{"name", "dtype", "shape", "nbytes", "sha256"})
      tensor["shape"].as_a.map(&.as_i.to_i32).should eq([1, 4_320 + valid_text_tokens, 16, 128])
    end

    accounting = receipt["named_buffer_accounting"].as_h
    accounting.keys.to_set.should eq(named_bytes.keys.to_set + Set{"opaque_mpsgraph_private_bytes"})
    accounting["reference_history_buffers"].as_i64.should eq(history_bytes)
    counters = receipt["execution_counters"].as_h
    counters.keys.to_set.should eq(Set{"encode_allocations", "encode_readbacks", "preterminal_waits", "terminal_waits"})
    counters["encode_allocations"].as_i64.should eq(0)
    counters["encode_readbacks"].as_i64.should eq(0)
    counters["preterminal_waits"].as_i.should eq(0)
    counters["terminal_waits"].as_i.should eq(1)

    expect_raises(ArgumentError) do
      HyNomasReferenceHistoryProbe.receipt_json(
        "a" * 40, "b" * 40, "c" * 64, "d" * 64, "e" * 64, "f" * 64,
        valid_text_tokens, "1" * 64, "history.safetensors", "2" * 64, artifact_bytes,
        records, named_bytes, 0_i64, 0_i64, 0, 2)
    end
    wrong_accounting = named_bytes.dup
    wrong_accounting["reference_history_buffers"] = history_bytes + 1
    expect_raises(ArgumentError) do
      HyNomasReferenceHistoryProbe.receipt_json(
        "a" * 40, "b" * 40, "c" * 64, "d" * 64, "e" * 64, "f" * 64,
        valid_text_tokens, "1" * 64, "history.safetensors", "2" * 64, artifact_bytes,
        records, wrong_accounting, 0_i64, 0_i64, 0, 1)
    end

    wrong_records = records.dup
    first = records[0]
    wrong_records[0] = HyNomasReferenceHistoryProbe::TensorArtifact.new(
      first.name, first.dtype, [1, 4_320, 16, 128], first.nbytes, first.sha256)
    expect_raises(ArgumentError) do
      HyNomasReferenceHistoryProbe.receipt_json(
        "a" * 40, "b" * 40, "c" * 64, "d" * 64, "e" * 64, "f" * 64,
        valid_text_tokens, "1" * 64, "history.safetensors", "2" * 64, artifact_bytes,
        wrong_records, named_bytes, 0_i64, 0_i64, 0, 1)
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceHistoryProbe.receipt_json(
        "a" * 40, "b" * 40, "c" * 64, "d" * 64, "e" * 64, "f" * 64,
        valid_text_tokens, "1" * 64, "history.safetensors", "2" * 64, artifact_bytes + 1,
        records, named_bytes, 0_i64, 0_i64, 0, 1)
    end
    wrong_total = named_bytes.dup
    wrong_total["total_named_bytes"] += 1
    expect_raises(ArgumentError) do
      HyNomasReferenceHistoryProbe.receipt_json(
        "a" * 40, "b" * 40, "c" * 64, "d" * 64, "e" * 64, "f" * 64,
        valid_text_tokens, "1" * 64, "history.safetensors", "2" * 64, artifact_bytes,
        records, wrong_total, 0_i64, 0_i64, 0, 1)
    end
  end

  it "rejects invalid active geometry and tensor names" do
    expect_raises(ArgumentError) { HyNomasReferenceHistoryProbe.active_rows(0) }
    expect_raises(ArgumentError) { HyNomasReferenceHistoryProbe.active_rows(1_281) }
    expect_raises(ArgumentError) { HyNomasReferenceHistoryProbe.geometry(5_600, 5_601) }
    expect_raises(ArgumentError) { HyNomasReferenceHistoryProbe.tensor_shape("history.q", 529) }
    expect_raises(ArgumentError) do
      HyNomasReferenceHistoryProbe.safetensors_file_bytes("", -1_i64)
    end
  end
end
