require "./spec_helper"
require "../src/ml/gguf/qwen_qbit_state_snapshot"

private alias QBitStateRecordKind = ML::GGUF::Qwen35StateSnapshot::RecordKind

private def qbit_state_bytes(values : Array(Float32)) : Bytes
  bytes = Bytes.new(values.size * sizeof(Float32))
  bytes.copy_from(Slice.new(values.to_unsafe.as(Pointer(UInt8)), bytes.size))
  bytes
end

private def qbit_state_floats(bytes : Bytes) : Array(Float32)
  values = Array(Float32).new(bytes.size // sizeof(Float32), 0.0_f32)
  Slice.new(values.to_unsafe.as(Pointer(UInt8)), bytes.size).copy_from(bytes)
  values
end

describe ML::GGUF::QwenQBitStateSnapshot do
  state_codec = ML::GGUF::QwenQBitStateSnapshot

  it "keeps live KV exact while encoding recurrent records as p7 tiles" do
    kv = [1.0_f32, -2.0_f32, 3.0_f32, -4.0_f32]
    recurrent = Array(Float32).new(13) { |i| (i - 6).to_f32 / 3.0_f32 }
    snapshot = ML::GGUF::Qwen35StateSnapshot::Snapshot.new(
      16,
      1,
      [5_i32],
      [
        ML::GGUF::Qwen35StateSnapshot::Record.new(0, QBitStateRecordKind::KCache, qbit_state_bytes(kv), ML::StorageMode::Shared),
        ML::GGUF::Qwen35StateSnapshot::Record.new(0, QBitStateRecordKind::ConvState, qbit_state_bytes(recurrent), ML::StorageMode::Shared),
      ],
    )

    encoded = state_codec.encode(snapshot, block_size: 8, precision: 7)
    encoded.records[0].raw.not_nil!.should eq(snapshot.records[0].bytes)
    encoded.records[0].qbit.should be_nil
    encoded.records[1].raw.should be_nil
    encoded.records[1].qbit.not_nil!.precision.should eq(7)

    decoded = state_codec.decode(encoded)
    decoded.records[0].bytes.should eq(snapshot.records[0].bytes)
    qbit_state_floats(decoded.records[1].bytes).each_with_index do |value, i|
      value.should be_close(recurrent[i], 0.05_f32)
    end

    native = ML::GGUF::QwenQBitNativeBlock.parse(state_codec.encode_native_recurrent(encoded, 19_u64))
    native.record_spans.map { |span| {span.cache_id, span.layer, span.kind, span.value_count} }.should eq([
      {19_u64, 0_i32, QBitStateRecordKind::ConvState.value, recurrent.size.to_i32},
    ])
  end

  it "validates the complete snapshot before restore admission" do
    raw = qbit_state_bytes([0.0_f32] * 8)
    record = ML::GGUF::QwenQBitStateSnapshot::EncodedRecord.new(
      0,
      QBitStateRecordKind::KCache,
      ML::StorageMode::Shared,
      raw.size,
      raw,
      nil,
    )
    malformed = ML::GGUF::QwenQBitStateSnapshot::Snapshot.new(8, 1, [0_i32], [record, record], 8, 7)
    expect_raises(ArgumentError, /duplicate/) { state_codec.validate(malformed) }
  end

  it "attaches an exact KV-only artifact without admitting recurrent records" do
    source = ML::GGUF::Qwen35StateSnapshot::Snapshot.new(
      16,
      1,
      [5_i32],
      [
        ML::GGUF::Qwen35StateSnapshot::Record.new(0, QBitStateRecordKind::KCache, qbit_state_bytes([1.0_f32, 2.0_f32]), ML::StorageMode::Shared),
        ML::GGUF::Qwen35StateSnapshot::Record.new(0, QBitStateRecordKind::ConvState, qbit_state_bytes([3.0_f32, 4.0_f32]), ML::StorageMode::Shared),
      ],
    )
    encoded = state_codec.encode(source, block_size: 8, precision: 7)
    external_kv = ML::GGUF::Qwen35StateSnapshot::Snapshot.new(
      16,
      1,
      [5_i32],
      [
        ML::GGUF::Qwen35StateSnapshot::Record.new(0, QBitStateRecordKind::KCache, qbit_state_bytes([9.0_f32, 10.0_f32]), ML::StorageMode::Shared),
      ],
    )
    artifact_bytes = ML::GGUF::Qwen35StateSnapshot.encode_artifact_bytes(external_kv)
    artifact = ML::GGUF::Qwen35StateSnapshot.decode_artifact_encoded_bytes(artifact_bytes, copy_payloads: false)

    attached = state_codec.with_exact_artifact(encoded, artifact)
    attached.records.find(&.kind.k_cache?).not_nil!.raw.not_nil!.should eq(qbit_state_bytes([9.0_f32, 10.0_f32]))
    attached.records.find(&.kind.conv_state?).not_nil!.qbit.should_not be_nil
    attached.backing_stores.should eq([artifact_bytes])

    recurrent_artifact = ML::GGUF::Qwen35StateSnapshot::Snapshot.new(
      16,
      1,
      [5_i32],
      [
        ML::GGUF::Qwen35StateSnapshot::Record.new(0, QBitStateRecordKind::ConvState, qbit_state_bytes([3.0_f32, 4.0_f32]), ML::StorageMode::Shared),
      ],
    )
    recurrent_bytes = ML::GGUF::Qwen35StateSnapshot.encode_artifact_bytes(recurrent_artifact)
    recurrent_encoded = ML::GGUF::Qwen35StateSnapshot.decode_artifact_encoded_bytes(recurrent_bytes, copy_payloads: false)
    expect_raises(ArgumentError, /KV-only/) { state_codec.with_exact_artifact(encoded, recurrent_encoded) }

    missing_artifact = ML::GGUF::Qwen35StateSnapshot::Snapshot.new(16, 1, [5_i32], [] of ML::GGUF::Qwen35StateSnapshot::Record)
    missing_bytes = ML::GGUF::Qwen35StateSnapshot.encode_artifact_bytes(missing_artifact)
    missing_encoded = ML::GGUF::Qwen35StateSnapshot.decode_artifact_encoded_bytes(missing_bytes, copy_payloads: false)
    expect_raises(ArgumentError, /record set mismatch/) { state_codec.with_exact_artifact(encoded, missing_encoded) }

    wrong_position_artifact = ML::GGUF::Qwen35StateSnapshot::Snapshot.new(
      16,
      1,
      [4_i32],
      external_kv.records,
    )
    wrong_position_bytes = ML::GGUF::Qwen35StateSnapshot.encode_artifact_bytes(wrong_position_artifact)
    wrong_position_encoded = ML::GGUF::Qwen35StateSnapshot.decode_artifact_encoded_bytes(wrong_position_bytes, copy_payloads: false)
    expect_raises(ArgumentError, /positions mismatch/) { state_codec.with_exact_artifact(encoded, wrong_position_encoded) }
  end
end
