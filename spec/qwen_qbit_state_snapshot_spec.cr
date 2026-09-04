require "./spec_helper"
require "../src/ml/gguf/qwen_qbit_state_snapshot"

private alias QBitStateRecordKind = ML::GGUF::Qwen35StateSnapshot::RecordKind

QWEN_9B_QBIT_STATE = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

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

  it "rejects RawF32 QBit restore into an F16 KV owner before route selection" do
    pending!("9B model metadata not present") unless File.exists?(QWEN_9B_QBIT_STATE)

    gguf = ML::GGUF::GGUFFile.new(QWEN_9B_QBIT_STATE, mmap_tensors: false)
    begin
      hp = ML::GGUF::Qwen35Hparams.new(gguf)
      state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 8, kv_cache_f16: true)
      snapshot = ML::GGUF::QwenQBitStateSnapshot::Snapshot.new(
        8,
        hp.n_layer,
        Array(Int32).new(hp.n_layer, 0_i32),
        [] of ML::GGUF::QwenQBitStateSnapshot::EncodedRecord,
        8,
        7,
      )

      expect_raises(ArgumentError, /RawF32 QBit snapshot cannot be restored into an F16 KV owner/) do
        state_codec.restore_into(snapshot, hp, state, prefer_metal: true)
      end
    ensure
      gguf.close
    end
  end

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

  it "encodes recurrent Native blocks one source record at a time" do
    records = [
      ML::GGUF::Qwen35StateSnapshot::Record.new(
        1,
        QBitStateRecordKind::ConvState,
        qbit_state_bytes(Array(Float32).new(13) { |i| (i - 6).to_f32 / 3.0_f32 }),
        ML::StorageMode::Shared,
      ),
      ML::GGUF::Qwen35StateSnapshot::Record.new(
        1,
        QBitStateRecordKind::SsmState,
        qbit_state_bytes(Array(Float32).new(29) { |i| (i - 14).to_f32 / 7.0_f32 }),
        ML::StorageMode::Shared,
      ),
    ]
    snapshot = ML::GGUF::Qwen35StateSnapshot::Snapshot.new(16, 2, [5_i32, 5_i32], records)
    reference = ML::GGUF::QwenQBitNativeBlock.parse_stream(
      state_codec.encode_native_recurrent(state_codec.encode(snapshot, block_size: 8, precision: 7), 19_u64)
    )

    encoder = ML::GGUF::QwenQBitStateSnapshot::NativeRecurrentStreamEncoder.new(
      cache_id: 19_u64,
      block_size: 8,
      precision: 7,
    )
    records.each { |record| encoder.append(record) }
    streamed = encoder.finish
    parsed = ML::GGUF::QwenQBitNativeBlock.parse_stream(streamed.bytes)

    streamed.record_count.should eq(2)
    streamed.source_byte_size.should eq(records.sum(0_i64, &.bytes.size.to_i64))
    streamed.peak_source_record_bytes.should eq(records.max_of(&.bytes.size).to_i64)
    streamed.peak_source_record_bytes.should be < streamed.source_byte_size
    parsed.blocks.size.should eq(records.size)
    ML::GGUF::QwenQBitNativeBlock.logical_sha256(parsed).should eq(
      ML::GGUF::QwenQBitNativeBlock.logical_sha256(reference)
    )
    parsed.record_spans.map { |span| {span.cache_id, span.layer, span.kind, span.value_count} }.should eq(
      reference.record_spans.map { |span| {span.cache_id, span.layer, span.kind, span.value_count} }
    )

    expect_raises(ArgumentError, /finished/) { encoder.append(records.first) }

    duplicate_encoder = ML::GGUF::QwenQBitStateSnapshot::NativeRecurrentStreamEncoder.new(block_size: 8)
    duplicate_encoder.append(records.first)
    expect_raises(ArgumentError, /duplicate/) { duplicate_encoder.append(records.first) }
    expect_raises(ArgumentError, /must not be empty/) do
      ML::GGUF::QwenQBitStateSnapshot::NativeRecurrentStreamEncoder.new(block_size: 8).finish
    end
  end

  it "pulls recurrent Native bytes without retaining the complete body" do
    records = [
      ML::GGUF::Qwen35StateSnapshot::Record.new(
        1,
        QBitStateRecordKind::ConvState,
        qbit_state_bytes(Array(Float32).new(13) { |i| (i - 6).to_f32 / 3.0_f32 }),
        ML::StorageMode::Shared,
      ),
      ML::GGUF::Qwen35StateSnapshot::Record.new(
        1,
        QBitStateRecordKind::SsmState,
        qbit_state_bytes(Array(Float32).new(29) { |i| (i - 14).to_f32 / 7.0_f32 }),
        ML::StorageMode::Shared,
      ),
    ]
    index = 0
    source = -> : ML::GGUF::Qwen35StateSnapshot::Record? do
      record = records[index]?
      index += 1 if record
      record
    end
    body = ML::GGUF::QwenQBitStateSnapshot::NativeRecurrentBody.new(
      source,
      cache_id: 19_u64,
      block_size: 8,
      precision: 7,
    )
    expect_raises(ArgumentError, /not fully consumed/) { body.summary }

    output = IO::Memory.new
    chunk = Bytes.new(3)
    while (read = body.read(chunk)) > 0
      output.write(chunk[0, read])
    end

    parsed = ML::GGUF::QwenQBitNativeBlock.parse_stream(output.to_slice)
    summary = body.summary
    body.byte_size.should eq(output.size)
    body.record_count.should eq(records.size)
    body.source_byte_size.should eq(records.sum(0_i64, &.bytes.size.to_i64))
    body.peak_source_record_bytes.should eq(records.max_of(&.bytes.size).to_i64)
    summary.row_count.should eq(parsed.row_count)
    summary.records.size.should eq(parsed.record_spans.size)
    summary.logical_sha256.should eq(ML::GGUF::QwenQBitNativeBlock.logical_sha256(parsed))
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

  it "keeps save-time recurrent compression at p6 through p8" do
    source = ML::GGUF::Qwen35StateSnapshot::Snapshot.new(
      16,
      1,
      [5_i32],
      [
        ML::GGUF::Qwen35StateSnapshot::Record.new(
          0,
          QBitStateRecordKind::ConvState,
          qbit_state_bytes([1.0_f32, 2.0_f32]),
          ML::StorageMode::Shared,
        ),
      ],
    )

    expect_raises(ArgumentError, /precision/) do
      state_codec.encode(source, block_size: 8, precision: 5)
    end
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
