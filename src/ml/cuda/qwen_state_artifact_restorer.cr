require "./qwen_mixed_stack_runner"
require "../gguf/qwen35_state_snapshot"

module ML::CUDA
  # Restores encoded Qwen `.qkv` state artifacts into a resident CUDA mixed stack.
  #
  # This is the production boundary for recurrent compressed artifacts: recurrent
  # records may be BF16 or block-INT8 and are decoded directly into device state;
  # KV records are expected to be raw Float32 records and are copied as-is.
  class QwenStateArtifactRestorer
    PTX = {{ read_file("src/ml/cuda/kernels/recurrent_int8_codec_probe.ptx") }}

    private record Segment,
      layer : Int32,
      kind : ML::GGUF::Qwen35StateSnapshot::RecordKind,
      codec : ML::GGUF::Qwen35StateSnapshot::RecordCodec,
      payload_offset : Int32,
      values : Int32,
      payload : Bytes

    getter h2d_ms : Float64 = 0.0
    getter kernel_ms : Float64 = 0.0
    getter kv_ms : Float64 = 0.0
    getter raw_recurrent_ms : Float64 = 0.0
    getter restore_ms : Float64 = 0.0

    def initialize(@snapshot : ML::GGUF::Qwen35StateSnapshot::EncodedSnapshot)
      @records = {} of Tuple(Int32, ML::GGUF::Qwen35StateSnapshot::RecordKind) => ML::GGUF::Qwen35StateSnapshot::EncodedRecord
      @segments = {} of Tuple(Int32, ML::GGUF::Qwen35StateSnapshot::RecordKind) => Segment
      @i8_payload_bytes_total = 0
      @bf16_payload_bytes_total = 0
      @has_i8 = false
      @has_bf16 = false
      @closed = false

      @snapshot.records.each do |record|
        key = {record.layer, record.kind}
        raise ArgumentError.new("duplicate Qwen encoded state record: layer=#{record.layer}, kind=#{record.kind}") if @records.has_key?(key)

        @records[key] = record
        next unless recurrent_record_kind?(record.kind)

        case record.codec
        in ML::GGUF::Qwen35StateSnapshot::RecordCodec::RawF32
          # Raw recurrent records are allowed for fallback/mixed future policies.
        in ML::GGUF::Qwen35StateSnapshot::RecordCodec::Bf16
          add_bf16_segment(record)
        in ML::GGUF::Qwen35StateSnapshot::RecordCodec::BlockI8
          add_i8_segment(record)
        end
      end

      @module = nil.as(CUDAModule?)
      @i8_fn = nil.as(KernelFunction?)
      @bf16_fn = nil.as(KernelFunction?)
      if @has_i8 || @has_bf16
        @module = CUDAModule.load(PTX, "qwen_state_artifact_restorer")
        @i8_fn = @module.not_nil!.function("recurrent_block_i8_interleaved_decode_f32") if @has_i8
        @bf16_fn = @module.not_nil!.function("recurrent_bf16_decode_f32") if @has_bf16
      end

      @i8_quant_device = @has_i8 ? DeviceBuffer.new(@i8_payload_bytes_total.to_u64) : nil
      @bf16_payload_device = @has_bf16 ? DeviceBuffer.new(@bf16_payload_bytes_total.to_u64) : nil

      @i8_q_ptr = Pointer(DevicePtr).malloc(1)
      @i8_out_ptr = Pointer(DevicePtr).malloc(1)
      @i8_values_ptr = Pointer(UInt32).malloc(1)
      @i8_block_ptr = Pointer(UInt32).malloc(1)
      @i8_params = Pointer(Void*).malloc(4)
      @i8_params[0] = @i8_q_ptr.as(Void*)
      @i8_params[1] = @i8_out_ptr.as(Void*)
      @i8_params[2] = @i8_values_ptr.as(Void*)
      @i8_params[3] = @i8_block_ptr.as(Void*)
      @i8_block_ptr.value = @snapshot.codec_block.to_u32

      @bf16_payload_ptr = Pointer(DevicePtr).malloc(1)
      @bf16_out_ptr = Pointer(DevicePtr).malloc(1)
      @bf16_values_ptr = Pointer(UInt32).malloc(1)
      @bf16_params = Pointer(Void*).malloc(3)
      @bf16_params[0] = @bf16_payload_ptr.as(Void*)
      @bf16_params[1] = @bf16_out_ptr.as(Void*)
      @bf16_params[2] = @bf16_values_ptr.as(Void*)
    end

    def restore(stack : QwenMixedStackRunner) : Nil
      t_total = Time.instant
      upload_encoded_payloads

      t_kernel = Time.instant
      stack.layer_ids.zip(stack.runners).each do |layer_id, runner|
        raise ArgumentError.new("Qwen encoded state snapshot position missing for layer #{layer_id}") if layer_id < 0 || layer_id >= @snapshot.positions.size

        case runner
        in QwenRecurrentLayerRunner
          restore_recurrent_record(layer_id, ML::GGUF::Qwen35StateSnapshot::RecordKind::ConvState, runner.conv_state_device_ptr, runner.conv_state_bytesize)
          restore_recurrent_record(layer_id, ML::GGUF::Qwen35StateSnapshot::RecordKind::SsmState, runner.ssm_state_device_ptr, runner.ssm_state_bytesize)
        in QwenFullAttnLayerRunner
          # KV stays raw for recurrent compressed artifacts; copy it after the
          # recurrent decode timer so attribution matches the probe split.
        end
      end
      CUDA.synchronize!("cuCtxSynchronize(qwen encoded artifact recurrent restore)") if @has_i8 || @has_bf16
      @kernel_ms = (Time.instant - t_kernel).total_milliseconds

      t_kv = Time.instant
      restore_kv_records(stack)
      @kv_ms = (Time.instant - t_kv).total_milliseconds
      @restore_ms = (Time.instant - t_total).total_milliseconds
    end

    def close : Nil
      return if @closed

      @i8_quant_device.try(&.close)
      @bf16_payload_device.try(&.close)
      @module.try(&.close)
      @closed = true
    end

    private def upload_encoded_payloads : Nil
      t_h2d = Time.instant
      if device = @i8_quant_device
        @segments.each_value do |segment|
          next unless segment.codec.block_i8?

          CUDA.copy_htod!(device.ptr + segment.payload_offset.to_u64, segment.payload.to_unsafe.as(Void*), segment.payload.size.to_u64, "qwen encoded artifact i8 interleaved")
        end
      end
      if device = @bf16_payload_device
        @segments.each_value do |segment|
          next unless segment.codec.bf16?

          CUDA.copy_htod!(device.ptr + segment.payload_offset.to_u64, segment.payload.to_unsafe.as(Void*), segment.payload.size.to_u64, "qwen encoded artifact bf16")
        end
      end
      @h2d_ms = (Time.instant - t_h2d).total_milliseconds
    end

    private def add_bf16_segment(record : ML::GGUF::Qwen35StateSnapshot::EncodedRecord) : Nil
      raise ArgumentError.new("corrupt BF16 recurrent artifact payload") unless record.payload.size * 2 == record.original_byte_size

      offset = @bf16_payload_bytes_total
      @bf16_payload_bytes_total += record.payload.size
      @segments[{record.layer, record.kind}] = Segment.new(record.layer, record.kind, record.codec, offset, record.original_byte_size // sizeof(Float32), record.payload)
      @has_bf16 = true
    end

    private def add_i8_segment(record : ML::GGUF::Qwen35StateSnapshot::EncodedRecord) : Nil
      block_size = @snapshot.codec_block
      raise ArgumentError.new("block INT8 recurrent artifact requires positive block size") unless block_size > 0

      values = record.original_byte_size // sizeof(Float32)
      q_offset = @i8_payload_bytes_total
      full_blocks = values // block_size
      tail = values % block_size
      expected_payload = full_blocks * (sizeof(Float32) + block_size)
      expected_payload += sizeof(Float32) + tail if tail > 0
      raise ArgumentError.new("corrupt INT8 recurrent artifact payload") unless record.payload.size == expected_payload

      @i8_payload_bytes_total += record.payload.size
      @segments[{record.layer, record.kind}] = Segment.new(record.layer, record.kind, record.codec, q_offset, values, record.payload)
      @has_i8 = true
    end

    private def restore_recurrent_record(layer_id : Int32,
                                         kind : ML::GGUF::Qwen35StateSnapshot::RecordKind,
                                         dst : DevicePtr,
                                         expected_bytesize : LibC::SizeT) : Nil
      record = encoded_record(layer_id, kind)
      raise ArgumentError.new("recurrent artifact size mismatch: layer=#{layer_id}, kind=#{kind}") unless record.original_byte_size.to_u64 == expected_bytesize.to_u64

      case record.codec
      in ML::GGUF::Qwen35StateSnapshot::RecordCodec::RawF32
        t_raw = Time.instant
        CUDA.copy_htod!(dst, record.payload.to_unsafe.as(Void*), expected_bytesize, "restore raw recurrent encoded artifact")
        @raw_recurrent_ms += (Time.instant - t_raw).total_milliseconds
      in ML::GGUF::Qwen35StateSnapshot::RecordCodec::Bf16
        segment = segment_for(layer_id, kind)
        @bf16_payload_ptr.value = @bf16_payload_device.not_nil!.ptr + segment.payload_offset.to_u64
        @bf16_out_ptr.value = dst
        @bf16_values_ptr.value = segment.values.to_u32
        launch_values(@bf16_fn.not_nil!, segment.values, @bf16_params, "qwen encoded artifact bf16 decode")
      in ML::GGUF::Qwen35StateSnapshot::RecordCodec::BlockI8
        segment = segment_for(layer_id, kind)
        @i8_q_ptr.value = @i8_quant_device.not_nil!.ptr + segment.payload_offset.to_u64
        @i8_out_ptr.value = dst
        @i8_values_ptr.value = segment.values.to_u32
        launch_values(@i8_fn.not_nil!, segment.values, @i8_params, "qwen encoded artifact i8 decode")
      end
    end

    private def restore_kv_records(stack : QwenMixedStackRunner) : Nil
      stack.layer_ids.zip(stack.runners).each do |layer_id, runner|
        next unless runner.is_a?(QwenFullAttnLayerRunner)

        position = @snapshot.positions[layer_id]
        restore_kv_record(runner, encoded_record(layer_id, ML::GGUF::Qwen35StateSnapshot::RecordKind::KCache), position, runner.k_cache_device_ptr, "restore encoded artifact k_cache")
        restore_kv_record(runner, encoded_record(layer_id, ML::GGUF::Qwen35StateSnapshot::RecordKind::VCache), position, runner.v_cache_device_ptr, "restore encoded artifact v_cache")
        runner.update_decode_position(position)
      end
    end

    private def restore_kv_record(runner : QwenFullAttnLayerRunner,
                                  record : ML::GGUF::Qwen35StateSnapshot::EncodedRecord,
                                  position : Int32,
                                  dst : DevicePtr,
                                  label : String) : Nil
      raise ArgumentError.new("#{label} must be raw in recurrent artifact") unless record.codec.raw_f32?
      actual = record.payload.size.to_u64
      full_bytesize = runner.kv_cache_bytesize.to_u64
      live_bytesize = runner.kv_cache_bytesize_for_tokens(position).to_u64
      unless actual == full_bytesize || actual == live_bytesize
        raise ArgumentError.new("#{label} size mismatch: artifact=#{actual}, runner_full=#{full_bytesize}, runner_live=#{live_bytesize}")
      end

      CUDA.copy_htod!(dst, record.payload.to_unsafe.as(Void*), actual, label)
    end

    private def encoded_record(layer : Int32, kind : ML::GGUF::Qwen35StateSnapshot::RecordKind) : ML::GGUF::Qwen35StateSnapshot::EncodedRecord
      @records[{layer, kind}]? || raise ArgumentError.new("missing Qwen encoded state record: layer=#{layer}, kind=#{kind}")
    end

    private def segment_for(layer : Int32, kind : ML::GGUF::Qwen35StateSnapshot::RecordKind) : Segment
      @segments[{layer, kind}]? || raise ArgumentError.new("missing Qwen encoded recurrent segment: layer=#{layer}, kind=#{kind}")
    end

    private def recurrent_record_kind?(kind : ML::GGUF::Qwen35StateSnapshot::RecordKind) : Bool
      case kind
      in ML::GGUF::Qwen35StateSnapshot::RecordKind::ConvState, ML::GGUF::Qwen35StateSnapshot::RecordKind::SsmState
        true
      in ML::GGUF::Qwen35StateSnapshot::RecordKind::KCache, ML::GGUF::Qwen35StateSnapshot::RecordKind::VCache
        false
      end
    end

    private def launch_values(fn : KernelFunction, values : Int32, params : Pointer(Void*), label : String) : Nil
      grid = ((values + 255) // 256).to_u32
      CUDA.launch!(fn, grid, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, params, label)
    end
  end
end
