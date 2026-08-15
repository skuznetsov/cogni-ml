require "./qwen_qbit_gaussian_codec"
require "./qwen_qbit_native_block"
require "./qwen35_metal"

{% unless flag?(:cpu_only) %}
  require "../metal/device"
  require "../metal/dispatch"
  require "../core/buffer"
{% end %}

module ML::GGUF
  # Direct p7 decoder for the experimental QBit cache path. One Metal dispatch
  # performs bit-plane reconstruction, centroid lookup, affine denormalization,
  # and the final write into a reusable Float32 state buffer.
  module QwenQBitMetalRestore
    extend self

    record Job,
      encoded : QwenQBitGaussianCodec::Encoded,
      destination : ML::MetalBuffer

    record NativeJob,
      span : QwenQBitNativeBlock::RecordSpan,
      destination : ML::MetalBuffer

    record NativeStreamJob,
      span : QwenQBitNativeBlock::StreamRecordSpan,
      destination : ML::MetalBuffer

    {% unless flag?(:cpu_only) %}
      SOURCE = {{ read_file("#{__DIR__}/kernels/qbit_restore_qwen35.metal") }}
      @@pipeline : ML::Metal::ComputePipeline?
      @@native_pipeline : ML::Metal::ComputePipeline?
    {% end %}

    def decode(encoded : QwenQBitGaussianCodec::Encoded,
               reusable : ML::MetalBuffer? = nil) : ML::MetalBuffer
      QwenQBitGaussianCodec.validate(encoded)
      raise ArgumentError.new("QBit Metal restore requires p7 precision") unless encoded.precision == 7

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        raise "Metal not available" unless Qwen35Metal.available?
        byte_size = encoded.value_count.to_i64 * sizeof(Float32)
        raise ArgumentError.new("QBit Metal restore cannot decode an empty payload") if byte_size == 0
        dst = reusable
        if dst.nil? || dst.size != byte_size
          dst = ML::MetalBuffer.new(byte_size, reusable.try(&.storage_mode) || ML::StorageMode::Shared)
        end

        decode_into([Job.new(encoded, dst)])
        dst
      {% end %}
    end

    def decode_into(jobs : Array(Job)) : Nil
      raise ArgumentError.new("QBit Metal restore batch must not be empty") if jobs.empty?
      jobs.each do |job|
        encoded = job.encoded
        QwenQBitGaussianCodec.validate(encoded)
        raise ArgumentError.new("QBit Metal restore requires p7 precision") unless encoded.precision == 7
        expected_size = encoded.value_count.to_i64 * sizeof(Float32)
        raise ArgumentError.new("QBit Metal restore destination size mismatch") unless job.destination.size == expected_size
      end

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        raise "Metal not available" unless Qwen35Metal.available?
        sources = [] of ML::MetalBuffer
        begin
          jobs.each do |job|
            payload = job.encoded.payload
            src = ML::MetalBuffer.new(payload.size.to_i64, ML::StorageMode::Shared)
            sources << src
            src.write_bytes(payload.to_unsafe, payload.size)
          end
          dispatch(sources, jobs)
        ensure
          sources.each(&.release)
        end
      {% end %}
    end

    # Restores directly from the plane-major ClickHouse Native/QBit layout.
    # The full block is uploaded once; every record is decoded into its own
    # prepared state buffer within one command buffer.
    def decode_native_into(block : QwenQBitNativeBlock::Parsed, jobs : Array(NativeJob)) : Nil
      raise ArgumentError.new("QBit Native Metal restore batch must not be empty") if jobs.empty?
      jobs.each do |job|
        span = job.span
        unless block.record_spans.includes?(span)
          raise ArgumentError.new("QBit Native Metal restore span is not present in the parsed block")
        end
        end_row = span.row_start.to_i64 + span.tile_count
        unless span.row_start >= 0 && span.tile_count > 0 && end_row <= block.row_count
          raise ArgumentError.new("QBit Native Metal restore row span is invalid")
        end
        minimum_count = (span.tile_count.to_i64 - 1) * block.block_size + 1
        maximum_count = span.tile_count.to_i64 * block.block_size
        unless span.value_count >= minimum_count && span.value_count <= maximum_count
          raise ArgumentError.new("QBit Native Metal restore value count is invalid")
        end
        expected_size = span.value_count.to_i64 * sizeof(Float32)
        raise ArgumentError.new("QBit Native Metal restore destination size mismatch") unless job.destination.size == expected_size
      end

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        raise "Metal not available" unless Qwen35Metal.available?
        source = ML::MetalBuffer.new(block.bytes.size.to_i64, ML::StorageMode::Shared)
        begin
          source.write_bytes(block.bytes.to_unsafe, block.bytes.size)
          dispatch_native(source, block, jobs)
        ensure
          source.release
        end
      {% end %}
    end

    # Restores records from an ordered sequence of ClickHouse Native blocks.
    # A logical record may cross block boundaries; every chunk writes at its
    # validated value offset in the same destination buffer and all chunks are
    # submitted through one command buffer.
    def decode_native_stream_into(stream : QwenQBitNativeBlock::Stream,
                                  jobs : Array(NativeStreamJob)) : Nil
      raise ArgumentError.new("QBit Native stream Metal restore batch must not be empty") if jobs.empty?
      jobs.each do |job|
        span = job.span
        unless stream.record_spans.includes?(span)
          raise ArgumentError.new("QBit Native stream Metal restore span is not present in the parsed stream")
        end
        unless span.tile_count > 0 && span.value_count > 0 && !span.chunks.empty?
          raise ArgumentError.new("QBit Native stream Metal restore span is invalid")
        end
        minimum_count = (span.tile_count.to_i64 - 1) * stream.block_size + 1
        maximum_count = span.tile_count.to_i64 * stream.block_size
        unless span.value_count >= minimum_count && span.value_count <= maximum_count
          raise ArgumentError.new("QBit Native stream Metal restore value count is invalid")
        end
        expected_size = span.value_count.to_i64 * sizeof(Float32)
        unless job.destination.size == expected_size
          raise ArgumentError.new("QBit Native stream Metal restore destination size mismatch")
        end

        expected_tile = 0_i64
        expected_value = 0_i64
        span.chunks.each do |chunk|
          unless chunk.block_index >= 0 && chunk.block_index < stream.blocks.size
            raise ArgumentError.new("QBit Native stream Metal restore block index is invalid")
          end
          block = stream.blocks[chunk.block_index]
          end_row = chunk.row_start.to_i64 + chunk.tile_count
          unless chunk.row_start >= 0 && chunk.tile_count > 0 && end_row <= block.row_count
            raise ArgumentError.new("QBit Native stream Metal restore row span is invalid")
          end
          unless chunk.tile_start == expected_tile && chunk.value_start == expected_value
            raise ArgumentError.new("QBit Native stream Metal restore chunk sequence is invalid")
          end
          minimum_chunk_count = (chunk.tile_count.to_i64 - 1) * stream.block_size + 1
          maximum_chunk_count = chunk.tile_count.to_i64 * stream.block_size
          unless chunk.value_count >= minimum_chunk_count && chunk.value_count <= maximum_chunk_count
            raise ArgumentError.new("QBit Native stream Metal restore chunk value count is invalid")
          end
          expected_tile += chunk.tile_count
          expected_value += chunk.value_count
        end
        unless expected_tile == span.tile_count && expected_value == span.value_count
          raise ArgumentError.new("QBit Native stream Metal restore chunk totals mismatch")
        end
      end

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        raise "Metal not available" unless Qwen35Metal.available?
        sources = [] of ML::MetalBuffer
        begin
          stream.blocks.each do |block|
            source = ML::MetalBuffer.new(block.bytes.size.to_i64, ML::StorageMode::Shared)
            sources << source
            source.write_bytes(block.bytes.to_unsafe, block.bytes.size)
          end
          dispatch_native_stream(sources, stream, jobs)
        ensure
          sources.each(&.release)
        end
      {% end %}
    end

    {% unless flag?(:cpu_only) %}
      private def pipeline : ML::Metal::ComputePipeline
        @@pipeline ||= ML::Metal::PipelineCache.get("qwen35_qbit_p7_decode_f32") {
          ML::Metal::ComputePipeline.new("qwen35_qbit_p7_decode_f32", SOURCE)
        }
      end

      private def native_pipeline : ML::Metal::ComputePipeline
        @@native_pipeline ||= ML::Metal::PipelineCache.get("qwen35_qbit_p7_native_decode_f32") {
          ML::Metal::ComputePipeline.new("qwen35_qbit_p7_native_decode_f32", SOURCE)
        }
      end

      private def dispatch(sources : Array(ML::MetalBuffer), jobs : Array(Job)) : Nil
        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        enc.set_pipeline(pipeline)
        jobs.each_with_index do |job, i|
          count = job.encoded.value_count
          enc.set_buffer(sources[i], 0)
          enc.set_buffer(job.destination, 1)
          enc.set_value(count.to_u32, 2)
          enc.set_value(job.encoded.block_size.to_u32, 3)
          enc.dispatch_1d(count, 256)
        end
        enc.end_encoding
        cmd.commit_and_wait
      end

      private def dispatch_native(source : ML::MetalBuffer,
                                  block : QwenQBitNativeBlock::Parsed,
                                  jobs : Array(NativeJob)) : Nil
        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        enc.set_pipeline(native_pipeline)
        jobs.each do |job|
          span = job.span
          enc.set_buffer(source, 0)
          enc.set_buffer(job.destination, 1)
          enc.set_value(span.value_count.to_u32, 2)
          enc.set_value(block.block_size.to_u32, 3)
          enc.set_value(span.row_start.to_u32, 4)
          enc.set_value(block.row_count.to_u32, 5)
          enc.set_value(block.mean_offset.to_u32, 6)
          enc.set_value(block.sigma_offset.to_u32, 7)
          enc.set_value(block.codes_offset.to_u32, 8)
          enc.dispatch_1d((span.value_count + 7) // 8, 256)
        end
        enc.end_encoding
        cmd.commit_and_wait
      end

      private def dispatch_native_stream(sources : Array(ML::MetalBuffer),
                                         stream : QwenQBitNativeBlock::Stream,
                                         jobs : Array(NativeStreamJob)) : Nil
        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        enc.set_pipeline(native_pipeline)
        jobs.each do |job|
          job.span.chunks.each do |chunk|
            block = stream.blocks[chunk.block_index]
            enc.set_buffer(sources[chunk.block_index], 0)
            enc.set_buffer(
              job.destination,
              1,
              ML::Metal::BufferAccess::Write,
              offset: chunk.value_start.to_i64 * sizeof(Float32),
            )
            enc.set_value(chunk.value_count.to_u32, 2)
            enc.set_value(block.block_size.to_u32, 3)
            enc.set_value(chunk.row_start.to_u32, 4)
            enc.set_value(block.row_count.to_u32, 5)
            enc.set_value(block.mean_offset.to_u32, 6)
            enc.set_value(block.sigma_offset.to_u32, 7)
            enc.set_value(block.codes_offset.to_u32, 8)
            enc.dispatch_1d((chunk.value_count + 7) // 8, 256)
          end
        end
        enc.end_encoding
        cmd.commit_and_wait
      end
    {% end %}
  end
end
