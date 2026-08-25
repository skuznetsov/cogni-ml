require "./qwen35_state_snapshot"
require "./qwen_qbit_gaussian_codec"
require "./qwen_qbit_metal_restore"
require "./qwen_qbit_native_writer"

module ML::GGUF
  # Default-off state wrapper for the p7 cache experiment. Live KV stays raw;
  # recurrent Conv/SSM records use QBit tiles and may restore directly into the
  # prepared Metal buffers consumed by Qwen35CPU.
  module QwenQBitStateSnapshot
    extend self

    class AdaptiveRestoreDeviceError < Exception
    end

    MIN_STATE_PRECISION           =   6
    MAX_STATE_PRECISION           =   8
    ADAPTIVE_RESTORE_CHUNK_TOKENS = 512

    alias RecordKind = Qwen35StateSnapshot::RecordKind

    record EncodedRecord,
      layer : Int32,
      kind : RecordKind,
      storage_mode : ML::StorageMode,
      original_byte_size : Int32,
      raw : Bytes?,
      qbit : QwenQBitGaussianCodec::Encoded?

    class Snapshot
      getter max_seq : Int32
      getter layer_count : Int32
      getter positions : Array(Int32)
      getter records : Array(EncodedRecord)
      getter block_size : Int32
      getter precision : Int32
      getter backing_stores : Array(Bytes)

      def initialize(@max_seq : Int32,
                     @layer_count : Int32,
                     @positions : Array(Int32),
                     @records : Array(EncodedRecord),
                     @block_size : Int32,
                     @precision : Int32,
                     @backing_stores : Array(Bytes) = [] of Bytes)
      end

      def payload_byte_size : Int64
        @records.sum(0_i64) do |record|
          if payload = record.raw
            payload.size.to_i64
          else
            record.qbit.not_nil!.payload.size.to_i64
          end
        end
      end
    end

    record NativeRecurrentEncoding,
      bytes : Bytes,
      record_count : Int32,
      source_byte_size : Int64,
      peak_source_record_bytes : Int64

    # Lazy HTTP request body. Each read holds at most one captured F32 record
    # and one encoded Native block; completed blocks contribute only compact
    # identities and an incremental logical digest.
    class NativeRecurrentBody < IO
      getter byte_size : Int64
      getter record_count : Int32
      getter source_byte_size : Int64
      getter peak_source_record_bytes : Int64

      @next_record : Proc(Qwen35StateSnapshot::Record?)
      @summary_builder : QwenQBitNativeBlock::SummaryBuilder
      @summary : QwenQBitNativeBlock::Summary?
      @seen : Set({Int32, UInt8})
      @block : Bytes
      @block_offset : Int32
      @eof : Bool

      def self.for_state(state : Qwen35CPU::State,
                         cache_id : UInt64 = 0_u64,
                         block_size : Int32 = 1024,
                         precision : Int32 = 7) : self
        new(Qwen35StateSnapshot.recurrent_record_source(state), cache_id, block_size, precision)
      end

      def initialize(@next_record : Proc(Qwen35StateSnapshot::Record?),
                     @cache_id : UInt64 = 0_u64,
                     @block_size : Int32 = 1024,
                     @precision : Int32 = 7)
        validate_native_parameters!(@block_size, @precision)
        @summary_builder = QwenQBitNativeBlock::SummaryBuilder.new(@block_size)
        @summary = nil
        @seen = Set({Int32, UInt8}).new
        @block = Bytes.empty
        @block_offset = 0
        @byte_size = 0_i64
        @record_count = 0
        @source_byte_size = 0_i64
        @peak_source_record_bytes = 0_i64
        @eof = false
      end

      def read(slice : Bytes) : Int32
        return 0 if slice.empty? || @eof
        load_next_block if @block_offset == @block.size
        return 0 if @eof

        count = Math.min(slice.size, @block.size - @block_offset)
        slice[0, count].copy_from(@block[@block_offset, count])
        @block_offset += count
        count
      end

      def write(slice : Bytes) : NoReturn
        raise IO::Error.new("QBit recurrent request body is read-only")
      end

      def summary : QwenQBitNativeBlock::Summary
        @summary || raise ArgumentError.new("QBit recurrent request body was not fully consumed")
      end

      private def load_next_block : Nil
        if record = @next_record.call
          key = {record.layer, record.kind.value}
          raise ArgumentError.new("duplicate QBit Native recurrent record") unless @seen.add?(key)
          block = QwenQBitStateSnapshot.encode_native_recurrent_record(
            record,
            @cache_id,
            @block_size,
            @precision,
          )
          @summary_builder.append(block)
          @block = block
          @block_offset = 0
          @byte_size += block.size
          @record_count += 1
          source_bytes = record.bytes.size.to_i64
          @source_byte_size += source_bytes
          @peak_source_record_bytes = source_bytes if source_bytes > @peak_source_record_bytes
        else
          @summary = @summary_builder.finish
          @block = Bytes.empty
          @block_offset = 0
          @eof = true
        end
      end

      private def validate_native_parameters!(block_size : Int32, precision : Int32) : Nil
        unless block_size > 0 && block_size <= UInt16::MAX && block_size % 8 == 0
          raise ArgumentError.new("QBit Native block size must be a positive UInt16 multiple of 8")
        end
        raise ArgumentError.new("QBit Native state encoding requires p7 precision") unless precision == 7
      end
    end

    # Builds one legal Native block per recurrent record. The final transport
    # body must remain resident until ClickHouse accepts it, but F32 capture,
    # Float32 conversion, and QBit payload ownership are bounded to one record.
    class NativeRecurrentStreamEncoder
      @io : IO::Memory
      @seen : Set({Int32, UInt8})
      @record_count : Int32
      @source_byte_size : Int64
      @peak_source_record_bytes : Int64
      @finished : Bool

      def initialize(@cache_id : UInt64 = 0_u64,
                     @block_size : Int32 = 1024,
                     @precision : Int32 = 7)
        unless @block_size > 0 && @block_size <= UInt16::MAX && @block_size % 8 == 0
          raise ArgumentError.new("QBit Native block size must be a positive UInt16 multiple of 8")
        end
        raise ArgumentError.new("QBit Native state encoding requires p7 precision") unless @precision == 7
        @io = IO::Memory.new
        @seen = Set({Int32, UInt8}).new
        @record_count = 0
        @source_byte_size = 0_i64
        @peak_source_record_bytes = 0_i64
        @finished = false
      end

      def append(record : Qwen35StateSnapshot::Record) : Nil
        raise ArgumentError.new("QBit Native recurrent stream encoder is finished") if @finished
        unless record.kind.conv_state? || record.kind.ssm_state?
          raise ArgumentError.new("QBit Native recurrent stream accepts recurrent records only")
        end
        key = {record.layer, record.kind.value}
        raise ArgumentError.new("duplicate QBit Native recurrent record") if @seen.includes?(key)

        block = QwenQBitStateSnapshot.encode_native_recurrent_record(
          record,
          @cache_id,
          @block_size,
          @precision,
        )
        @io.write(block)
        @seen.add(key)
        @record_count += 1
        source_bytes = record.bytes.size.to_i64
        @source_byte_size += source_bytes
        @peak_source_record_bytes = source_bytes if source_bytes > @peak_source_record_bytes
      end

      def finish : NativeRecurrentEncoding
        raise ArgumentError.new("QBit Native recurrent stream encoder is finished") if @finished
        raise ArgumentError.new("QBit Native recurrent stream must not be empty") if @record_count == 0
        @finished = true
        NativeRecurrentEncoding.new(
          @io.to_slice,
          @record_count,
          @source_byte_size,
          @peak_source_record_bytes,
        )
      end
    end

    def encode_native_recurrent_record(record : Qwen35StateSnapshot::Record,
                                       cache_id : UInt64 = 0_u64,
                                       block_size : Int32 = 1024,
                                       precision : Int32 = 7) : Bytes
      unless record.kind.conv_state? || record.kind.ssm_state?
        raise ArgumentError.new("QBit Native recurrent stream accepts recurrent records only")
      end
      unless block_size > 0 && block_size <= UInt16::MAX && block_size % 8 == 0
        raise ArgumentError.new("QBit Native block size must be a positive UInt16 multiple of 8")
      end
      raise ArgumentError.new("QBit Native state encoding requires p7 precision") unless precision == 7

      encoded = begin
        QwenQBitGaussianCodec.encode(floats_from(record.bytes), block_size, precision)
      rescue ex : ArgumentError
        raise ArgumentError.new(
          "QBit #{record.kind} layer #{record.layer} encoding failed: #{ex.message}"
        )
      end
      QwenQBitNativeWriter.encode([
        QwenQBitNativeWriter::Record.new(cache_id, record.layer, record.kind.value, encoded),
      ])
    end

    def encode(snapshot : Qwen35StateSnapshot::Snapshot,
               block_size : Int32 = 1024,
               precision : Int32 = 7) : Snapshot
      unless precision >= MIN_STATE_PRECISION && precision <= MAX_STATE_PRECISION
        raise ArgumentError.new("QBit state snapshot precision is unsupported")
      end
      records = snapshot.records.map do |record|
        if recurrent_record?(record.kind)
          encoded = begin
            QwenQBitGaussianCodec.encode(floats_from(record.bytes), block_size, precision)
          rescue ex : ArgumentError
            raise ArgumentError.new(
              "QBit #{record.kind} layer #{record.layer} encoding failed: #{ex.message}"
            )
          end
          EncodedRecord.new(record.layer, record.kind, record.storage_mode, record.bytes.size.to_i32, nil, encoded)
        else
          EncodedRecord.new(record.layer, record.kind, record.storage_mode, record.bytes.size.to_i32, record.bytes, nil)
        end
      end
      encoded = Snapshot.new(snapshot.max_seq, snapshot.layer_count, snapshot.positions.dup, records, block_size, precision)
      validate(encoded)
      encoded
    end

    # Replaces the in-memory exact records with a validated raw KV-only
    # artifact. The returned snapshot retains the artifact backing store so its
    # zero-copy payload slices stay live through device restore.
    def with_exact_artifact(snapshot : Snapshot,
                            artifact : Qwen35StateSnapshot::EncodedSnapshot) : Snapshot
      validate(snapshot)
      raise ArgumentError.new("QBit exact artifact max_seq mismatch") unless artifact.max_seq == snapshot.max_seq
      raise ArgumentError.new("QBit exact artifact layer count mismatch") unless artifact.layer_count == snapshot.layer_count
      raise ArgumentError.new("QBit exact artifact positions mismatch") unless artifact.positions == snapshot.positions
      raise ArgumentError.new("QBit exact artifact must use the raw codec") unless artifact.codec.raw_f32?

      exact_records = Hash({Int32, UInt8}, Qwen35StateSnapshot::EncodedRecord).new
      artifact.records.each do |record|
        raise ArgumentError.new("QBit exact artifact must be KV-only") if recurrent_record?(record.kind)
        raise ArgumentError.new("QBit exact artifact record must stay raw") unless record.codec.raw_f32?
        raise ArgumentError.new("QBit exact artifact record is truncated") unless record.payload.size == record.original_byte_size
        key = {record.layer, record.kind.value}
        raise ArgumentError.new("duplicate QBit exact artifact record") if exact_records.has_key?(key)
        exact_records[key] = record
      end

      expected_keys = snapshot.records.compact_map do |record|
        record.raw ? {record.layer, record.kind.value} : nil
      end.to_set
      raise ArgumentError.new("QBit exact artifact record set mismatch") unless exact_records.keys.to_set == expected_keys

      records = snapshot.records.map do |record|
        if record.raw
          exact = exact_records[{record.layer, record.kind.value}]
          raise ArgumentError.new("QBit exact artifact storage mode mismatch") unless exact.storage_mode == record.storage_mode
          raise ArgumentError.new("QBit exact artifact byte size mismatch") unless exact.original_byte_size == record.original_byte_size
          EncodedRecord.new(record.layer, record.kind, record.storage_mode, record.original_byte_size, exact.payload, nil)
        else
          record
        end
      end
      # `snapshot` was fully validated above; every replaced exact record has
      # now been checked against its expected key, mode, and byte size. Avoid a
      # second scan of the unchanged recurrent QBit tiles on the cache-hit path.
      backing_stores = snapshot.backing_stores + artifact.backing_stores
      attached = Snapshot.new(
        snapshot.max_seq,
        snapshot.layer_count,
        snapshot.positions.dup,
        records,
        snapshot.block_size,
        snapshot.precision,
        backing_stores,
      )
      attached
    end

    def decode(snapshot : Snapshot) : Qwen35StateSnapshot::Snapshot
      validate(snapshot)
      records = snapshot.records.map do |record|
        bytes = if raw = record.raw
                  raw
                else
                  bytes_from(QwenQBitGaussianCodec.decode(record.qbit.not_nil!))
                end
        Qwen35StateSnapshot::Record.new(record.layer, record.kind, bytes, record.storage_mode)
      end
      Qwen35StateSnapshot::Snapshot.new(snapshot.max_seq, snapshot.layer_count, snapshot.positions.dup, records)
    end

    def encode_native_recurrent(snapshot : Snapshot, cache_id : UInt64 = 0_u64) : Bytes
      validate(snapshot)
      raise ArgumentError.new("QBit Native state encoding requires p7 precision") unless snapshot.precision == 7
      records = snapshot.records.compact_map do |record|
        if encoded = record.qbit
          QwenQBitNativeWriter::Record.new(cache_id, record.layer, record.kind.value, encoded)
        end
      end
      QwenQBitNativeWriter.encode(records)
    end

    def encode_native_recurrent_streaming(state : Qwen35CPU::State,
                                          cache_id : UInt64 = 0_u64,
                                          block_size : Int32 = 1024,
                                          precision : Int32 = 7) : NativeRecurrentEncoding
      encoder = NativeRecurrentStreamEncoder.new(cache_id, block_size, precision)
      Qwen35StateSnapshot.each_recurrent_record(state) do |record|
        encoder.append(record)
      end
      encoder.finish
    end

    def restore_into(snapshot : Snapshot,
                     hp : Qwen35Hparams,
                     state : Qwen35CPU::State,
                     prefer_metal : Bool = Qwen35Metal.available?) : Nil
      validate_for_state(snapshot, hp, state)
      if prefer_metal && snapshot.precision != 7
        raise ArgumentError.new("QBit Metal state restore requires p7 precision")
      end
      unless prefer_metal && Qwen35Metal.available?
        Qwen35StateSnapshot.restore_into(decode(snapshot), hp, state, prefer_metal: false)
        return
      end

      {% if flag?(:cpu_only) %}
        Qwen35StateSnapshot.restore_into(decode(snapshot), hp, state, prefer_metal: false)
      {% else %}
        assignments = [] of NamedTuple(record: EncodedRecord, buffer: ML::MetalBuffer)
        jobs = [] of QwenQBitMetalRestore::Job
        snapshot.records.each do |record|
          reusable = state_buffer(state.layers[record.layer], record.kind)
          buffer = reusable
          if buffer.nil? || buffer.size != record.original_byte_size || buffer.storage_mode != record.storage_mode
            buffer = ML::MetalBuffer.new(record.original_byte_size.to_i64, record.storage_mode)
          end

          if raw = record.raw
            buffer.write_bytes(raw.to_unsafe, raw.size)
          else
            jobs << QwenQBitMetalRestore::Job.new(record.qbit.not_nil!, buffer)
          end
          assignments << {record: record, buffer: buffer}
        end
        QwenQBitMetalRestore.decode_into(jobs) unless jobs.empty?

        snapshot.positions.each_with_index { |position, i| state.layers[i].position = position }
        assignments.each { |assignment| assign_state_buffer(state.layers[assignment[:record].layer], assignment[:record].kind, assignment[:buffer]) }
      {% end %}
    end

    # Directly restores recurrent records from a validated plane-major Native
    # block while retaining exact live KV bytes from the snapshot envelope.
    # The caller must discard the target state if this method raises.
    def restore_native_into(snapshot : Snapshot,
                            block : QwenQBitNativeBlock::Parsed,
                            cache_id : UInt64,
                            hp : Qwen35Hparams,
                            state : Qwen35CPU::State) : Nil
      validate_for_state(snapshot, hp, state)
      raise ArgumentError.new("QBit Native state restore requires p7 precision") unless snapshot.precision == 7
      raise ArgumentError.new("QBit Native state block size mismatch") unless block.block_size == snapshot.block_size

      spans = Hash({Int32, UInt8}, QwenQBitNativeBlock::RecordSpan).new
      block.record_spans.each do |span|
        next unless span.cache_id == cache_id
        key = {span.layer, span.kind}
        raise ArgumentError.new("duplicate QBit Native state record") if spans.has_key?(key)
        spans[key] = span
      end

      expected_keys = snapshot.records.compact_map do |record|
        record.qbit ? {record.layer, record.kind.value} : nil
      end.to_set
      raise ArgumentError.new("QBit Native state record set mismatch") unless spans.keys.to_set == expected_keys

      snapshot.records.each do |record|
        next unless record.qbit
        span = spans[{record.layer, record.kind.value}]
        unless span.value_count.to_i64 * sizeof(Float32) == record.original_byte_size
          raise ArgumentError.new("QBit Native state value count mismatch")
        end
      end

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        raise "Metal not available" unless Qwen35Metal.available?
        assignments = [] of NamedTuple(record: EncodedRecord, buffer: ML::MetalBuffer)
        jobs = [] of QwenQBitMetalRestore::NativeJob
        snapshot.records.each do |record|
          reusable = state_buffer(state.layers[record.layer], record.kind)
          buffer = reusable
          if buffer.nil? || buffer.size != record.original_byte_size || buffer.storage_mode != record.storage_mode
            buffer = ML::MetalBuffer.new(record.original_byte_size.to_i64, record.storage_mode)
          end

          if raw = record.raw
            buffer.write_bytes(raw.to_unsafe, raw.size)
          else
            span = spans[{record.layer, record.kind.value}]
            jobs << QwenQBitMetalRestore::NativeJob.new(span, buffer)
          end
          assignments << {record: record, buffer: buffer}
        end
        QwenQBitMetalRestore.decode_native_into(block, jobs)

        snapshot.positions.each_with_index { |position, i| state.layers[i].position = position }
        assignments.each { |assignment| assign_state_buffer(state.layers[assignment[:record].layer], assignment[:record].kind, assignment[:buffer]) }
      {% end %}
    end

    # Directly restores recurrent records from a validated ordered Native
    # response stream. Logical records may cross ClickHouse block boundaries;
    # live KV still comes byte-exact from the snapshot envelope.
    def restore_native_stream_into(snapshot : Snapshot,
                                   stream : QwenQBitNativeBlock::Stream,
                                   cache_id : UInt64,
                                   hp : Qwen35Hparams,
                                   state : Qwen35CPU::State) : Nil
      validate_for_state(snapshot, hp, state)
      raise ArgumentError.new("QBit Native state restore requires p7 precision") unless snapshot.precision == 7
      raise ArgumentError.new("QBit Native state block size mismatch") unless stream.block_size == snapshot.block_size
      unless stream.record_spans.all? { |span| span.cache_id == cache_id }
        raise ArgumentError.new("unexpected QBit Native state cache identity")
      end

      spans = Hash({Int32, UInt8}, QwenQBitNativeBlock::StreamRecordSpan).new
      stream.record_spans.each do |span|
        next unless span.cache_id == cache_id
        key = {span.layer, span.kind}
        raise ArgumentError.new("duplicate QBit Native state record") if spans.has_key?(key)
        spans[key] = span
      end

      expected_keys = snapshot.records.compact_map do |record|
        record.qbit ? {record.layer, record.kind.value} : nil
      end.to_set
      raise ArgumentError.new("QBit Native state record set mismatch") unless spans.keys.to_set == expected_keys

      snapshot.records.each do |record|
        next unless record.qbit
        span = spans[{record.layer, record.kind.value}]
        unless span.value_count.to_i64 * sizeof(Float32) == record.original_byte_size
          raise ArgumentError.new("QBit Native state value count mismatch")
        end
      end

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        raise "Metal not available" unless Qwen35Metal.available?
        assignments = [] of NamedTuple(record: EncodedRecord, buffer: ML::MetalBuffer)
        jobs = [] of QwenQBitMetalRestore::NativeStreamJob
        snapshot.records.each do |record|
          reusable = state_buffer(state.layers[record.layer], record.kind)
          buffer = reusable
          if buffer.nil? || buffer.size != record.original_byte_size || buffer.storage_mode != record.storage_mode
            buffer = ML::MetalBuffer.new(record.original_byte_size.to_i64, record.storage_mode)
          end

          if raw = record.raw
            buffer.write_bytes(raw.to_unsafe, raw.size)
          else
            span = spans[{record.layer, record.kind.value}]
            jobs << QwenQBitMetalRestore::NativeStreamJob.new(span, buffer)
          end
          assignments << {record: record, buffer: buffer}
        end
        QwenQBitMetalRestore.decode_native_stream_into(stream, jobs)

        snapshot.positions.each_with_index { |position, i| state.layers[i].position = position }
        assignments.each { |assignment| assign_state_buffer(state.layers[assignment[:record].layer], assignment[:record].kind, assignment[:buffer]) }
      {% end %}
    end

    # Cold-runtime restore path: the strict cache envelope has already admitted
    # a Native recurrent stream and a raw exact-KV artifact, so no source-side
    # QBit Snapshot template exists in this process. Validate their complete
    # model-derived layout before mutating the prepared target state.
    # The caller must discard the target state if this method raises.
    def restore_admitted_native_stream_into(stream : QwenQBitNativeBlock::Stream,
                                            exact : Qwen35StateSnapshot::EncodedSnapshot,
                                            cache_id : UInt64,
                                            hp : Qwen35Hparams,
                                            state : Qwen35CPU::State) : Nil
      raise ArgumentError.new("QBit admitted state layer count mismatch") unless exact.layer_count == hp.n_layer && state.layers.size == hp.n_layer
      raise ArgumentError.new("QBit admitted state max_seq mismatch") unless exact.max_seq == state.max_seq
      raise ArgumentError.new("QBit admitted state position count mismatch") unless exact.positions.size == hp.n_layer
      raise ArgumentError.new("QBit admitted exact artifact must use raw Float32") unless exact.codec.raw_f32?
      unless stream.record_spans.all? { |span| span.cache_id == cache_id }
        raise ArgumentError.new("unexpected QBit Native state cache identity")
      end

      recurrent = Hash({Int32, UInt8}, QwenQBitNativeBlock::StreamRecordSpan).new
      stream.record_spans.each do |span|
        key = {span.layer, span.kind}
        raise ArgumentError.new("duplicate QBit Native state record") if recurrent.has_key?(key)
        recurrent[key] = span
      end
      exact_records = Hash({Int32, UInt8}, Qwen35StateSnapshot::EncodedRecord).new
      exact.records.each do |record|
        unless record.kind.k_cache? || record.kind.v_cache?
          raise ArgumentError.new("QBit admitted exact artifact must be KV-only")
        end
        raise ArgumentError.new("QBit admitted exact record must stay raw") unless record.codec.raw_f32?
        raise ArgumentError.new("QBit admitted exact record is truncated") unless record.payload.size == record.original_byte_size
        key = {record.layer, record.kind.value}
        raise ArgumentError.new("duplicate QBit admitted exact record") if exact_records.has_key?(key)
        exact_records[key] = record
      end

      expected_recurrent = Set({Int32, UInt8}).new
      expected_exact = Set({Int32, UInt8}).new
      kv_record_byte_size = state.max_seq.to_i64 * hp.head_dim * hp.n_head_kv * sizeof(Float32)
      qkv_dim = 2_i64 * hp.ssm_group_count * hp.ssm_state_size + hp.ssm_time_step_rank.to_i64 * hp.ssm_state_size
      conv_record_byte_size = (hp.ssm_conv_kernel - 1).to_i64 * qkv_dim * sizeof(Float32)
      ssm_record_byte_size = hp.ssm_time_step_rank.to_i64 * hp.ssm_state_size * hp.ssm_state_size * sizeof(Float32)
      hp.n_layer.times do |layer|
        if hp.full_attention?(layer)
          expected_exact.add({layer, RecordKind::KCache.value})
          expected_exact.add({layer, RecordKind::VCache.value})
        else
          expected_recurrent.add({layer, RecordKind::ConvState.value})
          expected_recurrent.add({layer, RecordKind::SsmState.value})
        end
      end
      raise ArgumentError.new("QBit admitted recurrent record set mismatch") unless recurrent.keys.to_set == expected_recurrent
      raise ArgumentError.new("QBit admitted exact record set mismatch") unless exact_records.keys.to_set == expected_exact

      recurrent.each do |key, span|
        kind = RecordKind.from_value(key[1])
        expected_size = kind.conv_state? ? conv_record_byte_size : ssm_record_byte_size
        unless span.value_count.to_i64 * sizeof(Float32) == expected_size
          raise ArgumentError.new("QBit admitted recurrent record byte size mismatch")
        end
        buffer = state_buffer(state.layers[key[0]], kind)
        unless buffer && buffer.size == expected_size
          raise ArgumentError.new("QBit admitted recurrent target buffer is not prepared")
        end
      end
      exact_records.each do |key, record|
        unless record.original_byte_size.to_i64 == kv_record_byte_size
          raise ArgumentError.new("QBit admitted exact record byte size mismatch")
        end
        buffer = state_buffer(state.layers[key[0]], record.kind)
        unless buffer && buffer.size == kv_record_byte_size
          raise ArgumentError.new("QBit admitted exact target buffer is not prepared")
        end
      end

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        raise "Metal not available" unless Qwen35Metal.available?
        jobs = [] of QwenQBitMetalRestore::NativeStreamJob
        recurrent.each do |key, span|
          kind = RecordKind.from_value(key[1])
          buffer = state_buffer(state.layers[key[0]], kind).not_nil!
          jobs << QwenQBitMetalRestore::NativeStreamJob.new(span, buffer)
        end

        exact_records.each do |key, record|
          buffer = state_buffer(state.layers[key[0]], record.kind).not_nil!
          buffer.write_bytes(record.payload.to_unsafe, record.payload.size)
        end
        QwenQBitMetalRestore.decode_native_stream_into(stream, jobs)
        exact.positions.each_with_index { |position, layer| state.layers[layer].position = position }
      {% end %}
    end

    # Cold restore into an already allocated adaptive resident state. Recurrent
    # records decode directly from the admitted Native stream into their Metal
    # owners. Each exact live-KV layer is uploaded only long enough for the GPU
    # packer to append it to the resident cache; no Float32 KV owner survives
    # this call. The caller must discard the target state if a device operation
    # raises after validation.
    def restore_admitted_native_stream_into_adaptive(stream : QwenQBitNativeBlock::Stream,
                                                     exact : Qwen35StateSnapshot::EncodedSnapshot,
                                                     cache_id : UInt64,
                                                     hp : Qwen35Hparams,
                                                     state : Qwen35CPU::State) : Nil
      raise ArgumentError.new("QBit adaptive state layer count mismatch") unless exact.layer_count == hp.n_layer && state.layers.size == hp.n_layer
      raise ArgumentError.new("QBit adaptive state max_seq mismatch") unless exact.max_seq == state.max_seq
      raise ArgumentError.new("QBit adaptive state position count mismatch") unless exact.positions.size == hp.n_layer
      raise ArgumentError.new("QBit adaptive exact artifact must use raw Float32") unless exact.codec.raw_f32?
      unless stream.record_spans.all? { |span| span.cache_id == cache_id }
        raise ArgumentError.new("unexpected QBit Native state cache identity")
      end

      live_tokens = exact.positions.first? || 0_i32
      unless live_tokens > 0 && live_tokens <= state.max_seq && exact.positions.all? { |position| position == live_tokens }
        raise ArgumentError.new("QBit adaptive state positions must name one live prefix")
      end
      unless state.adaptive_kv_layer_indices == hp.full_attention_layers
        raise ArgumentError.new("QBit adaptive restore requires every full-attention layer")
      end

      recurrent = Hash({Int32, UInt8}, QwenQBitNativeBlock::StreamRecordSpan).new
      stream.record_spans.each do |span|
        key = {span.layer, span.kind}
        raise ArgumentError.new("duplicate QBit Native state record") if recurrent.has_key?(key)
        recurrent[key] = span
      end
      exact_records = Hash({Int32, UInt8}, Qwen35StateSnapshot::EncodedRecord).new
      exact.records.each do |record|
        unless record.kind.k_cache? || record.kind.v_cache?
          raise ArgumentError.new("QBit adaptive exact artifact must be KV-only")
        end
        raise ArgumentError.new("QBit adaptive exact record must stay raw") unless record.codec.raw_f32?
        key = {record.layer, record.kind.value}
        raise ArgumentError.new("duplicate QBit adaptive exact record") if exact_records.has_key?(key)
        exact_records[key] = record
      end

      expected_recurrent = Set({Int32, UInt8}).new
      expected_exact = Set({Int32, UInt8}).new
      kv_record_byte_size = state.max_seq.to_i64 * hp.head_dim * hp.n_head_kv * sizeof(Float32)
      live_kv_byte_size = live_tokens.to_i64 * hp.head_dim * hp.n_head_kv * sizeof(Float32)
      qkv_dim = 2_i64 * hp.ssm_group_count * hp.ssm_state_size + hp.ssm_time_step_rank.to_i64 * hp.ssm_state_size
      conv_record_byte_size = (hp.ssm_conv_kernel - 1).to_i64 * qkv_dim * sizeof(Float32)
      ssm_record_byte_size = hp.ssm_time_step_rank.to_i64 * hp.ssm_state_size * hp.ssm_state_size * sizeof(Float32)
      hp.n_layer.times do |layer|
        if hp.full_attention?(layer)
          expected_exact.add({layer, RecordKind::KCache.value})
          expected_exact.add({layer, RecordKind::VCache.value})
        else
          expected_recurrent.add({layer, RecordKind::ConvState.value})
          expected_recurrent.add({layer, RecordKind::SsmState.value})
        end
      end
      raise ArgumentError.new("QBit adaptive recurrent record set mismatch") unless recurrent.keys.to_set == expected_recurrent
      raise ArgumentError.new("QBit adaptive exact record set mismatch") unless exact_records.keys.to_set == expected_exact

      recurrent.each do |key, span|
        kind = RecordKind.from_value(key[1])
        expected_size = kind.conv_state? ? conv_record_byte_size : ssm_record_byte_size
        unless span.value_count.to_i64 * sizeof(Float32) == expected_size
          raise ArgumentError.new("QBit adaptive recurrent record byte size mismatch")
        end
        layer = state.layers[key[0]]
        if layer.conv_state || layer.ssm_state
          raise ArgumentError.new("QBit adaptive recurrent state cannot coexist with a CPU owner")
        end
        buffer = state_buffer(layer, kind)
        unless buffer && buffer.size == expected_size
          raise ArgumentError.new("QBit adaptive recurrent target buffer is not prepared")
        end
      end
      exact_records.each do |key, record|
        unless record.original_byte_size.to_i64 == kv_record_byte_size
          raise ArgumentError.new("QBit adaptive exact record byte size mismatch")
        end
        unless record.payload.size.to_i64 == live_kv_byte_size || record.payload.size.to_i64 == kv_record_byte_size
          raise ArgumentError.new("QBit adaptive exact live prefix is truncated")
        end
        layer = state.layers[key[0]]
        if layer.k_cache || layer.v_cache || layer.k_cache_buf || layer.v_cache_buf
          raise ArgumentError.new("QBit adaptive KV cannot coexist with a Float32 owner")
        end
        cache = layer.adaptive_kv.not_nil!
        unless cache.cache_len == 0 && cache.max_seq == state.max_seq &&
               cache.n_head_kv == hp.n_head_kv && cache.head_dim == hp.head_dim
          raise ArgumentError.new("QBit adaptive target cache is not empty or has the wrong shape")
        end
      end

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        raise "Metal not available" unless Qwen35Metal.available?
        begin
          jobs = [] of QwenQBitMetalRestore::NativeStreamJob
          recurrent.each do |key, span|
            kind = RecordKind.from_value(key[1])
            buffer = state_buffer(state.layers[key[0]], kind).not_nil!
            jobs << QwenQBitMetalRestore::NativeStreamJob.new(span, buffer)
          end
          QwenQBitMetalRestore.decode_native_stream_into(stream, jobs)

          hp.full_attention_layers.each do |layer_index|
            k_record = exact_records[{layer_index, RecordKind::KCache.value}]
            v_record = exact_records[{layer_index, RecordKind::VCache.value}]
            token_offset = 0_i32
            while token_offset < live_tokens
              chunk_tokens = Math.min(ADAPTIVE_RESTORE_CHUNK_TOKENS, live_tokens - token_offset)
              chunk_byte_offset = token_offset.to_i64 * hp.head_dim * hp.n_head_kv * sizeof(Float32)
              chunk_byte_size = chunk_tokens.to_i64 * hp.head_dim * hp.n_head_kv * sizeof(Float32)
              k_source = ML::MetalBuffer.new(chunk_byte_size, ML::StorageMode::Shared)
              v_source = ML::MetalBuffer.new(chunk_byte_size, ML::StorageMode::Shared)
              begin
                k_source.write_bytes(k_record.payload[chunk_byte_offset.to_i, chunk_byte_size.to_i].to_unsafe, chunk_byte_size.to_i)
                v_source.write_bytes(v_record.payload[chunk_byte_offset.to_i, chunk_byte_size.to_i].to_unsafe, chunk_byte_size.to_i)
                QwenQBitAdaptiveResidentKV.append_from_metal(
                  state.layers[layer_index].adaptive_kv.not_nil!,
                  k_source,
                  v_source,
                  chunk_tokens,
                )
              ensure
                k_source.release
                v_source.release
              end
              token_offset += chunk_tokens
            end
          end
          exact.positions.each_with_index { |position, layer| state.layers[layer].position = position }
        rescue ex
          raise AdaptiveRestoreDeviceError.new("QBit adaptive device restore failed: #{ex.class}: #{ex.message}")
        end
      {% end %}
    end

    # Restore a cold artifact that already stores the live KV prefix in the
    # canonical adaptive representation. No Float32 KV staging or GPU repack is
    # needed; recurrent p7 still decodes directly into its final Metal owners.
    def restore_admitted_native_stream_into_adaptive_compact(
      stream : QwenQBitNativeBlock::Stream,
      compact : Qwen35StateSnapshot::EncodedSnapshot,
      cache_id : UInt64,
      hp : Qwen35Hparams,
      state : Qwen35CPU::State,
    ) : Nil
      unless compact.layer_count == hp.n_layer && state.layers.size == hp.n_layer
        raise ArgumentError.new("QBit compact adaptive state layer count mismatch")
      end
      raise ArgumentError.new("QBit compact adaptive state max_seq mismatch") unless compact.max_seq == state.max_seq
      raise ArgumentError.new("QBit compact adaptive state position count mismatch") unless compact.positions.size == hp.n_layer
      raise ArgumentError.new("QBit compact adaptive artifact must use raw framing") unless compact.codec.raw_f32?
      unless stream.record_spans.all? { |span| span.cache_id == cache_id }
        raise ArgumentError.new("unexpected QBit Native state cache identity")
      end

      live_tokens = compact.positions.first? || 0_i32
      unless live_tokens > 0 && live_tokens <= state.max_seq &&
             compact.positions.all? { |position| position == live_tokens }
        raise ArgumentError.new("QBit compact adaptive positions must name one live prefix")
      end
      unless state.adaptive_kv_layer_indices == hp.full_attention_layers
        raise ArgumentError.new("QBit compact adaptive restore requires every full-attention layer")
      end

      recurrent = Hash({Int32, UInt8}, QwenQBitNativeBlock::StreamRecordSpan).new
      stream.record_spans.each do |span|
        key = {span.layer, span.kind}
        raise ArgumentError.new("duplicate QBit Native state record") if recurrent.has_key?(key)
        recurrent[key] = span
      end
      kv_records = Hash({Int32, UInt8}, Qwen35StateSnapshot::EncodedRecord).new
      compact.records.each do |record|
        unless record.kind.k_cache? || record.kind.v_cache?
          raise ArgumentError.new("QBit compact adaptive artifact must be KV-only")
        end
        raise ArgumentError.new("QBit compact adaptive record must use raw framing") unless record.codec.raw_f32?
        key = {record.layer, record.kind.value}
        raise ArgumentError.new("duplicate QBit compact adaptive record") if kv_records.has_key?(key)
        kv_records[key] = record
      end

      expected_recurrent = Set({Int32, UInt8}).new
      expected_kv = Set({Int32, UInt8}).new
      kv_record_bytes = state.max_seq.to_i64 * hp.head_dim * hp.n_head_kv * sizeof(Float32)
      live_value_count = live_tokens.to_i64 * hp.head_dim * hp.n_head_kv
      if kv_record_bytes > Int32::MAX || live_value_count > Int32::MAX
        raise ArgumentError.new("QBit compact adaptive KV shape is too large")
      end
      qkv_dim = 2_i64 * hp.ssm_group_count * hp.ssm_state_size +
                hp.ssm_time_step_rank.to_i64 * hp.ssm_state_size
      conv_record_bytes = (hp.ssm_conv_kernel - 1).to_i64 * qkv_dim * sizeof(Float32)
      ssm_record_bytes = hp.ssm_time_step_rank.to_i64 * hp.ssm_state_size *
                         hp.ssm_state_size * sizeof(Float32)
      hp.n_layer.times do |layer|
        if hp.full_attention?(layer)
          expected_kv.add({layer, RecordKind::KCache.value})
          expected_kv.add({layer, RecordKind::VCache.value})
        else
          expected_recurrent.add({layer, RecordKind::ConvState.value})
          expected_recurrent.add({layer, RecordKind::SsmState.value})
        end
      end
      raise ArgumentError.new("QBit compact adaptive recurrent record set mismatch") unless recurrent.keys.to_set == expected_recurrent
      raise ArgumentError.new("QBit compact adaptive KV record set mismatch") unless kv_records.keys.to_set == expected_kv

      recurrent.each do |key, span|
        kind = RecordKind.from_value(key[1])
        expected_bytes = kind.conv_state? ? conv_record_bytes : ssm_record_bytes
        unless span.value_count.to_i64 * sizeof(Float32) == expected_bytes
          raise ArgumentError.new("QBit compact adaptive recurrent record byte size mismatch")
        end
        layer = state.layers[key[0]]
        if layer.conv_state || layer.ssm_state
          raise ArgumentError.new("QBit compact adaptive recurrent state cannot coexist with a CPU owner")
        end
        buffer = state_buffer(layer, kind)
        unless buffer && buffer.size == expected_bytes
          raise ArgumentError.new("QBit compact adaptive recurrent target buffer is not prepared")
        end
      end

      encoded_kv = Hash({Int32, UInt8}, QwenQBitAdaptiveKV::Encoded).new
      kv_records.each do |key, record|
        unless record.original_byte_size.to_i64 == kv_record_bytes
          raise ArgumentError.new("QBit compact adaptive record byte size mismatch")
        end
        layer = state.layers[key[0]]
        if layer.k_cache || layer.v_cache || layer.k_cache_buf || layer.v_cache_buf
          raise ArgumentError.new("QBit compact adaptive KV cannot coexist with a Float32 owner")
        end
        cache = layer.adaptive_kv.not_nil!
        unless cache.cache_len == 0 && cache.max_seq == state.max_seq &&
               cache.n_head_kv == hp.n_head_kv && cache.head_dim == hp.head_dim
          raise ArgumentError.new("QBit compact adaptive target cache is not empty or has the wrong shape")
        end
        encoded = QwenQBitAdaptiveKV::Encoded.new(
          live_value_count.to_i32,
          QwenQBitAdaptiveKV::ROW_VALUES,
          record.payload,
        )
        QwenQBitAdaptiveKV.validate(encoded)
        encoded_kv[key] = encoded
      end

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        raise "Metal not available" unless Qwen35Metal.available?
        begin
          jobs = recurrent.map do |key, span|
            kind = RecordKind.from_value(key[1])
            QwenQBitMetalRestore::NativeStreamJob.new(
              span,
              state_buffer(state.layers[key[0]], kind).not_nil!,
            )
          end
          QwenQBitMetalRestore.decode_native_stream_into(stream, jobs)
          hp.full_attention_layers.each do |layer_index|
            QwenQBitAdaptiveResidentKV.restore_snapshot!(
              state.layers[layer_index].adaptive_kv.not_nil!,
              encoded_kv[{layer_index, RecordKind::KCache.value}],
              encoded_kv[{layer_index, RecordKind::VCache.value}],
              live_tokens,
            )
          end
          compact.positions.each_with_index { |position, layer| state.layers[layer].position = position }
        rescue ex : ArgumentError
          raise ex
        rescue ex
          raise AdaptiveRestoreDeviceError.new("QBit compact adaptive device restore failed: #{ex.class}: #{ex.message}")
        end
      {% end %}
    end

    def validate(snapshot : Snapshot) : Nil
      raise ArgumentError.new("QBit state position count mismatch") unless snapshot.positions.size == snapshot.layer_count
      unless snapshot.precision >= MIN_STATE_PRECISION && snapshot.precision <= MAX_STATE_PRECISION
        raise ArgumentError.new("QBit state snapshot precision is unsupported")
      end
      seen = Set({Int32, UInt8}).new
      snapshot.records.each do |record|
        raise ArgumentError.new("QBit state record layer out of range") if record.layer < 0 || record.layer >= snapshot.layer_count
        key = {record.layer, record.kind.value}
        raise ArgumentError.new("duplicate QBit state record") unless seen.add?(key)
        raise ArgumentError.new("QBit state record byte size is not Float32-aligned") unless record.original_byte_size >= 0 && record.original_byte_size % sizeof(Float32) == 0

        if recurrent_record?(record.kind)
          raise ArgumentError.new("recurrent QBit state record cannot be raw") unless record.raw.nil? && record.qbit
          encoded = record.qbit.not_nil!
          raise ArgumentError.new("QBit state block size mismatch") unless encoded.block_size == snapshot.block_size
          raise ArgumentError.new("QBit state precision mismatch") unless encoded.precision == snapshot.precision
          raise ArgumentError.new("QBit state value count mismatch") unless encoded.value_count * sizeof(Float32) == record.original_byte_size
          QwenQBitGaussianCodec.validate(encoded)
        else
          raw = record.raw
          raise ArgumentError.new("live KV QBit state record must stay raw") unless raw && record.qbit.nil?
          raise ArgumentError.new("raw QBit state record size mismatch") unless raw.not_nil!.size == record.original_byte_size
        end
      end
    end

    private def validate_for_state(snapshot : Snapshot, hp : Qwen35Hparams, state : Qwen35CPU::State) : Nil
      validate(snapshot)
      raise ArgumentError.new("layer count mismatch: snapshot=#{snapshot.layer_count}, hp=#{hp.n_layer}") unless snapshot.layer_count == hp.n_layer
      raise ArgumentError.new("state layer count mismatch") unless snapshot.layer_count == state.layers.size
      raise ArgumentError.new("state max_seq mismatch") unless snapshot.max_seq == state.max_seq
    end

    private def recurrent_record?(kind : RecordKind) : Bool
      case kind
      in RecordKind::ConvState, RecordKind::SsmState
        true
      in RecordKind::KCache, RecordKind::VCache
        false
      end
    end

    private def floats_from(bytes : Bytes) : Array(Float32)
      raise ArgumentError.new("state record byte size is not Float32-aligned") unless bytes.size % sizeof(Float32) == 0
      values = Array(Float32).new(bytes.size // sizeof(Float32), 0.0_f32)
      Slice.new(values.to_unsafe.as(Pointer(UInt8)), bytes.size).copy_from(bytes)
      values
    end

    private def bytes_from(values : Array(Float32)) : Bytes
      bytes = Bytes.new(values.size * sizeof(Float32))
      bytes.copy_from(Slice.new(values.to_unsafe.as(Pointer(UInt8)), bytes.size))
      bytes
    end

    {% unless flag?(:cpu_only) %}
      private def state_buffer(layer : Qwen35CPU::LayerState, kind : RecordKind) : ML::MetalBuffer?
        case kind
        in RecordKind::KCache    then layer.k_cache_buf
        in RecordKind::VCache    then layer.v_cache_buf
        in RecordKind::ConvState then layer.conv_state_buf
        in RecordKind::SsmState  then layer.ssm_state_buf
        end
      end

      private def assign_state_buffer(layer : Qwen35CPU::LayerState, kind : RecordKind, buffer : ML::MetalBuffer) : Nil
        case kind
        in RecordKind::KCache    then layer.k_cache_buf = buffer
        in RecordKind::VCache    then layer.v_cache_buf = buffer
        in RecordKind::ConvState then layer.conv_state_buf = buffer
        in RecordKind::SsmState  then layer.ssm_state_buf = buffer
        end
      end
    {% end %}
  end
end
