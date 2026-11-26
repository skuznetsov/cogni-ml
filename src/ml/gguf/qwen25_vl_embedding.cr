require "../core/buffer"
require "../video/metal_mps_command_chain"
require "./qwen25_vl_weights"

{% unless flag?(:cpu_only) %}
  lib Qwen25VLEmbeddingFFI
    fun create = gs_qwen25_mps_embedding_gather_create(
      device : Void*, tokens : Int32, dim : Int32, unique_rows : Int32,
      compact_rows : Void*, row_indices : Void*, output : Void*,
      error : UInt8*, error_capacity : LibC::SizeT,
    ) : Void*
    fun encode = gs_qwen25_mps_embedding_gather_encode(
      plan : Void*, chain : Void*, error : UInt8*, error_capacity : LibC::SizeT,
    ) : Int32
    fun destroy = gs_qwen25_mps_embedding_gather_destroy(plan : Void*) : Void
  end
{% end %}

module ML::GGUF
  # Prepared, exact-bit BF16 embedding lookup. Source-row selection and
  # safetensors I/O happen before construction; only the gather itself runs
  # here, on the GPU and on the caller's MPSCommandChain.
  class Qwen25VLEmbeddingPlan
    struct NamedBytes
      getter compact_rows : Int64
      getter remapped_ids : Int64
      getter output : Int64
      getter total : Int64

      def initialize(@compact_rows : Int64, @remapped_ids : Int64,
                     @output : Int64, @total : Int64)
      end
    end

    @@retained = {} of UInt64 => Qwen25VLEmbeddingPlan

    getter token_count : Int32
    getter unique_row_count : Int32
    getter named_bytes : NamedBytes

    @native = Pointer(Void).null
    @compact_rows : MetalBuffer
    @index_buffer : MetalBuffer? = nil
    @output_buffer : MetalBuffer? = nil
    @owned_buffers = [] of MetalBuffer
    @error_text = Bytes.new(1024, 0_u8)
    @chain : Metal::MPSCommandChain? = nil
    @encoded = false
    @released = false

    # Named memory footprint of the compact table, remapped Int32 IDs, and
    # gathered BF16 output. `tokens` is also the input sequence length.
    def self.resident_bytes(config : Qwen25VLConfig, tokens : Int32,
                            unique_rows : Int32,
                            vocab_size : Int32 = Qwen25VLWeights::VOCAB_SIZE) : NamedBytes
      raise ArgumentError.new("embedding tokens outside configured context") unless 0 < tokens <= config.max_positions
      raise ArgumentError.new("embedding unique row count outside vocabulary") unless 0 < unique_rows <= vocab_size
      raise ArgumentError.new("embedding unique row count exceeds token count") if unique_rows > tokens
      raise ArgumentError.new("embedding vocabulary size must be positive") unless vocab_size > 0

      compact_rows = checked_product(unique_rows.to_i64, config.dim.to_i64, 2_i64)
      remapped_ids = checked_product(tokens.to_i64, sizeof(Int32).to_i64)
      output = checked_product(tokens.to_i64, config.dim.to_i64, 2_i64)
      elements = checked_product(tokens.to_i64, config.dim.to_i64)
      raise ArgumentError.new("embedding gather exceeds native index range") if elements > Int32::MAX
      total = checked_sum(compact_rows, remapped_ids, output)
      NamedBytes.new(compact_rows, remapped_ids, output, total)
    end

    # CPU-only preflight used by construction and tests. The compact table has
    # one BF16 row per unique source token; each index selects one such row.
    def self.validate_inputs(config : Qwen25VLConfig, tokens : Int32,
                             compact_rows_bytes : Int64, row_indices : Array(Int32),
                             vocab_size : Int32 = Qwen25VLWeights::VOCAB_SIZE) : NamedTuple(unique_rows: Int32, named_bytes: NamedBytes)
      raise ArgumentError.new("embedding tokens outside configured context") unless 0 < tokens <= config.max_positions
      raise ArgumentError.new("embedding row index count must equal tokens") unless row_indices.size == tokens
      raise ArgumentError.new("embedding vocabulary size must be positive") unless vocab_size > 0
      bytes_per_row = checked_product(config.dim.to_i64, 2_i64)
      raise ArgumentError.new("compact embedding table must contain whole BF16 rows") unless compact_rows_bytes > 0 && compact_rows_bytes % bytes_per_row == 0
      unique_rows64 = compact_rows_bytes // bytes_per_row
      raise ArgumentError.new("embedding unique row count outside vocabulary") unless 0 < unique_rows64 <= vocab_size
      raise ArgumentError.new("embedding unique row count exceeds token count") if unique_rows64 > tokens
      unique_rows = unique_rows64.to_i32
      referenced = Array(Bool).new(unique_rows, false)
      row_indices.each do |row|
        raise ArgumentError.new("embedding row index outside compact table") unless 0 <= row < unique_rows
        referenced[row] = true
      end
      raise ArgumentError.new("compact embedding table contains an unused row") unless referenced.all?
      bytes = resident_bytes(config, tokens, unique_rows, vocab_size)
      raise ArgumentError.new("compact embedding table byte count mismatch") unless bytes.compact_rows == compact_rows_bytes
      {unique_rows: unique_rows, named_bytes: bytes}
    end

    def self.validate_budget!(bytes : NamedBytes, max_named_bytes : Int64) : Nil
      raise ArgumentError.new("embedding gather exceeds admitted named byte budget") unless max_named_bytes >= bytes.total
    end

    # `compact_rows` is borrowed and must remain valid through the terminal
    # chain wait. The plan owns only the uploaded remapped IDs and output.
    # Budget validation happens before either owned MetalBuffer is allocated.
    def initialize(@config : Qwen25VLConfig, @token_count : Int32,
                   @compact_rows : MetalBuffer, row_indices : Array(Int32),
                   max_named_bytes : Int64,
                   vocab_size : Int32 = Qwen25VLWeights::VOCAB_SIZE)
      checked = self.class.validate_inputs(@config, @token_count,
        @compact_rows.size, row_indices, vocab_size)
      @unique_row_count = checked[:unique_rows]
      @named_bytes = checked[:named_bytes]
      self.class.validate_budget!(@named_bytes, max_named_bytes)
      raise ArgumentError.new("compact embedding table must be a valid shared buffer") unless @compact_rows.valid? && @compact_rows.storage_mode.shared?

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        begin
          @index_buffer = MetalBuffer.new(@named_bytes.remapped_ids)
          @owned_buffers << @index_buffer.not_nil!
          @index_buffer.not_nil!.write_bytes(row_indices.to_unsafe.as(Pointer(UInt8)), @named_bytes.remapped_ids.to_i)

          @output_buffer = MetalBuffer.new(@named_bytes.output)
          @owned_buffers << @output_buffer.not_nil!

          @native = Qwen25VLEmbeddingFFI.create(Metal::Device.instance.device_handle,
            @token_count, @config.dim, @unique_row_count, @compact_rows.handle,
            @index_buffer.not_nil!.handle, @output_buffer.not_nil!.handle,
            @error_text.to_unsafe, @error_text.size)
          raise "Qwen2.5-VL embedding gather preparation failed: #{error_message}" if @native.null?
          @@retained[object_id] = self
        rescue ex
          @native = Pointer(Void).null
          @owned_buffers.each(&.release)
          @owned_buffers.clear
          raise ex
        end
      {% end %}
    end

    def output : MetalBuffer
      check_live!
      @output_buffer || raise ArgumentError.new("embedding output has been released")
    end

    # Appends the exact-uint16 gather to the same active root. It neither
    # commits, waits, synchronizes nor allocates MetalBuffers.
    def encode(chain : Metal::MPSCommandChain) : Nil
      check_live!
      raise ArgumentError.new("cannot encode into a completed chain") if chain.completed?
      if previous = @chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("Qwen2.5-VL embedding plan belongs to another in-flight chain")
        end
        raise ArgumentError.new("Qwen2.5-VL embedding gather already encoded on this chain") if previous.same?(chain) && @encoded
      end
      raise ArgumentError.new("compact embedding table has been released") unless @compact_rows.valid?
      raise ArgumentError.new("embedding output or index buffer has been released") unless @output_buffer.try(&.valid?) && @index_buffer.try(&.valid?)

      @chain = chain
      @encoded = true
      chain.pause!
      {% unless flag?(:cpu_only) %}
        status = Qwen25VLEmbeddingFFI.encode(@native, chain.handle,
          @error_text.to_unsafe, @error_text.size)
        chain.resume!
        raise "Qwen2.5-VL embedding gather encode failed (#{status}): #{error_message}" unless status == 0
      {% end %}
    end

    def release : Nil
      return if @released
      if chain = @chain
        raise ArgumentError.new("terminal wait required before Qwen2.5-VL embedding release") unless chain.completed?
      end
      {% unless flag?(:cpu_only) %}
        Qwen25VLEmbeddingFFI.destroy(@native) unless @native.null?
      {% end %}
      @native = Pointer(Void).null
      @owned_buffers.each(&.release)
      @owned_buffers.clear
      @index_buffer = nil
      @output_buffer = nil
      @released = true
      @chain = nil
      @@retained.delete(object_id)
    end

    private def self.checked_product(*values : Int64) : Int64
      values.reduce(1_i64) do |product, value|
        raise ArgumentError.new("embedding byte count overflow") if value <= 0 || product > Int64::MAX // value
        product * value
      end
    end

    private def self.checked_sum(*values : Int64) : Int64
      values.reduce(0_i64) do |sum, value|
        raise ArgumentError.new("embedding byte count overflow") if value < 0 || sum > Int64::MAX - value
        sum + value
      end
    end

    private def check_live! : Nil
      raise ArgumentError.new("Qwen2.5-VL embedding plan released") if @released || @native.null?
    end

    private def error_message : String
      String.new(@error_text.to_unsafe)
    end
  end
end
