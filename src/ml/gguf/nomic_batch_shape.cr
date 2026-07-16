module ML::GGUF
  module NomicBatchShape
    extend self

    def bucket_enabled? : Bool
      ENV["NOMIC_BATCH_SEQ_BUCKET_OFF"]? != "1"
    end

    def bucket_seq_len(logical_len : Int32, context_len : Int32, enabled : Bool = bucket_enabled?) : Int32
      raise ArgumentError.new("logical_len must be positive") unless logical_len > 0
      raise ArgumentError.new("context_len must be positive") unless context_len > 0
      raise ArgumentError.new("logical_len exceeds context_len") if logical_len > context_len
      return logical_len unless enabled

      bucket = next_power_of_two(logical_len)
      bucket > context_len ? context_len : bucket
    end

    def padded_token_overhead(lengths : Array(Int32), physical_seq_len : Int32) : Int32
      raise ArgumentError.new("physical_seq_len must be positive") unless physical_seq_len > 0
      lengths.sum do |len|
        raise ArgumentError.new("lengths must be non-negative") if len < 0
        raise ArgumentError.new("length exceeds physical_seq_len") if len > physical_seq_len
        physical_seq_len - len
      end
    end

    private def next_power_of_two(value : Int32) : Int32
      v = value - 1
      v |= v >> 1
      v |= v >> 2
      v |= v >> 4
      v |= v >> 8
      v |= v >> 16
      v + 1
    end
  end
end
