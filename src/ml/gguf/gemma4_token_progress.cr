require "json"

module ML::GGUF
  # Reconstructs progress from llama-server's streaming token IDs. Token text is
  # deliberately ignored: a partial response may have empty content while
  # still carrying one or more generated token IDs.
  module Gemma4TokenProgress
    struct Point
      getter elapsed_ms : Int64
      getter generated_tokens : Int64

      def initialize(@elapsed_ms : Int64, @generated_tokens : Int64)
      end
    end

    class Collector
      getter token_ids : Array(Int64)
      getter points : Array(Point)
      getter requested : Int64

      @final_seen : Bool
      @finished : Bool
      @last_wall_elapsed_ms : Int64
      @last_monotonic_elapsed_ms : Int64

      def initialize(@start_epoch_ms : Int64, requested : Int32 | Int64)
        @requested = requested.to_i64
        raise ArgumentError.new("requested token count must be positive") unless @requested > 0
        @token_ids = [] of Int64
        @points = [Point.new(0_i64, 0_i64)]
        @final_seen = false
        @finished = false
        @last_wall_elapsed_ms = 0_i64
        @last_monotonic_elapsed_ms = 0_i64
      end

      def generated_tokens : Int64
        @token_ids.size.to_i64
      end

      # receipt_epoch_ms and monotonic_elapsed_ms must be observed on the same
      # host. The wall timestamp is retained for reporting; monotonic time is
      # used to detect a wall clock that moved backwards during collection.
      def consume(json : String, receipt_epoch_ms : Int64, monotonic_elapsed_ms : Int64) : Nil
        raise ArgumentError.new("stream already completed") if @final_seen || @finished
        event = JSON.parse(json).as_h
        stop = event["stop"].as_bool
        tokens = parse_token_ids(event["tokens"]?)
        reported = event["tokens_predicted"].as_i64
        elapsed = record_receipt!(receipt_epoch_ms, monotonic_elapsed_ms)

        if stop
          consume_final!(event, tokens, reported)
          @final_seen = true
        else
          append_delta!(tokens)
          unless reported == generated_tokens
            raise ArgumentError.new("stream token IDs disagree with tokens_predicted")
          end
          raise ArgumentError.new("stream exceeded requested token count") if generated_tokens > @requested
        end

        record_point!(elapsed, generated_tokens) unless elapsed.zero?
        nil
      end

      def finish!(end_epoch_ms : Int64, monotonic_elapsed_ms : Int64) : Nil
        raise ArgumentError.new("stream is truncated: final response event is missing") unless @final_seen
        raise ArgumentError.new("stream already completed") if @finished
        elapsed = record_receipt!(end_epoch_ms, monotonic_elapsed_ms)
        raise ArgumentError.new("stream ended before the final event") if elapsed < @points.last.elapsed_ms
        raise ArgumentError.new("completed token count disagrees with requested work") unless generated_tokens == @requested
        if elapsed.zero? && generated_tokens > 0
          raise ArgumentError.new("request window is shorter than one millisecond")
        end
        record_point!(elapsed, generated_tokens) unless elapsed.zero?
        @finished = true
        nil
      end

      def to_json(status : String? = nil) : JSON::Any
        available = @finished
        requested_status = status || (available ? "available" : "partial")
        unless {"available", "partial"}.includes?(requested_status)
          raise ArgumentError.new("token progress status must be available or partial")
        end
        if requested_status == "available" && !available
          raise ArgumentError.new("token progress is incomplete")
        end
        points_json = @points.map do |point|
          {elapsed_ms: point.elapsed_ms, generated_tokens: point.generated_tokens}
        end
        JSON.parse({
          status: requested_status,
          basis: "received_token_ids",
          timestamp_semantics: "observer_receipt_epoch_ms_relative_to_started_at",
          requested_tokens: @requested,
          final_tokens: available ? generated_tokens : nil,
          observed_tokens: generated_tokens,
          points: points_json,
        }.to_json)
      end

      private def parse_token_ids(value : JSON::Any?) : Array(Int64)
        return [] of Int64 unless value
        value.as_a.map do |item|
          id = item.as_i64
          raise ArgumentError.new("token IDs must be nonnegative") if id < 0
          id
        end
      end

      private def append_delta!(tokens : Array(Int64)) : Nil
        tokens.each { |token| @token_ids << token }
      end

      private def consume_final!(event : Hash(String, JSON::Any), tokens : Array(Int64), reported : Int64) : Nil
        unless reported == @requested
          raise ArgumentError.new("final tokens_predicted disagrees with requested work")
        end
        unless event["stop_type"].as_s == "limit"
          raise ArgumentError.new("backend did not stop at the fixed token limit")
        end
        raise ArgumentError.new("backend reported a truncated prompt") if event["truncated"]?.try(&.as_bool)
        raise ArgumentError.new("backend prompt usage is missing") unless event["tokens_evaluated"].as_i64 > 0
        timings = event["timings"].as_h
        unless timings["predicted_n"].as_i64 == @requested
          raise ArgumentError.new("final timings disagree with requested token count")
        end

        # The pinned server emits an empty final tokens array for streams. Some
        # compatible versions echo the full cumulative list; accept that only
        # when it is the prefix already observed plus optional unseen suffix.
        unless tokens.empty?
          if tokens == @token_ids
            # Duplicate full snapshot, not additional work.
          elsif tokens.size >= @token_ids.size && tokens[0, @token_ids.size] == @token_ids
            @token_ids.concat(tokens[@token_ids.size..-1])
          else
            raise ArgumentError.new("final token IDs disagree with streamed token IDs")
          end
        end
        unless generated_tokens == @requested
          raise ArgumentError.new("final token IDs disagree with fixed-work usage")
        end
        nil
      end

      private def record_receipt!(epoch_ms : Int64, monotonic_elapsed_ms : Int64) : Int64
        elapsed = epoch_ms - @start_epoch_ms
        raise ArgumentError.new("receipt timestamp precedes request start") if elapsed < 0
        if elapsed < @last_wall_elapsed_ms || monotonic_elapsed_ms < @last_monotonic_elapsed_ms
          raise ArgumentError.new("request clock moved backwards")
        end
        # Time.utc and Time.instant are sampled separately at request start;
        # tolerate small millisecond rounding skew, but reject a wall clock
        # clearly trailing the monotonic observation.
        if elapsed + 2_i64 < monotonic_elapsed_ms
          raise ArgumentError.new("wall clock moved backwards relative to monotonic time")
        end
        @last_wall_elapsed_ms = elapsed
        @last_monotonic_elapsed_ms = monotonic_elapsed_ms
        elapsed
      end

      private def record_point!(elapsed_ms : Int64, count : Int64) : Nil
        raise ArgumentError.new("generated token counter exceeds requested work") if count > @requested
        last = @points.last
        if elapsed_ms < last.elapsed_ms
          raise ArgumentError.new("token progress timestamps are not ordered")
        elsif elapsed_ms == last.elapsed_ms
          # Keep the initial [0, 0] anchor immutable. For positive timestamps,
          # coalesce same-millisecond receipts to the latest cumulative count.
          @points[-1] = Point.new(elapsed_ms, count) unless elapsed_ms.zero?
        else
          raise ArgumentError.new("generated token counter moved backwards") if count < last.generated_tokens
          @points << Point.new(elapsed_ms, count)
        end
        nil
      end
    end
  end
end
