require "digest/sha256"
require "json"

module ML::Video
  # One deterministic new take within an authenticated six-segment input bank.
  # The donor is outside the whole active run, so it remains initial noise
  # through every save/restart. Shared scene conditioning is unchanged.
  struct HyNomasSegmentNoiseEdit
    getter run_segments : Int32
    getter target_segment : Int32
    getter source_segment : Int32
    getter resident_segments : Int32

    def initialize(@run_segments : Int32, @target_segment : Int32,
                   @source_segment : Int32, @resident_segments : Int32 = 6)
      unless 2 <= @run_segments < @resident_segments <= 6 &&
             1 <= @target_segment < @run_segments &&
             @run_segments <= @source_segment < @resident_segments
        raise ArgumentError.new("Noise edit requires a noninitial active target and an unused future donor")
      end
    end

    def identity
      {schema: "cogni_ml.hy_nomas_segment_noise_edit/v1", run_segments: @run_segments,
       target_segment: @target_segment, source_segment: @source_segment,
       resident_segments: @resident_segments}
    end

    def source_for(segment : Int32) : Int32
      unless 0 <= segment < @run_segments
        raise ArgumentError.new("Noise edit segment is outside the active run")
      end
      segment == @target_segment ? @source_segment : segment
    end

    # Future edits cannot change an already-produced prefix. Once applied,
    # their identity is mandatory for every descendant checkpoint. The base
    # hash retains the executable, input, conditioning, weights and schedule
    # binding; this is not a cross-build migration or an equality relaxation.
    def bind_contract(base : String, next_segment : Int32) : String
      unless 1 <= next_segment < @run_segments
        raise ArgumentError.new("Noise edit contract requires an extendable boundary")
      end
      return base if next_segment <= @target_segment
      {schema: "cogni_ml.hy_nomas_edited_sequence_boundary/v1",
       base_contract_sha256: Digest::SHA256.hexdigest(base), noise_edit: identity}.to_json
    end
  end
end
