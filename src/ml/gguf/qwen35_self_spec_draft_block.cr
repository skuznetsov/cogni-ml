require "../core/buffer"
require "./qwen35_cpu"
require "./qwen35_metal"

module ML::GGUF
  # GPU draft block submitted by the self-spec scheduler.
  #
  # This object intentionally owns only completion/readback mechanics. Proposal
  # scheduling remains in the probe until the route dependencies are separated.
  class Qwen35SelfSpecDraftBlock
    getter submissions : Array(Qwen35Metal::DecodeWaveSubmission)
    getter state : Qwen35CPU::State
    getter lr_bufs : Hash(Int32, ML::MetalBuffer)
    getter full_current : Hash(Int32, Bool)
    getter use_updown : Bool
    getter use_noffn : Bool

    def initialize(@submissions, @state, @lr_bufs, @full_current, @use_updown, @use_noffn)
    end

    def wait!(limit : Int32? = nil) : Nil
      active = active_submissions(limit)
      waited_cmds = Set(UInt64).new
      active.each do |sub|
        sub.pending_cmds.each do |cmd|
          id = cmd.object_id
          next if waited_cmds.includes?(id)

          cmd.wait
          waited_cmds << id
        end
        id = sub.cmd.object_id
        unless waited_cmds.includes?(id)
          sub.cmd.wait
          waited_cmds << id
        end
      end
    end

    def top1_ids(limit : Int32, wait : Bool = true) : Array(Int32)
      wait!(limit) if wait
      active_submissions(limit).map do |sub|
        sub.top1_id_buf.not_nil!.contents.as(Pointer(UInt32)).value.to_i32
      end
    end

    def second_id(index : Int32) : Int32
      if buf = submissions[index].second_id_buf
        buf.contents.as(Pointer(UInt32)).value.to_i32
      else
        -1_i32
      end
    end

    def top2_margin(index : Int32) : Float64?
      sub = submissions[index]
      if top = sub.top1_value_buf
        if second = sub.second_value_buf
          top.contents.as(Pointer(Float32)).value.to_f64 - second.contents.as(Pointer(Float32)).value.to_f64
        end
      end
    end

    def drain! : Nil
      wait!
    end

    private def active_submissions(limit : Int32?) : Array(Qwen35Metal::DecodeWaveSubmission)
      if limit
        submissions[0, limit]
      else
        submissions
      end
    end
  end
end
