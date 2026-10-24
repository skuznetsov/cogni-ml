require "json"
require "../../src/ml/video/hy_nomas_block"
require "../../src/ml/video/hy_nomas_mps_activation"

# Frozen FFN inputs distinguish activation arithmetic from error propagated
# through the preceding projections. Reads occur only after the caller drains
# its existing command chain; this control appends no waits or Metal allocations.
class HyNomasActivationControls
  @plans = {} of String => ML::Video::HyNomasMPSActivation::GELUPlan

  def initialize(@config : ML::Video::HyNomasConfig, @arena : ML::Video::Arena,
                 variant : JSON::Any)
    begin
      {"img", "txt"}.each do |stream|
        tokens = stream == "img" ? @config.video_tokens : @config.text_tokens
        count = @config.batches * tokens * @config.ffn_dim
        prefix = "activation_control.#{stream}"
        record = variant["checkpoints"]["#{stream}.ff.hidden"]
        values = record["values"].as_a.map(&.as_f.to_f32)
        raise ArgumentError.new("Frozen GELU input extent mismatch") unless values.size == count
        input = @arena.upload("#{prefix}.input", values)
        output = @arena.alloc("#{prefix}.output", count)
        @plans[stream] = ML::Video::HyNomasMPSActivation::GELUPlan.new(
          [@config.batches, tokens, @config.ffn_dim], input, output)
      end
    rescue ex
      release
      raise ex
    end
  end

  def encode(chain : ML::Metal::MPSCommandChain) : Nil
    @plans.each_value { |plan| plan.encode(chain) }
  end

  def snapshots : Hash(String, Array(Float32))
    {"img", "txt"}.to_h do |stream|
      tokens = stream == "img" ? @config.video_tokens : @config.text_tokens
      count = @config.batches * tokens * @config.ffn_dim
      {stream, @arena["activation_control.#{stream}.output"].read(count)}
    end
  end

  def release(chain : ML::Metal::MPSCommandChain? = nil) : Nil
    @plans.each_value { |plan| plan.release(chain) }
    @plans.clear
  end
end
