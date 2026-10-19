require "./wan_s2v_transformer"

module ML::Video
  module WanS2VTransformer
    # Opt-in serial lifetime reuse across the attention block and its following
    # audio injector. All four block values are dead when the injector starts;
    # the next block overwrites them only after the injector has finished.
    # Both arenas form one exclusive workspace and must remain live together.
    REUSED_AUDIO_SCRATCH = {
      "wan_a_gather" => "wan_b_norm1",
      "wan_a_q"      => "wan_b_self_q",
      "wan_a_attn"   => "wan_b_self_attn",
      "wan_a_proj"   => "wan_b_self_proj",
    }

    def self.prepare_reused_scratch!(block_scratch : Arena, audio_scratch : Arena,
                                     cfg : WanS2VTransformerConfig) : Nil
      cfg.validate!
      unless !block_scratch.same?(audio_scratch) && block_scratch.buffers.empty? &&
             audio_scratch.buffers.empty? && block_scratch.resident_bytes == 0 &&
             audio_scratch.resident_bytes == 0
        raise ArgumentError.new("Reused Wan scratch requires two distinct empty arenas")
      end
      WanS2VBlock.prepare_scratch!(block_scratch, cfg.block_config)
      audio_scratch.upload("wan_a_zero_bias", Array(Float32).new(cfg.dim, 0.0_f32))
      WanS2VAudioInjector.scratch_plan(cfg.audio_config).each do |name, count|
        if source = REUSED_AUDIO_SCRATCH[name]?
          shared = block_scratch[source]
          MetalOps.check(shared, count)
          audio_scratch.buffers[name] = shared
        else
          audio_scratch.alloc(name, count)
        end
      end
    end

    def self.reused_scratch_bytes(cfg : WanS2VTransformerConfig) : Int64
      cfg.validate!
      unique_audio = WanS2VAudioInjector.scratch_plan(cfg.audio_config)
        .reject { |name, _| REUSED_AUDIO_SCRATCH.has_key?(name) }
        .sum { |_, count| count.to_i64 }
      WanS2VBlock.scratch_bytes(cfg.block_config) + (cfg.dim.to_i64 + unique_audio) * 4
    end

    # Setup-only union accounting: summing the two arenas separately would
    # count the four shared MetalBuffer objects twice.
    def self.workspace_resident_bytes(block_scratch : Arena, audio_scratch : Arena) : Int64
      buffers = [] of ML::MetalBuffer
      {block_scratch, audio_scratch}.each do |arena|
        arena.buffers.each_value do |buffer|
          next unless buffer.valid?
          buffers << buffer unless buffers.any? { |other| other.same?(buffer) }
        end
      end
      buffers.sum(&.size)
    end
  end
end
