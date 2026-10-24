# Two sequential blocks with common intermediate scratch and one prepared MPS
# workspace. Final output banks are distinct; within-block input/scratch aliases
# are rejected. The predecessor's output banks are the successor's inputs.
require "./hy_nomas_block_mps"

module ML::Video
  class HyNomasMPSSerialWorkspace
    @blocks = [] of HyNomasDualStreamBlock
    getter scratch_bytes : Int64 = 0_i64
    getter mps_workspace_bytes : Int64 = 0_i64

    @arena : Arena
    @views = [] of Arena
    @workspace : HyNomasMPSWorkspace? = nil
    @inputs : Array(ML::MetalBuffer)
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def initialize(config : HyNomasConfig, weights : Array(Arena),
                   img : ML::MetalBuffer, txt : ML::MetalBuffer,
                   vec : ML::MetalBuffer, vec_txt : ML::MetalBuffer,
                   cos : ML::MetalBuffer, sin : ML::MetalBuffer,
                   *, attention_query_chunk : Int32 = 0,
                   attention_heads_per_group : Int32 = 0)
      config.validate!
      raise ArgumentError.new("Hy-Nomas serial workspace requires exactly two weight Arenas") unless weights.size == 2
      if attention_query_chunk != 0 || attention_heads_per_group != 0
        HyNomasCompactAttentionMPSPlan.workspace_geometry(config.batches,
          config.joint_tokens, config.heads, config.head_dim,
          attention_query_chunk, attention_heads_per_group)
      end
      @inputs = [img, txt, vec, vec_txt, cos, sin]
      @arena = Arena.new
      begin
        HyNomasDualStreamBlock.prepare_scratch!(@arena, config)
        @blocks << HyNomasDualStreamBlock.new(config, weights[0], @arena)
        view = Arena.new
        @views << view
        HyNomasDualStreamBlock.scratch_plan(config).each do |name, count|
          buffer = if {"hy_nomas.img.output", "hy_nomas.txt.output"}.includes?(name)
                     @arena.alloc("serial.second.#{name}", count)
                   else
                     @arena[name]
                   end
          # Private, non-owning F32 map. It must be cleared, never released.
          # The first block retains the sole owning Arena through the rooted
          # MPS workspace; no caller receives the borrowed Arena itself.
          view.buffers[name] = buffer
        end
        @blocks << HyNomasDualStreamBlock.new(config, weights[1], view)
        @workspace = HyNomasMPSWorkspace.new(@blocks[0], img, txt, vec,
          vec_txt, cos, sin, attention_query_chunk: attention_query_chunk,
          attention_heads_per_group: attention_heads_per_group)
        workspace.register_serial_blocks!(@blocks,
          [{img, txt}, {@blocks[0].image_output, @blocks[0].text_output}])
        @scratch_bytes = @arena.resident_bytes
        @mps_workspace_bytes = workspace.resident_bytes
      rescue ex
        @workspace.try(&.release)
        @views.each { |borrowed| borrowed.buffers.clear }
        @arena.release
        @released = true
        raise ex
      end
    end

    def resident_bytes : Int64
      @scratch_bytes + @mps_workspace_bytes
    end

    # A tuple prevents callers from reordering the admitted block registry.
    def blocks : Tuple(HyNomasDualStreamBlock, HyNomasDualStreamBlock)
      {@blocks[0], @blocks[1]}
    end

    # after_block is a trusted diagnostic hook for preallocated GPU snapshots.
    # It may append copies to the same chain's current encoder, but must not
    # mutate bindings, commit command buffers, wait, or recursively encode.
    # Those diagnostic dispatches belong outside performance measurements.
    # If it raises after an append, the caller must drain the chain before
    # releasing this owner, exactly as for an encode failure.
    def encode(chain : ML::Metal::MPSCommandChain,
               *, after_block : Proc(Int32, HyNomasDualStreamBlock, Nil)? = nil) : Tuple(ML::MetalBuffer, ML::MetalBuffer)
      raise ArgumentError.new("Hy-Nomas serial workspace has been released") if @released
      chain.encoder
      if previous = @last_chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("Hy-Nomas serial workspace cannot span in-flight command chains")
        end
      end
      @last_chain = chain
      img, txt, vec, vec_txt, cos, sin = @inputs
      # Validate both registered views before the first GPU append, including
      # a successor whose owner map was changed after setup.
      workspace.validate_for!(@blocks[0], chain, img, txt, vec, vec_txt, cos, sin)
      workspace.validate_for!(@blocks[1], chain, @blocks[0].image_output,
        @blocks[0].text_output, vec, vec_txt, cos, sin)
      @blocks.each_with_index do |block, index|
        img, txt = block.encode_mps(chain, img, txt, vec, vec_txt, cos, sin, workspace)
        after_block.try(&.call(index.to_i32, block))
      end
      {img, txt}
    end

    def release : Nil
      return if @released
      if chain = @last_chain
        raise ArgumentError.new("Complete Hy-Nomas serial command chain before release") unless chain.completed?
      end
      workspace.release
      @views.each { |view| view.buffers.clear }
      @arena.release
      @released = true
    end

    private def workspace : HyNomasMPSWorkspace
      @workspace || raise ArgumentError.new("Hy-Nomas serial MPS plans are unavailable")
    end
  end
end
