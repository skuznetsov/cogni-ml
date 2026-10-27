# Two to eight sequential blocks with common intermediate scratch and one
# prepared MPS workspace. Final output banks are distinct; within-block
# input/scratch aliases are rejected. Each predecessor's output banks are the
# next block's inputs.
require "./hy_nomas_block_mps"

module ML::Video
  # Typed setup/encode contract for a trusted serial weight source. Validation
  # must be CPU-only and complete before the serial owner appends any command;
  # encode! may only append work to the caller's existing chain.
  abstract class HyNomasSerialWeightLoader
    abstract def validate_for!(block_index : Int32, arena : Arena) : Nil
    abstract def encode!(chain : ML::Metal::MPSCommandChain,
                         block_index : Int32, arena : Arena) : Nil
  end

  class HyNomasMPSSerialWorkspace
    @blocks = [] of HyNomasDualStreamBlock
    getter scratch_bytes : Int64 = 0_i64
    getter mps_workspace_bytes : Int64 = 0_i64

    @arena : Arena
    @views = [] of Arena
    @workspace : HyNomasMPSWorkspace? = nil
    @inputs : Array(ML::MetalBuffer)
    @config : HyNomasConfig
    @block_config : HyNomasConfig
    @compact_valid_text : Bool
    @compact_text_input : ML::MetalBuffer? = nil
    @copy_text_prefix = false
    @weight_arenas : Array(Arena)
    @weight_loader : HyNomasSerialWeightLoader?
    @block_inputs = [] of Tuple(ML::MetalBuffer, ML::MetalBuffer)
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    getter block_config : HyNomasConfig
    getter compact_text_input : ML::MetalBuffer?
    getter compaction_resident_bytes : Int64 = 0_i64

    # Exact additional F32 residency for the setup-time compact text input.
    # This pure estimate lets callers include it in memory admission before
    # constructing any Metal-backed owners.
    def self.compaction_resident_bytes(config : HyNomasConfig,
                                       valid_text_tokens : Int32,
                                       storage_tokens : Int32? = nil) : Int64
      compact = config.compact_valid_text_prefix(valid_text_tokens, storage_tokens)
      return 0_i64 if compact.text_rows == config.text_rows
      HyNomasDualStreamBlock.checked_count(compact.text_rows, compact.dim).to_i64 * sizeof(Float32)
    end

    def initialize(config : HyNomasConfig, weights : Array(Arena),
                   img : ML::MetalBuffer, txt : ML::MetalBuffer,
                   vec : ML::MetalBuffer, vec_txt : ML::MetalBuffer,
                   cos : ML::MetalBuffer, sin : ML::MetalBuffer,
                   *, attention_query_chunk : Int32 = 0,
                   attention_heads_per_group : Int32 = 0,
                   valid_text_tokens : Int32? = nil,
                   allow_shared_modulation : Bool = false,
                   full_dit : Bool = false,
                   compact_valid_text : Bool = false,
                   compact_text_tokens : Int32? = nil,
                   weight_loader : HyNomasSerialWeightLoader? = nil)
      config.validate!
      @config = config
      if !compact_valid_text && !compact_text_tokens.nil?
        raise ArgumentError.new("Hy-Nomas compact_text_tokens requires compact_valid_text")
      end
      compact_count = if compact_valid_text
                        valid_text_tokens || raise ArgumentError.new("Hy-Nomas valid-text compaction requires valid_text_tokens")
                      else
                        nil
                      end
      @block_config = if valid = compact_count
                        config.compact_valid_text_prefix(valid, compact_text_tokens)
                      else
                        config
                      end
      @compact_valid_text = compact_valid_text
      @compaction_resident_bytes = if valid = compact_count
                                     self.class.compaction_resident_bytes(config, valid, compact_text_tokens)
                                   else
                                     0_i64
                                   end
      attention_tokens = if compact_valid_text
                           HyNomasMPSWorkspace.attention_tokens_for(@block_config, compact_count)
                         else
                           HyNomasMPSWorkspace.attention_tokens_for(config, valid_text_tokens)
                         end
      count_valid = full_dit ? weights.size == 54 : (2..8).includes?(weights.size)
      unless count_valid
        expected = full_dit ? "exactly 54" : "between two and eight"
        raise ArgumentError.new("Hy-Nomas serial workspace requires #{expected} weight Arenas")
      end
      if full_dit
        unless loader = weight_loader
          raise ArgumentError.new("Hy-Nomas full DiT serial workspace requires a weight loader")
        end
        unless weights.all? { |arena| arena.same?(weights[0]) }
          raise ArgumentError.new("Hy-Nomas full DiT serial weights must share one fixed staging Arena")
        end
        @weight_loader = loader
      else
        if weight_loader
          raise ArgumentError.new("A serial weight loader requires full_dit mode")
        end
        @weight_loader = nil
      end
      @weight_arenas = weights.dup
      if attention_query_chunk != 0 || attention_heads_per_group != 0
        HyNomasCompactAttentionMPSPlan.workspace_geometry(@block_config.batches,
          attention_tokens, @block_config.heads, @block_config.head_dim,
          attention_query_chunk, attention_heads_per_group)
      end
      @inputs = [img, txt, vec, vec_txt, cos, sin]
      @arena = Arena.new
      begin
        block_txt = txt
        if compact_count
          if @block_config.text_tokens < config.text_tokens
            block_txt = @arena.alloc("serial.valid_text_input",
              HyNomasDualStreamBlock.checked_count(@block_config.text_rows, @block_config.dim))
            @copy_text_prefix = true
            unless block_txt.size == @compaction_resident_bytes
              raise ArgumentError.new("Hy-Nomas compact text input residency differs from admission")
            end
          end
          @compact_text_input = block_txt
        end
        HyNomasDualStreamBlock.prepare_scratch!(@arena, @block_config)
        @blocks << HyNomasDualStreamBlock.new(@block_config, weights[0], @arena)
        @block_inputs << {img, block_txt}
        plan = HyNomasDualStreamBlock.scratch_plan(@block_config)
        (1...weights.size).each do |index|
          view = Arena.new
          @views << view
          plan.each do |name, count|
            buffer = if {"hy_nomas.img.output", "hy_nomas.txt.output"}.includes?(name)
                       @arena.alloc("serial.block#{index}.#{name}", count)
                     else
                       @arena[name]
                     end
            # Private, non-owning F32 map. It must be cleared, never released.
            # The first block retains the sole owning Arena through the rooted
            # MPS workspace; no caller receives the borrowed Arena itself.
            view.buffers[name] = buffer
          end
          previous = @blocks[index - 1]
          @blocks << HyNomasDualStreamBlock.new(@block_config, weights[index], view)
          @block_inputs << {previous.image_output, previous.text_output}
        end
        @workspace = HyNomasMPSWorkspace.new(@blocks[0], img, block_txt, vec,
          vec_txt, cos, sin, attention_query_chunk: attention_query_chunk,
          attention_heads_per_group: attention_heads_per_group,
          allow_shared_modulation: allow_shared_modulation,
          valid_text_tokens: compact_valid_text ? compact_count : valid_text_tokens,
          text_rms_tokens: compact_valid_text ? config.text_tokens : nil)
        workspace.register_serial_blocks!(@blocks, @block_inputs,
          full_dit: full_dit)
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

    # Return a detached registry view so callers can inspect order without
    # reordering the owner's execution schedule. This CPU-side copy is never
    # created from the encode path.
    def blocks : Array(HyNomasDualStreamBlock)
      @blocks.dup
    end

    # Public CPU-only preflight for callers that have a GPU prelude to append.
    # This validates every registered block and loader binding without claiming
    # the chain or appending commands. `encode` repeats it immediately before
    # its own first append to close the validation/use gap.
    def validate_for!(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("Hy-Nomas serial workspace has been released") if @released
      chain.encoder
      if previous = @last_chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("Hy-Nomas serial workspace cannot span in-flight command chains")
        end
      end
      _, full_txt, vec, vec_txt, cos, sin = @inputs
      validate_compaction_input!(full_txt) if @compact_valid_text
      @blocks.each_with_index do |block, index|
        img, txt = @block_inputs[index]
        if loader = @weight_loader
          loader.validate_for!(index.to_i32, @weight_arenas[index])
        end
        workspace.validate_for!(block, chain, img, txt, vec, vec_txt, cos, sin,
          bind_chain: false)
      end
    end

    # after_block is a trusted diagnostic hook for preallocated GPU snapshots.
    # It may append copies to the same chain's current encoder, but must not
    # mutate bindings, commit command buffers, wait, or recursively encode.
    # Those diagnostic dispatches belong outside performance measurements.
    # If it raises after an append, the caller must drain the chain before
    # releasing this owner, exactly as for an encode failure.
    def encode(chain : ML::Metal::MPSCommandChain,
               *, after_block : Proc(Int32, HyNomasDualStreamBlock, Nil)? = nil) : Tuple(ML::MetalBuffer, ML::MetalBuffer)
      validate_for!(chain)
      _, _, vec, vec_txt, cos, sin = @inputs
      # Claim this chain only after the complete preflight succeeds. The first
      # block's normal backend validation then binds the workspace itself.
      @last_chain = chain
      if @copy_text_prefix
        source = @inputs[1]
        destination = @compact_text_input || raise ArgumentError.new("Hy-Nomas compact text input is unavailable")
        count = HyNomasDualStreamBlock.checked_count(@block_config.text_rows, @block_config.dim)
        MetalOps.copy(chain.encoder, source, destination, count)
      end
      @blocks.each_with_index do |block, index|
        img, txt = @block_inputs[index]
        @weight_loader.try(&.encode!(chain, index.to_i32, @weight_arenas[index]))
        img, txt = block.encode_mps(chain, img, txt, vec, vec_txt, cos, sin, workspace)
        after_block.try(&.call(index.to_i32, block))
      end
      {@blocks.last.image_output, @blocks.last.text_output}
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

    private def validate_compaction_input!(source : ML::MetalBuffer) : Nil
      expected_source = @config.text_rows.to_i64 * @config.dim
      MetalOps.check(source, expected_source)
      unless source.size == expected_source * sizeof(Float32)
        raise ArgumentError.new("Hy-Nomas full text input has an unexpected extent")
      end
      destination = @compact_text_input || raise ArgumentError.new("Hy-Nomas compact text input is unavailable")
      expected_destination = @block_config.text_rows.to_i64 * @block_config.dim
      MetalOps.check(destination, expected_destination)
      unless destination.size == expected_destination * sizeof(Float32)
        raise ArgumentError.new("Hy-Nomas compact text input has an unexpected extent")
      end
      if @copy_text_prefix
        MetalOps.distinct(source, destination)
        @arena.buffers.each_value { |buffer| MetalOps.distinct(source, buffer) }
      elsif !source.same?(destination)
        raise ArgumentError.new("Hy-Nomas full text input identity changed")
      end
    end
  end
end
