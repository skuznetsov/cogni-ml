require "./attention"
require "./linear"
require "./math"
require "../loopcd"

module ML::GGUF
  # One command per decode token, known continuation block, or causal prefill
  # tile. Only IDs and the requested logits cross the CPU/GPU boundary.
  class Inkling2Resident
    private DEPTH_EPS          = 1e-6_f32
    private DEPTH_DRAFT_PASSES =    3_i32
    private alias MathOps = Inkling2ResidentMath

    private record Visit, attention : Inkling2ResidentAttention,
      post_conv : Inkling2ResidentConvolution, ffn_conv : Inkling2ResidentConvolution do
      def begin_speculation(start : Int32) : Nil
        attention.begin_speculation(start)
        post_conv.begin_speculation(start)
        ffn_conv.begin_speculation(start)
      end

      def finish_speculation(end_position : Int32) : Nil
        attention.finish_speculation(end_position)
        post_conv.finish_speculation(end_position)
        ffn_conv.finish_speculation(end_position)
      end

      def rewind(end_position : Int32) : Nil
        attention.rewind(end_position)
        post_conv.rewind(end_position)
        ffn_conv.rewind(end_position)
      end

      def reset : Nil
        attention.reset
        post_conv.reset
        ffn_conv.reset
      end

      def finish : Nil
        attention.finish
      end

      def close : Nil
        attention.close
        post_conv.close
        ffn_conv.close
      end
    end

    private class Tile
      getter encoder : ML::Metal::ComputeEncoder
      getter start : Int32
      getter rows : Int32
      getter width : Int32
      getter elements : Int32
      getter? decode_block : Bool
      getter current : Inkling2View
      getter next_state : Inkling2View
      getter blocks : Inkling2View
      property block_count : Int32 = 0

      def initialize(@encoder, @scratch : Inkling2ResidentScratch, @start, @rows, @width, block_slots : Int32,
                     @decode_block : Bool)
        @elements = @rows * @width
        @current = hidden
        @next_state = hidden
        @blocks = @scratch.take(block_slots * @elements)
      end

      def hidden : Inkling2View
        @scratch.take(@elements)
      end

      def swap : Nil
        @current, @next_state = @next_state, @current
      end
    end

    @config : Inkling2Config
    @prelude_blocks : Int32
    @core_blocks : Int32
    @max_seq : Int32
    @block_slots : Int32
    @constants : ML::MetalBuffer
    @ids : ML::MetalBuffer
    @logits : ML::MetalBuffer
    @block_logits : ML::MetalBuffer? = nil
    @greedy_ids : ML::MetalBuffer? = nil
    @scratch : Inkling2ResidentScratch
    @visits = [] of Visit
    @reference_epilogue_visits = [] of Visit
    @float_offsets = {} of UInt64 => Int64
    @shared = [] of ML::MetalBuffer
    @depth_checkpoint : ML::MetalBuffer? = nil
    @depth_speculation : Bool
    @depth_frontier = 0_i32
    @depth_block_start : Int32? = nil
    @depth_draft_rows = 0_i32
    @depth_verified = false
    @depth_failed = false

    def self.supported?(weights : Inkling2Weights) : Bool
      config = weights.config
      max_head_dim = Inkling2ResidentAttention::MAX_HEAD_DIM
      return false unless config.head_dim <= max_head_dim && config.rel_dim <= max_head_dim
      embedding = weights.token_embd.type
      return false unless embedding.f32? || embedding.q8_0? || embedding.iq4_nl?
      return false unless Inkling2ResidentLinear.projection_supported?(weights.output)

      weights.layers.all? do |layer|
        attention = layer.attention
        projections = {attention.q, attention.k, attention.v, attention.gate, attention.sink,
                       attention.rel_proj, attention.output, attention.k_conv_proj, attention.v_conv_proj,
                       layer.router}
        projections.all? { |matrix| !matrix || Inkling2ResidentLinear.projection_supported?(matrix) } &&
          Inkling2ResidentLinear.moe_supported?(layer, config) &&
          (!layer.controller || controller_supported?(layer.controller.not_nil!))
      end
    end

    # The growable 241 KiB-per-row arena and half/M64 fast kernels are bounded
    # to the measured Mini layout. Other resident-compatible checkpoints keep
    # the paged allocator and their existing kernel fallbacks.
    def self.fast_prefill_supported?(weights : Inkling2Weights, passes : Int32) : Bool
      {% if flag?(:aarch64) %}
        config = weights.config
        return false unless passes == 3
        return false unless config.n_layer == 24 && config.n_embd == 2304 && config.res_block_size == 6
        return false unless config.core_start == 6 && config.core_end == 18 && config.gate_rank == 64
        return false unless config.n_head == 24 && config.n_head_kv == 8 && config.head_dim == 128 && config.rel_dim == 16
        return false unless config.n_expert == 128 && config.n_expert_used == 8 && config.expert_dim == 1152
        return false unless config.n_embd % 32 == 0 && config.expert_dim % 32 == 0
        supported?(weights)
      {% else %}
        false
      {% end %}
    end

    private def self.controller_supported?(controller : Inkling2LoopWeights) : Bool
      {controller.down, controller.gain, controller.forget}.all? do |matrices|
        matrices.all? { |matrix| Inkling2ResidentLinear.projection_supported?(matrix) }
      end
    end

    def initialize(@weights : Inkling2Weights, @passes : Int32, physical_visits : Array(Int32),
                   max_seq : Int32, @prefill_rows : Int32, *, @append_prefill_rows : Int32? = nil,
                   rollback_rows : Int32 = 0, @depth_speculation : Bool = false, @fast_prefill : Bool = false,
                   @loopcd : Inkling2LoopCD? = nil)
      @config = @weights.config
      @max_seq = max_seq
      @scratch = Inkling2ResidentScratch.new(@fast_prefill)
      @prelude_blocks = @config.core_start // @config.res_block_size
      @core_blocks = (@config.core_end - @config.core_start) // @config.res_block_size
      @block_slots = (@config.n_layer - 1) // @config.res_block_size + 1
      if @depth_speculation
        raise ArgumentError.new("Inkling 2 depth speculation requires exactly 10 target passes") unless @passes == 10
        raise ArgumentError.new("Inkling 2 depth speculation requires four rollback rows") unless rollback_rows == 4
        raise ArgumentError.new("Inkling 2 depth speculation requires at least one core block") unless @core_blocks > 0
      end
      prefill_mode = @fast_prefill ? Inkling2ResidentAttention::PrefillMode::Fast : Inkling2ResidentAttention::PrefillMode.from_env
      tile_routes, indirect_dispatch = if @fast_prefill
                                         {64, false}
                                       else
                                         case ENV["INKLING2_MOE_PREFILL_ROUTES"]?
                                         when nil, "16"     then {16, false}
                                         when "16_indirect" then {16, true}
                                         when "32"          then {32, false}
                                         when "64"          then {64, false}
                                         else
                                           raise ArgumentError.new("INKLING2_MOE_PREFILL_ROUTES must be 16, 16_indirect, 32, or 64")
                                         end
                                       end
      @linear = Inkling2ResidentLinear.new(@weights.gguf, prefill_expert_tile_routes: tile_routes,
        prefill_indirect_dispatch: indirect_dispatch, fast_prefill: @fast_prefill)
      @constants, @ids, @logits = prepare_buffers(physical_visits, max_seq, rollback_rows, prefill_mode)
    end

    def fast_prefill? : Bool
      @fast_prefill
    end

    def reserved_scratch_bytes : Int64
      @scratch.reserved_bytes
    end

    def forward(token : Int32, position : Int32, with_logits : Bool) : Array(Float32)?
      @ids.contents.as(Int32*).value = token
      evaluate(position, 1, with_logits, decode_step: true)
    end

    def forward_greedy(token : Int32, position : Int32) : Int32
      @ids.contents.as(Int32*).value = token
      output = greedy_ids_buffer
      submit_tile(position, 1, output, greedy: true, guide_hidden: true)
      output.contents.as(Int32*).value
    end

    def decode_block4(tokens : StaticArray(Int32, 4), position : Int32) : StaticArray(Array(Float32), 4)
      ensure_ordinary_mode! if @depth_speculation
      ids = @ids.contents.as(Int32*)
      tokens.each_with_index { |token, row| ids[row] = token }
      vocab = @config.vocab_size
      logits = @block_logits ||= allocate_shared(tokens.size.to_i64 * vocab * sizeof(Float32))
      submit_tile(position, tokens.size, logits, decode_block: true)
      read_block_logits(logits)
    end

    def decode_block4_greedy(tokens : StaticArray(Int32, 4), position : Int32) : StaticArray(Int32, 4)
      ensure_ordinary_mode! if @depth_speculation
      ids = @ids.contents.as(Int32*)
      tokens.each_with_index { |token, row| ids[row] = token }
      output = greedy_ids_buffer
      submit_tile(position, tokens.size, output, decode_block: true, greedy: true)
      data = output.contents.as(Int32*)
      StaticArray(Int32, 4).new { |row| data[row] }
    end

    def rewind_block(end_position : Int32) : Nil
      if @depth_speculation
        ensure_ordinary_mode!
        raise ArgumentError.new("Inkling 2 resident rewind end is outside the current frontier") unless 0 <= end_position <= @depth_frontier
      end
      @visits.each(&.rewind(end_position))
      @reference_epilogue_visits.each(&.rewind(end_position))
      @depth_frontier = end_position if @depth_speculation
    end

    def prefill(tokens : Indexable(Int32), position : Int32) : Array(Float32)
      ensure_ordinary_mode! if @depth_speculation
      tile_rows = @prefill_rows
      if (append_rows = @append_prefill_rows) && position > 0 && tokens.size >= append_rows
        tile_rows = append_rows
      end
      # Explicit wide tiles merge 128-row tiles. Keep a short final tile on
      # the same projection and expert paths as an explicit 128-row prefill.
      batched_end = tokens.size
      if @prefill_rows > Inkling2Engine::MAX_AUTO_PREFILL_TILE
        tail = tokens.size % Inkling2Engine::MAX_AUTO_PREFILL_TILE
        batched_end -= tail if tail < Inkling2ResidentLinear::MIN_GROUPED_ROWS
      end
      ids = @ids.contents.as(Int32*)
      offset = 0
      result = nil
      while offset < tokens.size
        end_offset = offset < batched_end ? batched_end : tokens.size
        rows = Math.min(tile_rows, end_offset - offset)
        rows.times { |row| ids[row] = tokens[offset + row] }
        result = evaluate(position + offset, rows, offset + rows == tokens.size)
        offset += rows
      end
      result.not_nil!
    end

    def key_value_at(visit : Int32, index : Int32, end_position : Int32)
      position = end_position - kv_count(visit, end_position) + index
      @visits[visit].attention.key_value_at(position, end_position)
    end

    def kv_count(visit : Int32, end_position : Int32) : Int32
      Math.min(@visits[visit].attention.capacity, end_position)
    end

    def draft_kv_count(visit : Int32, end_position : Int32) : Int32
      Math.min(draft_visit(visit, end_position).attention.capacity, end_position)
    end

    def draft_key_value_at(visit : Int32, index : Int32, end_position : Int32) : {Array(Float32), Array(Float32), Int32}
      owner = draft_visit(visit, end_position).attention
      count = Math.min(owner.capacity, end_position)
      raise IndexError.new("Inkling 2 draft attention cache index out of bounds") unless 0 <= index < count
      position = end_position - count + index
      owner.key_value_at(position, end_position)
    end

    def begin_depth_block(position : Int32) : Nil
      ensure_depth_enabled!
      ensure_depth_healthy!
      raise "Inkling 2 depth block is already active" if @depth_block_start
      unless position >= 0 && position == @depth_frontier
        raise ArgumentError.new("Inkling 2 depth block must begin at the current position #{@depth_frontier}")
      end
      if position.to_i64 + 4 > @max_seq
        raise ArgumentError.new("Inkling 2 depth block exceeds the context of #{@max_seq} tokens")
      end

      begin
        @visits.each(&.begin_speculation(position))
        @reference_epilogue_visits.each(&.begin_speculation(position))
        @depth_block_start = position
        @depth_draft_rows = 0
        @depth_verified = false
      rescue ex
        @depth_failed = true
        raise ex
      end
    end

    def draft_depth_row(token : Int32, position : Int32, *, logits : Bool) : Array(Float32)?
      ensure_depth_enabled!
      ensure_depth_healthy!
      start = active_depth_start
      unless position == start + @depth_draft_rows && @depth_draft_rows < 4
        raise ArgumentError.new("Inkling 2 depth draft rows must be consecutive and limited to four")
      end
      raise ArgumentError.new("Token #{token} is outside the Inkling 2 vocabulary") unless token.in?(0...@config.vocab_size)

      @ids.contents.as(Int32*).value = token
      submit_depth_draft_row(position, @depth_draft_rows, logits)
      @depth_draft_rows += 1
      logits ? @logits.read(@config.vocab_size) : nil
    end

    def verify_depth_block(position : Int32) : StaticArray(Array(Float32), 4)
      ensure_depth_enabled!
      ensure_depth_healthy!
      start = active_depth_start
      raise ArgumentError.new("Inkling 2 depth verification position does not match its draft block") unless position == start
      raise ArgumentError.new("Inkling 2 depth verification requires four draft rows") unless @depth_draft_rows == 4
      raise "Inkling 2 depth block has already been verified" if @depth_verified

      result = submit_depth_verification(position)
      @depth_verified = true
      result
    end

    def finish_depth_block(end_position : Int32) : Nil
      ensure_depth_enabled!
      ensure_depth_healthy!
      start = active_depth_start
      unless @depth_verified && end_position.in?(start..(start + 4))
        raise ArgumentError.new("Inkling 2 depth block must be verified and retain an end within its four rows")
      end

      begin
        @visits.each(&.finish_speculation(end_position))
        @reference_epilogue_visits.each(&.finish_speculation(end_position))
        @depth_frontier = end_position
        clear_depth_block
      rescue ex
        @depth_failed = true
        raise ex
      end
    end

    def reset : Nil
      @visits.each(&.reset)
      @reference_epilogue_visits.each(&.reset)
      @depth_frontier = 0 if @depth_speculation
      @depth_failed = false
      clear_depth_block
    end

    def close : Nil
      @visits.each(&.close)
      @visits.clear
      @reference_epilogue_visits.each(&.close)
      @reference_epilogue_visits.clear
      @shared.each(&.release)
      @shared.clear
      @depth_checkpoint.try(&.release)
      @depth_checkpoint = nil
      @scratch.close
      @linear.close
    end

    private def prepare_buffers(physical_visits : Array(Int32), max_seq : Int32, rollback_rows : Int32,
                                prefill_mode : Inkling2ResidentAttention::PrefillMode)
      physical_visits.each do |physical|
        sliding = @config.sliding?(physical)
        @visits << Visit.new(
          Inkling2ResidentAttention.new(@config, sliding, max_seq, rollback_rows: rollback_rows, prefill_mode: prefill_mode),
          Inkling2ResidentConvolution.new(@config.n_embd, @config.conv_size, rollback_rows: rollback_rows),
          Inkling2ResidentConvolution.new(@config.n_embd, @config.conv_size, rollback_rows: rollback_rows),
        )
      end
      if @depth_speculation || @loopcd.try(&.mode.logits?)
        (@config.core_end...@config.n_layer).each do |physical|
          sliding = @config.sliding?(physical)
          @reference_epilogue_visits << Visit.new(
            Inkling2ResidentAttention.new(@config, sliding, max_seq, rollback_rows: rollback_rows, prefill_mode: prefill_mode),
            Inkling2ResidentConvolution.new(@config.n_embd, @config.conv_size, rollback_rows: rollback_rows),
            Inkling2ResidentConvolution.new(@config.n_embd, @config.conv_size, rollback_rows: rollback_rows),
          )
        end
      end
      @depth_checkpoint = allocate_private(checkpoint_bytes) if @depth_speculation
      id_rows = Math.max(4, @append_prefill_rows || @prefill_rows)
      {upload_constants, allocate_shared(id_rows.to_i64 * sizeof(Int32)),
       allocate_shared(@config.vocab_size.to_i64 * sizeof(Float32))}
    rescue ex
      close
      raise ex
    end

    private def allocate_shared(bytes : Int64) : ML::MetalBuffer
      buffer = ML::MetalBuffer.new(bytes)
      @shared << buffer
      buffer
    end

    private def allocate_private(bytes : Int64) : ML::MetalBuffer
      ML::MetalBuffer.new(bytes, ML::StorageMode::Private)
    end

    private def checkpoint_bytes : Int64
      planes = 1_i64 + @prelude_blocks + @core_blocks
      elements = planes * 4_i64 * @config.n_embd
      raise ArgumentError.new("Inkling 2 depth checkpoint exceeds Int32 elements") if elements > Int32::MAX
      elements * sizeof(Float32)
    end

    private def checkpoint_view(plane : Int32, row : Int32 = 0) : Inkling2View
      raise ArgumentError.new("Inkling 2 depth checkpoint plane is out of bounds") unless plane.in?(0...checkpoint_plane_count)
      raise ArgumentError.new("Inkling 2 depth checkpoint row is out of bounds") unless row.in?(0...4)
      buffer = @depth_checkpoint || raise "Inkling 2 depth checkpoint is not allocated"
      offset = (plane.to_i64 * 4 + row) * @config.n_embd * sizeof(Float32)
      Inkling2View.new(buffer, offset)
    end

    private def checkpoint_plane_count : Int32
      1 + @prelude_blocks + @core_blocks
    end

    private def upload_constants : ML::MetalBuffer
      values = [@weights.token_embd_norm, @weights.output_norm, @weights.output_res_query]
      @weights.layers.each do |layer|
        attention = layer.attention
        values.concat([layer.q_norm, layer.k_norm, layer.v_norm, layer.ffn_res_query, layer.ffn_norm,
                       layer.ffn_post_conv, attention.q_clip, attention.rel_clip, attention.rel_bias,
                       attention.k_conv, attention.v_conv, attention.post_conv])
        layer.attn_res_query.try { |array| values << array }
        layer.router_bias.try { |array| values << array }
        layer.controller.try { |controller| values << controller.query }
      end

      size = 0_i64
      unique = values.uniq(&.object_id)
      unique.each do |array|
        @float_offsets[array.object_id] = size
        size += array.size.to_i64 * sizeof(Float32)
      end
      buffer = allocate_shared(size)
      base = buffer.contents.as(UInt8*)
      unique.each do |array|
        (base + @float_offsets[array.object_id]).as(Float32*).copy_from(array.to_unsafe, array.size)
      end
      buffer
    end

    private def constant(array : Array(Float32)) : Inkling2View
      Inkling2View.new(@constants, @float_offsets[array.object_id])
    end

    private def evaluate(position : Int32, rows : Int32, with_logits : Bool, *, decode_step : Bool = false) : Array(Float32)?
      submit_tile(position, rows, with_logits ? @logits : nil, guide_hidden: with_logits || decode_step)
      @logits.read(@config.vocab_size) if with_logits
    end

    private def greedy_ids_buffer : ML::MetalBuffer
      @greedy_ids ||= allocate_shared(4_i64 * sizeof(Int32))
    end

    private def submit_depth_draft_row(position : Int32, row : Int32, with_logits : Bool) : Nil
      submit_command do |encoder|
        encode_depth_draft_row(encoder, position, row, with_logits)
      end
    end

    private def encode_depth_draft_row(encoder : ML::Metal::ComputeEncoder, position : Int32,
                                       checkpoint_row : Int32, with_logits : Bool) : Nil
      rows = 1
      tile = Tile.new(encoder, @scratch, position, rows, @config.n_embd, @block_slots, false)
      prelude = tile.hidden
      banks = StaticArray[
        @scratch.take(@core_blocks * tile.elements),
        @scratch.take(@core_blocks * tile.elements),
      ]

      visit = encode_prelude(tile, prelude)
      MathOps.copy(encoder, prelude, checkpoint_view(0, checkpoint_row), tile.elements)
      @prelude_blocks.times do |plane|
        source = tile.blocks.slice(plane * tile.elements)
        MathOps.copy(encoder, source, checkpoint_view(1 + plane, checkpoint_row), tile.elements)
      end

      previous_bank = nil
      DEPTH_DRAFT_PASSES.times do |pass|
        visit = encode_core_pass(tile, visit, prelude, previous_bank, restore_prelude: pass > 0)

        bank = banks[pass % 2]
        encode_core_bank(tile, bank)
        previous_bank = bank

        if pass + 1 == DEPTH_DRAFT_PASSES
          @core_blocks.times do |plane|
            MathOps.copy(encoder, bank.slice(plane * tile.elements),
              checkpoint_view(1 + @prelude_blocks + plane, checkpoint_row), tile.elements)
          end
        end
      end

      encode_epilogue_branch(tile, @reference_epilogue_visits, with_logits ? @logits : nil)
    end

    private def submit_depth_verification(position : Int32) : StaticArray(Array(Float32), 4)
      logits = @block_logits ||= allocate_shared(4_i64 * @config.vocab_size * sizeof(Float32))
      submit_command do |encoder|
        encode_depth_verification(encoder, position, logits)
      end

      read_block_logits(logits)
    end

    private def read_block_logits(buffer : ML::MetalBuffer) : StaticArray(Array(Float32), 4)
      data = buffer.contents.as(Float32*)
      vocab = @config.vocab_size
      StaticArray(Array(Float32), 4).new do |row|
        result = Array(Float32).new(vocab, 0_f32)
        result.to_unsafe.copy_from(data + row * vocab, vocab)
        result
      end
    end

    private def encode_depth_verification(encoder : ML::Metal::ComputeEncoder, position : Int32,
                                          logits : ML::MetalBuffer) : Nil
      rows = 4
      tile = Tile.new(encoder, @scratch, position, rows, @config.n_embd, @block_slots, true)
      elements = tile.elements
      prelude = tile.hidden
      MathOps.copy(encoder, checkpoint_view(0), prelude, elements)
      @prelude_blocks.times do |plane|
        MathOps.copy(encoder, checkpoint_view(1 + plane), tile.blocks.slice(plane * elements), elements)
      end

      banks = StaticArray[
        @scratch.take(@core_blocks * elements),
        @scratch.take(@core_blocks * elements),
      ]
      previous_bank = checkpoint_view(1 + @prelude_blocks)
      visit = @config.core_start + DEPTH_DRAFT_PASSES * (@config.core_end - @config.core_start)
      (@passes - DEPTH_DRAFT_PASSES).times do |pass_offset|
        pass = DEPTH_DRAFT_PASSES + pass_offset
        visit = encode_core_pass(tile, visit, prelude, previous_bank, restore_prelude: true)

        if pass + 1 < @passes
          bank = banks[pass % 2]
          encode_core_bank(tile, bank)
          previous_bank = bank
        end
      end

      (@config.core_end...@config.n_layer).each do |physical|
        encode_layer(physical, visit, tile, nil)
        visit += 1
      end
      encode_logits(tile, logits)
    end

    private def submit_tile(position : Int32, rows : Int32, logits : ML::MetalBuffer?, *,
                            decode_block : Bool = false, greedy : Bool = false, guide_hidden : Bool = false) : Nil
      raise ArgumentError.new("Inkling 2 LoopCD supports only scalar decode") if decode_block && @loopcd
      if @depth_speculation
        ensure_ordinary_mode!
        raise ArgumentError.new("Inkling 2 resident tiles must be contiguous from position #{@depth_frontier}") unless position == @depth_frontier
      end
      submit_command(rows) do |encoder|
        encode_tile(encoder, position, rows, logits, decode_block, greedy, guide_hidden)
      end
      @depth_frontier = position + rows if @depth_speculation
    end

    private def submit_command(rows : Int32 = 0, & : ML::Metal::ComputeEncoder -> Nil) : Nil
      @scratch.prepare_command(rows)
      command = ML::Metal::CommandBuffer.new
      begin
        encoder = ML::Metal::ComputeEncoder.new(command)
        begin
          yield encoder
        ensure
          encoder.end_encoding
        end
        command.commit_and_wait
      rescue ex
        @depth_failed = true if @depth_speculation
        raise ex
      ensure
        command.discard unless command.committed?
        @visits.each(&.finish)
        @reference_epilogue_visits.each(&.finish)
      end
    end

    private def encode_tile(encoder : ML::Metal::ComputeEncoder, position : Int32, rows : Int32,
                            logits : ML::MetalBuffer?, decode_block : Bool, greedy : Bool, guide_hidden : Bool) : Nil
      tile = Tile.new(encoder, @scratch, position, rows, @config.n_embd, @block_slots, decode_block)
      prelude = tile.hidden if @passes > 1
      banks = if @passes > 1
                first = @scratch.take(@core_blocks * tile.elements)
                second = @fast_prefill && !decode_block && !@depth_speculation && rows > 1 ? first : @scratch.take(@core_blocks * tile.elements)
                StaticArray[first, second]
              end

      visit = encode_prelude(tile, prelude)

      previous_bank = nil
      loopcd = @loopcd
      # Keep the head inputs outside the reference epilogue's scratch mark.
      # Row zero is the target and row one is the pass-two reference.
      loopcd_inputs = @scratch.take(2 * tile.width) if logits && loopcd && loopcd.mode.logits?
      @passes.times do |pass|
        visit = encode_core_pass(tile, visit, prelude, previous_bank, restore_prelude: pass > 0)
        if pass + 1 < @passes
          # The first core block's input is the shared prelude. Save the
          # remaining normalized blocks and the terminal state for messaging.
          bank = banks.not_nil![pass % 2]
          encode_core_bank(tile, bank)
          previous_bank = bank
        end
        if @depth_speculation && pass + 1 == DEPTH_DRAFT_PASSES
          # Pass 4 restores the saved prelude and block count before it resumes
          # the target path, so the shallow epilogue can reuse this tile.
          encode_epilogue_branch(tile, @reference_epilogue_visits)
        elsif loopcd && loopcd.mode.logits? && pass + 1 == 2
          # Keep the reference convolution and KV history contiguous across
          # prefill tiles, including tiles where no logit row is requested.
          encode_epilogue_branch(tile, @reference_epilogue_visits, normalized: loopcd_inputs.try(&.slice(tile.width)))
        end
      end

      if loopcd && !loopcd.mode.logits? && guide_hidden
        encode_hidden_guidance(tile, previous_bank.not_nil!, loopcd)
      end

      (@config.core_end...@config.n_layer).each do |physical|
        encode_layer(physical, visit, tile, nil)
        visit += 1
      end
      encode_logits(tile, logits, greedy: greedy, loopcd_inputs: loopcd_inputs) if logits
    end

    private def encode_hidden_guidance(tile : Tile, previous_bank : Inkling2View, loopcd : Inkling2LoopCD) : Nil
      # Guide the final prompt row and each decode step. The next-state buffer
      # is available until the epilogue starts.
      row_offset = tile.elements - tile.width
      previous_terminal = previous_bank.slice((@core_blocks - 1) * tile.elements + row_offset)
      terminal = tile.current.slice(row_offset)
      normalized = tile.next_state.slice(row_offset)
      MathOps.rms(tile.encoder, terminal, normalized, nil, tile.width, 1, DEPTH_EPS)
      MathOps.extrapolate(tile.encoder, normalized, previous_terminal, terminal, tile.width, loopcd.strength)

      if loopcd.mode.hidden_blocks?
        # Passes share the prelude, including the first core block's input.
        (@core_blocks - 1).times do |block|
          final = tile.blocks.slice((@prelude_blocks + 1 + block) * tile.elements + row_offset)
          reference = previous_bank.slice(block * tile.elements + row_offset)
          MathOps.extrapolate(tile.encoder, final, reference, final, tile.width, loopcd.strength)
        end
      end
    end

    private def encode_output_normalized(tile : Tile, rows : Int32, output : Inkling2View? = nil) : Inkling2View
      mixed = mix(tile, tile.current, @weights.output_res_query, 1)
      selected = mixed.slice((tile.rows - rows) * tile.width)
      normalized = output || @scratch.take(rows * tile.width)
      MathOps.rms(tile.encoder, selected, normalized, constant(@weights.output_norm), tile.width, rows, @config.rms_eps)
      normalized
    end

    private def encode_logits(tile : Tile, buffer : ML::MetalBuffer, *, greedy : Bool = false,
                              loopcd_inputs : Inkling2View? = nil) : Nil
      encoder = tile.encoder
      rows = tile.decode_block? ? tile.rows : 1
      normalized = encode_output_normalized(tile, rows, loopcd_inputs)
      padded_vocab = @weights.output.out_dim
      vocab = @config.vocab_size
      projection_rows = loopcd_inputs && @weights.output.type.q8_0? ? 2 : rows
      logits = @scratch.take((loopcd_inputs ? 2 : rows) * padded_vocab)
      @linear.project(encoder, @weights.output, normalized, logits, projection_rows, decode_block: tile.decode_block?)
      output = Inkling2View.new(buffer)
      if loopcd_inputs
        reference = logits.slice(padded_vocab)
        if projection_rows == 1
          # Preserve the scalar reduction for F32 output weights.
          @linear.project(encoder, @weights.output, loopcd_inputs.slice(tile.width), reference, 1)
        end
        MathOps.softcap(encoder, reference, reference, @config.logit_softcap, vocab)
        scores = greedy ? logits : output
        MathOps.extrapolate(encoder, logits, reference, scores, vocab,
          @loopcd.not_nil!.strength, cap: @config.logit_softcap)
        if greedy
          MathOps.greedy_ids(encoder, @scratch, scores, output, 0_f32, vocab, padded_vocab, 1)
        end
      elsif greedy
        MathOps.greedy_ids(encoder, @scratch, logits, output, @config.logit_softcap, vocab, padded_vocab, rows)
      else
        rows.times do |row|
          MathOps.softcap(encoder, logits.slice(row * padded_vocab), output.slice(row * vocab), @config.logit_softcap, vocab)
        end
      end
    end

    private def draft_visit(visit : Int32, end_position : Int32) : Visit
      ensure_depth_enabled!
      ensure_depth_healthy!
      raise ArgumentError.new("Inkling 2 draft history end position is out of bounds") unless end_position.in?(0..@max_seq)
      unless visit.in?(0...@reference_epilogue_visits.size)
        raise IndexError.new("Inkling 2 draft layer visit out of bounds")
      end
      @reference_epilogue_visits[visit]
    end

    private def active_depth_start : Int32
      @depth_block_start || raise "Inkling 2 depth block has not begun"
    end

    private def clear_depth_block : Nil
      @depth_block_start = nil
      @depth_draft_rows = 0
      @depth_verified = false
    end

    private def ensure_depth_enabled! : Nil
      raise ArgumentError.new("Inkling 2 resident depth speculation is disabled") unless @depth_speculation
    end

    private def ensure_depth_healthy! : Nil
      raise "Inkling 2 resident depth block failed; reset the sequence" if @depth_failed
    end

    private def ensure_ordinary_mode! : Nil
      ensure_depth_healthy!
      raise "Inkling 2 depth block is active" if @depth_block_start
    end

    private def encode_prelude(tile : Tile, saved : Inkling2View?) : Int32
      encoder = tile.encoder
      mark = @scratch.mark
      embedding = tile.hidden
      matrix = @weights.token_embd
      unless MathOps.embed(encoder, @linear.view(matrix), matrix.type, Inkling2View.new(@ids), embedding, tile.width, tile.rows)
        raise "Inkling 2 resident embedding type was not admitted"
      end
      MathOps.rms(encoder, embedding, tile.current, constant(@weights.token_embd_norm), tile.width, tile.rows, @config.rms_eps)
      @scratch.rewind(mark)

      visit = 0
      (0...@config.core_start).each do |physical|
        encode_layer(physical, visit, tile, nil)
        visit += 1
      end
      if output = saved
        MathOps.copy(encoder, tile.current, output, tile.elements)
      end
      visit
    end

    private def encode_core_pass(tile : Tile, visit : Int32, prelude : Inkling2View?,
                                 previous_bank : Inkling2View?, *, restore_prelude : Bool) : Int32
      if restore_prelude
        tile.block_count = @prelude_blocks
        MathOps.copy(tile.encoder, prelude.not_nil!, tile.current, tile.elements)
      end
      (@config.core_start...@config.core_end).each do |physical|
        encode_layer(physical, visit, tile, previous_bank)
        visit += 1
      end
      visit
    end

    private def encode_core_bank(tile : Tile, bank : Inkling2View) : Nil
      if @core_blocks > 1
        source = tile.blocks.slice((@prelude_blocks + 1) * tile.elements)
        MathOps.copy(tile.encoder, source, bank, (@core_blocks - 1) * tile.elements)
      end
      terminal = bank.slice((@core_blocks - 1) * tile.elements)
      MathOps.rms(tile.encoder, tile.current, terminal, nil, tile.width, tile.rows, DEPTH_EPS)
    end

    private def encode_epilogue_branch(tile : Tile, visits : Array(Visit), logits : ML::MetalBuffer? = nil,
                                       *, normalized : Inkling2View? = nil) : Nil
      mark = @scratch.mark
      begin
        visit = 0
        (@config.core_end...@config.n_layer).each do |physical|
          encode_layer(physical, visit, tile, nil, visits)
          visit += 1
        end
        if normalized
          encode_output_normalized(tile, 1, normalized)
        elsif logits
          encode_logits(tile, logits)
        end
      ensure
        @scratch.rewind(mark)
      end
    end

    private def encode_layer(physical : Int32, visit_index : Int32, tile : Tile, previous_bank : Inkling2View?,
                             visits : Array(Visit) = @visits) : Nil
      layer = @weights.layers[physical]
      visit = visits[visit_index]
      encoder = tile.encoder
      mark = @scratch.mark
      residual = tile.hidden
      reuse_moe_input = @fast_prefill && !tile.decode_block? && !@depth_speculation && tile.rows > 1
      moe_input_mark = @scratch.mark if reuse_moe_input
      controller = previous_bank ? layer.controller.not_nil! : nil
      messages = if bank = previous_bank
                   output = @scratch.take(4 * tile.elements)
                   MathOps.depth_mix(encoder, bank, @core_blocks, nil, constant(controller.not_nil!.query),
                     output, tile.width, tile.rows, 4)
                   output
                 end

      attention_mark = @scratch.mark
      recycle_attention_inputs = @fast_prefill && !tile.decode_block? && !@depth_speculation && tile.rows > 1
      attended = attend(tile, layer.attention, visit, recycle_attention_inputs) do
        sliding = @config.sliding?(physical)
        query_count = sliding ? 1 : 3
        queries = layer.attn_res_query
        mixed = queries ? mix(tile, tile.current, queries, query_count) : tile.current
        q = mixed
        distinct_inputs = query_count == 3 && !queries.nil?
        k = distinct_inputs ? mixed.slice(tile.elements) : mixed
        v = distinct_inputs ? mixed.slice(2 * tile.elements) : mixed
        if controller && messages
          q, k, v = control_attention(tile, controller, {q, k, v}, messages)
        end
        q_in = normalize(tile, q, layer.q_norm)
        if sliding && !messages
          k_in = v_in = q_in
        else
          k_in = normalize(tile, k, layer.k_norm)
          v_in = normalize(tile, v, layer.v_norm)
        end
        {q_in, k_in, v_in}
      end
      if physical % @config.res_block_size == 0
        block = tile.blocks.slice(tile.block_count * tile.elements)
        MathOps.rms(encoder, tile.current, block, nil, tile.width, tile.rows, DEPTH_EPS)
        tile.block_count += 1
        MathOps.copy(encoder, attended, residual, tile.elements)
      else
        MathOps.add(encoder, tile.current, attended, residual, tile.elements)
      end
      @scratch.rewind(attention_mark)

      ffn = mix(tile, residual, layer.ffn_res_query, 1)
      if controller && messages
        ffn = control(tile, controller, 3, ffn, messages.slice(3 * tile.elements))
      end
      ffn_in = if input_mark = moe_input_mark
                 output = tile.next_state
                 MathOps.rms(encoder, ffn, output, constant(layer.ffn_norm), tile.width, tile.rows, @config.rms_eps)
                 @scratch.rewind(input_mark)
                 output
               else
                 normalize(tile, ffn, layer.ffn_norm)
               end
      bias = layer.router_bias.try { |array| constant(array) }
      routed = @linear.moe(encoder, @scratch, layer, @config, ffn_in, bias, tile.rows, decode_block: tile.decode_block?)
      convolved = visit.ffn_conv.encode(encoder, @scratch, tile.start, tile.rows, routed, constant(layer.ffn_post_conv))
      MathOps.add(encoder, residual, convolved, tile.next_state, tile.elements)
      tile.swap
      @scratch.rewind(mark)
    end

    private def attend(tile : Tile, weight : Inkling2AttentionWeights, visit : Visit,
                       recycle_inputs : Bool, & : -> Tuple(Inkling2View, Inkling2View, Inkling2View)) : Inkling2View
      matrices = {weight.q, weight.k, weight.v, weight.rel_proj, weight.gate, weight.sink, weight.k_conv_proj, weight.v_conv_proj}
      if recycle_inputs
        outputs = matrices.map { |matrix| matrix.try { |m| @scratch.take(m.out_dim * tile.rows) } }
        inputs_mark = @scratch.mark
      end
      q_in, k_in, v_in = yield
      inputs = {q_in, k_in, v_in, q_in, q_in, q_in, k_in, v_in}
      outputs ||= matrices.map { |matrix| matrix.try { |m| @scratch.take(m.out_dim * tile.rows) } }
      q, k, v, q_rel, gate, sink, dynamic_k, dynamic_v = outputs
      unless (tile.rows == 1 || tile.decode_block?) && decode_grouped_attention_projections? &&
             @linear.project_q8_group(tile.encoder, matrices, inputs, outputs, tile.rows)
        paired = !tile.decode_block? && tile.rows >= Inkling2ResidentLinear::MIN_GROUPED_ROWS &&
                 prefill_grouped_attention_projections? &&
                 @linear.project_q8_prefill_pair(tile.encoder, {weight.q, weight.gate}, q_in,
                   {q.not_nil!, gate.not_nil!}, tile.rows)
        matrices.each_with_index do |matrix, index|
          next if paired && (index == 0 || index == 4) # Q and gate
          if matrix
            @linear.project(tile.encoder, matrix, inputs[index], outputs[index].not_nil!, tile.rows,
              decode_block: tile.decode_block?)
          end
        end
      end
      if input_mark = inputs_mark
        @scratch.rewind(input_mark)
      end
      context = visit.attention.encode(tile.encoder, @scratch, tile.start, tile.rows,
        q.not_nil!, k.not_nil!, v.not_nil!, q_rel.not_nil!, gate.not_nil!, sink.not_nil!,
        constant(weight.q_clip), constant(weight.rel_clip), constant(weight.rel_bias),
        constant(weight.k_conv), constant(weight.v_conv), dynamic_k, dynamic_v, decode_block: tile.decode_block?)
      taps = constant(weight.post_conv)
      if tile.decode_block?
        if convolved = visit.post_conv.encode_q8_block(tile.encoder, @scratch, tile.start, tile.rows,
             @linear, weight.output, context, taps)
          return convolved
        end
      end
      projected = project(tile, weight.output, context)
      visit.post_conv.encode(tile.encoder, @scratch, tile.start, tile.rows, projected, taps)
    end

    # Override in a same-binary comparison with individual projection dispatches.
    private def decode_grouped_attention_projections? : Bool
      true
    end

    # Override in a same-binary comparison with individual prefill dispatches.
    private def prefill_grouped_attention_projections? : Bool
      true
    end

    private def mix(tile : Tile, input : Inkling2View, queries : Array(Float32), count : Int32) : Inkling2View
      result = @scratch.take(count * tile.elements)
      mark = @scratch.mark
      normalized = tile.hidden
      MathOps.rms(tile.encoder, input, normalized, nil, tile.width, tile.rows, DEPTH_EPS)
      MathOps.depth_mix(tile.encoder, tile.blocks, tile.block_count, normalized, constant(queries), result,
        tile.width, tile.rows, count)
      @scratch.rewind(mark)
      result
    end

    private def normalize(tile : Tile, input : Inkling2View, weight : Array(Float32)) : Inkling2View
      result = tile.hidden
      MathOps.rms(tile.encoder, input, result, constant(weight), tile.width, tile.rows, @config.rms_eps)
      result
    end

    private def project(tile : Tile, weight : QuantWeight, input : Inkling2View) : Inkling2View
      result = @scratch.take(weight.out_dim * tile.rows)
      @linear.project(tile.encoder, weight, input, result, tile.rows, decode_block: tile.decode_block?)
      result
    end

    # Override this predicate in a same-binary probe to compare fused scalar
    # controller evaluation with the reference projection-plus-epilogue path.
    private def decode_fused_control? : Bool
      true
    end

    # Keep this predicate as the same-binary A/B hook. The triplet dispatch is
    # enabled on Apple Silicon; block4 and multi-row paths do not consult it.
    private def decode_rms_concat_triplet? : Bool
      {% if flag?(:aarch64) %}
        true
      {% else %}
        false
      {% end %}
    end

    # Override this predicate in a same-binary probe to compare batched
    # prefill controller evaluation with the reference projection path.
    private def prefill_fused_control? : Bool
      true
    end

    private def control_attention(tile : Tile, controller : Inkling2LoopWeights,
                                  inputs : {Inkling2View, Inkling2View, Inkling2View},
                                  messages : Inkling2View) : {Inkling2View, Inkling2View, Inkling2View}
      downs = {controller.down[0], controller.down[1], controller.down[2]}
      unless tile.rows == 1 && Inkling2ResidentLinear.f32_triplet_supported?(downs)
        return {
          control(tile, controller, 0, inputs[0], messages),
          control(tile, controller, 1, inputs[1], messages.slice(tile.elements)),
          control(tile, controller, 2, inputs[2], messages.slice(2 * tile.elements)),
        }
      end

      results = {tile.hidden, tile.hidden, tile.hidden}
      mark = @scratch.mark
      triplet_rms_concat = decode_rms_concat_triplet?
      feature_dim = 2 * tile.elements
      rank_dim = downs[0].out_dim
      feature_elements = if triplet_rms_concat
                           MathOps.rms_concat_triplet_elements(tile.width)
                         else
                           3 * feature_dim
                         end
      rank_elements = if triplet_rms_concat
                        MathOps.triplet_elements(rank_dim, 3)
                      else
                        3 * rank_dim
                      end
      features = @scratch.take(feature_elements)
      ranks = @scratch.take(rank_elements)
      if triplet_rms_concat
        MathOps.rms_concat_triplet(tile.encoder, inputs, messages, features, tile.width, DEPTH_EPS)
      else
        inputs.each_with_index do |input, index|
          MathOps.rms_concat(tile.encoder, input, messages.slice(index * tile.elements),
            features.slice(index * feature_dim), tile.width, tile.rows, DEPTH_EPS)
        end
      end
      @linear.project_f32_triplet(tile.encoder, downs, features, ranks)
      apply_mark = @scratch.mark
      inputs.each_with_index do |input, index|
        apply_control(tile, controller, index, input, messages.slice(index * tile.elements),
          ranks.slice(index * rank_dim), results[index])
        @scratch.rewind(apply_mark)
      end
      @scratch.rewind(mark)
      results
    end

    private def control(tile : Tile, controller : Inkling2LoopWeights, index : Int32,
                        input : Inkling2View, message : Inkling2View) : Inkling2View
      result = tile.hidden
      mark = @scratch.mark
      features = @scratch.take(2 * tile.elements)
      MathOps.rms_concat(tile.encoder, input, message, features, tile.width, tile.rows, DEPTH_EPS)
      rank = @scratch.take(controller.down[index].out_dim * tile.rows)
      unless @linear.project_controller_down_mma(
               tile.encoder, controller.down[index], features, rank, tile.rows,
             )
        @linear.project(tile.encoder, controller.down[index], features, rank, tile.rows,
          decode_block: tile.decode_block?)
      end
      apply_control(tile, controller, index, input, message, rank, result)
      @scratch.rewind(mark)
      result
    end

    private def apply_control(tile : Tile, controller : Inkling2LoopWeights, index : Int32,
                              input : Inkling2View, message : Inkling2View,
                              rank : Inkling2View, result : Inkling2View) : Nil
      unless tile.decode_block?
        fused_control = tile.rows == 1 ? decode_fused_control? : prefill_fused_control?
        if fused_control &&
           @linear.project_f32_control_pair(tile.encoder, {controller.gain[index], controller.forget[index]},
             rank, input, message, result, tile.width, rows: tile.rows)
          return
        end
      end

      gain = project(tile, controller.gain[index], rank)
      forget = project(tile, controller.forget[index], rank)
      MathOps.controls(tile.encoder, input, message, gain, forget, result, tile.elements)
    end
  end
end
