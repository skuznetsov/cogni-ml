require "./inkling2_resident_attention"
require "./inkling2_resident_linear"
require "./inkling2_resident_math"

module ML::GGUF
  # One command per decode token or causal prefill tile. Only IDs and the
  # requested final logits cross the CPU/GPU boundary.
  class Inkling2Resident
    private DEPTH_EPS = 1e-6_f32
    private alias MathOps = Inkling2ResidentMath

    private record Visit, attention : Inkling2ResidentAttention,
      post_conv : Inkling2ResidentConvolution, ffn_conv : Inkling2ResidentConvolution do
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
      getter current : Inkling2View
      getter next_state : Inkling2View
      getter blocks : Inkling2View
      property block_count : Int32 = 0

      def initialize(@encoder, @scratch : Inkling2ResidentScratch, @start, @rows, @width, block_slots : Int32)
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
    @block_slots : Int32
    @constants : ML::MetalBuffer
    @ids : ML::MetalBuffer
    @logits : ML::MetalBuffer
    @scratch = Inkling2ResidentScratch.new
    @visits = [] of Visit
    @float_offsets = {} of UInt64 => Int64
    @shared = [] of ML::MetalBuffer

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

    private def self.controller_supported?(controller : Inkling2LoopWeights) : Bool
      {controller.down, controller.gain, controller.forget}.all? do |matrices|
        matrices.all? { |matrix| Inkling2ResidentLinear.projection_supported?(matrix) }
      end
    end

    def initialize(@weights : Inkling2Weights, @passes : Int32, physical_visits : Array(Int32),
                   max_seq : Int32, @prefill_rows : Int32, *, @append_prefill_rows : Int32? = nil)
      @config = @weights.config
      @prelude_blocks = @config.core_start // @config.res_block_size
      @core_blocks = (@config.core_end - @config.core_start) // @config.res_block_size
      @block_slots = (@config.n_layer - 1) // @config.res_block_size + 1
      @linear = Inkling2ResidentLinear.new(@weights.gguf)
      @constants, @ids, @logits = prepare_buffers(physical_visits, max_seq)
    end

    def forward(token : Int32, position : Int32, with_logits : Bool) : Array(Float32)?
      @ids.contents.as(Int32*).value = token
      evaluate(position, 1, with_logits)
    end

    def prefill(tokens : Indexable(Int32), position : Int32) : Array(Float32)
      tile_rows = @prefill_rows
      if (append_rows = @append_prefill_rows) && position > 0 && tokens.size >= append_rows
        tile_rows = append_rows
      end
      ids = @ids.contents.as(Int32*)
      offset = 0
      result = nil
      while offset < tokens.size
        rows = Math.min(tile_rows, tokens.size - offset)
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

    def reset : Nil
      @visits.each do |visit|
        visit.attention.reset
        visit.post_conv.reset
        visit.ffn_conv.reset
      end
    end

    def close : Nil
      @visits.each(&.close)
      @visits.clear
      @shared.each(&.release)
      @shared.clear
      @scratch.close
      @linear.close
    end

    private def prepare_buffers(physical_visits : Array(Int32), max_seq : Int32)
      physical_visits.each do |physical|
        sliding = @config.sliding?(physical)
        @visits << Visit.new(
          Inkling2ResidentAttention.new(@config, sliding, max_seq),
          Inkling2ResidentConvolution.new(@config.n_embd, @config.conv_size),
          Inkling2ResidentConvolution.new(@config.n_embd, @config.conv_size),
        )
      end
      id_rows = @append_prefill_rows || @prefill_rows
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

    private def evaluate(position : Int32, rows : Int32, with_logits : Bool) : Array(Float32)?
      @scratch.reset
      command = ML::Metal::CommandBuffer.new
      begin
        encoder = ML::Metal::ComputeEncoder.new(command)
        begin
          encode_tile(encoder, position, rows, with_logits)
        ensure
          encoder.end_encoding
        end
        command.commit_and_wait
      ensure
        command.discard unless command.committed?
        @visits.each { |visit| visit.attention.finish }
      end
      @logits.read(@config.vocab_size) if with_logits
    end

    private def encode_tile(encoder : ML::Metal::ComputeEncoder, position : Int32, rows : Int32, with_logits : Bool) : Nil
      tile = Tile.new(encoder, @scratch, position, rows, @config.n_embd, @block_slots)
      prelude = tile.hidden if @passes > 1
      banks = StaticArray[@scratch.take(@core_blocks * tile.elements), @scratch.take(@core_blocks * tile.elements)] if @passes > 1

      mark = @scratch.mark
      embedding = tile.hidden
      matrix = @weights.token_embd
      unless MathOps.embed(encoder, @linear.view(matrix), matrix.type, Inkling2View.new(@ids), embedding, tile.width, rows)
        raise "Inkling 2 resident embedding type was not admitted"
      end
      MathOps.rms(encoder, embedding, tile.current, constant(@weights.token_embd_norm), tile.width, rows, @config.rms_eps)
      @scratch.rewind(mark)

      visit = 0
      (0...@config.core_start).each do |physical|
        encode_layer(physical, visit, tile, nil)
        visit += 1
      end
      if saved = prelude
        MathOps.copy(encoder, tile.current, saved, tile.elements)
      end

      previous_bank = nil
      @passes.times do |pass|
        if pass > 0
          tile.block_count = @prelude_blocks
          MathOps.copy(encoder, prelude.not_nil!, tile.current, tile.elements)
        end
        (@config.core_start...@config.core_end).each do |physical|
          encode_layer(physical, visit, tile, previous_bank)
          visit += 1
        end
        if pass + 1 < @passes
          # The first core block's input is the shared prelude. Save the
          # remaining normalized blocks and the terminal state for messaging.
          bank = banks.not_nil![pass % 2]
          if @core_blocks > 1
            source = tile.blocks.slice((@prelude_blocks + 1) * tile.elements)
            MathOps.copy(encoder, source, bank, (@core_blocks - 1) * tile.elements)
          end
          terminal = bank.slice((@core_blocks - 1) * tile.elements)
          MathOps.rms(encoder, tile.current, terminal, nil, tile.width, rows, DEPTH_EPS)
          previous_bank = bank
        end
      end

      (@config.core_end...@config.n_layer).each do |physical|
        encode_layer(physical, visit, tile, nil)
        visit += 1
      end
      if with_logits
        mixed = mix(tile, tile.current, @weights.output_res_query, 1)
        last = mixed.slice((rows - 1) * tile.width)
        normalized = @scratch.take(tile.width)
        MathOps.rms(encoder, last, normalized, constant(@weights.output_norm), tile.width, 1, @config.rms_eps)
        logits = @scratch.take(@weights.output.out_dim)
        @linear.project(encoder, @weights.output, normalized, logits, 1)
        MathOps.softcap(encoder, logits, Inkling2View.new(@logits), @config.logit_softcap, @config.vocab_size)
      end
    end

    private def encode_layer(physical : Int32, visit_index : Int32, tile : Tile, previous_bank : Inkling2View?) : Nil
      layer = @weights.layers[physical]
      visit = @visits[visit_index]
      encoder = tile.encoder
      mark = @scratch.mark
      residual = tile.hidden
      controller = previous_bank ? layer.controller.not_nil! : nil
      messages = if bank = previous_bank
                   output = @scratch.take(4 * tile.elements)
                   MathOps.depth_mix(encoder, bank, @core_blocks, nil, constant(controller.not_nil!.query),
                     output, tile.width, tile.rows, 4)
                   output
                 end

      attention_mark = @scratch.mark
      sliding = @config.sliding?(physical)
      query_count = sliding ? 1 : 3
      queries = layer.attn_res_query
      mixed = queries ? mix(tile, tile.current, queries, query_count) : tile.current
      q = mixed
      distinct_inputs = query_count == 3 && !queries.nil?
      k = distinct_inputs ? mixed.slice(tile.elements) : mixed
      v = distinct_inputs ? mixed.slice(2 * tile.elements) : mixed
      if controller && messages
        q = control(tile, controller, 0, q, messages)
        k = control(tile, controller, 1, k, messages.slice(tile.elements))
        v = control(tile, controller, 2, v, messages.slice(2 * tile.elements))
      end
      q_in = normalize(tile, q, layer.q_norm)
      if sliding && !messages
        k_in = v_in = q_in
      else
        k_in = normalize(tile, k, layer.k_norm)
        v_in = normalize(tile, v, layer.v_norm)
      end
      attended = attend(tile, layer.attention, visit, q_in, k_in, v_in)
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
      ffn_in = normalize(tile, ffn, layer.ffn_norm)
      bias = layer.router_bias.try { |array| constant(array) }
      routed = @linear.moe(encoder, @scratch, layer, @config, ffn_in, bias, tile.rows)
      convolved = visit.ffn_conv.encode(encoder, @scratch, tile.start, tile.rows, routed, constant(layer.ffn_post_conv))
      MathOps.add(encoder, residual, convolved, tile.next_state, tile.elements)
      tile.swap
      @scratch.rewind(mark)
    end

    private def attend(tile : Tile, weight : Inkling2AttentionWeights, visit : Visit,
                       q_in : Inkling2View, k_in : Inkling2View, v_in : Inkling2View) : Inkling2View
      q = project(tile, weight.q, q_in)
      k = project(tile, weight.k, k_in)
      v = project(tile, weight.v, v_in)
      q_rel = project(tile, weight.rel_proj, q_in)
      gate = project(tile, weight.gate, q_in)
      sink = project(tile, weight.sink, q_in)
      dynamic_k = weight.k_conv_proj.try { |matrix| project(tile, matrix, k_in) }
      dynamic_v = weight.v_conv_proj.try { |matrix| project(tile, matrix, v_in) }
      context = visit.attention.encode(tile.encoder, @scratch, tile.start, tile.rows, q, k, v, q_rel, gate, sink,
        constant(weight.q_clip), constant(weight.rel_clip), constant(weight.rel_bias),
        constant(weight.k_conv), constant(weight.v_conv), dynamic_k, dynamic_v)
      projected = project(tile, weight.output, context)
      visit.post_conv.encode(tile.encoder, @scratch, tile.start, tile.rows, projected, constant(weight.post_conv))
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
      @linear.project(tile.encoder, weight, input, result, tile.rows)
      result
    end

    private def control(tile : Tile, controller : Inkling2LoopWeights, index : Int32,
                        input : Inkling2View, message : Inkling2View) : Inkling2View
      result = tile.hidden
      mark = @scratch.mark
      features = @scratch.take(2 * tile.elements)
      MathOps.rms_concat(tile.encoder, input, message, features, tile.width, tile.rows, DEPTH_EPS)
      rank = project(tile, controller.down[index], features)
      gain = project(tile, controller.gain[index], rank)
      forget = project(tile, controller.forget[index], rank)
      MathOps.controls(tile.encoder, input, message, gain, forget, result, tile.elements)
      @scratch.rewind(mark)
      result
    end
  end
end
