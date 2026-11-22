require "./inkling2_weights"
require "./inkling2_matmul"
require "./inkling2_attention"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  require "./inkling2_resident"
{% end %}

module ML::GGUF
  module Inkling2Math
    extend self

    DEPTH_EPS = 1e-6_f32

    def rms_norm(input : Array(Float32), eps : Float32, weight : Array(Float32)? = nil) : Array(Float32)
      mean_square = input.sum(0.0) { |value| value.to_f64 * value } / input.size
      scale = (1.0 / Math.sqrt(mean_square + eps)).to_f32
      if weight
        Array(Float32).new(input.size) { |i| input[i] * scale * weight[i] }
      else
        input.map { |value| value * scale }
      end
    end

    def normalize_bank(bank : Array(Array(Float32))) : Array(Array(Float32))
      bank.map { |state| rms_norm(state, DEPTH_EPS) }
    end

    # Depth attention uses normalized block states for both keys and values.
    def depth_mix(bank : Array(Array(Float32)), queries : Array(Float32)) : Array(Array(Float32))
      width = bank.first.size
      scale = (1.0 / Math.sqrt(width)).to_f32
      Array(Array(Float32)).new(queries.size // width) do |query|
        start = query * width
        scores = bank.map do |state|
          dot = 0_f32
          width.times { |i| dot += state[i] * queries[start + i] }
          dot * scale
        end
        max = scores.max
        scores.map! { |score| Math.exp(score - max) }
        sum = scores.sum
        result = Array(Float32).new(width, 0_f32)
        bank.each_with_index do |state, i|
          weight = scores[i] / sum
          width.times { |j| result[j] += weight * state[j] }
        end
        result
      end
    end

    def relative_delta(previous : Array(Array(Float32)), current : Array(Array(Float32))) : Float32
      change = 0.0
      energy = 0.0
      previous.each_with_index do |state, row|
        state.each_with_index do |value, i|
          difference = current[row][i].to_f64 - value
          change += difference * difference
          energy += value.to_f64 * value
        end
      end
      (Math.sqrt(change) / Math.max(Math.sqrt(energy), DEPTH_EPS)).to_f32
    end

    def below_loop_threshold?(previous : Float32?, current : Float32, threshold : Float32) : Bool
      return false unless previous

      previous <= threshold && current <= threshold
    end

    def sigmoid(value : Float32) : Float32
      1_f32 / (1_f32 + Math.exp(-value))
    end

    # Correction bias affects selection; combine with the original sigmoid scores.
    def route(logits : Array(Float32), bias : Array(Float32)?, picked : Int32, scale : Float32) : Array({Int32, Float32})
      probabilities = logits.map { |logit| sigmoid(logit) }
      order = probabilities.map_with_index do |probability, id|
        score = bias ? probability + bias[id] : probability
        {-score, id}
      end
      order.sort!
      sum = 0_f32
      picked.times { |i| sum += probabilities[order[i][1]] }
      normalization = sum + 1e-20_f32
      Array({Int32, Float32}).new(picked) do |i|
        id = order[i][1]
        {id, probabilities[id] / normalization * scale}
      end
    end

    def situ(gate : Float32, up : Float32, beta : Float32, linear_beta : Float32) : Float32
      bounded_gate = beta * Math.tanh(gate / beta)
      bounded_up = linear_beta * Math.tanh(up / linear_beta)
      bounded_gate * sigmoid(gate) * bounded_up
    end
  end

  # One causal sequence. An optional threshold selects each token's exit depth.
  # A later token can extend the core history of earlier tokens on demand.
  class Inkling2Engine
    DEFAULT_PREFILL_TILE =  32
    MAX_PREFILL_TILE     = 128

    private DEPTH_BLOCK_ROWS = 4

    alias LoopTrace = Proc(Int32, Int32, Float32, Nil)

    private alias CoreBank = Array(Array(Float32))
    private record PendingCore, prelude : Array(Float32), blocks : CoreBank, bank : CoreBank

    getter weights : Inkling2Weights
    getter passes : Int32
    getter position : Int32 = 0
    getter max_seq : Int32
    getter backfilled_passes : Int64 = 0_i64
    getter? depth_speculation : Bool

    @closed = false
    @failed = false
    @attention : Array(Inkling2AttentionState)
    @ffn_conv : Array(Inkling2Convolution)
    @matmul : Inkling2Matmul?
    {% if flag?(:darwin) && !flag?(:cpu_only) %}
      @resident : Inkling2Resident? = nil
    {% end %}
    @core_prefix = [] of Int32
    @pending_core = {} of Int32 => PendingCore

    def initialize(path : String, @max_seq : Int32, *, metal : Bool, resident : Bool = true, passes : Int32? = nil,
                   @trace : LoopTrace? = nil, @stop_delta : Float32? = nil,
                   prefill_tile : Int32? = nil, auto_prefill_tile : Bool = false, @greedy_verifier : Bool = false,
                   @depth_speculation : Bool = false)
      if @greedy_verifier && @depth_speculation
        raise ArgumentError.new("Inkling 2 cannot combine depth speculation and n-gram verification")
      end
      if prefill_tile && auto_prefill_tile
        raise ArgumentError.new("Inkling 2 cannot combine automatic and explicit prefill tile sizes")
      end
      if prefill_tile && !prefill_tile.in?(1..MAX_PREFILL_TILE)
        raise ArgumentError.new("Inkling 2 prefill tile must be between 1 and #{MAX_PREFILL_TILE} tokens")
      end
      @weights = Inkling2Weights.new(path)
      @passes, @attention, @ffn_conv, @matmul = prepare(metal, resident, passes, prefill_tile, auto_prefill_tile)
      @core_prefix = Array(Int32).new(@passes, 0) if @stop_delta
    end

    def self.resolve_passes(configured : Int32, varying : Bool, requested : Int32?) : Int32
      passes = requested || configured
      if varying
        raise ArgumentError.new("Research checkpoint supports 1 to 10 passes in this engine") unless passes.in?(1..10)
      elsif passes != configured
        raise ArgumentError.new("This checkpoint requires #{configured} passes. Use TML-d24-varloop-research to vary them.")
      end
      passes
    end

    def self.greedy_token_id(logits : Array(Float32)) : Int32
      best_id = -1
      best_score = -Float32::INFINITY
      logits.each_with_index do |score, index|
        if score.finite? && score > best_score
          best_score = score
          best_id = index
        end
      end
      raise "Inkling 2 returned no finite logits" if best_id < 0
      best_id
    end

    def forward(token : Int32, *, logits : Bool = true) : Array(Float32)?
      validate_active!
      validate_token!(token)
      raise ArgumentError.new("Inkling 2 context of #{@max_seq} tokens is full") if @position >= @max_seq

      begin
        result = {% if flag?(:darwin) && !flag?(:cpu_only) %}
                   if backend = @resident
                     backend.forward(token, @position, logits)
                   else
                     run_token(token, logits)
                   end
                 {% else %}
                   run_token(token, logits)
                 {% end %}
        @position += 1
        result
      rescue ex
        @failed = true
        raise ex
      end
    end

    # Evaluate four known continuation tokens with the serial decode numerics.
    # Commits all four to the sequence and returns the next-token logits after each.
    def decode_block4(tokens : StaticArray(Int32, 4)) : StaticArray(Array(Float32), 4)
      evaluate_block4(tokens) { |rows| {rows, tokens.size} }
    end

    # The caller supplies the exact pending token in row zero. Commit it and
    # the remaining consecutive candidates that match the target's greedy IDs.
    # Return the committed row count and the exact next pending token ID.
    def verify_greedy_block4(tokens : StaticArray(Int32, 4)) : {Int32, Int32}
      raise ArgumentError.new("Inkling 2 greedy verification was not enabled") unless @greedy_verifier
      evaluate_block4(tokens) do |rows|
        accepted = 1
        expected = self.class.greedy_token_id(rows[0])
        while accepted < tokens.size && tokens[accepted] == expected
          expected = self.class.greedy_token_id(rows[accepted])
          accepted += 1
        end
        { {accepted, expected}, accepted }
      end
    end

    private def evaluate_block4(tokens : StaticArray(Int32, 4), & : StaticArray(Array(Float32), 4) -> Tuple(T, Int32)) : T forall T
      validate_active!
      raise ArgumentError.new("Inkling 2 decode block exceeds the context of #{@max_seq} tokens") if tokens.size > @max_seq - @position
      tokens.each { |token| validate_token!(token) }

      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        backend = @resident || raise ArgumentError.new("Inkling 2 block decode requires fixed-depth resident Metal")
        begin
          result, accepted = yield backend.decode_block4(tokens, @position)
          backend.rewind_block(@position + accepted) if accepted < tokens.size
          @position += accepted
          result
        rescue ex
          @failed = true
          raise ex
        end
      {% else %}
        raise ArgumentError.new("Inkling 2 block decode requires fixed-depth resident Metal")
      {% end %}
    end

    # Force all inputs for parity checks, retaining only the requested prefix.
    def decode_depth_block4(tokens : StaticArray(Int32, 4), *, commit : Int32 = 4) : StaticArray(Array(Float32), 4)
      unless commit.in?(1..tokens.size)
        raise ArgumentError.new("Inkling 2 depth block commit must be between 1 and #{tokens.size}")
      end
      evaluate_depth_block(tokens[0], candidates: tokens) { |_, rows| {rows, commit} }
    end

    # Draft after the exact pending input, then return its accepted block prefix
    # and the next ID from the ten-pass model. EOS stays pending for the caller.
    def verify_depth_block4(first_id : Int32, *, eos_id : Int32) : {StaticArray(Int32, 4), Int32, Int32}
      raise ArgumentError.new("Inkling 2 depth block cannot start at EOS") if first_id == eos_id
      evaluate_depth_block(first_id, eos_id: eos_id) do |tokens, rows|
        accepted = 1
        expected = self.class.greedy_token_id(rows[0])
        while accepted < tokens.size && expected != eos_id && tokens[accepted] == expected
          expected = self.class.greedy_token_id(rows[accepted])
          accepted += 1
        end
        { {tokens, accepted, expected}, accepted }
      end
    end

    private def evaluate_depth_block(first_id : Int32, *, candidates : StaticArray(Int32, 4)? = nil, eos_id : Int32? = nil,
                                     & : StaticArray(Int32, 4), StaticArray(Array(Float32), 4) -> Tuple(T, Int32)) : T forall T
      validate_active!
      raise ArgumentError.new("Inkling 2 depth speculation was not enabled") unless @depth_speculation
      if DEPTH_BLOCK_ROWS > @max_seq - @position
        raise ArgumentError.new("Inkling 2 depth block exceeds the context of #{@max_seq} tokens")
      end
      if candidates
        candidates.each { |token| validate_token!(token) }
      else
        validate_token!(first_id)
      end

      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        backend = @resident.not_nil!
        tokens = candidates || StaticArray(Int32, 4).new(first_id)
        begin
          backend.begin_depth_block(@position)
          tokens.size.times do |row|
            token = tokens[row]
            needs_candidate = candidates.nil? && row < tokens.size - 1
            logits = backend.draft_depth_row(token, @position + row, logits: needs_candidate && token != eos_id)
            # A speculative EOS fills the remaining slots. The target clips the
            # retained prefix before EOS or at the first unequal candidate.
            tokens[row + 1] = logits ? self.class.greedy_token_id(logits) : token if needs_candidate
          end
          result, accepted = yield tokens, backend.verify_depth_block(@position)
          backend.finish_depth_block(@position + accepted)
          @position += accepted
          result
        rescue ex
          @failed = true
          raise ex
        end
      {% else %}
        raise ArgumentError.new("Inkling 2 depth speculation requires resident Metal")
      {% end %}
    end

    # Extend this sequence and return only its last prompt row's logits.
    def prefill(tokens : Indexable(Int32)) : Array(Float32)
      validate_active!
      raise ArgumentError.new("Inkling 2 prompt is empty") if tokens.empty?
      if tokens.size > @max_seq - @position
        raise ArgumentError.new("Inkling 2 prefill exceeds the context of #{@max_seq} tokens")
      end
      tokens.each { |token| validate_token!(token) }

      begin
        {% if flag?(:darwin) && !flag?(:cpu_only) %}
          if backend = @resident
            result = backend.prefill(tokens, @position)
            @position += tokens.size
            return result
          end
        {% end %}
        result = nil
        tokens.each_with_index do |token, index|
          result = run_token(token, index == tokens.size - 1)
          @position += 1
        end
        result.not_nil!
      rescue ex
        @failed = true
        raise ex
      end
    end

    def resident? : Bool
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        !@resident.nil?
      {% else %}
        false
      {% end %}
    end

    def layer_visits : Int32
      visit_for_pass(@passes) + config.n_layer - config.core_end
    end

    def kv_count(visit : Int32) : Int32
      validate_visit!(visit)
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        if backend = @resident
          return backend.kv_count(visit, @position)
        end
      {% end %}
      @attention[visit].count
    end

    # A snapshot in chronological order, scoped to one physical layer visit.
    def key_value_at(visit : Int32, index : Int32) : {Array(Float32), Array(Float32), Int32}
      count = kv_count(visit)
      raise IndexError.new("Inkling 2 attention cache index out of bounds") unless 0 <= index < count
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        if backend = @resident
          return backend.key_value_at(visit, index, @position)
        end
      {% end %}
      key, value, position = @attention[visit].key_value_at(index)
      {key.dup, value.dup, position}
    end

    def draft_layer_visits : Int32
      @depth_speculation ? config.n_layer - config.core_end : 0
    end

    def draft_kv_count(visit : Int32) : Int32
      validate_draft_visit!(visit)
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        @resident.not_nil!.draft_kv_count(visit, @position)
      {% else %}
        raise ArgumentError.new("Inkling 2 depth speculation requires resident Metal")
      {% end %}
    end

    def draft_key_value_at(visit : Int32, index : Int32) : {Array(Float32), Array(Float32), Int32}
      count = draft_kv_count(visit)
      raise IndexError.new("Inkling 2 draft cache index out of bounds") unless 0 <= index < count
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        @resident.not_nil!.draft_key_value_at(visit, index, @position)
      {% else %}
        raise ArgumentError.new("Inkling 2 depth speculation requires resident Metal")
      {% end %}
    end

    def reset : Nil
      raise "Inkling 2 engine is closed" if @closed

      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        @resident.try(&.reset)
      {% end %}
      @attention.each(&.reset)
      @ffn_conv.each(&.reset)
      @core_prefix.fill(0)
      @pending_core.clear
      @backfilled_passes = 0_i64
      @position = 0
      @failed = false
    end

    def close : Nil
      return if @closed

      @closed = true
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        @resident.try(&.close)
      {% end %}
      @matmul.try(&.close)
      @weights.close
      @attention.clear
      @ffn_conv.clear
      @pending_core.clear
    end

    private def config : Inkling2Config
      @weights.config
    end

    private def validate_active! : Nil
      raise "Inkling 2 engine is closed" if @closed
      raise "Inkling 2 sequence must be reset after a failed forward pass" if @failed
    end

    private def validate_token!(token : Int32) : Nil
      raise ArgumentError.new("Token #{token} is outside the Inkling 2 vocabulary") unless token.in?(0...config.vocab_size)
    end

    private def validate_visit!(visit : Int32) : Nil
      validate_active!
      raise IndexError.new("Inkling 2 layer visit out of bounds") unless 0 <= visit < layer_visits
    end

    private def validate_draft_visit!(visit : Int32) : Nil
      validate_active!
      raise IndexError.new("Inkling 2 draft layer visit out of bounds") unless 0 <= visit < draft_layer_visits
    end

    private def prepare(metal : Bool, resident : Bool, passes : Int32?, prefill_tile : Int32?, auto_prefill_tile : Bool)
      unless @max_seq.in?(1..config.context_length)
        raise ArgumentError.new("Context must be between 1 and #{config.context_length} tokens")
      end
      loops = self.class.resolve_passes(config.passes, config.varying?, passes)
      if @depth_speculation && (!config.varying? || loops != 10)
        raise ArgumentError.new("Inkling 2 depth speculation requires the variable-loop research checkpoint at 10 passes")
      end
      if threshold = @stop_delta
        raise ArgumentError.new("Adaptive depth requires TML-d24-varloop-research") unless config.varying?
        raise ArgumentError.new("Adaptive depth requires a cap of at least 3 passes") if loops < 3
        unless threshold.finite? && threshold > 0
          raise ArgumentError.new("The stopping threshold must be finite and positive")
        end
      end
      visits = (0...config.core_start).to_a
      loops.times { visits.concat((config.core_start...config.core_end).to_a) }
      visits.concat((config.core_end...config.n_layer).to_a)
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        if metal && resident && !@stop_delta && !@trace && Inkling2Resident.supported?(@weights)
          rows = auto_prefill_tile ? MAX_PREFILL_TILE : (prefill_tile || DEFAULT_PREFILL_TILE)
          append_rows = Inkling2ResidentLinear::MAX_GROUPED_ROWS if auto_prefill_tile
          @resident = Inkling2Resident.new(@weights, loops, visits, @max_seq, rows,
            append_prefill_rows: append_rows, rollback_rows: (@greedy_verifier || @depth_speculation) ? DEPTH_BLOCK_ROWS : 0,
            depth_speculation: @depth_speculation)
          return {loops, [] of Inkling2AttentionState, [] of Inkling2Convolution, nil}
        end
      {% end %}
      raise ArgumentError.new("Inkling 2 greedy verification requires fixed-depth resident Metal") if @greedy_verifier
      raise ArgumentError.new("Inkling 2 depth speculation requires fixed-depth resident Metal") if @depth_speculation
      if prefill_tile || auto_prefill_tile
        raise ArgumentError.new("Setting the Inkling 2 prefill tile requires resident Metal")
      end
      attention = visits.map { |layer| Inkling2AttentionState.new(config, config.sliding?(layer), @max_seq) }
      convolutions = visits.map { Inkling2Convolution.new(config.n_embd, config.conv_size) }
      {loops, attention, convolutions, make_matmul(metal)}
    rescue ex
      @weights.close
      raise ex
    end

    private def make_matmul(metal : Bool) : Inkling2Matmul
      return Inkling2CPUMatmul.new unless metal

      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        Inkling2MetalMatmul.new(@weights.gguf)
      {% else %}
        raise ArgumentError.new("Metal is unavailable in this build. Use --cpu for CPU inference.")
      {% end %}
    end

    private def run_token(token : Int32, with_logits : Bool) : Array(Float32)?
      current = Inkling2Math.rms_norm(@weights.embedding_row(token), config.rms_eps, @weights.token_embd_norm)
      blocks = CoreBank.new
      config.core_start.times do |physical|
        current = layer_forward(physical, physical, current, blocks, @position)
      end

      prelude = current
      prelude_blocks = blocks
      bank = nil
      previous_delta = nil
      threshold = @stop_delta
      trace = @trace
      @passes.times do |pass|
        fill_core_prefix(pass) if threshold
        current, blocks = run_core_pass(pass, @position, prelude, prelude_blocks, bank)
        @core_prefix[pass] = @position + 1 if threshold

        completed = pass + 1
        next if completed == @passes && !trace

        next_bank = core_message_bank(prelude_blocks, blocks, current)
        if bank && (trace || threshold)
          delta = Inkling2Math.relative_delta(bank, next_bank)
          trace.try(&.call(@position, completed, delta))
          if threshold && completed < @passes && Inkling2Math.below_loop_threshold?(previous_delta, delta, threshold)
            @pending_core[@position] = PendingCore.new(prelude, prelude_blocks, next_bank)
            break
          end
          previous_delta = delta
        end
        bank = next_bank
      end

      visit = visit_for_pass(@passes)
      (config.core_end...config.n_layer).each do |physical|
        current = layer_forward(physical, visit, current, blocks, @position)
        visit += 1
      end
      return unless with_logits

      mixed = mix(blocks, current, @weights.output_res_query).first
      normalized = Inkling2Math.rms_norm(mixed, config.rms_eps, @weights.output_norm)
      logits = @matmul.not_nil!.multiply(@weights.output, normalized)
      cap = config.logit_softcap
      logits = logits[0, config.vocab_size]
      logits.map! { |value| cap * Math.tanh(value / cap) } if cap > 0
      logits
    end

    private def visit_for_pass(pass : Int32) : Int32
      config.core_start + pass * (config.core_end - config.core_start)
    end

    private def run_core_pass(pass : Int32, position : Int32, prelude : Array(Float32), prelude_blocks : CoreBank,
                              bank : CoreBank?) : {Array(Float32), CoreBank}
      current = prelude
      blocks = prelude_blocks.dup
      visit = visit_for_pass(pass)
      (config.core_start...config.core_end).each do |physical|
        current = layer_forward(physical, visit, current, blocks, position, bank)
        visit += 1
      end
      {current, blocks}
    end

    private def core_message_bank(prelude_blocks : CoreBank, blocks : CoreBank, current : Array(Float32)) : CoreBank
      # The first core block's input is the prelude, shared across passes.
      Inkling2Math.normalize_bank(blocks[(prelude_blocks.size + 1)..] + [current])
    end

    private def fill_core_prefix(pass : Int32) : Nil
      # The preceding pass already covers this prefix. Fill only the missing
      # positions at this depth, in order, and leave the epilogue history intact.
      position = @core_prefix[pass]
      while position < @position
        pending = @pending_core[position]
        current, blocks = run_core_pass(pass, position, pending.prelude, pending.blocks, pending.bank)
        if pass + 1 == @passes
          @pending_core.delete(position)
        else
          bank = core_message_bank(pending.blocks, blocks, current)
          @pending_core[position] = PendingCore.new(pending.prelude, pending.blocks, bank)
        end
        @backfilled_passes += 1
        position += 1
      end
    end

    private def mix(blocks : Array(Array(Float32)), current : Array(Float32), queries : Array(Float32)) : Array(Array(Float32))
      Inkling2Math.depth_mix(Inkling2Math.normalize_bank(blocks + [current]), queries)
    end

    private def layer_forward(physical : Int32, visit : Int32, current : Array(Float32), blocks : CoreBank,
                              position : Int32, bank : CoreBank? = nil) : Array(Float32)
      layer = @weights.layers[physical]
      sliding = config.sliding?(physical)
      inputs = if query = layer.attn_res_query
                 mixed = mix(blocks, current, query)
                 sliding ? [mixed.first, mixed.first, mixed.first] : mixed
               else
                 [current, current, current]
               end
      controller = bank ? layer.controller.not_nil! : nil
      messages = if bank && controller
                   Inkling2Math.depth_mix(bank, controller.query)
                 end
      if controller && messages
        inputs = apply_controls(controller, inputs, messages, 0)
      end

      q_in = Inkling2Math.rms_norm(inputs[0], config.rms_eps, layer.q_norm)
      if sliding && !messages
        k_in = v_in = q_in
      else
        k_in = Inkling2Math.rms_norm(inputs[1], config.rms_eps, layer.k_norm)
        v_in = Inkling2Math.rms_norm(inputs[2], config.rms_eps, layer.v_norm)
      end
      attended = Inkling2Attention.forward(layer.attention, config, @attention[visit], @matmul.not_nil!, q_in, k_in, v_in, position)

      if physical % config.res_block_size == 0
        blocks << current
        residual = attended
      else
        residual = Array(Float32).new(config.n_embd) { |i| current[i] + attended[i] }
      end
      mixed = mix(blocks, residual, layer.ffn_res_query)
      if controller && messages
        mixed = apply_controls(controller, mixed, messages, 3)
      end
      ffn_in = Inkling2Math.rms_norm(mixed.first, config.rms_eps, layer.ffn_norm)
      ffn_out = moe(layer, ffn_in)
      ffn_out = @ffn_conv[visit].apply(ffn_out, layer.ffn_post_conv)
      Array(Float32).new(config.n_embd) { |i| residual[i] + ffn_out[i] }
    end

    private def apply_controls(loop : Inkling2LoopWeights, inputs : Array(Array(Float32)), messages : Array(Array(Float32)),
                               first : Int32) : Array(Array(Float32))
      down = inputs.map_with_index do |input, i|
        id = first + i
        features = Inkling2Math.rms_norm(input, Inkling2Math::DEPTH_EPS) + Inkling2Math.rms_norm(messages[id], Inkling2Math::DEPTH_EPS)
        {loop.down[id], features}
      end
      matmul = @matmul.not_nil!
      gate = matmul.multiply_many(down)
      jobs = [] of {QuantWeight, Array(Float32)}
      gate.each_with_index do |state, i|
        jobs << {loop.gain[first + i], state} << {loop.forget[first + i], state}
      end
      coefficients = matmul.multiply_many(jobs)
      inputs.map_with_index do |input, i|
        gain, forget, message = coefficients[2 * i], coefficients[2 * i + 1], messages[first + i]
        Array(Float32).new(input.size) do |j|
          input[j] - Math.tanh(forget[j]) * input[j] + Math.tanh(gain[j]) * message[j]
        end
      end
    end

    private def moe(layer : Inkling2LayerWeights, input : Array(Float32)) : Array(Float32)
      matmul = @matmul.not_nil!
      if result = matmul.try_moe(layer, input, config)
        return result
      end

      experts = Inkling2Math.route(matmul.multiply(layer.router, input), layer.router_bias, config.n_expert_used, config.expert_scale)
      activated = matmul.multiply_many(experts.map { |id, _| {layer.gate_up.expert(id), input} })
      width = config.expert_dim
      down = experts.map_with_index do |(id, _), i|
        gate_up = activated[i]
        gated = Array(Float32).new(width) do |j|
          Inkling2Math.situ(gate_up[j], gate_up[width + j], config.situ_beta, config.situ_linear_beta)
        end
        {layer.down.expert(id), gated}
      end
      values = matmul.multiply_many(down)
      result = Array(Float32).new(config.n_embd, 0_f32)
      experts.each_with_index do |(_, weight), i|
        value = values[i]
        config.n_embd.times { |j| result[j] += weight * value[j] }
      end
      result
    end
  end
end
