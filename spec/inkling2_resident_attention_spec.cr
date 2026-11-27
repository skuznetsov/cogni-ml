require "./spec_helper"
require "../src/ml/gguf/inkling2_weights"
require "../src/ml/gguf/inkling2_matmul"
require "../src/ml/gguf/inkling2_attention"
require "../src/ml/gguf/inkling2_metal_view"
require "../src/ml/gguf/inkling2_resident_attention"
require "../src/ml/gguf/inkling2_resident_math"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  # Keep dispatch assertions local to this GPU spec. PipelineCache has no
  # production-facing key lookup API; checking the fresh cache proves which
  # named pipeline the just-completed encode requested.
  class ML::Metal::PipelineCache
    def self.resident_spec_contains?(name : String) : Bool
      @@cache.has_key?(name)
    end
  end

  private record ResidentProjectedRow,
    q : Array(Float32),
    k : Array(Float32),
    v : Array(Float32),
    q_rel : Array(Float32),
    gate : Array(Float32),
    sink : Array(Float32),
    dynamic_k : Array(Float32)?,
    dynamic_v : Array(Float32)?

  private record ResidentAttentionRun,
    output : Array(Float32),
    snapshots : Array(Tuple(Array(Float32), Array(Float32), Int32))

  private def resident_test_config(
    *,
    sliding : Bool,
    n_head : Int32,
    n_head_kv : Int32,
    head_dim : Int32,
    rel_dim : Int32,
    window : Int32,
    max_seq : Int32,
  ) : ML::GGUF::Inkling2Config
    n_embd = n_head * head_dim
    ML::GGUF::Inkling2Config.new(
      n_layer: 1_i32,
      n_embd: n_embd,
      n_head: n_head,
      n_head_kv: n_head_kv,
      head_dim: head_dim,
      rel_dim: rel_dim,
      rel_extent: 4_i32,
      sliding_window: window,
      conv_size: 2_i32,
      log_scale_floor: 2_i32,
      log_scale_alpha: 0.17_f32,
      qk_clip: 0.35_f32,
      rel_clip: 0.9_f32,
      rms_eps: 1e-5_f32,
      n_expert: 1_i32,
      n_expert_used: 1_i32,
      expert_dim: 1_i32,
      expert_scale: 1_f32,
      situ_beta: 1_f32,
      situ_linear_beta: 1_f32,
      res_block_size: 1_i32,
      vocab_size: 2_i32,
      padded_vocab_size: 2_i32,
      context_length: max_seq,
      core_start: 0_i32,
      core_end: 1_i32,
      passes: 3_i32,
      gate_rank: 1_i32,
      logit_softcap: 30_f32,
      varying: false,
      sliding_pattern: [sliding],
    )
  end

  private def resident_f32_weight(label : String, out_dim : Int32, in_dim : Int32, values : Array(Float32)) : ML::GGUF::QuantWeight
    raise "bad resident test matrix size for #{label}" unless values.size == out_dim * in_dim
    bytes = Bytes.new(values.size * sizeof(Float32), 0_u8)
    bytes.to_unsafe.as(Float32*).copy_from(values.to_unsafe, values.size)
    ML::GGUF::QuantWeight.new(bytes, ML::GGUF::TensorType::F32, out_dim, in_dim, "resident-test:#{label}")
  end

  private def resident_matrix(rows : Int32, columns : Int32, salt : Int32) : Array(Float32)
    Array(Float32).new(rows * columns) do |index|
      value = ((index * 7 + salt * 11) % 19 - 9).to_f32
      value * 0.025_f32
    end
  end

  private def resident_attention_weights(config : ML::GGUF::Inkling2Config) : ML::GGUF::Inkling2AttentionWeights
    q_width = config.n_head * config.head_dim
    kv_width = config.n_head_kv * config.head_dim
    relative_extent = config.sliding?(0) ? config.sliding_window : config.rel_extent
    output = Array(Float32).new(config.n_embd * q_width, 0_f32)
    Math.min(config.n_embd, q_width).times { |index| output[index * q_width + index] = 1_f32 }
    dynamic_k = if config.sliding?(0)
                  nil
                else
                  resident_f32_weight("dynamic-k", config.conv_size * config.n_head_kv, config.n_embd,
                    resident_matrix(config.conv_size * config.n_head_kv, config.n_embd, 8))
                end
    dynamic_v = if config.sliding?(0)
                  nil
                else
                  resident_f32_weight("dynamic-v", config.conv_size * config.n_head_kv, config.n_embd,
                    resident_matrix(config.conv_size * config.n_head_kv, config.n_embd, 9))
                end
    relative_projection = resident_matrix(config.n_head * config.rel_dim, config.n_embd, 6)
      .map { |value| value * 25_f32 }

    ML::GGUF::Inkling2AttentionWeights.new(
      q: resident_f32_weight("q", q_width, config.n_embd, resident_matrix(q_width, config.n_embd, 1)),
      k: resident_f32_weight("k", kv_width, config.n_embd, resident_matrix(kv_width, config.n_embd, 2)),
      v: resident_f32_weight("v", kv_width, config.n_embd, resident_matrix(kv_width, config.n_embd, 3)),
      output: resident_f32_weight("output", config.n_embd, q_width, output),
      gate: resident_f32_weight("gate", q_width, config.n_embd, resident_matrix(q_width, config.n_embd, 4)),
      sink: resident_f32_weight("sink", config.n_head, config.n_embd, resident_matrix(config.n_head, config.n_embd, 5)),
      rel_proj: resident_f32_weight("relative", config.n_head * config.rel_dim, config.n_embd,
        relative_projection),
      q_clip: Array(Float32).new(q_width) { |index| 0.7_f32 + (index % 4).to_f32 * 0.1_f32 },
      rel_bias: Array(Float32).new(relative_extent * config.rel_dim) { |index| (index % 5 - 2).to_f32 * 0.09_f32 },
      rel_clip: Array(Float32).new(config.n_head * config.rel_dim) { |index| 0.8_f32 + (index % 3).to_f32 * 0.1_f32 },
      k_conv: resident_matrix(kv_width, config.conv_size, 10),
      v_conv: resident_matrix(kv_width, config.conv_size, 11),
      post_conv: Array(Float32).new(config.n_embd * config.conv_size, 0_f32),
      k_conv_proj: dynamic_k,
      v_conv_proj: dynamic_v,
    )
  end

  private def resident_projected_rows(
    weights : ML::GGUF::Inkling2AttentionWeights,
    matmul : ML::GGUF::Inkling2CPUMatmul,
    inputs : Array({Array(Float32), Array(Float32), Array(Float32)}),
  ) : Array(ResidentProjectedRow)
    inputs.map do |q_in, k_in, v_in|
      jobs = [
        {weights.q, q_in}, {weights.k, k_in}, {weights.v, v_in},
        {weights.rel_proj, q_in}, {weights.gate, q_in}, {weights.sink, q_in},
      ]
      if k_projection = weights.k_conv_proj
        jobs << {k_projection, k_in}
        jobs << {weights.v_conv_proj.not_nil!, v_in}
      end
      projections = matmul.multiply_many(jobs)
      ResidentProjectedRow.new(
        projections[0], projections[1], projections[2], projections[3], projections[4], projections[5],
        weights.k_conv_proj ? projections[6] : nil,
        weights.v_conv_proj ? projections[7] : nil,
      )
    end
  end

  private def resident_inputs(config : ML::GGUF::Inkling2Config, rows : Int32) : Array({Array(Float32), Array(Float32), Array(Float32)})
    Array({Array(Float32), Array(Float32), Array(Float32)}).new(rows) do |row|
      make = ->(salt : Int32) do
        Array(Float32).new(config.n_embd) do |column|
          (((row + 2) * 13 + column * 5 + salt * 7) % 23 - 11).to_f32 * 0.06_f32
        end
      end
      {make.call(1), make.call(2), make.call(3)}
    end
  end

  private def resident_flatten(rows : Array(ResidentProjectedRow), &block : ResidentProjectedRow -> Array(Float32)) : Array(Float32)
    result = [] of Float32
    rows.each { |row| result.concat(yield row) }
    result
  end

  private def resident_shared_view(values : Array(Float32), buffers : Array(ML::MetalBuffer)) : ML::GGUF::Inkling2View
    buffer = ML::MetalBuffer.from_array(values)
    buffers << buffer
    ML::GGUF::Inkling2View.new(buffer)
  end

  private def resident_run_tile(
    attention : ML::GGUF::Inkling2ResidentAttention,
    start : Int32,
    projected : Array(ResidentProjectedRow),
    weights : ML::GGUF::Inkling2AttentionWeights,
    config : ML::GGUF::Inkling2Config,
    *,
    decode_block : Bool = false,
  ) : Array(Float32)
    rows = projected.size.to_i32
    q_width = config.n_head * config.head_dim
    output_count = rows * q_width
    buffers = [] of ML::MetalBuffer
    scratch = ML::GGUF::Inkling2ResidentScratch.new
    output = ML::MetalBuffer.new(output_count.to_i64 * sizeof(Float32), ML::StorageMode::Shared)
    begin
      q = resident_shared_view(resident_flatten(projected) { |row| row.q }, buffers)
      k = resident_shared_view(resident_flatten(projected) { |row| row.k }, buffers)
      v = resident_shared_view(resident_flatten(projected) { |row| row.v }, buffers)
      q_rel = resident_shared_view(resident_flatten(projected) { |row| row.q_rel }, buffers)
      gate = resident_shared_view(resident_flatten(projected) { |row| row.gate }, buffers)
      sink = resident_shared_view(resident_flatten(projected) { |row| row.sink }, buffers)
      q_clip = resident_shared_view(weights.q_clip, buffers)
      rel_clip = resident_shared_view(weights.rel_clip, buffers)
      relative_bias = resident_shared_view(weights.rel_bias, buffers)
      k_taps = resident_shared_view(weights.k_conv, buffers)
      v_taps = resident_shared_view(weights.v_conv, buffers)
      dynamic_k = if projected.first.dynamic_k
                    resident_shared_view(resident_flatten(projected) { |row| row.dynamic_k.not_nil! }, buffers)
                  end
      dynamic_v = if projected.first.dynamic_v
                    resident_shared_view(resident_flatten(projected) { |row| row.dynamic_v.not_nil! }, buffers)
                  end

      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      attended = nil.as(ML::GGUF::Inkling2View?)
      begin
        attended = attention.encode(encoder, scratch, start, rows, q, k, v, q_rel, gate, sink,
          q_clip, rel_clip, relative_bias, k_taps, v_taps, dynamic_k, dynamic_v,
          decode_block: decode_block)
        ML::GGUF::Inkling2ResidentMath.copy(encoder, attended.not_nil!, ML::GGUF::Inkling2View.new(output), output_count)
      ensure
        encoder.end_encoding
      end
      command.commit_and_wait
      attention.finish
      output.read(output_count)
    ensure
      scratch.close
      buffers.each(&.release)
      output.release
    end
  end

  private def resident_run_prefill_mode(
    config : ML::GGUF::Inkling2Config,
    weights : ML::GGUF::Inkling2AttentionWeights,
    projected : Array(ResidentProjectedRow),
    tiles : Array(Tuple(Int32, Int32)),
    *,
    sliding : Bool,
    max_seq : Int32,
    expected_pipeline : String,
    rollback_rows : Int32 = 0,
    prefill_mode : ML::GGUF::Inkling2ResidentAttention::PrefillMode? = nil,
    snapshot_positions : Array(Int32) = [] of Int32,
    expect_regular_fallback : Bool = false,
  ) : ResidentAttentionRun
    ML::Metal::PipelineCache.clear
    attention = if mode = prefill_mode
                  ML::GGUF::Inkling2ResidentAttention.new(config, sliding, max_seq,
                    rollback_rows: rollback_rows, prefill_mode: mode)
                else
                  ML::GGUF::Inkling2ResidentAttention.new(config, sliding, max_seq,
                    rollback_rows: rollback_rows)
                end
    begin
      output = [] of Float32
      tiles.each do |start, rows|
        output.concat(resident_run_tile(attention, start, projected[start, rows], weights, config))
      end
      ML::Metal::PipelineCache.resident_spec_contains?(expected_pipeline).should be_true
      if expect_regular_fallback
        ML::Metal::PipelineCache.resident_spec_contains?("inkling2_resident_attention").should be_true
      end
      snapshots = snapshot_positions.map do |position|
        attention.key_value_at(position, max_seq)
      end
      ResidentAttentionRun.new(output, snapshots)
    ensure
      attention.close
    end
  end

  private def resident_bits(values : Array(Float32)) : Array(UInt32)
    values.map(&.unsafe_as(UInt32))
  end

  private def resident_check_scalar_convolution_pair(
    start : Int32,
    reference : {ML::GGUF::Inkling2ResidentConvolution, ML::GGUF::Inkling2ResidentConvolution},
    paired : {ML::GGUF::Inkling2ResidentConvolution, ML::GGUF::Inkling2ResidentConvolution},
    seed : Int32,
  ) : Nil
    width = reference[0].channels
    taps = reference[0].taps
    heads = 3_i32
    buffers = [] of ML::MetalBuffer
    scratch = ML::GGUF::Inkling2ResidentScratch.new
    readback = ML::MetalBuffer.new(4_i64 * width * sizeof(Float32), ML::StorageMode::Shared)
    command = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(command)
    begin
      inputs = StaticArray(ML::GGUF::Inkling2View, 2).new do |side|
        view = resident_shared_view(Array(Float32).new(width + 4) do |index|
          Math.sin((index + 1) * (seed + 1) * (side + 1) * 0.013).to_f32
        end, buffers)
        ML::GGUF::Inkling2View.new(view.buffer, 4_i64 * sizeof(Float32))
      end
      weights = StaticArray(ML::GGUF::Inkling2View, 2).new do |side|
        view = resident_shared_view(Array(Float32).new(width * taps + 4) do |index|
          Math.cos((index + 1) * (side + 1) * 0.077).to_f32
        end, buffers)
        ML::GGUF::Inkling2View.new(view.buffer, 4_i64 * sizeof(Float32))
      end
      dynamic_taps = resident_shared_view(Array(Float32).new(heads * taps + 4) do |index|
        Math.sin((seed + index + 1) * 0.17).to_f32
      end, buffers)
      dynamic = StaticArray(ML::GGUF::Inkling2View?, 2).new do |side|
        ML::GGUF::Inkling2View.new(dynamic_taps.buffer, 4_i64 * sizeof(Float32)) if seed % 3 != side
      end

      begin
        expected = StaticArray(ML::GGUF::Inkling2View, 2).new do |side|
          reference[side].encode(encoder, scratch, start, 1, inputs[side], weights[side],
            dynamic[side], dynamic_heads: dynamic[side] ? heads : 0)
        end
        actual = paired[0].encode_scalar_pair(encoder, scratch, start, paired[1],
          {inputs[0], inputs[1]}, {weights[0], weights[1]}, {dynamic[0], dynamic[1]}, heads)
        {expected[0], expected[1], actual[0], actual[1]}.each_with_index do |view, index|
          destination = ML::GGUF::Inkling2View.new(readback, index.to_i64 * width * sizeof(Float32))
          ML::GGUF::Inkling2ResidentMath.copy(encoder, view, destination, width)
        end
      ensure
        encoder.end_encoding
      end
      command.commit_and_wait
      bits = resident_bits(readback.read(4 * width))
      2.times do |side|
        bits[(side + 2) * width, width].should eq bits[side * width, width]
      end
    ensure
      scratch.close
      readback.release
      buffers.each(&.release)
    end
  end

  private def resident_encode_convolution_tile(
    convolution : ML::GGUF::Inkling2ResidentConvolution,
    start : Int32,
    input_values : Array(Float32),
    tap_values : Array(Float32),
  ) : Nil
    buffers = [] of ML::MetalBuffer
    scratch = ML::GGUF::Inkling2ResidentScratch.new
    command = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(command)
    begin
      input = resident_shared_view(input_values, buffers)
      taps = resident_shared_view(tap_values, buffers)
      begin
        convolution.encode(encoder, scratch, start, input_values.size.to_i32, input, taps)
      ensure
        encoder.end_encoding
      end
      command.commit_and_wait
    ensure
      scratch.close
      buffers.each(&.release)
    end
  end

  private def resident_run_convolution_tile(
    convolution : ML::GGUF::Inkling2ResidentConvolution,
    start : Int32,
    input_values : Array(Float32),
    tap_values : Array(Float32),
  ) : Array(Float32)
    count = input_values.size.to_i32
    buffers = [] of ML::MetalBuffer
    scratch = ML::GGUF::Inkling2ResidentScratch.new
    output = ML::MetalBuffer.new(count.to_i64 * sizeof(Float32), ML::StorageMode::Shared)
    command = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(command)
    begin
      input = resident_shared_view(input_values, buffers)
      taps = resident_shared_view(tap_values, buffers)
      begin
        result = convolution.encode(encoder, scratch, start, count, input, taps)
        ML::GGUF::Inkling2ResidentMath.copy(encoder, result, ML::GGUF::Inkling2View.new(output), count)
      ensure
        encoder.end_encoding
      end
      command.commit_and_wait
      output.read(count)
    ensure
      scratch.close
      buffers.each(&.release)
      output.release
    end
  end

  if ML::Metal::Device.init!
    describe ML::GGUF::Inkling2ResidentAttention do
      it "keeps 128-channel full-prefill results bitwise across regular, scores, and shared-KV modes" do
        config = resident_test_config(sliding: false, n_head: 3_i32, n_head_kv: 1_i32,
          head_dim: 128_i32, rel_dim: 16_i32, window: 4_i32, max_seq: 7_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        projected = resident_projected_rows(weights, matmul, resident_inputs(config, 7_i32))
        # Three-row prefill tiles exercise odd row/head counts; the final
        # one-row tail must remain on the regular path.
        tiles = [{0_i32, 3_i32}, {3_i32, 3_i32}, {6_i32, 1_i32}]

        default_regular = resident_run_prefill_mode(config, weights, projected, tiles,
          sliding: false, max_seq: 7_i32, expected_pipeline: "inkling2_resident_attention",
          expect_regular_fallback: true)
        explicit_regular = resident_run_prefill_mode(config, weights, projected, tiles,
          sliding: false, max_seq: 7_i32, prefill_mode: ML::GGUF::Inkling2ResidentAttention::PrefillMode::Regular,
          expected_pipeline: "inkling2_resident_attention", expect_regular_fallback: true)
        scores = resident_run_prefill_mode(config, weights, projected, tiles,
          sliding: false, max_seq: 7_i32, prefill_mode: ML::GGUF::Inkling2ResidentAttention::PrefillMode::Scores,
          expected_pipeline: "inkling2_resident_attention_prefill_scores", expect_regular_fallback: true)
        shared_kv = resident_run_prefill_mode(config, weights, projected, tiles,
          sliding: false, max_seq: 7_i32, prefill_mode: ML::GGUF::Inkling2ResidentAttention::PrefillMode::SharedKV,
          expected_pipeline: "inkling2_resident_attention_prefill_shared_kv", expect_regular_fallback: true)

        resident_bits(default_regular.output).should eq(resident_bits(explicit_regular.output))
        resident_bits(scores.output).should eq(resident_bits(explicit_regular.output))
        resident_bits(shared_kv.output).should eq(resident_bits(explicit_regular.output))
      end

      it "matches scores across ordered prefill spans and the reserved KV ring wrap" do
        config = resident_test_config(sliding: true, n_head: 3_i32, n_head_kv: 1_i32,
          head_dim: 128_i32, rel_dim: 16_i32, window: 580_i32, max_seq: 707_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        projected = resident_projected_rows(weights, matmul, resident_inputs(config, 707_i32))
        tiles = [{0_i32, 512_i32}, {512_i32, 65_i32}, {577_i32, 65_i32}, {642_i32, 65_i32}]
        retained_positions = [127_i32, 511_i32, 581_i32, 582_i32, 706_i32]

        scores = resident_run_prefill_mode(config, weights, projected, tiles,
          sliding: true, max_seq: 707_i32, snapshot_positions: retained_positions, rollback_rows: 2_i32,
          prefill_mode: ML::GGUF::Inkling2ResidentAttention::PrefillMode::Scores,
          expected_pipeline: "inkling2_resident_attention_prefill_scores")
        # The second tile starts inside the logical window; the third reads
        # cached keys on both sides of the physical wrap at position 582.
        ordered = resident_run_prefill_mode(config, weights, projected, [{0_i32, 577_i32}, {577_i32, 65_i32}, {642_i32, 65_i32}],
          sliding: true, max_seq: 707_i32, snapshot_positions: retained_positions, rollback_rows: 2_i32,
          prefill_mode: ML::GGUF::Inkling2ResidentAttention::PrefillMode::Ordered,
          expected_pipeline: "inkling2_prefill_ordered_softmax_pv")
        ML::Metal::PipelineCache.resident_spec_contains?("inkling2_prefill_ordered_qk_gqa3").should be_true

        ML::Metal::PipelineCache.resident_spec_contains?("inkling2_resident_attention_prefill_scores").should be_false
        resident_bits(ordered.output).should eq(resident_bits(scores.output))
        ordered.snapshots.zip(scores.snapshots).each do |actual, expected|
          actual[2].should eq(expected[2])
          resident_bits(actual[0]).should eq(resident_bits(expected[0]))
          resident_bits(actual[1]).should eq(resident_bits(expected[1]))
        end
      end

      it "keeps global ordered panels bitwise across the cached prefix and a one-row tail" do
        config = resident_test_config(sliding: false, n_head: 3_i32, n_head_kv: 1_i32,
          head_dim: 128_i32, rel_dim: 16_i32, window: 4_i32, max_seq: 1_088_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        projected = resident_projected_rows(weights, matmul, resident_inputs(config, 1_088_i32))
        # The final span starts at key 1023, so its first panel crosses
        # the cache/tile boundary inside a key group and ends with 32+32+1 rows.
        tiles = [{0_i32, 511_i32}, {511_i32, 512_i32}, {1_023_i32, 65_i32}]
        retained_positions = [0_i32, 1_022_i32, 1_023_i32, 1_055_i32, 1_087_i32]

        regular = resident_run_prefill_mode(config, weights, projected, tiles,
          sliding: false, max_seq: 1_088_i32, snapshot_positions: retained_positions,
          expected_pipeline: "inkling2_resident_attention")
        ordered = resident_run_prefill_mode(config, weights, projected, tiles,
          sliding: false, max_seq: 1_088_i32, snapshot_positions: retained_positions,
          prefill_mode: ML::GGUF::Inkling2ResidentAttention::PrefillMode::Ordered,
          expected_pipeline: "inkling2_prefill_global_qk_gqa3")
        ML::Metal::PipelineCache.resident_spec_contains?("inkling2_prefill_global_softmax_pv").should be_true
        resident_bits(ordered.output).should eq(resident_bits(regular.output))
        ordered.snapshots.zip(regular.snapshots).each do |actual, expected|
          actual[2].should eq(expected[2])
          resident_bits(actual[0]).should eq(resident_bits(expected[0]))
          resident_bits(actual[1]).should eq(resident_bits(expected[1]))
        end
      end

      it "keeps per-head ordered prefill bitwise when four queries have no common keys" do
        config = resident_test_config(sliding: true, n_head: 2_i32, n_head_kv: 1_i32,
          head_dim: 128_i32, rel_dim: 16_i32, window: 3_i32, max_seq: 130_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        projected = resident_projected_rows(weights, matmul, resident_inputs(config, 130_i32))
        tiles = [{0_i32, 65_i32}, {65_i32, 65_i32}]

        scores = resident_run_prefill_mode(config, weights, projected, tiles,
          sliding: true, max_seq: 130_i32, rollback_rows: 2_i32,
          prefill_mode: ML::GGUF::Inkling2ResidentAttention::PrefillMode::Scores,
          expected_pipeline: "inkling2_resident_attention_prefill_scores")
        ordered = resident_run_prefill_mode(config, weights, projected, tiles,
          sliding: true, max_seq: 130_i32, rollback_rows: 2_i32,
          prefill_mode: ML::GGUF::Inkling2ResidentAttention::PrefillMode::Ordered,
          expected_pipeline: "inkling2_prefill_ordered_qk")

        ML::Metal::PipelineCache.resident_spec_contains?("inkling2_resident_attention_prefill_scores").should be_false
        resident_bits(ordered.output).should eq(resident_bits(scores.output))
      end

      it "keeps 128-channel sliding-prefill results bitwise through reserved-ring wrap" do
        config = resident_test_config(sliding: true, n_head: 3_i32, n_head_kv: 1_i32,
          head_dim: 128_i32, rel_dim: 16_i32, window: 3_i32, max_seq: 9_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        inputs = resident_inputs(config, 9_i32)
        projected = resident_projected_rows(weights, matmul, inputs)
        tiles = [{0_i32, 3_i32}, {3_i32, 3_i32}, {6_i32, 2_i32}, {8_i32, 1_i32}]
        retained_positions = [6_i32, 7_i32, 8_i32]

        expected = [] of Float32
        cpu = ML::GGUF::Inkling2AttentionState.new(config, true, 9_i32)
        inputs.each_with_index do |(q_in, k_in, v_in), position|
          output = ML::GGUF::Inkling2Attention.forward(weights, config, cpu, matmul,
            q_in, k_in, v_in, position.to_i32)
          output.each { |value| expected << value * Math.sqrt(config.conv_size + 1).to_f32 }
        end
        expected_snapshots = [0_i32, 1_i32, 2_i32].map { |index| cpu.key_value_at(index) }

        regular = resident_run_prefill_mode(config, weights, projected, tiles,
          sliding: true, max_seq: 9_i32, rollback_rows: 2_i32,
          expected_pipeline: "inkling2_resident_attention", snapshot_positions: retained_positions)
        scores = resident_run_prefill_mode(config, weights, projected, tiles,
          sliding: true, max_seq: 9_i32, rollback_rows: 2_i32,
          prefill_mode: ML::GGUF::Inkling2ResidentAttention::PrefillMode::Scores,
          expected_pipeline: "inkling2_resident_attention_prefill_scores",
          snapshot_positions: retained_positions, expect_regular_fallback: true)
        shared_kv = resident_run_prefill_mode(config, weights, projected, tiles,
          sliding: true, max_seq: 9_i32, rollback_rows: 2_i32,
          prefill_mode: ML::GGUF::Inkling2ResidentAttention::PrefillMode::SharedKV,
          expected_pipeline: "inkling2_resident_attention_prefill_shared_kv",
          snapshot_positions: retained_positions, expect_regular_fallback: true)

        # The logical window is three rows and the reserve grows the physical
        # modulo ring to five slots, so positions 5..8 cross its head.
        resident_bits(scores.output).should eq(resident_bits(regular.output))
        resident_bits(shared_kv.output).should eq(resident_bits(regular.output))
        expected.size.times { |index| (regular.output[index] - expected[index]).abs.should be < 4e-4_f32 }
        {scores, shared_kv}.each do |candidate|
          candidate.snapshots.size.should eq(3)
          candidate.snapshots.each_with_index do |(key, value, position), index|
            regular_key, regular_value, regular_position = regular.snapshots[index]
            position.should eq(retained_positions[index])
            position.should eq(regular_position)
            resident_bits(key).should eq(resident_bits(regular_key))
            resident_bits(value).should eq(resident_bits(regular_value))
            expected_key, expected_value, expected_position = expected_snapshots[index]
            position.should eq(expected_position)
            key.size.times { |channel| (key[channel] - expected_key[channel]).abs.should be < 3e-4_f32 }
            value.size.times { |channel| (value[channel] - expected_value[channel]).abs.should be < 3e-4_f32 }
          end
        end
      end

      it "matches sliding attention across tile boundaries, ring wrap, oversized batches, reset, and KV snapshots" do
        config = resident_test_config(sliding: true, n_head: 4_i32, n_head_kv: 2_i32,
          head_dim: 3_i32, rel_dim: 1_i32, window: 3_i32, max_seq: 16_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        inputs = resident_inputs(config, 9_i32)
        projected = resident_projected_rows(weights, matmul, inputs)
        gpu = ML::GGUF::Inkling2ResidentAttention.new(config, true, 16_i32)
        cpu = ML::GGUF::Inkling2AttentionState.new(config, true, 16_i32)
        expected = [] of Float32

        inputs.each_with_index do |(q_in, k_in, v_in), position|
          output = ML::GGUF::Inkling2Attention.forward(weights, config, cpu, matmul, q_in, k_in, v_in, position.to_i32)
          output.each { |value| expected << value * Math.sqrt(config.conv_size + 1).to_f32 }
        end

        actual = resident_run_tile(gpu, 0_i32, projected[0, 5], weights, config)
        actual.concat(resident_run_tile(gpu, 5_i32, projected[5, 4], weights, config))
        actual.size.times { |index| (actual[index] - expected[index]).abs.should be < 3e-4_f32 }

        3.times do |chronological|
          key, value, position = cpu.key_value_at(chronological.to_i32)
          snapshot_key, snapshot_value, snapshot_position = gpu.key_value_at(position, 9_i32)
          snapshot_position.should eq position
          key.size.times { |channel| (snapshot_key[channel] - key[channel]).abs.should be < 3e-4_f32 }
          value.size.times { |channel| (snapshot_value[channel] - value[channel]).abs.should be < 3e-4_f32 }
        end
        expect_raises(IndexError) { gpu.key_value_at(4_i32, 5_i32) }

        gpu.reset
        cpu.reset
        reset_expected = ML::GGUF::Inkling2Attention.forward(weights, config, cpu, matmul,
          inputs[0][0], inputs[0][1], inputs[0][2], 0_i32)
        reset_actual = resident_run_tile(gpu, 0_i32, [projected[0]], weights, config)
        reset_actual.size.times do |index|
          (reset_actual[index] - reset_expected[index] * Math.sqrt(config.conv_size + 1).to_f32).abs.should be < 3e-4_f32
        end
      ensure
        gpu.try(&.close)
      end

      it "preserves the old full-attention prefix when cache growth crosses 256 rows" do
        config = resident_test_config(sliding: false, n_head: 1_i32, n_head_kv: 1_i32,
          head_dim: 1_i32, rel_dim: 1_i32, window: 4_i32, max_seq: 300_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        inputs = resident_inputs(config, 258_i32)
        projected = resident_projected_rows(weights, matmul, inputs)
        gpu = ML::GGUF::Inkling2ResidentAttention.new(config, false, 300_i32)
        cpu = ML::GGUF::Inkling2AttentionState.new(config, false, 300_i32)
        expected = [] of Float32

        inputs.each_with_index do |(q_in, k_in, v_in), position|
          output = ML::GGUF::Inkling2Attention.forward(weights, config, cpu, matmul, q_in, k_in, v_in, position.to_i32)
          expected << output[0] * Math.sqrt(config.conv_size + 1).to_f32
        end

        actual = resident_run_tile(gpu, 0_i32, projected[0, 256], weights, config)
        actual.concat(resident_run_tile(gpu, 256_i32, projected[256, 2], weights, config))
        actual.size.times { |index| (actual[index] - expected[index]).abs.should be < 4e-4_f32 }

        {0_i32, 127_i32, 255_i32, 257_i32}.each do |position|
          key, value, actual_position = gpu.key_value_at(position, 258_i32)
          expected_key, expected_value, expected_position = cpu.key_value_at(position)
          actual_position.should eq position
          expected_position.should eq position
          (key[0] - expected_key[0]).abs.should be < 3e-4_f32
          (value[0] - expected_value[0]).abs.should be < 3e-4_f32
        end
      ensure
        gpu.try(&.close)
      end

      it "matches split-K decode after 64 keys through a retained sliding ring" do
        config = resident_test_config(sliding: true, n_head: 2_i32, n_head_kv: 1_i32,
          head_dim: 2_i32, rel_dim: 2_i32, window: 70_i32, max_seq: 90_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        inputs = resident_inputs(config, 80_i32)
        projected = resident_projected_rows(weights, matmul, inputs)
        gpu = ML::GGUF::Inkling2ResidentAttention.new(config, true, 90_i32)
        cpu = ML::GGUF::Inkling2AttentionState.new(config, true, 90_i32)
        expected = [] of Float32

        inputs.each_with_index do |(q_in, k_in, v_in), position|
          output = ML::GGUF::Inkling2Attention.forward(weights, config, cpu, matmul,
            q_in, k_in, v_in, position.to_i32)
          output.each { |value| expected << value * Math.sqrt(config.conv_size + 1).to_f32 }
        end

        actual = [] of Float32
        projected.each_with_index do |row, position|
          actual.concat(resident_run_tile(gpu, position.to_i32, [row], weights, config))
        end
        actual.size.times { |index| (actual[index] - expected[index]).abs.should be < 5e-4_f32 }

        {10_i32, 42_i32, 79_i32}.each do |position|
          snapshot_key, snapshot_value, snapshot_position = gpu.key_value_at(position, 80_i32)
          expected_key, expected_value, expected_position = cpu.key_value_at(position - 10)
          snapshot_position.should eq position
          expected_position.should eq position
          snapshot_key.size.times { |channel| (snapshot_key[channel] - expected_key[channel]).abs.should be < 3e-4_f32 }
          snapshot_value.size.times { |channel| (snapshot_value[channel] - expected_value[channel]).abs.should be < 3e-4_f32 }
        end
        expect_raises(IndexError) { gpu.key_value_at(9_i32, 80_i32) }
        expect_raises(IndexError) { gpu.key_value_at(9_i32, 10_i32) }
      ensure
        gpu.try(&.close)
      end

      it "matches serial attention across the 64-key split threshold in a four-row decode block" do
        config = resident_test_config(sliding: false, n_head: 2_i32, n_head_kv: 1_i32,
          head_dim: 2_i32, rel_dim: 2_i32, window: 70_i32, max_seq: 70_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        inputs = resident_inputs(config, 66_i32)
        projected = resident_projected_rows(weights, matmul, inputs)
        blocked = ML::GGUF::Inkling2ResidentAttention.new(config, false, 70_i32)
        serial = ML::GGUF::Inkling2ResidentAttention.new(config, false, 70_i32)
        begin
          # Both paths share a 62-row prefix. The following four rows expose
          # 63, 64, 65, and 66 keys, so the block crosses the >64 split-K gate.
          resident_run_tile(blocked, 0_i32, projected[0, 62], weights, config)
          resident_run_tile(serial, 0_i32, projected[0, 62], weights, config)

          expected = [] of Float32
          4.times do |row|
            expected.concat(resident_run_tile(serial, 62 + row, [projected[62 + row]], weights, config))
          end
          actual = resident_run_tile(blocked, 62_i32, projected[62, 4], weights, config, decode_block: true)
          actual.size.times { |index| (actual[index] - expected[index]).abs.should be < 2e-6_f32 }

          {62_i32, 63_i32, 64_i32, 65_i32}.each do |position|
            block_key, block_value, _block_position = blocked.key_value_at(position, 66_i32)
            serial_key, serial_value, _serial_position = serial.key_value_at(position, 66_i32)
            block_key.size.times { |channel| (block_key[channel] - serial_key[channel]).abs.should be < 2e-6_f32 }
            block_value.size.times { |channel| (block_value[channel] - serial_value[channel]).abs.should be < 2e-6_f32 }
          end
        ensure
          blocked.close
          serial.close
        end
      end

      it "matches serial attention when a four-row decode block crosses the 1024-slot KV ring" do
        config = resident_test_config(sliding: true, n_head: 1_i32, n_head_kv: 1_i32,
          head_dim: 1_i32, rel_dim: 1_i32, window: 1024_i32, max_seq: 1026_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        inputs = resident_inputs(config, 1026_i32)
        projected = resident_projected_rows(weights, matmul, inputs)
        blocked = ML::GGUF::Inkling2ResidentAttention.new(config, true, 1026_i32)
        serial = ML::GGUF::Inkling2ResidentAttention.new(config, true, 1026_i32)
        begin
          resident_run_tile(blocked, 0_i32, projected[0, 1022], weights, config)
          resident_run_tile(serial, 0_i32, projected[0, 1022], weights, config)

          expected = [] of Float32
          4.times do |row|
            expected.concat(resident_run_tile(serial, 1022 + row, [projected[1022 + row]], weights, config))
          end
          actual = resident_run_tile(blocked, 1022_i32, projected[1022, 4], weights, config, decode_block: true)
          actual.size.times { |index| (actual[index] - expected[index]).abs.should be < 3e-6_f32 }

          {2_i32, 1023_i32, 1024_i32, 1025_i32}.each do |position|
            block_key, block_value, _block_position = blocked.key_value_at(position, 1026_i32)
            serial_key, serial_value, _serial_position = serial.key_value_at(position, 1026_i32)
            (block_key[0] - serial_key[0]).abs.should be < 2e-6_f32
            (block_value[0] - serial_value[0]).abs.should be < 2e-6_f32
          end
        ensure
          blocked.close
          serial.close
        end
      end

      it "preserves 128-channel split-K results across a four-row decode block and a reserved KV ring" do
        config = resident_test_config(sliding: true, n_head: 2_i32, n_head_kv: 1_i32,
          head_dim: 128_i32, rel_dim: 16_i32, window: 1024_i32, max_seq: 1026_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        pattern = resident_projected_rows(weights, matmul, resident_inputs(config, 23_i32))
        projected = Array(ResidentProjectedRow).new(1026) { |row| pattern[row % pattern.size] }
        blocked = ML::GGUF::Inkling2ResidentAttention.new(config, true, 1026_i32, rollback_rows: 4_i32)
        serial = ML::GGUF::Inkling2ResidentAttention.new(config, true, 1026_i32)
        begin
          resident_run_tile(blocked, 0_i32, projected[0, 1022], weights, config)
          resident_run_tile(serial, 0_i32, projected[0, 1022], weights, config)

          ML::Metal::PipelineCache.clear
          expected = [] of Float32
          4.times do |row|
            expected.concat(resident_run_tile(serial, 1022 + row, [projected[1022 + row]], weights, config))
          end
          {% if flag?(:aarch64) %}
            ML::Metal::PipelineCache.resident_spec_contains?("inkling2_resident_attention_split_qk8").should be_true
          {% end %}

          ML::Metal::PipelineCache.clear
          actual = resident_run_tile(blocked, 1022_i32, projected[1022, 4], weights, config, decode_block: true)
          ML::Metal::PipelineCache.resident_spec_contains?("inkling2_resident_attention_split_parallel").should be_true
          actual.map(&.unsafe_as(UInt32)).should eq(expected.map(&.unsafe_as(UInt32)))

          {2_i32, 1023_i32, 1024_i32, 1025_i32}.each do |position|
            block_key, block_value, block_position = blocked.key_value_at(position, 1026_i32)
            serial_key, serial_value, serial_position = serial.key_value_at(position, 1026_i32)
            block_position.should eq(serial_position)
            block_key.map(&.unsafe_as(UInt32)).should eq(serial_key.map(&.unsafe_as(UInt32)))
            block_value.map(&.unsafe_as(UInt32)).should eq(serial_value.map(&.unsafe_as(UInt32)))
          end
        ensure
          blocked.close
          serial.close
        end
      end

      it "preserves split-K attention output and KV after rewinding across a physical ring wrap" do
        config = resident_test_config(sliding: true, n_head: 2_i32, n_head_kv: 1_i32,
          head_dim: 128_i32, rel_dim: 16_i32, window: 70_i32, max_seq: 153_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        pattern = resident_projected_rows(weights, matmul, resident_inputs(config, 23_i32))
        projected = Array(ResidentProjectedRow).new(151) { |row| pattern[row % pattern.size] }
        blocked = ML::GGUF::Inkling2ResidentAttention.new(config, true, 153_i32, rollback_rows: 4_i32)
        serial = ML::GGUF::Inkling2ResidentAttention.new(config, true, 153_i32, rollback_rows: 4_i32)
        begin
          ML::Metal::PipelineCache.clear
          # The 74-slot physical ring wraps at 148 during the four-row
          # append. Keep rows 147 and 148, then replace the rejected suffix.
          resident_run_tile(blocked, 0_i32, projected[0, 147], weights, config)
          resident_run_tile(serial, 0_i32, projected[0, 147], weights, config)
          block = resident_run_tile(blocked, 147_i32, projected[147, 4], weights, config, decode_block: true)
          kept = resident_run_tile(serial, 147_i32, [projected[147]], weights, config)
          kept.concat(resident_run_tile(serial, 148_i32, [projected[148]], weights, config))
          resident_bits(block[0, kept.size]).should eq(resident_bits(kept))
          blocked.rewind(149_i32)

          # Replay starts just after the wrap, so the last partition spans
          # both cache ranges and the new tile.
          replay = pattern.first(4)
          actual = resident_run_tile(blocked, 149_i32, replay, weights, config, decode_block: true)
          expected = [] of Float32
          replay.each_with_index do |row, offset|
            expected.concat(resident_run_tile(serial, 149 + offset, [row], weights, config))
          end
          resident_bits(actual).should eq(resident_bits(expected))
          ML::Metal::PipelineCache.resident_spec_contains?("inkling2_resident_attention_split_parallel").should be_true

          {83_i32, 147_i32, 148_i32, 149_i32, 152_i32}.each do |position|
            block_key, block_value, block_position = blocked.key_value_at(position, 153_i32)
            serial_key, serial_value, serial_position = serial.key_value_at(position, 153_i32)
            block_position.should eq(serial_position)
            resident_bits(block_key).should eq(resident_bits(serial_key))
            resident_bits(block_value).should eq(resident_bits(serial_value))
          end
        ensure
          blocked.close
          serial.close
        end
      end

      it "keeps global block and scalar decode bitwise beyond 512 cached scores per partition" do
        start = 2047_i32
        block_rows = 4_i32
        end_position = start + block_rows
        max_seq = end_position + 1
        config = resident_test_config(sliding: false, n_head: 1_i32, n_head_kv: 1_i32,
          head_dim: 128_i32, rel_dim: 16_i32, window: 1024_i32, max_seq: max_seq)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        pattern = resident_projected_rows(weights, matmul, resident_inputs(config, 23_i32))
        prefix = Array(ResidentProjectedRow).new(start) { |row| pattern[row % pattern.size] }
        continuation = pattern.first(block_rows)
        reference = ML::GGUF::Inkling2ResidentAttention.new(config, false, max_seq)
        fast = ML::GGUF::Inkling2ResidentAttention.new(config, false, max_seq,
          prefill_mode: ML::GGUF::Inkling2ResidentAttention::PrefillMode::Fast)
        begin
          resident_run_tile(reference, 0_i32, prefix, weights, config)
          resident_run_tile(fast, 0_i32, prefix, weights, config)

          ML::Metal::PipelineCache.clear
          expected = resident_run_tile(reference, start, continuation, weights, config, decode_block: true)
          ML::Metal::PipelineCache.resident_spec_contains?("inkling2_resident_attention_split").should be_true

          actual = resident_run_tile(fast, start, continuation, weights, config, decode_block: true)
          ML::Metal::PipelineCache.resident_spec_contains?("inkling2_resident_attention_split_parallel").should be_true
          resident_bits(actual).should eq(resident_bits(expected))

          ML::Metal::PipelineCache.clear
          next_row = [pattern[block_rows]]
          expected_scalar = resident_run_tile(reference, end_position, next_row, weights, config)
          actual_scalar = resident_run_tile(fast, end_position, next_row, weights, config)
          {% if flag?(:aarch64) %}
            ML::Metal::PipelineCache.resident_spec_contains?("inkling2_resident_attention_split_qk8").should be_true
          {% end %}
          resident_bits(actual_scalar).should eq(resident_bits(expected_scalar))

          (start...max_seq).each do |position|
            actual_k, actual_v, _ = fast.key_value_at(position, max_seq)
            expected_k, expected_v, _ = reference.key_value_at(position, max_seq)
            resident_bits(actual_k).should eq(resident_bits(expected_k))
            resident_bits(actual_v).should eq(resident_bits(expected_v))
          end
        ensure
          reference.close
          fast.close
        end
      end

      it "rewinds a speculative sliding block without losing the accepted branch across cache wrap" do
        config = resident_test_config(sliding: true, n_head: 1_i32, n_head_kv: 1_i32,
          head_dim: 1_i32, rel_dim: 1_i32, window: 3_i32, max_seq: 23_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        original_inputs = resident_inputs(config, 23_i32)
        alternate_inputs = resident_inputs(config, 2_i32).map do |(q_in, k_in, v_in)|
          {q_in.map { |value| value * 1.7_f32 + 0.1_f32 },
           k_in.map { |value| value * -0.8_f32 + 0.2_f32 },
           v_in.map { |value| value * 1.3_f32 - 0.15_f32 }}
        end
        original_projected = resident_projected_rows(weights, matmul, original_inputs)
        alternate_projected = resident_projected_rows(weights, matmul, alternate_inputs)
        gpu = ML::GGUF::Inkling2ResidentAttention.new(config, true, 23_i32, rollback_rows: 4_i32)
        cpu = ML::GGUF::Inkling2AttentionState.new(config, true, 23_i32)
        expected = [] of Float32

        begin
          # The speculative rows 19..22 cross the seven-slot physical KV ring
          # while preserving the accepted logical window of three rows.
          resident_run_tile(gpu, 0_i32, original_projected[0, 19], weights, config)
          resident_run_tile(gpu, 19_i32, original_projected[19, 4], weights, config)
          gpu.rewind(21_i32)

          21.times do |position|
            q_in, k_in, v_in = original_inputs[position]
            output = ML::GGUF::Inkling2Attention.forward(weights, config, cpu, matmul,
              q_in, k_in, v_in, position.to_i32)
            output.each { |value| expected << value * Math.sqrt(config.conv_size + 1).to_f32 }
          end
          2.times do |row|
            q_in, k_in, v_in = alternate_inputs[row]
            output = ML::GGUF::Inkling2Attention.forward(weights, config, cpu, matmul,
              q_in, k_in, v_in, (21 + row).to_i32)
            output.each { |value| expected << value * Math.sqrt(config.conv_size + 1).to_f32 }
          end

          actual = resident_run_tile(gpu, 21_i32, [alternate_projected[0]], weights, config)
          # This rewind is within the reserve and before the current end, but
          # it crosses the latest appended tile's start and would revisit
          # history already changed by the accepted branch row.
          expect_raises(ArgumentError) { gpu.rewind(18_i32) }
          actual.concat(resident_run_tile(gpu, 22_i32, [alternate_projected[1]], weights, config))
          2.times do |row|
            q_width = config.n_head * config.head_dim
            q_width.times do |channel|
              (actual[row * q_width + channel] - expected[(21 + row) * q_width + channel]).abs.should be < 4e-5_f32
            end
          end

          {20_i32, 21_i32, 22_i32}.each_with_index do |position, chronological|
            actual_key, actual_value, actual_position = gpu.key_value_at(position, 23_i32)
            expected_key, expected_value, expected_position = cpu.key_value_at(chronological.to_i32)
            actual_position.should eq position
            expected_position.should eq position
            (actual_key[0] - expected_key[0]).abs.should be < 4e-5_f32
            (actual_value[0] - expected_value[0]).abs.should be < 4e-5_f32
          end
        ensure
          gpu.close
        end
      end

      it "retains a partial prefix across four scalar speculative writes over ring wrap" do
        config = resident_test_config(sliding: true, n_head: 1_i32, n_head_kv: 1_i32,
          head_dim: 1_i32, rel_dim: 1_i32, window: 3_i32, max_seq: 23_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        original_inputs = resident_inputs(config, 23_i32)
        alternate_inputs = resident_inputs(config, 1_i32).map do |(q_in, k_in, v_in)|
          {q_in.map { |value| value * 1.7_f32 + 0.1_f32 },
           k_in.map { |value| value * -0.8_f32 + 0.2_f32 },
           v_in.map { |value| value * 1.3_f32 - 0.15_f32 }}
        end
        original_projected = resident_projected_rows(weights, matmul, original_inputs)
        alternate_projected = resident_projected_rows(weights, matmul, alternate_inputs)
        gpu = ML::GGUF::Inkling2ResidentAttention.new(config, true, 23_i32, rollback_rows: 4_i32)
        cpu = ML::GGUF::Inkling2AttentionState.new(config, true, 23_i32)

        begin
          resident_run_tile(gpu, 0_i32, original_projected[0, 19], weights, config)
          expect_raises(ArgumentError) { gpu.begin_speculation(18_i32) }
          gpu.begin_speculation(19_i32)
          4.times do |offset|
            resident_run_tile(gpu, 19_i32 + offset, [original_projected[19 + offset]], weights, config)
          end
          expect_raises(ArgumentError) { gpu.rewind(22_i32) }
          expect_raises(ArgumentError) do
            resident_run_tile(gpu, 23_i32, [original_projected[22]], weights, config)
          end
          expect_raises(ArgumentError) { gpu.finish_speculation(18_i32) }
          gpu.finish_speculation(21_i32)

          21.times do |position|
            q_in, k_in, v_in = original_inputs[position]
            ML::GGUF::Inkling2Attention.forward(weights, config, cpu, matmul,
              q_in, k_in, v_in, position.to_i32)
          end
          q_in, k_in, v_in = alternate_inputs[0]
          expected = ML::GGUF::Inkling2Attention.forward(weights, config, cpu, matmul, q_in, k_in, v_in, 21_i32)
            .map { |value| value * Math.sqrt(config.conv_size + 1).to_f32 }
          actual = resident_run_tile(gpu, 21_i32, [alternate_projected[0]], weights, config)
          actual.size.times { |index| (actual[index] - expected[index]).abs.should be < 4e-5_f32 }

          {19_i32, 20_i32, 21_i32}.each_with_index do |position, chronological|
            actual_key, actual_value, actual_position = gpu.key_value_at(position, 22_i32)
            expected_key, expected_value, expected_position = cpu.key_value_at(chronological.to_i32)
            actual_position.should eq position
            expected_position.should eq position
            (actual_key[0] - expected_key[0]).abs.should be < 4e-5_f32
            (actual_value[0] - expected_value[0]).abs.should be < 4e-5_f32
          end
        ensure
          gpu.close
        end
      end

      it "matches separate scalar K/V convolutions across a history wrap and partial keep" do
        {0_i32, 4_i32}.each do |reserve|
          reference = {ML::GGUF::Inkling2ResidentConvolution.new(129, 3, rollback_rows: reserve),
                       ML::GGUF::Inkling2ResidentConvolution.new(129, 3, rollback_rows: reserve)}
          paired = {ML::GGUF::Inkling2ResidentConvolution.new(129, 3, rollback_rows: reserve),
                    ML::GGUF::Inkling2ResidentConvolution.new(129, 3, rollback_rows: reserve)}
          begin
            5.times { |position| resident_check_scalar_convolution_pair(position, reference, paired, position) }
            if reserve > 0
              {reference, paired}.each { |convolutions| convolutions.each(&.begin_speculation(5)) }
              5.upto(8) { |position| resident_check_scalar_convolution_pair(position, reference, paired, position) }
              {reference, paired}.each { |convolutions| convolutions.each(&.finish_speculation(7)) }
              7.upto(10) { |position| resident_check_scalar_convolution_pair(position, reference, paired, position + 20) }
            end
            resident_check_scalar_convolution_pair(0, reference, paired, 42)
          ensure
            {reference, paired}.each { |convolutions| convolutions.each(&.close) }
          end
        end
      end

      it "supports a bounded partial speculation span on standalone convolution" do
        convolution = ML::GGUF::Inkling2ResidentConvolution.new(1_i32, 2_i32, rollback_rows: 4_i32)
        taps = [0.65_f32, -0.25_f32]
        inputs = Array(Float32).new(12) { |index| (index % 11 - 5).to_f32 * 0.07_f32 }
        replacement = -0.31_f32

        begin
          resident_run_convolution_tile(convolution, 0_i32, inputs[0, 8], taps)
          expect_raises(ArgumentError) { convolution.begin_speculation(7_i32) }
          convolution.begin_speculation(8_i32)
          4.times do |offset|
            resident_run_convolution_tile(convolution, 8_i32 + offset, [inputs[8 + offset]], taps)
          end
          expect_raises(ArgumentError) { convolution.rewind(11_i32) }
          expect_raises(ArgumentError) do
            resident_run_convolution_tile(convolution, 12_i32, [replacement], taps)
          end
          expect_raises(ArgumentError) { convolution.finish_speculation(7_i32) }
          convolution.finish_speculation(10_i32)

          actual = resident_run_convolution_tile(convolution, 10_i32, [replacement], taps)
          expected = (replacement + inputs[9] * taps[0] + replacement * taps[1]) / Math.sqrt(3_f32).to_f32
          (actual[0] - expected).abs.should be < 2e-6_f32
        ensure
          convolution.close
        end
      end

      it "keeps standalone convolution rewind inside the latest appended tile" do
        convolution = ML::GGUF::Inkling2ResidentConvolution.new(1_i32, 2_i32, rollback_rows: 4_i32)
        taps = [0.65_f32, -0.25_f32]
        inputs = Array(Float32).new(68) { |index| (index % 11 - 5).to_f32 * 0.07_f32 }
        begin
          resident_encode_convolution_tile(convolution, 0_i32, inputs[0, 64], taps)
          resident_encode_convolution_tile(convolution, 64_i32, inputs[64, 4], taps)
          convolution.rewind(65_i32)
          resident_encode_convolution_tile(convolution, 65_i32, [0.31_f32], taps)

          expect_raises(ArgumentError) { convolution.rewind(62_i32) }
        ensure
          convolution.close
        end
      end
    end
  end
{% end %}
