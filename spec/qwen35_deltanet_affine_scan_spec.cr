require "./spec_helper"
require "../src/ml/gguf/qwen35_deltanet_block_scan"

private alias BlockScan = ML::GGUF::Qwen35DeltaNetBlockScan
private alias DeltaInputs = ML::GGUF::Qwen35DeltaNetBlockScan::DeltaInputs

private def random_state(rng : Random, s : Int32) : BlockScan::Matrix
  Array.new(s) do
    Array.new(s) { ((rng.next_float - 0.5) * 0.2) }
  end
end

private def random_inputs(rng : Random, n_tokens : Int32, s : Int32,
                          kv_scale : Float64 = 0.5,
                          g_base : Float64 = 0.88,
                          g_span : Float64 = 0.10) : Array(DeltaInputs)
  Array.new(n_tokens) do
    DeltaInputs.new(
      Array.new(s) { ((rng.next_float - 0.5) * kv_scale) },
      Array.new(s) { ((rng.next_float - 0.5) * kv_scale) },
      Array.new(s) { ((rng.next_float - 0.5) * kv_scale) },
      g_base + g_span * rng.next_float,
      rng.next_float
    )
  end
end

describe "Qwen35 DeltaNet affine block scan algebra" do
  it "matches serial DeltaNet state updates via composed affine transforms" do
    s = 8
    rng = Random.new(0xA551_u64)
    state = random_state(rng, s)
    inputs = random_inputs(rng, 12, s, kv_scale: 0.6, g_base: 0.90, g_span: 0.08)

    serial = state
    inputs.each { |inp| serial, _ = BlockScan.serial_delta_step(serial, inp, 1.0) }

    composed = BlockScan.compose_all(inputs)
    scanned = BlockScan.apply_affine(state, composed)

    BlockScan.max_abs_delta(serial, scanned).should be < 1.0e-10
  end

  it "supports block-prefix replay that reproduces serial intermediate outputs" do
    s = 8
    block_size = 4
    scale = 1.0 / Math.sqrt(s.to_f64)
    rng = Random.new(0xB10C_u64)
    initial = random_state(rng, s)
    inputs = random_inputs(rng, 12, s)

    serial_state, serial_y = BlockScan.replay_block(initial, inputs, scale)

    blocks = inputs.each_slice(block_size).to_a
    block_summaries = blocks.map { |block| BlockScan.compose_all(block) }

    prefix_state = initial
    replay_y = [] of Array(Float64)
    blocks.each_with_index do |block, i|
      replayed_state, block_y = BlockScan.replay_block(prefix_state, block, scale)
      replay_y.concat(block_y)

      prefix_state = BlockScan.apply_affine(prefix_state, block_summaries[i])
      BlockScan.max_abs_delta(replayed_state, prefix_state).should be < 1.0e-10
    end

    BlockScan.max_abs_delta(serial_state, prefix_state).should be < 1.0e-10
    replay_y.size.should eq(serial_y.size)
    replay_y.each_with_index do |row, i|
      row.each_with_index do |v, j|
        (v - serial_y[i][j]).abs.should be < 1.0e-10
      end
    end
  end

  it "represents the block additive term as low-rank transformed V/K factors" do
    s = 8
    rng = Random.new(0x10B10C_u64)
    inputs = random_inputs(rng, 4, s)

    dense_block = BlockScan.compose_all(inputs)
    compact_b = BlockScan.dense_low_rank_b_for_block(inputs)

    BlockScan.max_abs_delta(dense_block.b, compact_b).should be < 1.0e-10
  end

  it "applies compact summaries without materializing dense B" do
    s = 8
    rng = Random.new(0xC0A9AC7_u64)
    state = random_state(rng, s)
    inputs = random_inputs(rng, 6, s)

    dense_block = BlockScan.compose_all(inputs)
    compact = BlockScan.compact_summary_for_block(inputs)

    dense_out = BlockScan.apply_affine(state, dense_block)
    compact_out = BlockScan.apply_compact(state, compact)

    BlockScan.max_abs_delta(dense_out, compact_out).should be < 1.0e-10
  end

  it "composes compact summaries without materializing dense B" do
    s = 8
    rng = Random.new(0xC04C7_u64)
    state = random_state(rng, s)
    first = random_inputs(rng, 4, s)
    second = random_inputs(rng, 5, s)

    dense = BlockScan.compose(BlockScan.compose_all(first), BlockScan.compose_all(second))
    compact = BlockScan.compose_compact(
      BlockScan.compact_summary_for_block(first),
      BlockScan.compact_summary_for_block(second)
    )

    dense_out = BlockScan.apply_affine(state, dense)
    compact_out = BlockScan.apply_compact(state, compact)

    BlockScan.max_abs_delta(dense.b, BlockScan.dense_b_from_compact(compact)).should be < 1.0e-10
    BlockScan.max_abs_delta(dense_out, compact_out).should be < 1.0e-10
  end

  it "represents the transition matrix A as compact rank updates" do
    s = 8
    rng = Random.new(0xA11_u64)
    inputs = random_inputs(rng, 6, s)

    dense = BlockScan.compose_all(inputs)
    compact_a = BlockScan.dense_a_from_transition(
      BlockScan.compact_transition_for_block(inputs), s
    )

    BlockScan.max_abs_delta(dense.a, compact_a).should be < 1.0e-10
  end

  it "applies fully compact summaries without materializing dense A or dense B" do
    s = 8
    rng = Random.new(0xFA11C0_u64)
    state = random_state(rng, s)
    inputs = random_inputs(rng, 6, s)

    dense = BlockScan.compose_all(inputs)
    fully_compact = BlockScan.fully_compact_summary_for_block(inputs)

    dense_out = BlockScan.apply_affine(state, dense)
    compact_out = BlockScan.apply_fully_compact(state, fully_compact)

    BlockScan.max_abs_delta(dense_out, compact_out).should be < 1.0e-10
  end

  it "composes fully compact summaries without materializing dense A or B" do
    s = 8
    rng = Random.new(0xF011C0_u64)
    state = random_state(rng, s)
    first = random_inputs(rng, 4, s)
    second = random_inputs(rng, 5, s)

    dense = BlockScan.compose(BlockScan.compose_all(first), BlockScan.compose_all(second))
    fully_compact = BlockScan.compose_fully_compact(
      BlockScan.fully_compact_summary_for_block(first),
      BlockScan.fully_compact_summary_for_block(second)
    )

    dense_a = BlockScan.dense_a_from_transition(fully_compact.transition, s)
    dense_out = BlockScan.apply_affine(state, dense)
    compact_out = BlockScan.apply_fully_compact(state, fully_compact)

    BlockScan.max_abs_delta(dense.a, dense_a).should be < 1.0e-10
    BlockScan.max_abs_delta(dense.b, BlockScan.dense_b_from_compact(
      BlockScan::CompactDeltaSummary.new(dense_a, fully_compact.b_lefts, fully_compact.b_rights)
    )).should be < 1.0e-10
    BlockScan.max_abs_delta(dense_out, compact_out).should be < 1.0e-10
  end

  it "compresses fully compact summaries to row-basis rank <= state size exactly" do
    s = 8
    rng = Random.new(0xC0FE55_u64)
    state = random_state(rng, s)
    inputs = random_inputs(rng, 16, s)

    dense = BlockScan.compose_all(inputs)
    full = BlockScan.fully_compact_summary_for_block(inputs)
    compressed = BlockScan.compress_fully_compact_row_basis(full, s)

    compressed.transition.u_cols.size.should be <= s
    compressed.transition.v_cols.size.should be <= s
    compressed.b_lefts.size.should be <= s
    compressed.b_rights.size.should be <= s

    dense_a = BlockScan.dense_a_from_transition(compressed.transition, s)
    dense_out = BlockScan.apply_affine(state, dense)
    compact_out = BlockScan.apply_fully_compact(state, compressed)

    BlockScan.max_abs_delta(dense.a, dense_a).should be < 1.0e-10
    BlockScan.max_abs_delta(dense_out, compact_out).should be < 1.0e-10
  end

  it "keeps prefix composition rank-capped when compressing after each compose" do
    s = 8
    block_size = 4
    rng = Random.new(0xC0A9E_u64)
    state = random_state(rng, s)
    inputs = random_inputs(rng, 32, s)
    blocks = inputs.each_slice(block_size).to_a

    dense = blocks.map { |block| BlockScan.compose_all(block) }
      .reduce { |acc, tr| BlockScan.compose(acc, tr) }
    compressed = blocks.map { |block| BlockScan.fully_compact_summary_for_block(block) }
      .reduce do |acc, summary|
        BlockScan.compose_fully_compact_compressed(acc, summary, s)
      end

    compressed.transition.u_cols.size.should be <= s
    compressed.b_lefts.size.should be <= s

    dense_a = BlockScan.dense_a_from_transition(compressed.transition, s)
    dense_out = BlockScan.apply_affine(state, dense)
    compact_out = BlockScan.apply_fully_compact(state, compressed)

    BlockScan.max_abs_delta(dense.a, dense_a).should be < 1.0e-10
    BlockScan.max_abs_delta(dense_out, compact_out).should be < 1.0e-10
  end

  it "compresses compact factors by left-basis without materializing dense matrices" do
    s = 8
    rng = Random.new(0xFAC70B_u64)
    state = random_state(rng, s)
    inputs = random_inputs(rng, 24, s)

    dense = BlockScan.compose_all(inputs)
    full = BlockScan.fully_compact_summary_for_block(inputs)
    compressed = BlockScan.compress_fully_compact_factor_basis(full)

    compressed.transition.u_cols.size.should be <= s
    compressed.b_lefts.size.should be <= s

    dense_a = BlockScan.dense_a_from_transition(compressed.transition, s)
    dense_out = BlockScan.apply_affine(state, dense)
    compact_out = BlockScan.apply_fully_compact(state, compressed)

    BlockScan.max_abs_delta(dense.a, dense_a).should be < 1.0e-8
    BlockScan.max_abs_delta(dense_out, compact_out).should be < 1.0e-8
  end

  it "keeps prefix composition rank-capped with factor-basis compression" do
    s = 8
    block_size = 4
    rng = Random.new(0xC0FFEE_u64)
    state = random_state(rng, s)
    inputs = random_inputs(rng, 48, s)
    blocks = inputs.each_slice(block_size).to_a

    dense = blocks.map { |block| BlockScan.compose_all(block) }
      .reduce { |acc, tr| BlockScan.compose(acc, tr) }
    compressed = blocks.map { |block| BlockScan.fully_compact_summary_for_block(block) }
      .reduce do |acc, summary|
        BlockScan.compose_fully_compact_factor_compressed(acc, summary)
      end

    compressed.transition.u_cols.size.should be <= s
    compressed.b_lefts.size.should be <= s

    dense_a = BlockScan.dense_a_from_transition(compressed.transition, s)
    dense_out = BlockScan.apply_affine(state, dense)
    compact_out = BlockScan.apply_fully_compact(state, compressed)

    BlockScan.max_abs_delta(dense.a, dense_a).should be < 1.0e-8
    BlockScan.max_abs_delta(dense_out, compact_out).should be < 1.0e-8
  end

  it "replays block outputs through adjoint transformed queries" do
    s = 8
    scale = 1.0 / Math.sqrt(s.to_f64)
    rng = Random.new(0xAD501_u64)
    state = random_state(rng, s)
    inputs = random_inputs(rng, 10, s)

    serial_state, serial_y = BlockScan.replay_block(state, inputs, scale)
    adjoint_state, adjoint_y = BlockScan.adjoint_replay_block(state, inputs, scale)

    BlockScan.max_abs_delta(serial_state, adjoint_state).should be < 1.0e-10
    adjoint_y.size.should eq(serial_y.size)
    serial_y.each_with_index do |row, t|
      row.each_with_index do |v, d|
        (v - adjoint_y[t][d]).abs.should be < 1.0e-10
      end
    end
  end

  it "supports block-prefix adjoint replay for intermediate outputs" do
    s = 8
    block_size = 4
    scale = 1.0 / Math.sqrt(s.to_f64)
    rng = Random.new(0xAD502_u64)
    initial = random_state(rng, s)
    inputs = random_inputs(rng, 16, s)

    serial_state, serial_y = BlockScan.replay_block(initial, inputs, scale)

    prefix_state = initial
    replay_y = [] of Array(Float64)
    inputs.each_slice(block_size) do |block|
      prefix_state, block_y = BlockScan.adjoint_replay_block(prefix_state, block, scale)
      replay_y.concat(block_y)
    end

    BlockScan.max_abs_delta(serial_state, prefix_state).should be < 1.0e-10
    replay_y.size.should eq(serial_y.size)
    replay_y.each_with_index do |row, t|
      row.each_with_index do |v, d|
        (v - serial_y[t][d]).abs.should be < 1.0e-10
      end
    end
  end
end
