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
end
