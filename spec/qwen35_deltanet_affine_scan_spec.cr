require "./spec_helper"

private alias Matrix = Array(Array(Float64))

private struct DeltaInputs
  getter k : Array(Float64)
  getter v : Array(Float64)
  getter q : Array(Float64)
  getter g : Float64
  getter beta : Float64

  def initialize(@k : Array(Float64), @v : Array(Float64), @q : Array(Float64),
                 @g : Float64, @beta : Float64)
  end
end

private struct AffineDelta
  getter a : Matrix
  getter b : Matrix

  def initialize(@a : Matrix, @b : Matrix)
  end
end

private def zeros(rows : Int32, cols : Int32) : Matrix
  Array.new(rows) { Array.new(cols, 0.0) }
end

private def identity(n : Int32) : Matrix
  m = zeros(n, n)
  n.times { |i| m[i][i] = 1.0 }
  m
end

private def matmul(a : Matrix, b : Matrix) : Matrix
  rows = a.size
  mid = b.size
  cols = b[0].size
  out = zeros(rows, cols)
  rows.times do |i|
    cols.times do |j|
      acc = 0.0
      mid.times { |k| acc += a[i][k] * b[k][j] }
      out[i][j] = acc
    end
  end
  out
end

private def matadd(a : Matrix, b : Matrix) : Matrix
  rows = a.size
  cols = a[0].size
  out = zeros(rows, cols)
  rows.times do |i|
    cols.times { |j| out[i][j] = a[i][j] + b[i][j] }
  end
  out
end

private def dot(a : Array(Float64), b : Array(Float64)) : Float64
  acc = 0.0
  a.size.times { |i| acc += a[i] * b[i] }
  acc
end

private def max_abs_delta(a : Matrix, b : Matrix) : Float64
  max = 0.0
  a.size.times do |i|
    a[i].size.times do |j|
      d = (a[i][j] - b[i][j]).abs
      max = d if d > max
    end
  end
  max
end

private def serial_delta_step(state : Matrix, inp : DeltaInputs, scale : Float64) : Tuple(Matrix, Array(Float64))
  s = state.size
  next_state = zeros(s, s)
  y = Array.new(s, 0.0)

  s.times do |d2|
    decayed = Array.new(s, 0.0)
    s.times { |d1| decayed[d1] = state[d2][d1] * inp.g }

    sk = dot(decayed, inp.k)
    delt = inp.beta * (inp.v[d2] - sk)

    s.times { |d1| next_state[d2][d1] = decayed[d1] + inp.k[d1] * delt }
    y[d2] = dot(next_state[d2], inp.q) * scale
  end

  {next_state, y}
end

private def affine_for(inp : DeltaInputs) : AffineDelta
  s = inp.k.size
  a = identity(s)
  s.times do |i|
    s.times do |j|
      # Row-vector convention: r_t = r_{t-1} * g*(I - beta*K*K^T) + beta*V_t*K.
      a[i][j] = inp.g * (a[i][j] - inp.beta * inp.k[i] * inp.k[j])
    end
  end

  b = zeros(s, s)
  s.times do |d2|
    s.times { |d1| b[d2][d1] = inp.beta * inp.v[d2] * inp.k[d1] }
  end

  AffineDelta.new(a, b)
end

private def compose(first : AffineDelta, second : AffineDelta) : AffineDelta
  # Apply `first`, then `second`:
  #   (S*A1 + B1)*A2 + B2 = S*(A1*A2) + (B1*A2 + B2)
  AffineDelta.new(
    matmul(first.a, second.a),
    matadd(matmul(first.b, second.a), second.b)
  )
end

private def apply_affine(state : Matrix, tr : AffineDelta) : Matrix
  matadd(matmul(state, tr.a), tr.b)
end

private def vec_matmul(v : Array(Float64), m : Matrix) : Array(Float64)
  cols = m[0].size
  out = Array.new(cols, 0.0)
  cols.times do |j|
    acc = 0.0
    v.size.times { |i| acc += v[i] * m[i][j] }
    out[j] = acc
  end
  out
end

private def dense_low_rank_b_for_block(inputs : Array(DeltaInputs)) : Matrix
  s = inputs[0].k.size
  suffix = identity(s)
  transformed_keys = Array(Array(Float64)).new(inputs.size) { Array.new(s, 0.0) }

  (inputs.size - 1).downto(0) do |i|
    inp = inputs[i]
    transformed_keys[i] = vec_matmul(inp.k, suffix).map { |x| x * inp.beta }
    suffix = matmul(affine_for(inp).a, suffix)
  end

  b = zeros(s, s)
  inputs.each_with_index do |inp, t|
    s.times do |d2|
      s.times do |d1|
        b[d2][d1] += inp.v[d2] * transformed_keys[t][d1]
      end
    end
  end
  b
end

private def replay_block(state : Matrix, inputs : Array(DeltaInputs), scale : Float64) : Tuple(Matrix, Array(Array(Float64)))
  ys = [] of Array(Float64)
  cur = state
  inputs.each do |inp|
    cur, y = serial_delta_step(cur, inp, scale)
    ys << y
  end
  {cur, ys}
end

describe "Qwen35 DeltaNet affine block scan algebra" do
  it "matches serial DeltaNet state updates via composed affine transforms" do
    s = 8
    n_tokens = 12
    rng = Random.new(0xA551_u64)

    state = Array.new(s) do
      Array.new(s) { ((rng.next_float - 0.5) * 0.2) }
    end
    inputs = Array.new(n_tokens) do
      DeltaInputs.new(
        Array.new(s) { ((rng.next_float - 0.5) * 0.6) },
        Array.new(s) { ((rng.next_float - 0.5) * 0.6) },
        Array.new(s) { ((rng.next_float - 0.5) * 0.6) },
        0.90 + 0.08 * rng.next_float,
        rng.next_float
      )
    end

    serial = state
    inputs.each { |inp| serial, _ = serial_delta_step(serial, inp, 1.0) }

    composed = inputs.map { |inp| affine_for(inp) }.reduce { |acc, tr| compose(acc, tr) }
    scanned = apply_affine(state, composed)

    max_abs_delta(serial, scanned).should be < 1.0e-10
  end

  it "supports block-prefix replay that reproduces serial intermediate outputs" do
    s = 8
    block_size = 4
    n_tokens = 12
    scale = 1.0 / Math.sqrt(s.to_f64)
    rng = Random.new(0xB10C_u64)

    initial = Array.new(s) do
      Array.new(s) { ((rng.next_float - 0.5) * 0.2) }
    end
    inputs = Array.new(n_tokens) do
      DeltaInputs.new(
        Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
        Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
        Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
        0.88 + 0.10 * rng.next_float,
        rng.next_float
      )
    end

    serial_state, serial_y = replay_block(initial, inputs, scale)

    blocks = inputs.each_slice(block_size).to_a
    block_summaries = blocks.map do |block|
      block.map { |inp| affine_for(inp) }.reduce { |acc, tr| compose(acc, tr) }
    end

    prefix_state = initial
    replay_y = [] of Array(Float64)
    blocks.each_with_index do |block, i|
      replayed_state, block_y = replay_block(prefix_state, block, scale)
      replay_y.concat(block_y)

      prefix_state = apply_affine(prefix_state, block_summaries[i])
      max_abs_delta(replayed_state, prefix_state).should be < 1.0e-10
    end

    max_abs_delta(serial_state, prefix_state).should be < 1.0e-10
    replay_y.size.should eq(serial_y.size)
    replay_y.each_with_index do |row, i|
      row.each_with_index do |v, j|
        (v - serial_y[i][j]).abs.should be < 1.0e-10
      end
    end
  end

  it "represents the block additive term as low-rank transformed V/K factors" do
    s = 8
    block_size = 4
    rng = Random.new(0x10B10C_u64)
    inputs = Array.new(block_size) do
      DeltaInputs.new(
        Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
        Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
        Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
        0.88 + 0.10 * rng.next_float,
        rng.next_float
      )
    end

    dense_block = inputs.map { |inp| affine_for(inp) }.reduce { |acc, tr| compose(acc, tr) }
    compact_b = dense_low_rank_b_for_block(inputs)

    max_abs_delta(dense_block.b, compact_b).should be < 1.0e-10
  end
end
