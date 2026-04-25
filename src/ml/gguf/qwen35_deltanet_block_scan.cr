module ML
  module GGUF
    module Qwen35DeltaNetBlockScan
      alias Matrix = Array(Array(Float64))

      struct DeltaInputs
        getter k : Array(Float64)
        getter v : Array(Float64)
        getter q : Array(Float64)
        getter g : Float64
        getter beta : Float64

        def initialize(@k : Array(Float64), @v : Array(Float64), @q : Array(Float64),
                       @g : Float64, @beta : Float64)
        end
      end

      struct AffineDelta
        getter a : Matrix
        getter b : Matrix

        def initialize(@a : Matrix, @b : Matrix)
        end
      end

      def self.zeros(rows : Int32, cols : Int32) : Matrix
        Array.new(rows) { Array.new(cols, 0.0) }
      end

      def self.identity(n : Int32) : Matrix
        m = zeros(n, n)
        n.times { |i| m[i][i] = 1.0 }
        m
      end

      def self.matmul(a : Matrix, b : Matrix) : Matrix
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

      def self.matadd(a : Matrix, b : Matrix) : Matrix
        rows = a.size
        cols = a[0].size
        out = zeros(rows, cols)
        rows.times do |i|
          cols.times { |j| out[i][j] = a[i][j] + b[i][j] }
        end
        out
      end

      def self.dot(a : Array(Float64), b : Array(Float64)) : Float64
        acc = 0.0
        a.size.times { |i| acc += a[i] * b[i] }
        acc
      end

      def self.max_abs_delta(a : Matrix, b : Matrix) : Float64
        max = 0.0
        a.size.times do |i|
          a[i].size.times do |j|
            d = (a[i][j] - b[i][j]).abs
            max = d if d > max
          end
        end
        max
      end

      def self.serial_delta_step(state : Matrix, inp : DeltaInputs, scale : Float64) : Tuple(Matrix, Array(Float64))
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

      def self.affine_for(inp : DeltaInputs) : AffineDelta
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

      def self.compose(first : AffineDelta, second : AffineDelta) : AffineDelta
        # Apply `first`, then `second`:
        #   (S*A1 + B1)*A2 + B2 = S*(A1*A2) + (B1*A2 + B2)
        AffineDelta.new(
          matmul(first.a, second.a),
          matadd(matmul(first.b, second.a), second.b)
        )
      end

      def self.apply_affine(state : Matrix, tr : AffineDelta) : Matrix
        matadd(matmul(state, tr.a), tr.b)
      end

      def self.vec_matmul(v : Array(Float64), m : Matrix) : Array(Float64)
        cols = m[0].size
        out = Array.new(cols, 0.0)
        cols.times do |j|
          acc = 0.0
          v.size.times { |i| acc += v[i] * m[i][j] }
          out[j] = acc
        end
        out
      end

      def self.dense_low_rank_b_for_block(inputs : Array(DeltaInputs)) : Matrix
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

      def self.replay_block(state : Matrix, inputs : Array(DeltaInputs), scale : Float64) : Tuple(Matrix, Array(Array(Float64)))
        ys = [] of Array(Float64)
        cur = state
        inputs.each do |inp|
          cur, y = serial_delta_step(cur, inp, scale)
          ys << y
        end
        {cur, ys}
      end

      def self.replay_final_state(state : Matrix, inputs : Array(DeltaInputs), scale : Float64) : Matrix
        replay_block(state, inputs, scale)[0]
      end

      def self.compose_all(inputs : Array(DeltaInputs)) : AffineDelta
        inputs.map { |inp| affine_for(inp) }.reduce { |acc, tr| compose(acc, tr) }
      end
    end
  end
end
