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

      struct CompactDeltaSummary
        getter a : Matrix
        getter b_lefts : Array(Array(Float64))
        getter b_rights : Array(Array(Float64))

        def initialize(@a : Matrix,
                       @b_lefts : Array(Array(Float64)),
                       @b_rights : Array(Array(Float64)))
          raise ArgumentError.new("compact B factors size mismatch") unless @b_lefts.size == @b_rights.size
        end
      end

      struct CompactTransition
        getter gamma : Float64
        getter u_cols : Array(Array(Float64))
        getter v_cols : Array(Array(Float64))

        def initialize(@gamma : Float64,
                       @u_cols : Array(Array(Float64)),
                       @v_cols : Array(Array(Float64)))
          raise ArgumentError.new("compact transition factors size mismatch") unless @u_cols.size == @v_cols.size
        end
      end

      struct FullyCompactDeltaSummary
        getter transition : CompactTransition
        getter b_lefts : Array(Array(Float64))
        getter b_rights : Array(Array(Float64))

        def initialize(@transition : CompactTransition,
                       @b_lefts : Array(Array(Float64)),
                       @b_rights : Array(Array(Float64)))
          raise ArgumentError.new("compact B factors size mismatch") unless @b_lefts.size == @b_rights.size
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

      def self.vec_dot(a : Array(Float64), b : Array(Float64)) : Float64
        dot(a, b)
      end

      def self.vec_add_scaled(a : Array(Float64), b : Array(Float64), scale : Float64) : Array(Float64)
        out = Array.new(a.size, 0.0)
        a.size.times { |i| out[i] = a[i] + b[i] * scale }
        out
      end

      def self.apply_transition_to_vec(v : Array(Float64), tr : CompactTransition) : Array(Float64)
        result = v.dup
        tr.u_cols.each_with_index do |u, idx|
          coeff = vec_dot(v, u)
          right = tr.v_cols[idx]
          result = vec_add_scaled(result, right, coeff)
        end
        result.map { |x| x * tr.gamma }
      end

      def self.dense_a_from_transition(tr : CompactTransition, s : Int32) : Matrix
        a = identity(s)
        tr.u_cols.each_with_index do |u, idx|
          v = tr.v_cols[idx]
          s.times do |i|
            s.times do |j|
              a[i][j] += u[i] * v[j]
            end
          end
        end
        s.times do |i|
          s.times { |j| a[i][j] *= tr.gamma }
        end
        a
      end

      def self.row_basis_factors(m : Matrix) : Tuple(Array(Array(Float64)), Array(Array(Float64)))
        rows = m.size
        cols = m[0].size
        lefts = [] of Array(Float64)
        rights = [] of Array(Float64)

        rows.times do |i|
          next if m[i].all? { |x| x == 0.0 }

          left = Array.new(rows, 0.0)
          left[i] = 1.0
          right = Array.new(cols) { |j| m[i][j] }
          lefts << left
          rights << right
        end

        {lefts, rights}
      end

      def self.compress_transition_row_basis(tr : CompactTransition, s : Int32) : CompactTransition
        raise ArgumentError.new("cannot row-compress a zero-gamma transition") if tr.gamma == 0.0

        dense_a = dense_a_from_transition(tr, s)
        delta = zeros(s, s)
        s.times do |i|
          s.times do |j|
            delta[i][j] = dense_a[i][j] / tr.gamma
          end
          delta[i][i] -= 1.0
        end

        u_cols, v_cols = row_basis_factors(delta)
        CompactTransition.new(tr.gamma, u_cols, v_cols)
      end

      def self.compress_b_row_basis(lefts : Array(Array(Float64)),
                                    rights : Array(Array(Float64)),
                                    s : Int32) : Tuple(Array(Array(Float64)), Array(Array(Float64)))
        dense = zeros(s, s)
        lefts.each_with_index do |left, idx|
          right = rights[idx]
          s.times do |i|
            s.times do |j|
              dense[i][j] += left[i] * right[j]
            end
          end
        end

        row_basis_factors(dense)
      end

      def self.compact_transition_for_block(inputs : Array(DeltaInputs)) : CompactTransition
        raise ArgumentError.new("inputs must not be empty") if inputs.empty?

        gamma = 1.0
        u_cols = [] of Array(Float64)
        v_cols = [] of Array(Float64)

        inputs.each do |inp|
          u = inp.k.map { |x| -inp.beta * x }
          v = inp.k
          if u_cols.empty?
            new_u = u
          else
            # (I + U V^T)(I + u v^T) = I + U V^T + (u + U(V^T u)) v^T.
            new_u = u.dup
            u_cols.each_with_index do |old_u, idx|
              coeff = vec_dot(v_cols[idx], u)
              new_u = vec_add_scaled(new_u, old_u, coeff)
            end
          end
          u_cols << new_u
          v_cols << v
          gamma *= inp.g
        end

        CompactTransition.new(gamma, u_cols, v_cols)
      end

      def self.compose_transition(first : CompactTransition, second : CompactTransition) : CompactTransition
        new_u = first.u_cols.dup
        new_v = first.v_cols.dup

        second.u_cols.each_with_index do |u2, idx|
          transformed_u = u2.dup
          first.u_cols.each_with_index do |u1, j|
            coeff = vec_dot(first.v_cols[j], u2)
            transformed_u = vec_add_scaled(transformed_u, u1, coeff)
          end
          new_u << transformed_u
          new_v << second.v_cols[idx]
        end

        CompactTransition.new(first.gamma * second.gamma, new_u, new_v)
      end

      def self.dense_low_rank_b_for_block(inputs : Array(DeltaInputs)) : Matrix
        dense_b_from_compact(compact_summary_for_block(inputs))
      end

      def self.compact_summary_for_block(inputs : Array(DeltaInputs)) : CompactDeltaSummary
        raise ArgumentError.new("inputs must not be empty") if inputs.empty?

        s = inputs[0].k.size
        suffix = identity(s)
        lefts = Array(Array(Float64)).new(inputs.size) { Array.new(s, 0.0) }
        rights = Array(Array(Float64)).new(inputs.size) { Array.new(s, 0.0) }

        (inputs.size - 1).downto(0) do |i|
          inp = inputs[i]
          lefts[i] = inp.v
          rights[i] = vec_matmul(inp.k, suffix).map { |x| x * inp.beta }
          suffix = matmul(affine_for(inp).a, suffix)
        end

        CompactDeltaSummary.new(suffix, lefts, rights)
      end

      def self.fully_compact_summary_for_block(inputs : Array(DeltaInputs)) : FullyCompactDeltaSummary
        dense_b_summary = compact_summary_for_block(inputs)
        FullyCompactDeltaSummary.new(
          compact_transition_for_block(inputs),
          dense_b_summary.b_lefts,
          dense_b_summary.b_rights
        )
      end

      def self.dense_b_from_compact(summary : CompactDeltaSummary) : Matrix
        s = summary.a.size
        b = zeros(s, s)
        summary.b_lefts.each_with_index do |left, idx|
          right = summary.b_rights[idx]
          s.times do |d2|
            s.times do |d1|
              b[d2][d1] += left[d2] * right[d1]
            end
          end
        end
        b
      end

      def self.apply_compact(state : Matrix, summary : CompactDeltaSummary) : Matrix
        out = matmul(state, summary.a)
        s = out.size
        summary.b_lefts.each_with_index do |left, idx|
          right = summary.b_rights[idx]
          s.times do |d2|
            s.times do |d1|
              out[d2][d1] += left[d2] * right[d1]
            end
          end
        end
        out
      end

      def self.apply_fully_compact(state : Matrix, summary : FullyCompactDeltaSummary) : Matrix
        s = state.size
        out = zeros(s, s)
        state.each_with_index do |row, i|
          out[i] = apply_transition_to_vec(row, summary.transition)
        end
        summary.b_lefts.each_with_index do |left, idx|
          right = summary.b_rights[idx]
          s.times do |d2|
            s.times do |d1|
              out[d2][d1] += left[d2] * right[d1]
            end
          end
        end
        out
      end

      def self.compose_compact(first : CompactDeltaSummary, second : CompactDeltaSummary) : CompactDeltaSummary
        transformed_rights = first.b_rights.map { |right| vec_matmul(right, second.a) }
        CompactDeltaSummary.new(
          matmul(first.a, second.a),
          first.b_lefts + second.b_lefts,
          transformed_rights + second.b_rights
        )
      end

      def self.compose_fully_compact(first : FullyCompactDeltaSummary,
                                     second : FullyCompactDeltaSummary) : FullyCompactDeltaSummary
        transformed_rights = first.b_rights.map { |right| apply_transition_to_vec(right, second.transition) }
        FullyCompactDeltaSummary.new(
          compose_transition(first.transition, second.transition),
          first.b_lefts + second.b_lefts,
          transformed_rights + second.b_rights
        )
      end

      def self.compress_fully_compact_row_basis(summary : FullyCompactDeltaSummary, s : Int32) : FullyCompactDeltaSummary
        transition = compress_transition_row_basis(summary.transition, s)
        b_lefts, b_rights = compress_b_row_basis(summary.b_lefts, summary.b_rights, s)
        FullyCompactDeltaSummary.new(transition, b_lefts, b_rights)
      end

      def self.compose_fully_compact_compressed(first : FullyCompactDeltaSummary,
                                                second : FullyCompactDeltaSummary,
                                                s : Int32) : FullyCompactDeltaSummary
        compress_fully_compact_row_basis(compose_fully_compact(first, second), s)
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
