require "spec"
require "../src/ml/audio/drama3/resident_batch"
require "../src/ml/audio/drama3/backbone_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private def drama3_batch_values(count : Int32, seed : Int32 = 1) : Array(Float32)
    Array(Float32).new(count) do |index|
      (((index * 17 + seed * 23 + index // 7 * 11) % 61) - 30).to_f32 / 29_f32
    end
  end

  private def drama3_batch_bf16_matrix(&)
    path = File.tempname("drama3-batch-bf16", ".bin")
    io = IO::Memory.new
    io.write(Bytes.new(22, 0_u8))
    values = drama3_batch_values(28, 9)
    values.each do |value|
      io.write_bytes((value.unsafe_as(UInt32) >> 16).to_u16, IO::ByteFormat::LittleEndian)
    end
    File.write(path, io.to_slice)
    owner = ML::Audio::Drama3::MappedFile.new(path)
    tensor = ML::Audio::Drama3::Tensor.new([4, 7], "BF16", owner.slice(22_i64, 56_i64), owner)
    begin
      yield ML::Audio::Drama3::Matrix.new(tensor)
    ensure
      ML::Audio::Drama3::MetalBackend.reset!
      File.delete(path) if File.exists?(path)
    end
  end

  private def drama3_batch_packed_matrix(&)
    path = File.tempname("drama3-batch-packed", ".bin")
    experts, output_rows, cols, bits = 3, 3, 64, 4
    words_per_row = cols // (32 // bits)
    words = Array(UInt32).new(experts * output_rows * words_per_row) do |index|
      word = 0_u32
      (32 // bits).times do |lane|
        quantized = ((index * 7 + lane * 5 + index // 3) % 16).to_u32
        word |= quantized << (lane * bits)
      end
      word
    end
    io = IO::Memory.new
    io.write(Bytes.new(22, 0_u8))
    words.each { |word| io.write_bytes(word, IO::ByteFormat::LittleEndian) }
    File.write(path, io.to_slice)
    owner = ML::Audio::Drama3::MappedFile.new(path)
    packed = ML::Audio::Drama3::Tensor.new([experts, output_rows, words_per_row], "U32",
      owner.slice(22_i64, (words.size * 4).to_i64), owner)
    scale_values = Array(Float32).new(experts * output_rows) { |index| (index % 5 + 1).to_f32 / 13_f32 }
    bias_values = Array(Float32).new(experts * output_rows) { |index| (index % 7 - 3).to_f32 / 17_f32 }
    scales = ML::Audio::Drama3::Tensor.new([experts, output_rows, 1], scale_values)
    biases = ML::Audio::Drama3::Tensor.new([experts, output_rows, 1], bias_values)
    matrix = ML::Audio::Drama3::Matrix.new(packed, scales, biases, bits, 64, 1)
    begin
      yield matrix
    ensure
      ML::Audio::Drama3::MetalBackend.reset!
      File.delete(path) if File.exists?(path)
    end
  end

  private def drama3_batch_check_dense(context : ML::Audio::Drama3::Resident::Context,
                                       matrix : ML::Audio::Drama3::Matrix,
                                       rows : Int32, values : Array(Float32), reduction : String) : Nil
    parent = context.upload([101_f32] + values + [102_f32])
    input = parent.slice(1, values.size)
    batched = context.matvec_rows(matrix, input, rows, dense_reduction: reduction)
    expected = [] of UInt32
    rows.times do |row|
      single = context.matvec(matrix, input.slice(row * matrix.cols, matrix.cols), dense_reduction: reduction)
      context.read(single, "terminal").each { |value| expected << value.unsafe_as(UInt32) }
    end
    context.read(batched, "terminal").map(&.unsafe_as(UInt32)).should eq(expected)
  end

  describe "DRAMA-3 resident batch primitives" do
    it "matches dense F32 and BF16 rows bitwise for each reduction at graph depths zero and two" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        rows, cols, output_rows = 3, 7, 4
        expert_values = drama3_batch_values(2 * output_rows * cols, 4)
        matrix = ML::Audio::Drama3::Matrix.new(
          ML::Audio::Drama3::Tensor.new([2, output_rows, cols], expert_values), nil, nil, 4, 64, 1)
        input_values = drama3_batch_values(rows * cols, 6)

        ["0", "2"].each do |depth|
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth
          ["tree", "simd", "simd32"].each do |reduction|
            drama3_batch_check_dense(ML::Audio::Drama3::Resident::Context.new,
              matrix, rows, input_values, reduction)
            drama3_batch_bf16_matrix do |bf16|
              drama3_batch_check_dense(ML::Audio::Drama3::Resident::Context.new,
                bf16, rows, input_values, reduction)
            end
          end
        end
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
        if previous_depth
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous_depth
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
      end
    end

    it "keeps packed projections and flattened per-token expert ranks on the existing affine path" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        drama3_batch_packed_matrix do |matrix|
          rows, top_k = 3, 2
          shared_values = drama3_batch_values(rows * matrix.cols, 14)
          ids_values = [2_f32, 0_f32, 1_f32, 2_f32, 1_f32, 0_f32]
          context = ML::Audio::Drama3::Resident::Context.new
          input_parent = context.upload([77_f32] + shared_values + [88_f32])
          input = input_parent.slice(1, shared_values.size)
          ids_parent = context.upload([-1_f32] + ids_values + [-1_f32])
          ids = ids_parent.slice(1, ids_values.size)

          projection = context.matvec_rows(matrix, input, rows)
          projection_expected = [] of UInt32
          rows.times do |row|
            single = context.matvec(matrix, input.slice(row * matrix.cols, matrix.cols))
            context.read(single, "terminal").each { |value| projection_expected << value.unsafe_as(UInt32) }
          end
          context.read(projection, "terminal").map(&.unsafe_as(UInt32)).should eq(projection_expected)

          batched = context.experts_rows(matrix, input, ids, rows, top_k)
          expected = [] of UInt32
          ids_values.each_with_index do |_, flat_rank|
            token = flat_rank // top_k
            single = context.expert(matrix, input.slice(token * matrix.cols, matrix.cols), ids.slice(flat_rank, 1))
            context.read(single, "terminal").each { |value| expected << value.unsafe_as(UInt32) }
          end
          context.read(batched, "terminal").map(&.unsafe_as(UInt32)).should eq(expected)

          per_rank_values = drama3_batch_values(ids.count * matrix.cols, 21)
          per_rank_parent = context.upload([66_f32] + per_rank_values + [99_f32])
          per_rank = per_rank_parent.slice(1, per_rank_values.size)
          per_rank_batch = context.experts_rows(matrix, per_rank, ids, rows, top_k, per_rank_input: true)
          per_rank_expected = [] of UInt32
          ids_values.each_with_index do |_, flat_rank|
            single = context.expert(matrix,
              per_rank.slice(flat_rank * matrix.cols, matrix.cols), ids.slice(flat_rank, 1))
            context.read(single, "terminal").each { |value| per_rank_expected << value.unsafe_as(UInt32) }
          end
          context.read(per_rank_batch, "terminal").map(&.unsafe_as(UInt32)).should eq(per_rank_expected)
        end
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
      end
    end

    it "preserves route tie order, softmax weights, expert-rank sum order, and offset-aware shared gates" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        context = ML::Audio::Drama3::Resident::Context.new
        rows, experts, top_k = 3, 4, 3
        score_values = [
          0_f32, 0_f32, 0_f32, 0_f32,
          0.5_f32, -1_f32, 0.5_f32, 0.2_f32,
          -2_f32, -1_f32, -1_f32, 1.5_f32,
        ]
        score_parent = context.upload([123_f32] + score_values + [456_f32])
        scores = score_parent.slice(1, score_values.size)
        legacy_routes = Hash(Bool, Array(Array(Float32))).new

        [true, false].each do |normalize|
          ids, weights = context.route_rows(scores, rows, experts, top_k, normalize)
          expected_ids = [] of UInt32
          expected_weights = [] of UInt32
          routes = [] of Array(Float32)
          rows.times do |row|
            row_scores = scores.slice(row * experts, experts)
            legacy = context.alloc(top_k * 2)
            context.dispatch("dr3_route", ML::Audio::Drama3::BackboneResident::SOURCE,
              [row_scores, legacy], [experts.to_u64, top_k.to_u64, normalize ? 1_u64 : 0_u64], 1)
            values = context.read(legacy, "terminal")
            routes << values
            top_k.times do |rank|
              expected_ids << values[rank].unsafe_as(UInt32)
              expected_weights << values[top_k + rank].unsafe_as(UInt32)
            end
          end
          context.read(ids, "terminal").map(&.unsafe_as(UInt32)).should eq(expected_ids)
          context.read(weights, "terminal").map(&.unsafe_as(UInt32)).should eq(expected_weights)
          legacy_routes[normalize] = routes
        end

        route_ids, route_weights = context.route_rows(scores, rows, experts, top_k, true)
        hidden = 5
        down_values = drama3_batch_values(rows * top_k * hidden, 31)
        down_parent = context.upload([55_f32] + down_values + [66_f32])
        down = down_parent.slice(1, down_values.size)
        summed = context.expert_sum_rows(down, route_weights, rows, hidden, top_k)
        sum_expected = [] of UInt32
        rows.times do |row|
          routing = context.upload(legacy_routes[true][row])
          legacy_sum = context.alloc(hidden)
          context.dispatch("dr3_expert_sum", ML::Audio::Drama3::BackboneResident::SOURCE,
            [down.slice(row * top_k * hidden, top_k * hidden), routing, legacy_sum],
            [hidden.to_u64, top_k.to_u64], hidden)
          context.read(legacy_sum, "terminal").each { |value| sum_expected << value.unsafe_as(UInt32) }
        end
        context.read(summed, "terminal").map(&.unsafe_as(UInt32)).should eq(sum_expected)
        context.read(route_ids, "terminal").size.should eq(rows * top_k)

        gate_rows, gate_hidden = 3, 5
        initial_values = drama3_batch_values(gate_rows * gate_hidden, 41)
        shared_values = drama3_batch_values(gate_rows * gate_hidden, 43)
        gate_values = [-0.75_f32, 0_f32, 1.25_f32]
        result_parent = context.upload([7_f32] + initial_values + [8_f32])
        shared_parent = context.upload([9_f32] + shared_values + [10_f32])
        gate_parent = context.upload([11_f32] + gate_values + [12_f32])
        result, shared, gate = result_parent.slice(1, initial_values.size), shared_parent.slice(1, shared_values.size), gate_parent.slice(1, gate_values.size)
        context.weighted_add_rows!(result, shared, gate, gate_rows, gate_hidden)
        weighted_expected = [] of UInt32
        gate_rows.times do |row|
          single_result = context.upload(initial_values[row * gate_hidden, gate_hidden])
          context.dispatch("dr3_weighted_add", ML::Audio::Drama3::BackboneResident::SOURCE,
            [single_result, shared.slice(row * gate_hidden, gate_hidden), gate.slice(row, 1)],
            [gate_hidden.to_u64, 1_u64], gate_hidden)
          context.read(single_result, "terminal").each { |value| weighted_expected << value.unsafe_as(UInt32) }
        end
        context.read(result, "terminal").map(&.unsafe_as(UInt32)).should eq(weighted_expected)
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
      end
    end

    it "rejects foreign, invalid-shape, and overflowing row inputs before submission" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        context = ML::Audio::Drama3::Resident::Context.new
        foreign_context = ML::Audio::Drama3::Resident::Context.new
        matrix = ML::Audio::Drama3::Matrix.new(2, 3, drama3_batch_values(6))
        input = context.upload([1_f32, 2_f32, 3_f32])
        foreign = foreign_context.upload([1_f32, 2_f32, 3_f32])
        allocations = context.allocation_stats[:allocations]
        expect_raises(ArgumentError, /different resident context/) { context.matvec_rows(matrix, foreign, 1) }
        expect_raises(ArgumentError, /row count/) { context.matvec_rows(matrix, input, 0) }
        expect_raises(ArgumentError, /rows width/) { context.matvec_rows(matrix, input, 2) }
        expect_raises(ArgumentError, /Int32 element capacity/) { context.matvec_rows(matrix, input, Int32::MAX) }
        expect_raises(ArgumentError, /route dimensions/) { context.route_rows(input, 1, 2, 3, true) }
        context.allocation_stats[:allocations].should eq(allocations)
        context.graph_stats["submitted"].should eq(0_i64)

        drama3_batch_packed_matrix do |packed|
          ids = context.upload([0_f32, 1_f32])
          short = context.upload([1_f32])
          packed_allocations = context.allocation_stats[:allocations]
          expect_raises(ArgumentError, /input width/) { context.experts_rows(packed, short, ids, 1, 2) }
          expect_raises(ArgumentError, /ID shape/) { context.experts_rows(packed, input, ids, 2, 2) }
          context.allocation_stats[:allocations].should eq(packed_allocations)
        end
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
      end
    end
  end
{% end %}
