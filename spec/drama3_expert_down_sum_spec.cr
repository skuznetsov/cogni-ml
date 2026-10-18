require "spec"
require "../src/ml/audio/drama3/resident"
require "../src/ml/audio/drama3/backbone_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private def drama3_expert_down_sum_with_device(&block : -> T) : T forall T
    keys = ["DRAMA3_BACKEND", "DRAMA3_COGNIGRAPH_MAX_INFLIGHT", "DRAMA3_METAL_TIMELINE",
            "DRAMA3_EXPERT_THREADS", "DRAMA3_EXPERT_PACKED_ROWS", "DRAMA3_EXPERT_DOWN_SUM_FUSION",
            "DRAMA3_PROFILE"]
    previous = keys.to_h { |key| {key, ENV[key]?} }
    begin
      ENV["DRAMA3_BACKEND"] = "metal"
      ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "0"
      ENV["DRAMA3_METAL_TIMELINE"] = "0"
      ENV["DRAMA3_EXPERT_THREADS"] = "64"
      ENV["DRAMA3_EXPERT_PACKED_ROWS"] = "0"
      ENV.delete("DRAMA3_EXPERT_DOWN_SUM_FUSION")
      ENV.delete("DRAMA3_PROFILE")
      ML::Audio::Drama3::Profile.configure_from_env!
      device = ML::Metal::Device.instance
      pending!("Metal is unavailable or lacks unified memory") unless device.available? && device.has_unified_memory?
      yield
    ensure
      ML::Audio::Drama3::MetalBackend.reset!
      keys.each do |key|
        if value = previous[key]
          ENV[key] = value.not_nil!
        else
          ENV.delete(key)
        end
      end
      ML::Audio::Drama3::Profile.configure_from_env!
    end
  end

  private def drama3_expert_down_sum_f16_bits(value : Float32) : UInt16
    return 0_u16 if value == 0_f32
    bits = value.unsafe_as(UInt32)
    exponent = ((bits >> 23) & 0xff).to_i32 - 127 + 15
    raise ArgumentError.new("Q4 fixture F16 value must be a normal number") unless exponent > 0 && exponent < 31
    sign = (bits >> 16) & 0x8000
    fraction = (bits & 0x7fffff) >> 13
    (sign | (exponent.to_u32 << 10) | fraction).to_u16
  end

  private def drama3_expert_down_sum_write_float(io : IO, dtype : String, value : Float32) : Nil
    case dtype
    when "F32"
      io.write_bytes(value, IO::ByteFormat::LittleEndian)
    when "BF16"
      io.write_bytes((value.unsafe_as(UInt32) >> 16).to_u16, IO::ByteFormat::LittleEndian)
    when "F16"
      io.write_bytes(drama3_expert_down_sum_f16_bits(value), IO::ByteFormat::LittleEndian)
    else
      raise ArgumentError.new("unsupported expert down-sum fixture dtype #{dtype}")
    end
  end

  private def drama3_expert_down_sum_fixture(rows : Int32, groups : Int32, bits : Int32,
                                             scale_dtype : String = "F32", bias_dtype : String = "F32",
                                             experts : Int32 = 5)
    group_size = 64
    cols = groups * group_size
    per_word = 32 // bits
    word_count = experts * rows * cols // per_word
    affine_count = experts * rows * groups
    weight_offset = 22_i64
    weight_bytes = word_count.to_i64 * 4_i64
    scale_bytes = affine_count.to_i64 * (scale_dtype == "F32" ? 4_i64 : 2_i64)
    bias_bytes = affine_count.to_i64 * (bias_dtype == "F32" ? 4_i64 : 2_i64)
    scale_offset = weight_offset + weight_bytes + 1_i64
    bias_offset = scale_offset + scale_bytes + 2_i64
    path = File.tempname("drama3-expert-down-sum", ".bin")

    io = IO::Memory.new
    io.write(Bytes.new(weight_offset.to_i, 0_u8))
    word_count.times do |index|
      word = 0_u32
      per_word.times do |lane|
        quantized = (index * 13 + lane * 7 + index // 11) % (1 << bits)
        word |= quantized.to_u32 << (lane * bits)
      end
      io.write_bytes(word, IO::ByteFormat::LittleEndian)
    end
    io.write(Bytes.new(1, 0xa5_u8))
    affine_count.times do |index|
      drama3_expert_down_sum_write_float(io, scale_dtype, [0.25_f32, 0.5_f32, 0.75_f32, 1_f32][index % 4])
    end
    io.write(Bytes.new(2, 0x5a_u8))
    affine_count.times do |index|
      drama3_expert_down_sum_write_float(io, bias_dtype, [-0.125_f32, -0.0625_f32, 0_f32, 0.0625_f32, 0.125_f32][index % 5])
    end
    File.write(path, io.to_slice)

    owner = ML::Audio::Drama3::MappedFile.new(path)
    packed = ML::Audio::Drama3::Tensor.new([experts, rows, cols // per_word], "U32",
      owner.slice(weight_offset, weight_bytes), owner)
    scales = ML::Audio::Drama3::Tensor.new([experts, rows, groups], scale_dtype,
      owner.slice(scale_offset, scale_bytes), owner)
    biases = ML::Audio::Drama3::Tensor.new([experts, rows, groups], bias_dtype,
      owner.slice(bias_offset, bias_bytes), owner)
    matrix = ML::Audio::Drama3::Matrix.new(packed, scales, biases, bits, group_size)
    {path: path, owner: owner, matrix: matrix, weight_offset: weight_offset,
     scale_offset: scale_offset, bias_offset: bias_offset}
  end

  private DRAMA3_EXPERT_DOWN_SUM_RECEIPT_SOURCE = ML::Audio::Drama3::Resident::SOURCE + <<-METAL
    inline uint dr3_down_sum_status(float value) {
        if (isnan(value)) return 2u;
        if (isinf(value)) return 1u;
        return 0u;
    }

    kernel void dr3_down_sum_compare(
        device const float* expected [[buffer(0)]], device const float* actual [[buffer(1)]],
        device atomic_uint* receipt [[buffer(2)]], constant ulong* params [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
        if (ulong(index) >= params[0]) return;
        float a = expected[index], b = actual[index];
        uint base = uint(params[1]);
        if (!isnan(a) && !isnan(b) && as_type<uint>(a) != as_type<uint>(b))
            atomic_fetch_add_explicit(&receipt[base], 1u, memory_order_relaxed);
        if (isfinite(a)) atomic_fetch_add_explicit(&receipt[base + 1], 1u, memory_order_relaxed);
        if (isfinite(b)) atomic_fetch_add_explicit(&receipt[base + 2], 1u, memory_order_relaxed);
        if (isnan(a)) atomic_fetch_add_explicit(&receipt[base + 3], 1u, memory_order_relaxed);
        if (isnan(b)) atomic_fetch_add_explicit(&receipt[base + 4], 1u, memory_order_relaxed);
        if (isinf(a)) atomic_fetch_add_explicit(&receipt[base + 5], 1u, memory_order_relaxed);
        if (isinf(b)) atomic_fetch_add_explicit(&receipt[base + 6], 1u, memory_order_relaxed);
        if (dr3_down_sum_status(a) != dr3_down_sum_status(b))
            atomic_fetch_add_explicit(&receipt[base + 7], 1u, memory_order_relaxed);
    }
    METAL

  private def drama3_expert_down_sum_receipt_compare(context : ML::Audio::Drama3::Resident::Context,
                                                     expected : ML::Audio::Drama3::Resident::Buffer,
                                                     actual : ML::Audio::Drama3::Resident::Buffer,
                                                     receipt : ML::Audio::Drama3::Resident::Buffer,
                                                     base : UInt64) : Nil
    expected.count.should eq(actual.count)
    encoder = context.encoder
    encoder.set_pipeline(ML::Metal::PipelineCache.get_or_compile(
      "dr3_down_sum_compare", DRAMA3_EXPERT_DOWN_SUM_RECEIPT_SOURCE,
    ))
    [expected, actual, receipt].each_with_index do |buffer, index|
      encoder.set_buffer(buffer.buffer, index, offset: buffer.offset)
    end
    params = [actual.count.to_u64, base]
    encoder.set_bytes(params.to_unsafe.as(Void*), 16, 3)
    encoder.dispatch_1d(actual.count, 256)
  end

  private def drama3_expert_down_sum_receipt_accepts?(counts : Array(UInt32), base : Int32,
                                                      count : Int32, nan_count : UInt32 = 0_u32,
                                                      inf_count : UInt32 = 0_u32) : Bool
    finite_count = count.to_u32 - nan_count - inf_count
    counts[base] == 0_u32 &&
      counts[base + 1] == finite_count && counts[base + 2] == finite_count &&
      counts[base + 3] == nan_count && counts[base + 4] == nan_count &&
      counts[base + 5] == inf_count && counts[base + 6] == inf_count &&
      counts[base + 7] == 0_u32
  end

  private def drama3_expert_down_sum_gpu_receipt(context : ML::Audio::Drama3::Resident::Context,
                                                 expected : ML::Audio::Drama3::Resident::Buffer,
                                                 actual : ML::Audio::Drama3::Resident::Buffer,
                                                 expected_down : ML::Audio::Drama3::Resident::Buffer?,
                                                 actual_down : ML::Audio::Drama3::Resident::Buffer?,
                                                 output_nan_count : UInt32 = 0_u32,
                                                 down_nan_count : UInt32 = 0_u32) : Nil
    receipt = context.upload(Array(Float32).new(24, 0_f32))
    drama3_expert_down_sum_receipt_compare(context, expected, actual, receipt, 0_u64)
    if expected_down && actual_down
      drama3_expert_down_sum_receipt_compare(context, expected_down.not_nil!, actual_down.not_nil!, receipt, 8_u64)
    else
      expected_down.should be_nil
      actual_down.should be_nil
    end

    # A one-bit difference and a finite-vs-NaN pair must fail the same
    # acceptance rule used for the implementation comparison.
    one = 1_f32
    one_ulp_up = (one.unsafe_as(UInt32) + 1_u32).unsafe_as(Float32)
    negative_expected = context.upload([one, Float32::NAN])
    negative_actual = context.upload([one_ulp_up, 0_f32])
    drama3_expert_down_sum_receipt_compare(context, negative_expected, negative_actual, receipt, 16_u64)

    counts = context.read(receipt, "terminal").map(&.unsafe_as(UInt32))
    drama3_expert_down_sum_receipt_accepts?(counts, 0, actual.count, output_nan_count).should be_true
    if expected_down && actual_down
      drama3_expert_down_sum_receipt_accepts?(counts, 8, actual_down.not_nil!.count, down_nan_count).should be_true
    end
    counts[16, 8].should eq([1_u32, 1_u32, 2_u32, 1_u32, 0_u32, 0_u32, 0_u32, 1_u32])
    drama3_expert_down_sum_receipt_accepts?(counts, 16, 2).should be_false
    context.readbacks.keys.should eq(["terminal"])
    context.readbacks["terminal"].should eq(96_i64)
    context.graph_stats["submitted"].should eq(1_i64)
    context.graph_stats["completed"].should eq(1_i64)
    context.graph_stats["completion_waits"].should eq(1_i64)
    context.graph_stats["pending"].should eq(0_i64)
  end

  private def drama3_expert_down_sum_compare_case(
    fixture : NamedTuple(path: String,
      owner: ML::Audio::Drama3::MappedFile,
      matrix: ML::Audio::Drama3::Matrix,
      weight_offset: Int64, scale_offset: Int64, bias_offset: Int64),
    inputs : Array(Float32), id_values : Array(Float32),
    weight_values : Array(Float32), requested_threads : String = "64",
    reduction : String = "native", capture_down : Bool = false,
    fusion_enabled : Bool = true, expect_fused : Bool = true,
    output_nan_count : UInt32 = 0_u32,
    down_nan_count : UInt32 = 0_u32,
  ) : Nil
    ENV["DRAMA3_EXPERT_THREADS"] = requested_threads
    ENV["DRAMA3_EXPERT_DOWN_SUM_FUSION"] = fusion_enabled ? "1" : "0"
    context = ML::Audio::Drama3::Resident::Context.new
    begin
      context.expert_down_sum_fusion_enabled.should eq(fusion_enabled)
      matrix = fixture[:matrix]
      x = context.upload(inputs)
      ids = context.upload(id_values)
      weights = context.upload(weight_values)

      expected_down = context.experts(matrix, x, ids, per_rank_input: true, reduction: reduction)
      routing = context.upload(id_values + weight_values)
      expected_output = context.alloc(matrix.rows)
      context.dispatch("dr3_expert_sum", ML::Audio::Drama3::BackboneResident::SOURCE,
        [expected_down, routing, expected_output], [matrix.rows.to_u64, id_values.size.to_u64], matrix.rows)

      before_allocations = context.allocation_stats[:allocations]
      before_fused_dispatches = context.expert_down_sum_fused_dispatches
      actual = context.experts_down_sum(matrix, x, ids, weights, reduction: reduction,
        capture_down: capture_down)
      actual[:output].count.should eq(matrix.rows)
      actual[:down].try(&.count).should eq(capture_down ? matrix.rows * id_values.size : nil)
      allocated = context.allocation_stats[:allocations] - before_allocations
      allocated.should eq(expect_fused ? (capture_down ? 2_i64 : 1_i64) : 2_i64)
      context.expert_down_sum_fused_dispatches.should eq(before_fused_dispatches + (expect_fused ? 1_i64 : 0_i64))

      drama3_expert_down_sum_gpu_receipt(context, expected_output, actual[:output],
        capture_down ? expected_down : nil, actual[:down], output_nan_count, down_nan_count)
    ensure
      context.abort
    end
  end

  describe "DRAMA-3 expert down-sum fusion" do
    it "strictly validates and freezes the opt-in selector without GPU work" do
      drama3_expert_down_sum_with_device do
        ENV.delete("DRAMA3_EXPERT_DOWN_SUM_FUSION")
        default_context = ML::Audio::Drama3::Resident::Context.new
        begin
          default_context.expert_down_sum_fusion_enabled.should be_false
          ENV["DRAMA3_EXPERT_DOWN_SUM_FUSION"] = "1"
          default_context.expert_down_sum_fusion_enabled.should be_false
          default_context.allocation_stats[:allocations].should eq(0_i64)
          default_context.graph_stats["submitted"].should eq(0_i64)
          default_context.readbacks.should be_empty

          enabled_context = ML::Audio::Drama3::Resident::Context.new
          begin
            enabled_context.expert_down_sum_fusion_enabled.should be_true
            ENV["DRAMA3_EXPERT_DOWN_SUM_FUSION"] = "0"
            enabled_context.expert_down_sum_fusion_enabled.should be_true
            enabled_context.allocation_stats[:allocations].should eq(0_i64)
            enabled_context.graph_stats["submitted"].should eq(0_i64)
            enabled_context.readbacks.should be_empty
          ensure
            enabled_context.abort
          end

          ["", "true", "2"].each do |invalid|
            ENV["DRAMA3_EXPERT_DOWN_SUM_FUSION"] = invalid
            expect_raises(ArgumentError, /DRAMA3_EXPERT_DOWN_SUM_FUSION/) do
              ML::Audio::Drama3::Resident::Context.new
            end
          end
        ensure
          default_context.abort
        end
      end
    end

    it "matches the ordered experts plus legacy weighted-sum reference at fused geometry boundaries" do
      drama3_expert_down_sum_with_device do
        cases = [
          {4, 32, 8, "F32", "BF16"}, # rank*group lanes exactly 256
          {8, 8, 3, "BF16", "F16"},
          {4, 1, 1, "F16", "F32"},
        ]
        cases.each_with_index do |case_data, case_index|
          bits, groups, ranks, scale_dtype, bias_dtype = case_data
          fixture = drama3_expert_down_sum_fixture(3, groups, bits, scale_dtype, bias_dtype)
          begin
            ML::Audio::Drama3::MetalBackend.tensor_buffer(fixture[:matrix].tensor, true).offset.should eq(fixture[:weight_offset])
            ML::Audio::Drama3::MetalBackend.tensor_buffer(fixture[:matrix].scales.not_nil!, true).offset.should eq(fixture[:scale_offset])
            ML::Audio::Drama3::MetalBackend.tensor_buffer(fixture[:matrix].biases.not_nil!, true).offset.should eq(fixture[:bias_offset])
            fixture[:scale_offset].odd?.should be_true
            fixture[:bias_offset].odd?.should be_true

            ids = Array(Float32).new(ranks) { |rank| [4, 1, 3, 0, 2][(rank * 3 + 1) % 5].to_f32 }
            cols = groups * 64
            inputs = Array(Float32).new(cols * ranks) do |index|
              ((index * 17 + index // 5) % 37 - 18).to_f32 / 64_f32
            end
            weights = Array(Float32).new(ranks) do |rank|
              [-0.75_f32, 0.5_f32, 1_f32, -0.25_f32][rank % 4]
            end
            drama3_expert_down_sum_compare_case(fixture, inputs, ids, weights,
              capture_down: case_index.odd?, fusion_enabled: true, expect_fused: true)
          ensure
            File.delete(fixture[:path]) if File.exists?(fixture[:path])
          end
        end
      end
    end

    it "takes the ordered fallback for unsupported geometry, MLX reduction, and a disabled selector" do
      drama3_expert_down_sum_with_device do
        cases = [
          {3, 2, "64", 4, "native", true},   # group count does not divide 64
          {33, 1, "64", 4, "native", false}, # above the fused maximum
          {4, 9, "64", 8, "native", true},   # too many ranks
          {4, 2, "32", 4, "native", false},  # effective expert thread count is 32
          {8, 2, "64", 4, "mlx_q4", false},  # explicit MLX reduction always falls back
        ]
        cases.each do |case_data|
          groups, ranks, requested_threads, bits, reduction, capture_down = case_data
          rows = reduction == "mlx_q4" ? 8 : 3
          fixture = drama3_expert_down_sum_fixture(rows, groups, bits)
          begin
            ids = Array(Float32).new(ranks) { |rank| [3, 0, 4, 1, 2][rank % 5].to_f32 }
            cols = groups * 64
            inputs = Array(Float32).new(cols * ranks) do |index|
              ((index * 11 + index // 7) % 29 - 14).to_f32 / 32_f32
            end
            weights = Array(Float32).new(ranks) { |rank| (rank - 2).to_f32 / 4_f32 }
            drama3_expert_down_sum_compare_case(fixture, inputs, ids, weights,
              requested_threads: requested_threads, reduction: reduction, capture_down: capture_down,
              fusion_enabled: true, expect_fused: false)
          ensure
            File.delete(fixture[:path]) if File.exists?(fixture[:path])
          end
        end

        fixture = drama3_expert_down_sum_fixture(3, 1, 4)
        begin
          drama3_expert_down_sum_compare_case(fixture, Array(Float32).new(64) { |index| index.to_f32 / 64_f32 },
            [1_f32], [0.5_f32], fusion_enabled: false, expect_fused: false)
        ensure
          File.delete(fixture[:path]) if File.exists?(fixture[:path])
        end
      end
    end

    it "preserves cancellation and signed-zero order and classifies invalid expert IDs as NaN" do
      drama3_expert_down_sum_with_device do
        path = File.tempname("drama3-expert-down-sum-unit", ".bin")
        io = IO::Memory.new
        8.times { io.write_bytes(0_u32, IO::ByteFormat::LittleEndian) }
        File.write(path, io.to_slice)
        owner = ML::Audio::Drama3::MappedFile.new(path)
        packed = ML::Audio::Drama3::Tensor.new([1, 1, 8], "U32", owner.slice(0_i64, 32_i64), owner)
        scales = ML::Audio::Drama3::Tensor.new([1, 1, 1], [0_f32])
        biases = ML::Audio::Drama3::Tensor.new([1, 1, 1], [1_f32])
        unit = {path: path, owner: owner,
                matrix: ML::Audio::Drama3::Matrix.new(packed, scales, biases, 4, 64),
                weight_offset: 0_i64, scale_offset: 0_i64, bias_offset: 0_i64}
        begin
          # Every down value is exactly one; the selected rank fold therefore
          # rounds 1e20 + 1 - 1e20 to +0 before the signed-zero products.
          drama3_expert_down_sum_compare_case(unit, Array(Float32).new(5 * 64, 1_f32 / 64_f32),
            Array(Float32).new(5, 0_f32), [1e20_f32, 1_f32, -1e20_f32, -0_f32, 0_f32],
            capture_down: true, fusion_enabled: true, expect_fused: true)
        ensure
          File.delete(path) if File.exists?(path)
        end

        fixture = drama3_expert_down_sum_fixture(3, 4, 8)
        begin
          ids = [0_f32, -0.0_f32, -1_f32, Float32::NAN, 2.5_f32, 5_f32, Float32::INFINITY]
          inputs = Array(Float32).new(ids.size * fixture[:matrix].cols) do |index|
            ((index * 5 + 3) % 19 - 9).to_f32 / 16_f32
          end
          weights = [0.25_f32, -0.25_f32, 0.5_f32, -0.75_f32, 1_f32, 0.125_f32, -0.125_f32]
          drama3_expert_down_sum_compare_case(fixture, inputs, ids, weights,
            capture_down: true, fusion_enabled: true, expect_fused: true,
            output_nan_count: fixture[:matrix].rows.to_u32,
            down_nan_count: (5 * fixture[:matrix].rows).to_u32)
        ensure
          File.delete(fixture[:path]) if File.exists?(fixture[:path])
        end
      end
    end

    it "rejects foreign buffers and bad input/weight shapes before allocations or submission" do
      drama3_expert_down_sum_with_device do
        ENV["DRAMA3_EXPERT_DOWN_SUM_FUSION"] = "1"
        fixture = drama3_expert_down_sum_fixture(3, 2, 4)
        context = ML::Audio::Drama3::Resident::Context.new
        foreign = ML::Audio::Drama3::Resident::Context.new
        begin
          ids = context.upload([0_f32, 1_f32])
          weights = context.upload([0.5_f32, -0.25_f32])
          x = context.upload(Array(Float32).new(2 * fixture[:matrix].cols, 0.125_f32))
          short_x = context.upload(Array(Float32).new(2 * fixture[:matrix].cols - 1, 0.125_f32))
          short_weights = context.upload([0.5_f32])
          foreign_x = foreign.upload(Array(Float32).new(2 * fixture[:matrix].cols, 0.125_f32))
          foreign_ids = foreign.upload([0_f32, 1_f32])
          foreign_weights = foreign.upload([0.5_f32, -0.25_f32])
          before_allocations = context.allocation_stats[:allocations]
          before_graph = context.graph_stats.dup
          before_readbacks = context.readbacks.dup

          expect_raises(ArgumentError) { context.experts_down_sum(fixture[:matrix], foreign_x, ids, weights) }
          expect_raises(ArgumentError) { context.experts_down_sum(fixture[:matrix], x, foreign_ids, weights) }
          expect_raises(ArgumentError) { context.experts_down_sum(fixture[:matrix], x, ids, foreign_weights) }
          expect_raises(ArgumentError, /input width/) { context.experts_down_sum(fixture[:matrix], short_x, ids, weights) }
          expect_raises(ArgumentError) { context.experts_down_sum(fixture[:matrix], x, ids, short_weights) }

          context.allocation_stats[:allocations].should eq(before_allocations)
          context.graph_stats.should eq(before_graph)
          context.readbacks.should eq(before_readbacks)
        ensure
          context.abort
          foreign.abort
          File.delete(fixture[:path]) if File.exists?(fixture[:path])
        end
      end
    end
  end
{% end %}
