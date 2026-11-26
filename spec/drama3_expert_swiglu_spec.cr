require "spec"
require "../src/ml/audio/drama3/resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private alias Drama3ExpertSwigluPair = Tuple(
    ML::Audio::Drama3::Resident::Buffer,
    ML::Audio::Drama3::Resident::Buffer,
  )

  private def drama3_expert_swiglu_with_device(&block : -> T) : T forall T
    keys = ["DRAMA3_BACKEND", "DRAMA3_DENSE_REDUCTION", "DRAMA3_COGNIGRAPH_MAX_INFLIGHT",
            "DRAMA3_METAL_TIMELINE", "DRAMA3_METAL_RESOURCES", "DRAMA3_EXPERT_THREADS",
            "DRAMA3_EXPERT_PACKED_ROWS", "DRAMA3_EXPERT_DOWN_SUM_FUSION",
            "DRAMA3_EXPERT_SWIGLU_FUSION", "DRAMA3_PROFILE"]
    previous = keys.to_h { |key| {key, ENV[key]?} }
    begin
      ENV["DRAMA3_BACKEND"] = "metal"
      ENV["DRAMA3_DENSE_REDUCTION"] = "tree"
      ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "0"
      ENV["DRAMA3_METAL_TIMELINE"] = "0"
      ENV["DRAMA3_METAL_RESOURCES"] = "0"
      ENV["DRAMA3_EXPERT_THREADS"] = "64"
      ENV["DRAMA3_EXPERT_PACKED_ROWS"] = "0"
      ENV["DRAMA3_EXPERT_DOWN_SUM_FUSION"] = "0"
      ENV.delete("DRAMA3_EXPERT_SWIGLU_FUSION")
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

  private def drama3_expert_swiglu_f16_bits(value : Float32) : UInt16
    bits = value.unsafe_as(UInt32)
    sign = (bits >> 16) & 0x8000
    return sign.to_u16 if value == 0_f32
    exponent = ((bits >> 23) & 0xff).to_i32 - 127 + 15
    raise ArgumentError.new("SwiGLU affine F16 fixture must be a normal value") unless exponent > 0 && exponent < 31
    fraction = (bits & 0x7fffff) >> 13
    (sign | (exponent.to_u32 << 10) | fraction).to_u16
  end

  private def drama3_expert_swiglu_write(io : IO, dtype : String, value : Float32) : Nil
    case dtype
    when "F32"
      io.write_bytes(value, IO::ByteFormat::LittleEndian)
    when "BF16"
      io.write_bytes((value.unsafe_as(UInt32) >> 16).to_u16, IO::ByteFormat::LittleEndian)
    when "F16"
      io.write_bytes(drama3_expert_swiglu_f16_bits(value), IO::ByteFormat::LittleEndian)
    else
      raise ArgumentError.new("unsupported SwiGLU fixture dtype #{dtype}")
    end
  end

  private def drama3_expert_swiglu_affine_fixture(rows : Int32, groups : Int32, bits : Int32,
                                                  scale_dtype : String, bias_dtype : String,
                                                  projection : Int32, unit_weights : Bool = false,
                                                  experts : Int32 = 5)
    group_size = 64
    cols = groups * group_size
    per_word = 32 // bits
    word_count = experts * rows * cols // per_word
    affine_count = experts * rows * groups
    weight_offset = (24 + projection * 8).to_i64
    weight_bytes = word_count.to_i64 * 4_i64
    scale_bytes = affine_count.to_i64 * (scale_dtype == "F32" ? 4_i64 : 2_i64)
    bias_bytes = affine_count.to_i64 * (bias_dtype == "F32" ? 4_i64 : 2_i64)
    scale_offset = weight_offset + weight_bytes + 1_i64
    bias_offset = scale_offset + scale_bytes + 2_i64
    path = File.tempname("drama3-expert-swiglu", ".bin")

    io = IO::Memory.new
    io.write(Bytes.new(weight_offset.to_i, 0_u8))
    word_count.times do |index|
      word = 0_u32
      per_word.times do |lane|
        quantized = unit_weights ? (1 << bits) - 1 : (index * 13 + lane * 7 + index // 11) % (1 << bits)
        word |= quantized.to_u32 << (lane * bits)
      end
      io.write_bytes(word, IO::ByteFormat::LittleEndian)
    end
    io.write(Bytes.new(1, 0xa5_u8))
    affine_count.times do |index|
      value = if unit_weights
                1_f32 / ((1 << bits) - 1).to_f32
              else
                [0.25_f32, 0.5_f32, 0.75_f32, 1_f32][(index + projection) % 4]
              end
      drama3_expert_swiglu_write(io, scale_dtype, value)
    end
    io.write(Bytes.new(2, 0x5a_u8))
    affine_count.times do |index|
      value = unit_weights ? 0_f32 : [-0.125_f32, -0.0625_f32, 0_f32, 0.0625_f32, 0.125_f32][(index * 3 + projection) % 5]
      drama3_expert_swiglu_write(io, bias_dtype, value)
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

  private def drama3_expert_swiglu_dense_fixture(rows : Int32, cols : Int32, dtype : String,
                                                 projection : Int32, values : Array(Float32),
                                                 offset_adjustment : Int32 = 0, experts : Int32 = 1)
    raise ArgumentError.new("SwiGLU dense fixture shape") unless values.size == rows * cols * experts
    offset = (24 + projection * 8 + offset_adjustment).to_i64
    width = dtype == "F32" ? 4_i64 : 2_i64
    path = File.tempname("drama3-dense-swiglu", ".bin")
    io = IO::Memory.new
    io.write(Bytes.new(offset.to_i, 0_u8))
    values.each { |value| drama3_expert_swiglu_write(io, dtype, value) }
    File.write(path, io.to_slice)
    owner = ML::Audio::Drama3::MappedFile.new(path)
    shape = experts == 1 ? [rows, cols] : [experts, rows, cols]
    tensor = ML::Audio::Drama3::Tensor.new(shape, dtype,
      owner.slice(offset, values.size.to_i64 * width), owner)
    {path: path, owner: owner, matrix: ML::Audio::Drama3::Matrix.new(tensor), offset: offset}
  end

  private DRAMA3_EXPERT_SWIGLU_RECEIPT_SOURCE = ML::Audio::Drama3::Resident::SOURCE + <<-METAL
    kernel void dr3_expert_swiglu_receipt(
        device const float* expected [[buffer(0)]], device const float* actual [[buffer(1)]],
        device atomic_uint* receipt [[buffer(2)]], constant ulong* params [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
        if (ulong(index) >= params[0]) return;
        float a = expected[index], b = actual[index];
        uint base = uint(params[1]);
        bool a_nan = isnan(a), b_nan = isnan(b);
        bool a_inf = isinf(a), b_inf = isinf(b);
        if (!a_nan && !b_nan && as_type<uint>(a) != as_type<uint>(b))
            atomic_fetch_add_explicit(&receipt[base], 1u, memory_order_relaxed);
        if (isfinite(a)) atomic_fetch_add_explicit(&receipt[base + 1], 1u, memory_order_relaxed);
        if (isfinite(b)) atomic_fetch_add_explicit(&receipt[base + 2], 1u, memory_order_relaxed);
        if (a_nan) atomic_fetch_add_explicit(&receipt[base + 3], 1u, memory_order_relaxed);
        if (b_nan) atomic_fetch_add_explicit(&receipt[base + 4], 1u, memory_order_relaxed);
        if (a_inf) atomic_fetch_add_explicit(&receipt[base + 5], 1u, memory_order_relaxed);
        if (b_inf) atomic_fetch_add_explicit(&receipt[base + 6], 1u, memory_order_relaxed);
        uint a_class = a_nan ? 1u : (a_inf ? 2u : 0u);
        uint b_class = b_nan ? 1u : (b_inf ? 2u : 0u);
        if (a_class != b_class) atomic_fetch_add_explicit(&receipt[base + 7], 1u, memory_order_relaxed);
    }
    METAL

  private def drama3_expert_swiglu_receipt_pair(context : ML::Audio::Drama3::Resident::Context,
                                                expected : ML::Audio::Drama3::Resident::Buffer,
                                                actual : ML::Audio::Drama3::Resident::Buffer,
                                                receipt : ML::Audio::Drama3::Resident::Buffer,
                                                base : UInt64) : Nil
    expected.count.should eq(actual.count)
    context.dispatch("dr3_expert_swiglu_receipt", DRAMA3_EXPERT_SWIGLU_RECEIPT_SOURCE,
      [expected, actual, receipt], [expected.count.to_u64, base], expected.count)
  end

  private def drama3_expert_swiglu_receipt_accepts?(counts : Array(UInt32), base : Int32) : Bool
    counts[base] == 0_u32 &&
      counts[base + 1] == counts[base + 2] &&
      counts[base + 3] == counts[base + 4] &&
      counts[base + 5] == counts[base + 6] &&
      counts[base + 7] == 0_u32
  end

  private def drama3_expert_swiglu_compare(context : ML::Audio::Drama3::Resident::Context,
                                           pairs : Array(Drama3ExpertSwigluPair),
                                           before_graph : Hash(String, Int64),
                                           before_readbacks : Hash(String, Int64)) : Nil
    receipt = context.upload(Array(Float32).new((pairs.size + 1) * 8, 0_f32))
    pairs.each_with_index do |pair, index|
      drama3_expert_swiglu_receipt_pair(context, pair[0], pair[1], receipt, (index * 8).to_u64)
    end

    # Seed one ULP and finite/NaN defects so an all-clear receipt is diagnostic.
    one = 1_f32
    one_ulp_up = (one.unsafe_as(UInt32) + 1_u32).unsafe_as(Float32)
    negative_expected = context.upload([one, Float32::NAN])
    negative_actual = context.upload([one_ulp_up, 0_f32])
    control_base = pairs.size * 8
    drama3_expert_swiglu_receipt_pair(context, negative_expected, negative_actual,
      receipt, control_base.to_u64)

    graph = context.graph_stats
    graph["submitted"].should eq(before_graph["submitted"])
    graph["completion_waits"].should eq(before_graph["completion_waits"])
    context.readbacks.should eq(before_readbacks)
    counts = context.read(receipt, "terminal").map(&.unsafe_as(UInt32))
    pairs.each_index do |index|
      drama3_expert_swiglu_receipt_accepts?(counts, index * 8).should be_true
    end
    drama3_expert_swiglu_receipt_accepts?(counts, control_base).should be_false
    context.readbacks.keys.should eq(["terminal"])
    context.readbacks["terminal"].should eq(receipt.count.to_i64 * 4_i64)
    context.graph_stats["submitted"].should eq(1_i64)
    context.graph_stats["completed"].should eq(1_i64)
    context.graph_stats["completion_waits"].should eq(1_i64)
    context.graph_stats["pending"].should eq(0_i64)
  end

  private def drama3_expert_swiglu_pending_unchanged(context : ML::Audio::Drama3::Resident::Context,
                                                     graph : Hash(String, Int64),
                                                     readbacks : Hash(String, Int64)) : Nil
    current = context.graph_stats
    current["submitted"].should eq(graph["submitted"])
    current["completion_waits"].should eq(graph["completion_waits"])
    current["pending"].should eq(graph["pending"])
    context.readbacks.should eq(readbacks)
  end

  private def drama3_expert_swiglu_upload_view(context : ML::Audio::Drama3::Resident::Context,
                                               values : Array(Float32)) : ML::Audio::Drama3::Resident::Buffer
    storage = context.upload([123.0_f32] + values + [-321.0_f32])
    storage.slice(1, values.size)
  end

  private def drama3_expert_swiglu_input_values(cols : Int32) : Array(Float32)
    Array(Float32).new(cols) { |index| (((index * 17 + index // 5) % 37) - 18).to_f32 / 32_f32 }
  end

  private def drama3_expert_swiglu_compare_expert_case(
    gate : ML::Audio::Drama3::Matrix, up : ML::Audio::Drama3::Matrix,
    inputs : Array(Float32), ids : Array(Float32), selector : String? = "1",
    expected_fused : Bool = true, capture : Bool = true, threads : String = "64",
    mutate_selector_after_context : String? = nil,
  ) : Nil
    ENV["DRAMA3_EXPERT_THREADS"] = threads
    if value = selector
      ENV["DRAMA3_EXPERT_SWIGLU_FUSION"] = value
    else
      ENV.delete("DRAMA3_EXPERT_SWIGLU_FUSION")
    end
    context = ML::Audio::Drama3::Resident::Context.new
    begin
      x = drama3_expert_swiglu_upload_view(context, inputs)
      selected_ids = drama3_expert_swiglu_upload_view(context, ids)
      mutate_selector_after_context.try { |value| ENV["DRAMA3_EXPERT_SWIGLU_FUSION"] = value }
      before_graph = context.graph_stats
      before_readbacks = context.readbacks.dup

      expected_gate = context.experts(gate, x, selected_ids)
      expected_up = context.experts(up, x, selected_ids)
      expected_output = context.mul(context.silu(expected_gate), expected_up)
      drama3_expert_swiglu_pending_unchanged(context, before_graph, before_readbacks)

      before_fused = context.expert_swiglu_fused_dispatches
      actual = context.experts_swiglu(gate, up, x, selected_ids,
        capture_projections: capture)
      actual[:output].count.should eq(gate.rows * ids.size)
      actual[:gate].try(&.count).should eq(capture ? expected_gate.count : nil)
      actual[:up].try(&.count).should eq(capture ? expected_up.count : nil)
      context.expert_swiglu_fused_dispatches.should eq(before_fused + (expected_fused ? 1_i64 : 0_i64))
      drama3_expert_swiglu_pending_unchanged(context, before_graph, before_readbacks)

      pairs = [] of Drama3ExpertSwigluPair
      pairs << {expected_output, actual[:output]}
      if capture
        pairs << {expected_gate, actual[:gate].not_nil!}
        pairs << {expected_up, actual[:up].not_nil!}
      end
      drama3_expert_swiglu_compare(context, pairs, before_graph, before_readbacks)
    ensure
      context.abort
    end
  end

  private def drama3_expert_swiglu_compare_dense_case(
    gate : ML::Audio::Drama3::Matrix, up : ML::Audio::Drama3::Matrix,
    inputs : Array(Float32), reduction : String? = nil, expected_fused : Bool = true,
    capture : Bool = true,
  ) : Nil
    ENV["DRAMA3_EXPERT_SWIGLU_FUSION"] = "1"
    context = ML::Audio::Drama3::Resident::Context.new
    begin
      x = drama3_expert_swiglu_upload_view(context, inputs)
      before_graph = context.graph_stats
      before_readbacks = context.readbacks.dup

      expected_gate = context.matvec(gate, x, dense_reduction: reduction)
      expected_up = context.matvec(up, x, dense_reduction: reduction)
      expected_output = context.mul(context.silu(expected_gate), expected_up)
      drama3_expert_swiglu_pending_unchanged(context, before_graph, before_readbacks)

      before_fused = context.dense_swiglu_fused_dispatches
      actual = context.matvec_swiglu(gate, up, x, dense_reduction: reduction,
        capture_projections: capture)
      actual[:output].count.should eq(gate.rows)
      actual[:gate].try(&.count).should eq(capture ? expected_gate.count : nil)
      actual[:up].try(&.count).should eq(capture ? expected_up.count : nil)
      context.dense_swiglu_fused_dispatches.should eq(before_fused + (expected_fused ? 1_i64 : 0_i64))
      drama3_expert_swiglu_pending_unchanged(context, before_graph, before_readbacks)

      pairs = [] of Drama3ExpertSwigluPair
      pairs << {expected_output, actual[:output]}
      if capture
        pairs << {expected_gate, actual[:gate].not_nil!}
        pairs << {expected_up, actual[:up].not_nil!}
      end
      drama3_expert_swiglu_compare(context, pairs, before_graph, before_readbacks)
    ensure
      context.abort
    end
  end

  describe "DRAMA-3 resident expert SwiGLU" do
    it "matches separate Q4/Q8 expert, stable SiLU, and multiply calls" do
      drama3_expert_swiglu_with_device do
        gate_fixture = drama3_expert_swiglu_affine_fixture(3, 3, 4, "F32", "BF16", 0)
        up_fixture = drama3_expert_swiglu_affine_fixture(3, 3, 8, "F16", "F32", 1)
        begin
          gate_fixture[:weight_offset].should_not eq(up_fixture[:weight_offset])
          gate_fixture[:scale_offset].odd?.should be_true
          gate_fixture[:bias_offset].odd?.should be_true
          up_fixture[:scale_offset].odd?.should be_true
          drama3_expert_swiglu_compare_expert_case(gate_fixture[:matrix], up_fixture[:matrix],
            drama3_expert_swiglu_input_values(3 * 64), [4_f32, 1_f32, 4_f32, 0_f32], capture: true)
        ensure
          File.delete(gate_fixture[:path]) if File.exists?(gate_fixture[:path])
          File.delete(up_fixture[:path]) if File.exists?(up_fixture[:path])
        end

        gate_fixture = drama3_expert_swiglu_affine_fixture(3, 32, 8, "BF16", "F16", 0)
        up_fixture = drama3_expert_swiglu_affine_fixture(3, 32, 4, "F32", "BF16", 1)
        begin
          drama3_expert_swiglu_compare_expert_case(gate_fixture[:matrix], up_fixture[:matrix],
            drama3_expert_swiglu_input_values(32 * 64), [3_f32, 0_f32, 2_f32], capture: false)
        ensure
          File.delete(gate_fixture[:path]) if File.exists?(gate_fixture[:path])
          File.delete(up_fixture[:path]) if File.exists?(up_fixture[:path])
        end
      end
    end

    it "preserves exact group cancellation and invalid, duplicate, and shuffled expert IDs" do
      drama3_expert_swiglu_with_device do
        gate_fixture = drama3_expert_swiglu_affine_fixture(3, 3, 4, "F32", "F32", 0, unit_weights: true)
        up_fixture = drama3_expert_swiglu_affine_fixture(3, 3, 4, "F32", "F32", 1, unit_weights: true)
        begin
          cancellation = Array(Float32).new(3 * 64, 0_f32)
          cancellation[0] = 2048_f32
          cancellation[64] = 0.25_f32
          cancellation[128] = -2048_f32
          drama3_expert_swiglu_compare_expert_case(gate_fixture[:matrix], up_fixture[:matrix],
            cancellation, [4_f32, 4_f32, 0_f32, 2_f32], capture: true)

          invalid_ids = [-0.0_f32, -1_f32, Float32::NAN, 1.5_f32, 5_f32, Float32::INFINITY]
          drama3_expert_swiglu_compare_expert_case(gate_fixture[:matrix], up_fixture[:matrix],
            drama3_expert_swiglu_input_values(3 * 64), invalid_ids, capture: true)
        ensure
          File.delete(gate_fixture[:path]) if File.exists?(gate_fixture[:path])
          File.delete(up_fixture[:path]) if File.exists?(up_fixture[:path])
        end
      end
    end

    it "uses the default-off frozen selector and guards unsupported expert geometry" do
      drama3_expert_swiglu_with_device do
        small_gate = drama3_expert_swiglu_affine_fixture(3, 1, 4, "F32", "F32", 0)
        small_up = drama3_expert_swiglu_affine_fixture(3, 1, 8, "F16", "BF16", 1)
        begin
          drama3_expert_swiglu_compare_expert_case(small_gate[:matrix], small_up[:matrix],
            drama3_expert_swiglu_input_values(64), [1_f32, 2_f32],
            selector: nil, expected_fused: false, mutate_selector_after_context: "1")
          drama3_expert_swiglu_compare_expert_case(small_gate[:matrix], small_up[:matrix],
            drama3_expert_swiglu_input_values(64), [1_f32, 2_f32],
            selector: "1", expected_fused: true, mutate_selector_after_context: "0")
          drama3_expert_swiglu_compare_expert_case(small_gate[:matrix], small_up[:matrix],
            drama3_expert_swiglu_input_values(64), [1_f32, 2_f32],
            selector: "0", expected_fused: false)
          drama3_expert_swiglu_compare_expert_case(small_gate[:matrix], small_up[:matrix],
            drama3_expert_swiglu_input_values(64), [1_f32, 2_f32],
            selector: "1", expected_fused: false, threads: "32")
          drama3_expert_swiglu_compare_expert_case(small_gate[:matrix], small_up[:matrix],
            drama3_expert_swiglu_input_values(64), Array(Float32).new(33) { |index| (index % 5).to_f32 },
            expected_fused: false)
        ensure
          File.delete(small_gate[:path]) if File.exists?(small_gate[:path])
          File.delete(small_up[:path]) if File.exists?(small_up[:path])
        end

        wide_gate = drama3_expert_swiglu_affine_fixture(3, 33, 4, "F32", "F32", 0)
        wide_up = drama3_expert_swiglu_affine_fixture(3, 33, 8, "F16", "BF16", 1)
        begin
          drama3_expert_swiglu_compare_expert_case(wide_gate[:matrix], wide_up[:matrix],
            drama3_expert_swiglu_input_values(33 * 64), [4_f32, 1_f32], expected_fused: false)
        ensure
          File.delete(wide_gate[:path]) if File.exists?(wide_gate[:path])
          File.delete(wide_up[:path]) if File.exists?(wide_up[:path])
        end

        ["", "true", "2"].each do |invalid|
          ENV["DRAMA3_EXPERT_SWIGLU_FUSION"] = invalid
          expect_raises(ArgumentError, /DRAMA3_EXPERT_SWIGLU_FUSION/) do
            ML::Audio::Drama3::Resident::Context.new
          end
        end
      end
    end

    it "fuses aligned BF16 tree matvecs, including BF16 specials, and falls back otherwise" do
      drama3_expert_swiglu_with_device do
        cols = 64
        values0 = Array(Float32).new(3 * cols) { |index| (((index * 7 + 3) % 19) - 9).to_f32 / 8_f32 }
        values1 = Array(Float32).new(3 * cols) { |index| (((index * 11 + 5) % 23) - 11).to_f32 / 8_f32 }
        gate_fixture = drama3_expert_swiglu_dense_fixture(3, cols, "BF16", 0, values0)
        up_fixture = drama3_expert_swiglu_dense_fixture(3, cols, "BF16", 1, values1)
        begin
          gate_fixture[:offset].even?.should be_true
          up_fixture[:offset].even?.should be_true
          drama3_expert_swiglu_compare_dense_case(gate_fixture[:matrix], up_fixture[:matrix],
            Array(Float32).new(cols, 0.5_f32), capture: true)
        ensure
          File.delete(gate_fixture[:path]) if File.exists?(gate_fixture[:path])
          File.delete(up_fixture[:path]) if File.exists?(up_fixture[:path])
        end

        special0 = Array(Float32).new(5 * cols, 0_f32)
        special1 = Array(Float32).new(5 * cols, 1_f32)
        negative_zero = 0x80000000_u32.unsafe_as(Float32)
        cols.times { |index| special0[index] = negative_zero }
        special0[cols] = 0x00010000_u32.unsafe_as(Float32) # BF16 subnormal
        special0[2 * cols] = Float32::NAN
        special0[3 * cols] = Float32::INFINITY
        special0[4 * cols] = -Float32::INFINITY
        gate_fixture = drama3_expert_swiglu_dense_fixture(5, cols, "BF16", 0, special0)
        up_fixture = drama3_expert_swiglu_dense_fixture(5, cols, "BF16", 1, special1)
        begin
          drama3_expert_swiglu_compare_dense_case(gate_fixture[:matrix], up_fixture[:matrix],
            Array(Float32).new(cols, 1_f32), capture: true)
          # The aligned BF16/tree route is ineligible for simd reduction. Keep
          # these mapped fixtures live while checking that ordered fallback.
          drama3_expert_swiglu_compare_dense_case(gate_fixture[:matrix], up_fixture[:matrix],
            Array(Float32).new(cols, 0.25_f32), reduction: "simd", expected_fused: false)
        ensure
          File.delete(gate_fixture[:path]) if File.exists?(gate_fixture[:path])
          File.delete(up_fixture[:path]) if File.exists?(up_fixture[:path])
        end

        f32_gate = drama3_expert_swiglu_dense_fixture(3, cols, "F32", 0, values0)
        f32_up = drama3_expert_swiglu_dense_fixture(3, cols, "F32", 1, values1)
        begin
          drama3_expert_swiglu_compare_dense_case(f32_gate[:matrix], f32_up[:matrix],
            Array(Float32).new(cols, 0.25_f32), expected_fused: false, capture: false)
        ensure
          File.delete(f32_gate[:path]) if File.exists?(f32_gate[:path])
          File.delete(f32_up[:path]) if File.exists?(f32_up[:path])
        end

        misaligned_gate = drama3_expert_swiglu_dense_fixture(3, cols, "BF16", 0, values0, offset_adjustment: 1)
        aligned_up = drama3_expert_swiglu_dense_fixture(3, cols, "BF16", 1, values1)
        begin
          misaligned_gate[:offset].odd?.should be_true
          drama3_expert_swiglu_compare_dense_case(misaligned_gate[:matrix], aligned_up[:matrix],
            Array(Float32).new(cols, 0.5_f32), expected_fused: false)
        ensure
          File.delete(misaligned_gate[:path]) if File.exists?(misaligned_gate[:path])
          File.delete(aligned_up[:path]) if File.exists?(aligned_up[:path])
        end

        ranked_gate = drama3_expert_swiglu_dense_fixture(3, cols, "BF16", 0, values0 + values0, experts: 2)
        ranked_up = drama3_expert_swiglu_dense_fixture(3, cols, "BF16", 1, values1 + values1, experts: 2)
        begin
          drama3_expert_swiglu_compare_dense_case(ranked_gate[:matrix], ranked_up[:matrix],
            Array(Float32).new(cols, 0.5_f32), expected_fused: false)
        ensure
          File.delete(ranked_gate[:path]) if File.exists?(ranked_gate[:path])
          File.delete(ranked_up[:path]) if File.exists?(ranked_up[:path])
        end
      end
    end

    it "rejects foreign and malformed resident inputs before allocation or encoding" do
      drama3_expert_swiglu_with_device do
        ENV["DRAMA3_EXPERT_SWIGLU_FUSION"] = "1"
        fixture = drama3_expert_swiglu_affine_fixture(3, 2, 4, "F32", "BF16", 0)
        mismatched = drama3_expert_swiglu_affine_fixture(4, 2, 4, "F32", "BF16", 1)
        different_expert_count = drama3_expert_swiglu_affine_fixture(3, 2, 4, "F32", "BF16", 1, experts: 4)
        dense_gate = drama3_expert_swiglu_dense_fixture(3, 64, "BF16", 0,
          drama3_expert_swiglu_input_values(3 * 64))
        dense_up = drama3_expert_swiglu_dense_fixture(4, 64, "BF16", 1,
          drama3_expert_swiglu_input_values(4 * 64))
        context = ML::Audio::Drama3::Resident::Context.new
        foreign = ML::Audio::Drama3::Resident::Context.new
        begin
          ids = context.upload([0_f32, 1_f32])
          x = context.upload(Array(Float32).new(fixture[:matrix].cols, 0.125_f32))
          short_x = context.upload(Array(Float32).new(fixture[:matrix].cols - 1, 0.125_f32))
          foreign_x = foreign.upload(Array(Float32).new(fixture[:matrix].cols, 0.125_f32))
          foreign_ids = foreign.upload([0_f32, 1_f32])
          dense_x = context.upload(Array(Float32).new(64, 0.125_f32))
          before_allocations = context.allocation_stats[:allocations]
          before_graph = context.graph_stats.dup
          before_readbacks = context.readbacks.dup
          before_fused = context.expert_swiglu_fused_dispatches
          before_pipelines = ML::Metal::PipelineCache.entry_count

          expect_raises(ArgumentError, /different resident context/) do
            context.experts_swiglu(fixture[:matrix], fixture[:matrix], foreign_x, ids)
          end
          expect_raises(ArgumentError, /different resident context/) do
            context.experts_swiglu(fixture[:matrix], fixture[:matrix], x, foreign_ids)
          end
          expect_raises(ArgumentError, /input width/) do
            context.experts_swiglu(fixture[:matrix], fixture[:matrix], short_x, ids)
          end
          expect_raises(ArgumentError) do
            context.experts_swiglu(fixture[:matrix], mismatched[:matrix], x, ids)
          end
          expect_raises(ArgumentError) do
            context.experts_swiglu(fixture[:matrix], different_expert_count[:matrix], x, ids)
          end
          expect_raises(ArgumentError) do
            context.matvec_swiglu(dense_gate[:matrix], dense_up[:matrix], dense_x)
          end

          context.allocation_stats[:allocations].should eq(before_allocations)
          context.graph_stats.should eq(before_graph)
          context.readbacks.should eq(before_readbacks)
          context.expert_swiglu_fused_dispatches.should eq(before_fused)
          ML::Metal::PipelineCache.entry_count.should eq(before_pipelines)
        ensure
          context.abort
          foreign.abort
          File.delete(fixture[:path]) if File.exists?(fixture[:path])
          File.delete(mismatched[:path]) if File.exists?(mismatched[:path])
          File.delete(different_expert_count[:path]) if File.exists?(different_expert_count[:path])
          File.delete(dense_gate[:path]) if File.exists?(dense_gate[:path])
          File.delete(dense_up[:path]) if File.exists?(dense_up[:path])
        end
      end
    end
  end
{% end %}
