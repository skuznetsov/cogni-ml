require "spec"
require "../src/ml/audio/drama3/resident"
require "../src/ml/audio/drama3/backbone_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private alias Drama3DenseFusionGuard = NamedTuple(
    storage: ML::Audio::Drama3::Resident::Buffer,
    values: ML::Audio::Drama3::Resident::Buffer,
  )

  private DRAMA3_DENSE_FUSION_RECEIPT_SOURCE = ML::Audio::Drama3::Resident::SOURCE + <<-METAL
    inline void dr3_dense_fusion_record(device atomic_uint* receipt,
                                         uint bit_slot, uint finite_slot,
                                         float reference, float actual) {
        if (as_type<uint>(reference) != as_type<uint>(actual)) {
            atomic_fetch_add_explicit(&receipt[bit_slot], 1u, memory_order_relaxed);
        }
        if (!isfinite(reference) || !isfinite(actual)) {
            atomic_fetch_add_explicit(&receipt[finite_slot], 1u, memory_order_relaxed);
        }
    }

    kernel void dr3_dense_fusion_compare(
        device const float* reference [[buffer(0)]], device const float* actual [[buffer(1)]],
        device const float* reference_storage [[buffer(2)]], device const float* actual_storage [[buffer(3)]],
        device atomic_uint* receipt [[buffer(4)]], constant ulong* params [[buffer(5)]],
        uint index [[thread_position_in_grid]]) {
        ulong count = params[0];
        if (ulong(index) >= count) return;
        dr3_dense_fusion_record(receipt, 0, 1, reference[index], actual[index]);
        if (index == 0) {
            uint guard_bits = uint(params[1]);
            if (as_type<uint>(reference_storage[0]) != guard_bits) {
                atomic_fetch_add_explicit(&receipt[4], 1u, memory_order_relaxed);
            }
            if (as_type<uint>(reference_storage[count + 1]) != guard_bits) {
                atomic_fetch_add_explicit(&receipt[4], 1u, memory_order_relaxed);
            }
            if (as_type<uint>(actual_storage[0]) != guard_bits) {
                atomic_fetch_add_explicit(&receipt[4], 1u, memory_order_relaxed);
            }
            if (as_type<uint>(actual_storage[count + 1]) != guard_bits) {
                atomic_fetch_add_explicit(&receipt[4], 1u, memory_order_relaxed);
            }
        }
    }

    kernel void dr3_dense_fusion_negative_control(
        device atomic_uint* receipt [[buffer(0)]], uint index [[thread_position_in_grid]]) {
        if (index != 0) return;
        float one = 1.0f;
        float one_ulp_up = as_type<float>(as_type<uint>(one) + 1u);
        dr3_dense_fusion_record(receipt, 2, 3, one, one_ulp_up);
        dr3_dense_fusion_record(receipt, 2, 3, as_type<float>(0x7fc00000u), 0.0f);
    }

    kernel void dr3_dense_fusion_cancellation_witness(
        device const float* reference [[buffer(0)]], device const float* actual [[buffer(1)]],
        device atomic_uint* receipt [[buffer(2)]], constant ulong* params [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
        if (index != 0) return;
        uint expected = uint(params[0]);
        if (as_type<uint>(reference[0]) != expected || as_type<uint>(actual[0]) != expected) {
            atomic_fetch_add_explicit(&receipt[5], 1u, memory_order_relaxed);
        }
    }
    METAL

  private def drama3_dense_fusion_with_device(&block : -> T) : T forall T
    keys = ["DRAMA3_BACKEND", "DRAMA3_DENSE_REDUCTION", "DRAMA3_BACKBONE_DENSE_REDUCTION",
            "DRAMA3_BACKBONE_DENSE_FUSION", "DRAMA3_COGNIGRAPH_MAX_INFLIGHT",
            "DRAMA3_METAL_TIMELINE", "DRAMA3_METAL_RESOURCES", "DRAMA3_DELTA_TRANSPOSE",
            "DRAMA3_MOE_BATCH", "DRAMA3_EXPERT_DOWN_REDUCTION", "DRAMA3_EXPERT_THREADS",
            "DRAMA3_EXPERT_PACKED_ROWS"]
    previous = keys.to_h { |key| {key, ENV[key]?} }
    begin
      ENV["DRAMA3_BACKEND"] = "metal"
      ENV["DRAMA3_DENSE_REDUCTION"] = "tree"
      ENV["DRAMA3_BACKBONE_DENSE_REDUCTION"] = "tree"
      ENV["DRAMA3_BACKBONE_DENSE_FUSION"] = "0"
      ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "0"
      ENV["DRAMA3_METAL_TIMELINE"] = "0"
      ENV["DRAMA3_METAL_RESOURCES"] = "0"
      ENV["DRAMA3_DELTA_TRANSPOSE"] = "1"
      ENV["DRAMA3_MOE_BATCH"] = "1"
      ENV["DRAMA3_EXPERT_DOWN_REDUCTION"] = "native"
      ENV["DRAMA3_EXPERT_THREADS"] = "64"
      ENV["DRAMA3_EXPERT_PACKED_ROWS"] = "0"
      device = ML::Metal::Device.instance
      pending!("Metal is unavailable or lacks unified memory") unless device.available? && device.has_unified_memory?
      ML::Audio::Drama3::MetalBackend.enabled?.should be_true
      yield
    ensure
      keys.each do |key|
        if value = previous[key]
          ENV[key] = value.not_nil!
        else
          ENV.delete(key)
        end
      end
    end
  end

  private def drama3_dense_fusion_values(projection : Int32, rows : Int32, cols : Int32) : Array(Float32)
    Array(Float32).new(rows * cols) do |index|
      row, col = index // cols, index % cols
      if projection == 0 && row == 1 && cols >= 33
        case col
        when  0 then 2048_f32
        when  1 then 0.25_f32
        when 32 then -2048_f32
        else         0_f32
        end
      elsif row == rows - 1 && col == cols - 1
        -0.75_f32
      else
        (((index * 13 + projection * 7 + 5) % 17) - 8).to_f32 / 8_f32
      end
    end
  end

  private def drama3_dense_fusion_input(cols : Int32) : Array(Float32)
    values = Array(Float32).new(cols) { |index| (((index * 7 + 3) % 19) - 9).to_f32 / 8_f32 }
    values[0] = 8192_f32
    values[1] = 1_f32
    values[32] = 8192_f32 if cols >= 33
    values
  end

  private def drama3_dense_fusion_f16_bits(value : Float32) : UInt16
    return 0_u16 if value == 0_f32
    bits = value.unsafe_as(UInt32)
    exponent = ((bits >> 23) & 0xff).to_i32 - 127 + 15
    raise ArgumentError.new("DRAMA-3 dense fusion F16 fixture must be a normal value") unless exponent > 0 && exponent < 31
    sign = (bits >> 16) & 0x8000
    fraction = (bits & 0x7fffff) >> 13
    (sign | (exponent.to_u32 << 10) | fraction).to_u16
  end

  private def drama3_dense_fusion_write_value(io : IO, dtype : String, value : Float32) : Nil
    case dtype
    when "F32"
      io.write_bytes(value, IO::ByteFormat::LittleEndian)
    when "BF16"
      io.write_bytes((value.unsafe_as(UInt32) >> 16).to_u16, IO::ByteFormat::LittleEndian)
    when "F16"
      io.write_bytes(drama3_dense_fusion_f16_bits(value), IO::ByteFormat::LittleEndian)
    else
      raise ArgumentError.new("unsupported DRAMA-3 dense fusion dtype #{dtype}")
    end
  end

  private def drama3_dense_fusion_dense_fixture(projection : Int32, rows : Int32, cols : Int32,
                                                dtype : String, offset : Int32)
    values = drama3_dense_fusion_values(projection, rows, cols)
    path = File.tempname("drama3-dense-fusion", ".bin")
    io = IO::Memory.new
    io.write(Bytes.new(offset, 0_u8))
    values.each { |value| drama3_dense_fusion_write_value(io, dtype, value) }
    File.write(path, io.to_slice)
    owner = ML::Audio::Drama3::MappedFile.new(path)
    byte_width = dtype == "F32" ? 4 : 2
    tensor = ML::Audio::Drama3::Tensor.new(
      [rows, cols], dtype, owner.slice(offset.to_i64, (values.size * byte_width).to_i64), owner,
    )
    {path: path, owner: owner, tensor: tensor, matrix: ML::Audio::Drama3::Matrix.new(tensor)}
  end

  private def drama3_dense_fusion_q4_fixture(rows : Int32 = 7, cols : Int32 = 64, offset : Int32 = 24)
    words = rows * (cols // 8)
    path = File.tempname("drama3-dense-fusion-q4", ".bin")
    io = IO::Memory.new
    io.write(Bytes.new(offset, 0_u8))
    words.times do |index|
      word = 0_u32
      8.times do |lane|
        q = (index * 5 + lane * 3 + 1) % 16
        word |= q.to_u32 << (lane * 4)
      end
      io.write_bytes(word, IO::ByteFormat::LittleEndian)
    end
    File.write(path, io.to_slice)
    owner = ML::Audio::Drama3::MappedFile.new(path)
    tensor = ML::Audio::Drama3::Tensor.new([rows, cols // 8], "U32", owner.slice(offset.to_i64, words.to_i64 * 4), owner)
    scales = ML::Audio::Drama3::Tensor.new([rows, 1], Array(Float32).new(rows, 0.125_f32))
    biases = ML::Audio::Drama3::Tensor.new([rows, 1], Array(Float32).new(rows, -0.03125_f32))
    matrix = ML::Audio::Drama3::Matrix.new(tensor, scales, biases, 4, 64)
    {path: path, owner: owner, tensor: tensor, matrix: matrix}
  end

  private def drama3_dense_fusion_guarded_copy(context : ML::Audio::Drama3::Resident::Context,
                                               source : ML::Audio::Drama3::Resident::Buffer,
                                               sentinel : Float32) : Drama3DenseFusionGuard
    storage = context.upload(Array(Float32).new(source.count + 2, sentinel))
    values = storage.slice(1, source.count)
    context.copy(source, values)
    {storage: storage, values: values}
  end

  private def drama3_dense_fusion_empty_guard(context : ML::Audio::Drama3::Resident::Context,
                                              count : Int32, sentinel : Float32) : Drama3DenseFusionGuard
    storage = context.upload(Array(Float32).new(count + 2, sentinel))
    {storage: storage, values: storage.slice(1, count)}
  end

  private def drama3_dense_fusion_compare(context : ML::Audio::Drama3::Resident::Context,
                                          reference : Drama3DenseFusionGuard,
                                          actual : Drama3DenseFusionGuard,
                                          receipt : ML::Audio::Drama3::Resident::Buffer,
                                          sentinel : Float32) : Nil
    count = reference[:values].count
    reference[:storage].count.should eq(count + 2)
    actual[:values].count.should eq(count)
    actual[:storage].count.should eq(count + 2)
    context.dispatch(
      "dr3_dense_fusion_compare", DRAMA3_DENSE_FUSION_RECEIPT_SOURCE,
      [reference[:values], actual[:values], reference[:storage], actual[:storage], receipt],
      [count.to_u64, sentinel.unsafe_as(UInt32).to_u64], count,
    )
  end

  private def drama3_dense_fusion_cancellation_witness(context : ML::Audio::Drama3::Resident::Context,
                                                       reference : ML::Audio::Drama3::Resident::Buffer,
                                                       actual : ML::Audio::Drama3::Resident::Buffer,
                                                       receipt : ML::Audio::Drama3::Resident::Buffer) : Nil
    context.dispatch(
      "dr3_dense_fusion_cancellation_witness", DRAMA3_DENSE_FUSION_RECEIPT_SOURCE,
      [reference, actual, receipt], [0.25_f32.unsafe_as(UInt32).to_u64], 1,
    )
  end

  private def drama3_dense_fusion_direct_offset_outputs(
    context : ML::Audio::Drama3::Resident::Context,
    matrices : Tuple(ML::Audio::Drama3::Matrix, ML::Audio::Drama3::Matrix,
                     ML::Audio::Drama3::Matrix, ML::Audio::Drama3::Matrix),
    input : ML::Audio::Drama3::Resident::Buffer,
    sentinel : Float32,
  ) : Array(Drama3DenseFusionGuard)
    weights = {
      ML::Audio::Drama3::MetalBackend.tensor_buffer(matrices[0].tensor, true),
      ML::Audio::Drama3::MetalBackend.tensor_buffer(matrices[1].tensor, true),
      ML::Audio::Drama3::MetalBackend.tensor_buffer(matrices[2].tensor, true),
      ML::Audio::Drama3::MetalBackend.tensor_buffer(matrices[3].tensor, true),
    }
    outputs = Array(Drama3DenseFusionGuard).new
    matrices.each do |matrix|
      outputs << drama3_dense_fusion_empty_guard(context, matrix.rows, sentinel)
    end
    params = [] of UInt64
    matrices.each_with_index do |matrix, index|
      params.concat([matrix.rows.to_u64, matrix.cols.to_u64, weights[index].offset.to_u64,
                     context.encoding(matrix.tensor), matrix.expert.to_u64])
    end
    total_rows = matrices[0].rows + matrices[1].rows + matrices[2].rows + matrices[3].rows

    encoder = context.encoder
    encoder.set_pipeline(context.pipeline("dr3_dense_four_tree", ML::Audio::Drama3::Resident::SOURCE))
    weights.each_with_index { |weight, index| encoder.set_buffer(weight.buffer, index) }
    encoder.set_buffer(input.buffer, 4, offset: input.offset)
    outputs.each_with_index { |output, index| encoder.set_buffer(output[:values].buffer, index + 5, offset: output[:values].offset) }
    encoder.set_bytes(params.to_unsafe.as(Void*), (params.size * 8).to_i32, 9)
    encoder.dispatch_threadgroups({total_rows, 1, 1}, {64, 1, 1})
    outputs
  end

  private def drama3_dense_fusion_compare_case(cols : Int32, offsets : Array(Int32),
                                               reduction : String = "tree", expect_fused : Bool = true,
                                               exercise_output_offsets : Bool = false,
                                               quantized_projection : Bool = false) : Nil
    rows = [5, 3, 7, 2]
    dtypes = ["F32", "BF16", "F16", "F32"]
    fixtures = Array(NamedTuple(path: String, owner: ML::Audio::Drama3::MappedFile,
      tensor: ML::Audio::Drama3::Tensor, matrix: ML::Audio::Drama3::Matrix)).new
    begin
      4.times do |projection|
        if quantized_projection && projection == 2
          fixtures << drama3_dense_fusion_q4_fixture(rows[projection], cols, offsets[projection])
        else
          fixtures << drama3_dense_fusion_dense_fixture(projection.to_i32, rows[projection], cols,
            dtypes[projection], offsets[projection])
        end
      end
      matrices = {fixtures[0][:matrix], fixtures[1][:matrix], fixtures[2][:matrix], fixtures[3][:matrix]}

      ML::Metal::PipelineCache.clear
      ML::Audio::Drama3::MetalBackend.reset!
      ENV["DRAMA3_DENSE_REDUCTION"] = "tree"
      context = ML::Audio::Drama3::Resident::Context.new
      guarded_input = Array(Float32).new(cols + 2, -37_f32)
      drama3_dense_fusion_input(cols).each_with_index { |value, index| guarded_input[index + 1] = value }
      input = context.upload(guarded_input).slice(1, cols)

      baseline = context.matvec_four(matrices, input, dense_reduction: reduction, fused: false)
      before_candidate = ML::Metal::PipelineCache.entry_count
      actual = context.matvec_four(matrices, input, dense_reduction: reduction, fused: true)
      (ML::Metal::PipelineCache.entry_count - before_candidate).should eq(expect_fused ? 1 : 0)

      receipt = context.upload(Array(Float32).new(6, 0_f32))
      sentinel = -113.25_f32
      baseline_guarded = Array(Drama3DenseFusionGuard).new
      actual_guarded = Array(Drama3DenseFusionGuard).new
      4.times do |projection|
        baseline_guarded << drama3_dense_fusion_guarded_copy(context, baseline[projection], sentinel)
        actual_guarded << drama3_dense_fusion_guarded_copy(context, actual[projection], sentinel)
        drama3_dense_fusion_compare(context, baseline_guarded[projection], actual_guarded[projection], receipt, sentinel)
      end

      if cols >= 33
        drama3_dense_fusion_cancellation_witness(
          context, baseline[0].slice(1, 1), actual[0].slice(1, 1), receipt,
        )
      end

      if exercise_output_offsets
        expect_fused.should be_true
        offset_outputs = drama3_dense_fusion_direct_offset_outputs(context, matrices, input, sentinel)
        4.times do |projection|
          drama3_dense_fusion_compare(context, baseline_guarded[projection], offset_outputs[projection], receipt, sentinel)
        end
        if cols >= 33
          drama3_dense_fusion_cancellation_witness(
            context, baseline[0].slice(1, 1), offset_outputs[0][:values].slice(1, 1), receipt,
          )
        end
      end

      context.dispatch("dr3_dense_fusion_negative_control", DRAMA3_DENSE_FUSION_RECEIPT_SOURCE,
        [receipt], [] of UInt64, 1, 1)
      counts = context.read(receipt, "terminal").map(&.unsafe_as(UInt32))
      counts.should eq([0_u32, 0_u32, 2_u32, 1_u32, 0_u32, 0_u32])
      # The real pair must pass exact-bit and finite checks; the injected
      # one-ULP and NaN mismatches must fail those same receipt slots.
      (counts[0] == 0 && counts[1] == 0 && counts[4] == 0 && counts[5] == 0).should be_true
      (counts[2] == 0 && counts[3] == 0).should be_false
      context.readbacks.keys.should eq(["terminal"])
      context.readbacks["terminal"].should eq(24_i64)
      graph = context.graph_stats
      graph["submitted"].should eq(1_i64)
      graph["completed"].should eq(1_i64)
      graph["completion_waits"].should eq(1_i64)
      graph["pending"].should eq(0_i64)
    ensure
      ML::Audio::Drama3::MetalBackend.reset!
      ML::Metal::PipelineCache.clear
      fixtures.each { |fixture| File.delete(fixture[:path]) if File.exists?(fixture[:path]) }
    end
  end

  private def drama3_dense_fusion_tiny_config : ML::Audio::Drama3::BackboneConfig
    ML::Audio::Drama3::BackboneConfig.new(JSON.parse(<<-JSON
      {
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "full_attention_interval": 1,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 1,
        "linear_key_head_dim": 2,
        "linear_value_head_dim": 2,
        "linear_conv_kernel_dim": 2,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5}
      }
      JSON
    ))
  end

  describe "DRAMA-3 resident dense four-projection fusion" do
    it "matches tree matvecs bitwise across projection boundaries, dtypes, and lane tails" do
      drama3_dense_fusion_with_device do
        [31, 32, 33, 63, 64, 65, 2048].each do |cols|
          drama3_dense_fusion_compare_case(cols, [24, 26, 28, 32])
        end
        # Odd raw byte offsets exercise the byte-decoding path and the test-only
        # nonzero output views exercise the fused kernel's output binding offsets.
        drama3_dense_fusion_compare_case(65, [21, 23, 25, 27], exercise_output_offsets: true)
      ensure
        ML::Audio::Drama3::MetalBackend.reset!
      end
    end

    it "falls back to four established matvecs for SIMD and affine Q4 inputs" do
      drama3_dense_fusion_with_device do
        drama3_dense_fusion_compare_case(65, [24, 26, 28, 32], reduction: "simd", expect_fused: false)
        drama3_dense_fusion_compare_case(64, [24, 26, 28, 32], expect_fused: false, quantized_projection: true)
      ensure
        ML::Audio::Drama3::MetalBackend.reset!
      end
    end

    it "rejects wrong owners and widths before allocation, pipeline lookup, or dispatch" do
      drama3_dense_fusion_with_device do
        ENV["DRAMA3_DENSE_REDUCTION"] = "tree"
        ML::Metal::PipelineCache.clear
        context = ML::Audio::Drama3::Resident::Context.new
        other = ML::Audio::Drama3::Resident::Context.new
        matrices = {
          ML::Audio::Drama3::Matrix.new(2, 4, Array(Float32).new(8, 1_f32)),
          ML::Audio::Drama3::Matrix.new(3, 4, Array(Float32).new(12, 1_f32)),
          ML::Audio::Drama3::Matrix.new(1, 4, Array(Float32).new(4, 1_f32)),
          ML::Audio::Drama3::Matrix.new(5, 4, Array(Float32).new(20, 1_f32)),
        }
        foreign_input = other.upload(Array(Float32).new(4, 1_f32))
        before_allocations = context.allocation_stats[:allocations]
        before_graph = context.graph_stats.dup
        before_pipelines = ML::Metal::PipelineCache.entry_count
        expect_raises(ArgumentError, /buffer belongs to a different resident context/) do
          context.matvec_four(matrices, foreign_input, fused: true)
        end
        context.allocation_stats[:allocations].should eq(before_allocations)
        context.graph_stats.should eq(before_graph)
        ML::Metal::PipelineCache.entry_count.should eq(before_pipelines)

        local_wrong_width = context.upload([1_f32, 2_f32, 3_f32])
        before_allocations = context.allocation_stats[:allocations]
        before_graph = context.graph_stats.dup
        before_pipelines = ML::Metal::PipelineCache.entry_count
        expect_raises(ArgumentError, /resident matvec width/) do
          context.matvec_four(matrices, local_wrong_width, fused: true)
        end
        context.allocation_stats[:allocations].should eq(before_allocations)
        context.graph_stats.should eq(before_graph)
        ML::Metal::PipelineCache.entry_count.should eq(before_pipelines)
        context.readbacks.should be_empty
      ensure
        ML::Audio::Drama3::MetalBackend.reset!
        ML::Metal::PipelineCache.clear
      end
    end

    it "freezes the strict backbone selector at construction and rejects malformed values" do
      keys = ["DRAMA3_BACKEND", "DRAMA3_DENSE_REDUCTION", "DRAMA3_BACKBONE_DENSE_REDUCTION",
              "DRAMA3_BACKBONE_DENSE_FUSION", "DRAMA3_COGNIGRAPH_MAX_INFLIGHT",
              "DRAMA3_METAL_TIMELINE", "DRAMA3_METAL_RESOURCES", "DRAMA3_DELTA_TRANSPOSE",
              "DRAMA3_MOE_BATCH", "DRAMA3_EXPERT_DOWN_REDUCTION"]
      previous = keys.to_h { |key| {key, ENV[key]?} }
      begin
        drama3_dense_fusion_with_device do
          ENV["DRAMA3_DENSE_REDUCTION"] = "tree"
          ENV["DRAMA3_BACKBONE_DENSE_REDUCTION"] = "tree"
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "0"
          ENV["DRAMA3_METAL_TIMELINE"] = "0"
          ENV["DRAMA3_METAL_RESOURCES"] = "0"
          ENV["DRAMA3_DELTA_TRANSPOSE"] = "1"
          ENV["DRAMA3_MOE_BATCH"] = "1"
          ENV["DRAMA3_EXPERT_DOWN_REDUCTION"] = "native"
          config = drama3_dense_fusion_tiny_config
          store = ML::Audio::Drama3::WeightStore.new({} of String => ML::Audio::Drama3::Tensor)

          ENV["DRAMA3_BACKBONE_DENSE_FUSION"] = "0"
          default_context = ML::Audio::Drama3::Resident::Context.new
          default_model = ML::Audio::Drama3::BackboneResident.new(config, store, default_context, 1)
          ENV["DRAMA3_BACKBONE_DENSE_FUSION"] = "1"
          default_model.dense_fusion_enabled.should be_false

          enabled_context = ML::Audio::Drama3::Resident::Context.new
          enabled_model = ML::Audio::Drama3::BackboneResident.new(config, store, enabled_context, 1)
          enabled_model.dense_fusion_enabled.should be_true
          ENV["DRAMA3_BACKBONE_DENSE_FUSION"] = "0"
          enabled_model.dense_fusion_enabled.should be_true
          default_context.graph_stats["submitted"].should eq(0_i64)
          enabled_context.graph_stats["submitted"].should eq(0_i64)

          invalid_context = ML::Audio::Drama3::Resident::Context.new
          ENV["DRAMA3_BACKBONE_DENSE_FUSION"] = "true"
          before_allocations = invalid_context.allocation_stats[:allocations]
          before_graph = invalid_context.graph_stats.dup
          expect_raises(ArgumentError, /DRAMA3_BACKBONE_DENSE_FUSION must be 0 or 1/) do
            ML::Audio::Drama3::BackboneResident.new(config, store, invalid_context, 1)
          end
          invalid_context.allocation_stats[:allocations].should eq(before_allocations)
          invalid_context.graph_stats.should eq(before_graph)
          invalid_context.readbacks.should be_empty
        ensure
          ML::Audio::Drama3::MetalBackend.reset!
        end
      ensure
        keys.each do |key|
          if value = previous[key]
            ENV[key] = value.not_nil!
          else
            ENV.delete(key)
          end
        end
      end
    end
  end
{% end %}
