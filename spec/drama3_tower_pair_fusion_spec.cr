require "spec"
require "../src/ml/audio/drama3/tower_resident"

include ML::Audio::Drama3

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private alias Drama3TowerPairGuard = NamedTuple(
    storage: Resident::Buffer,
    values: Resident::Buffer,
  )

  private alias Drama3TowerPairFixture = NamedTuple(
    path: String,
    owner: MappedFile,
    tensor: Tensor,
    matrix: Matrix,
  )

  private DRAMA3_TOWER_PAIR_RECEIPT_SOURCE = Resident::SOURCE + <<-METAL
    inline void dr3_tower_pair_record(device atomic_uint* receipt,
                                      uint bit_slot, uint finite_slot,
                                      float reference, float actual) {
        if (as_type<uint>(reference) != as_type<uint>(actual)) {
            atomic_fetch_add_explicit(&receipt[bit_slot], 1u, memory_order_relaxed);
        }
        if (!isfinite(reference) || !isfinite(actual)) {
            atomic_fetch_add_explicit(&receipt[finite_slot], 1u, memory_order_relaxed);
        }
    }

    kernel void dr3_tower_pair_compare(
        device const float* reference [[buffer(0)]], device const float* actual [[buffer(1)]],
        device const float* reference_storage [[buffer(2)]], device const float* actual_storage [[buffer(3)]],
        device atomic_uint* receipt [[buffer(4)]], constant ulong* params [[buffer(5)]],
        uint index [[thread_position_in_grid]]) {
        if (ulong(index) >= params[0]) return;
        dr3_tower_pair_record(receipt, 0, 1, reference[index], actual[index]);
        if (index == 0) {
            uint guard_bits = uint(params[1]);
            if (as_type<uint>(reference_storage[0]) != guard_bits ||
                as_type<uint>(reference_storage[params[0] + 1]) != guard_bits ||
                as_type<uint>(actual_storage[0]) != guard_bits ||
                as_type<uint>(actual_storage[params[0] + 1]) != guard_bits) {
                atomic_fetch_add_explicit(&receipt[4], 1u, memory_order_relaxed);
            }
        }
    }

    kernel void dr3_tower_pair_negative_control(
        device atomic_uint* receipt [[buffer(0)]], uint index [[thread_position_in_grid]]) {
        if (index != 0) return;
        float one = 1.0f;
        float one_ulp_up = as_type<float>(as_type<uint>(one) + 1u);
        dr3_tower_pair_record(receipt, 2, 3, one, one_ulp_up);
        dr3_tower_pair_record(receipt, 2, 3, as_type<float>(0x7fc00000u), 0.0f);
    }

    kernel void dr3_tower_pair_cancellation_witness(
        device const float* reference [[buffer(0)]], device const float* actual [[buffer(1)]],
        device atomic_uint* receipt [[buffer(2)]], constant ulong* params [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
        if (index != 0) return;
        if (as_type<uint>(reference[0]) != uint(params[0]) ||
            as_type<uint>(actual[0]) != uint(params[0])) {
            atomic_fetch_add_explicit(&receipt[5], 1u, memory_order_relaxed);
        }
    }
    METAL

  private def drama3_tower_pair_with_device(&block : -> T) : T forall T
    keys = ["DRAMA3_BACKEND", "DRAMA3_DENSE_REDUCTION", "DRAMA3_COGNIGRAPH_MAX_INFLIGHT",
            "DRAMA3_METAL_TIMELINE", "DRAMA3_METAL_RESOURCES", "DRAMA3_EXPERT_THREADS",
            "DRAMA3_EXPERT_PACKED_ROWS", "DRAMA3_TOWER_EMBEDDING_FUSION",
            "DRAMA3_TOWER_FFN_PAIR_FUSION", "DRAMA3_DELTA_TRANSPOSE", "DRAMA3_MOE_BATCH",
            "DRAMA3_EXPERT_DOWN_REDUCTION"]
    previous = keys.to_h { |key| {key, ENV[key]?} }
    begin
      ENV["DRAMA3_BACKEND"] = "metal"
      ENV["DRAMA3_DENSE_REDUCTION"] = "tree"
      ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "0"
      ENV["DRAMA3_METAL_TIMELINE"] = "0"
      ENV["DRAMA3_METAL_RESOURCES"] = "0"
      ENV["DRAMA3_EXPERT_THREADS"] = "64"
      ENV["DRAMA3_EXPERT_PACKED_ROWS"] = "0"
      ENV["DRAMA3_TOWER_EMBEDDING_FUSION"] = "0"
      ENV["DRAMA3_TOWER_FFN_PAIR_FUSION"] = "0"
      ENV["DRAMA3_DELTA_TRANSPOSE"] = "1"
      ENV["DRAMA3_MOE_BATCH"] = "1"
      ENV["DRAMA3_EXPERT_DOWN_REDUCTION"] = "native"
      device = ML::Metal::Device.instance
      pending!("Metal is unavailable or lacks unified memory") unless device.available? && device.has_unified_memory?
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

  private def drama3_tower_pair_weight_value(projection : Int32, row : Int32,
                                             col : Int32, cols : Int32) : Float32
    if projection == 0 && row == 1 && cols >= 33
      case col
      when  0 then 2048_f32
      when  1 then 0.25_f32
      when 32 then -2048_f32
      else         0_f32
      end
    elsif row == 0 && col == cols - 1
      -0.75_f32
    else
      (((row * cols + col * 13 + projection * 7 + 5) % 17) - 8).to_f32 / 8_f32
    end
  end

  private def drama3_tower_pair_input_values(cols : Int32) : Array(Float32)
    values = Array(Float32).new(cols) { |index| (((index * 7 + 3) % 19) - 9).to_f32 / 8_f32 }
    values[0] = 8192_f32
    values[1] = 1_f32 if cols > 1
    values[32] = 8192_f32 if cols >= 33
    values
  end

  private def drama3_tower_pair_f16_bits(value : Float32) : UInt16
    return 0_u16 if value == 0_f32
    bits = value.unsafe_as(UInt32)
    exponent = ((bits >> 23) & 0xff).to_i32 - 127 + 15
    raise ArgumentError.new("DRAMA-3 tower pair F16 fixture must be a normal value") unless exponent > 0 && exponent < 31
    sign = (bits >> 16) & 0x8000
    fraction = (bits & 0x7fffff) >> 13
    (sign | (exponent.to_u32 << 10) | fraction).to_u16
  end

  private def drama3_tower_pair_write_value(io : IO, dtype : String, value : Float32) : Nil
    case dtype
    when "F32"
      io.write_bytes(value, IO::ByteFormat::LittleEndian)
    when "BF16"
      io.write_bytes((value.unsafe_as(UInt32) >> 16).to_u16, IO::ByteFormat::LittleEndian)
    when "F16"
      io.write_bytes(drama3_tower_pair_f16_bits(value), IO::ByteFormat::LittleEndian)
    else
      raise ArgumentError.new("unsupported DRAMA-3 tower pair dtype #{dtype}")
    end
  end

  private def drama3_tower_pair_dense_fixture(projection : Int32, rows : Int32, cols : Int32,
                                              dtype : String, offset : Int32,
                                              payload : Array(Float32)? = nil) : Drama3TowerPairFixture
    values = payload || Array(Float32).new(rows * cols) do |index|
      drama3_tower_pair_weight_value(projection, index // cols, index % cols, cols)
    end
    raise ArgumentError.new("pair fixture payload shape") unless values.size == rows * cols
    path = File.tempname("drama3-tower-pair", ".bin")
    io = IO::Memory.new
    io.write(Bytes.new(offset, 0_u8))
    values.each { |value| drama3_tower_pair_write_value(io, dtype, value) }
    File.write(path, io.to_slice)
    owner = MappedFile.new(path)
    byte_width = dtype == "F32" ? 4 : 2
    tensor = Tensor.new([rows, cols], dtype, owner.slice(offset.to_i64, (values.size * byte_width).to_i64), owner)
    {path: path, owner: owner, tensor: tensor, matrix: Matrix.new(tensor)}
  end

  private def drama3_tower_pair_q4_fixture(rows : Int32 = 7, cols : Int32 = 64,
                                           offset : Int32 = 24) : Drama3TowerPairFixture
    words = rows * (cols // 8)
    path = File.tempname("drama3-tower-pair-q4", ".bin")
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
    owner = MappedFile.new(path)
    tensor = Tensor.new([rows, cols // 8], "U32", owner.slice(offset.to_i64, words.to_i64 * 4), owner)
    scales = Tensor.new([rows, 1], Array(Float32).new(rows, 0.125_f32))
    biases = Tensor.new([rows, 1], Array(Float32).new(rows, -0.03125_f32))
    matrix = Matrix.new(tensor, scales, biases, 4, 64)
    {path: path, owner: owner, tensor: tensor, matrix: matrix}
  end

  private def drama3_tower_pair_guarded_copy(context : Resident::Context,
                                             source : Resident::Buffer,
                                             sentinel : Float32) : Drama3TowerPairGuard
    storage = context.upload(Array(Float32).new(source.count + 2, sentinel))
    values = storage.slice(1, source.count)
    context.copy(source, values)
    {storage: storage, values: values}
  end

  private def drama3_tower_pair_empty_guard(context : Resident::Context,
                                            count : Int32,
                                            sentinel : Float32) : Drama3TowerPairGuard
    storage = context.upload(Array(Float32).new(count + 2, sentinel))
    {storage: storage, values: storage.slice(1, count)}
  end

  private def drama3_tower_pair_compare(context : Resident::Context,
                                        expected : Drama3TowerPairGuard,
                                        actual : Drama3TowerPairGuard,
                                        receipt : Resident::Buffer,
                                        sentinel : Float32) : Nil
    count = expected[:values].count
    expected[:storage].count.should eq(count + 2)
    actual[:values].count.should eq(count)
    actual[:storage].count.should eq(count + 2)
    context.dispatch(
      "dr3_tower_pair_compare", DRAMA3_TOWER_PAIR_RECEIPT_SOURCE,
      [expected[:values], actual[:values], expected[:storage], actual[:storage], receipt],
      [count.to_u64, sentinel.unsafe_as(UInt32).to_u64], count,
    )
  end

  private def drama3_tower_pair_cancellation_witness(context : Resident::Context,
                                                     expected : Resident::Buffer,
                                                     actual : Resident::Buffer,
                                                     receipt : Resident::Buffer) : Nil
    context.dispatch(
      "dr3_tower_pair_cancellation_witness", DRAMA3_TOWER_PAIR_RECEIPT_SOURCE,
      [expected, actual, receipt], [0.25_f32.unsafe_as(UInt32).to_u64], 1,
    )
  end

  private def drama3_tower_pair_direct_outputs(
    context : Resident::Context,
    matrices : Tuple(Matrix, Matrix),
    input : Resident::Buffer,
    sentinel : Float32,
    paired_rows : Bool = false,
  ) : Tuple(Drama3TowerPairGuard, Drama3TowerPairGuard)
    weights = {
      MetalBackend.tensor_buffer(matrices[0].tensor, true),
      MetalBackend.tensor_buffer(matrices[1].tensor, true),
    }
    outputs = {
      drama3_tower_pair_empty_guard(context, matrices[0].rows, sentinel),
      drama3_tower_pair_empty_guard(context, matrices[1].rows, sentinel),
    }
    params = [] of UInt64
    matrices.each_with_index do |matrix, index|
      params.concat([matrix.rows.to_u64, matrix.cols.to_u64, weights[index].offset.to_u64,
                     context.encoding(matrix.tensor), matrix.expert.to_u64])
    end
    total_rows = matrices[0].rows + matrices[1].rows

    encoder = context.encoder
    name = paired_rows ? "dr3_dense_pair_bf16_aligned" : "dr3_dense_pair_tree"
    encoder.set_pipeline(context.pipeline(name, Resident::SOURCE))
    weights.each_with_index { |weight, index| encoder.set_buffer(weight.buffer, index) }
    encoder.set_buffer(input.buffer, 2, offset: input.offset)
    outputs.each_with_index { |output, index| encoder.set_buffer(output[:values].buffer, index + 3, offset: output[:values].offset) }
    encoder.set_bytes(params.to_unsafe.as(Void*), (params.size * 8).to_i32, 5)
    encoder.dispatch_threadgroups({paired_rows ? matrices[0].rows : total_rows, 1, 1}, {64, 1, 1})
    outputs
  end

  private def drama3_tower_pair_compare_case(
    context : Resident::Context,
    matrices : Tuple(Matrix, Matrix),
    cols : Int32,
    receipt : Resident::Buffer,
    reduction : String = "tree",
    expect_fused : Bool = true,
    expected_pipeline_delta : Int32 = 0,
    output_offsets : Bool = false,
    cancellation : Bool = false,
    expected_pipeline_name : String? = nil,
  ) : Nil
    sentinel = -113.25_f32
    guarded_input = Array(Float32).new(cols + 2, -37_f32)
    drama3_tower_pair_input_values(cols).each_with_index { |value, index| guarded_input[index + 1] = value }
    input = context.upload(guarded_input).slice(1, cols)

    expected = context.matvec_pair(matrices, input, dense_reduction: reduction, fused: false)
    if expected_pipeline_name
      # Drain the oracle command before clearing pipeline objects; the cache is
      # empty at candidate dispatch so the next key proves the selected branch.
      context.finish
      ML::Metal::PipelineCache.clear
    end
    before_candidate = ML::Metal::PipelineCache.entry_count
    actual = context.matvec_pair(matrices, input, dense_reduction: reduction, fused: true)
    (ML::Metal::PipelineCache.entry_count - before_candidate).should eq(expected_pipeline_delta)
    if name = expected_pipeline_name
      selected = ML::Metal::PipelineCache.get(name) do
        raise "candidate did not compile expected pair kernel #{name}"
      end
      selected.name.should eq(name)
    end
    actual[0].count.should eq(matrices[0].rows)
    actual[1].count.should eq(matrices[1].rows)
    expected_guarded = {
      drama3_tower_pair_guarded_copy(context, expected[0], sentinel),
      drama3_tower_pair_guarded_copy(context, expected[1], sentinel),
    }
    actual_guarded = {
      drama3_tower_pair_guarded_copy(context, actual[0], sentinel),
      drama3_tower_pair_guarded_copy(context, actual[1], sentinel),
    }
    2.times do |index|
      drama3_tower_pair_compare(context, expected_guarded[index], actual_guarded[index], receipt, sentinel)
    end

    if cancellation
      cancellation.should be_true
      drama3_tower_pair_cancellation_witness(context, expected[0].slice(1, 1), actual[0].slice(1, 1), receipt)
    end

    if output_offsets
      expect_fused.should be_true
      direct = drama3_tower_pair_direct_outputs(context, matrices, input, sentinel,
        paired_rows: expected_pipeline_name == "dr3_dense_pair_bf16_aligned")
      2.times do |index|
        drama3_tower_pair_compare(context, expected_guarded[index], direct[index], receipt, sentinel)
      end
      if cancellation
        drama3_tower_pair_cancellation_witness(context, expected[0].slice(1, 1), direct[0][:values].slice(1, 1), receipt)
      end
    end
  end

  private def drama3_tower_pair_tiny_config : TowerConfig
    TowerConfig.new(
      vocab_size: 4,
      num_codebooks: 3,
      dim: 4,
      n_head: 2,
      n_local_heads: 1,
      head_dim: 2,
      intermediate_size: 6,
      norm_eps: 1e-6,
      rope_base: 100.0,
      rotary_percent: 1.0,
      shared_enrichment_layers: 1,
      vq_backbone_codebooks: 1,
      semantic_begin: 10,
      semantic_end: 13,
    )
  end

  private def drama3_tower_pair_tiny_store(config : TowerConfig) : WeightStore
    tensors = {} of String => Tensor
    seed = 1
    add_matrix = ->(name : String, rows : Int32, cols : Int32) do
      values = Array(Float32).new(rows * cols) { |i| (((i + seed * 3) % 11) - 5).to_f32 * 0.017_f32 }
      tensors[name] = Tensor.new([rows, cols], values)
      seed += 1
    end
    add_vector = ->(name : String, size : Int32) do
      values = Array(Float32).new(size) { |i| (((i + seed * 2) % 7) - 3).to_f32 * 0.006_f32 }
      tensors[name] = Tensor.new([size], values)
      seed += 1
    end

    add_matrix.call("tower.codebook_embeddings.weight", config.vocab_size * config.num_codebooks, config.dim)
    layer_count = config.shared_enrichment_layers + config.num_codebooks - 1
    layer_count.times do |index|
      layer_prefix = index < config.shared_enrichment_layers ? "tower.shared_layers.#{index}" : "tower.residual_layers.#{index - config.shared_enrichment_layers}"
      embedding = index < config.shared_enrichment_layers ? "semantic_embedding" : "codebook_embedding"
      add_matrix.call("#{layer_prefix}.#{embedding}.weight", config.vocab_size, config.dim)
      {"enorm", "hnorm", "final_layernorm"}.each { |name| add_vector.call("#{layer_prefix}.#{name}.weight", config.dim) }
      add_matrix.call("#{layer_prefix}.eh_proj.weight", config.dim, config.dim * 2)

      transformer = "#{layer_prefix}.transformer_layer"
      add_vector.call("#{transformer}.attention_norm.weight", config.dim)
      add_vector.call("#{transformer}.ffn_norm.weight", config.dim)
      q_width = config.n_head * config.head_dim
      kv_width = config.n_local_heads * config.head_dim
      add_matrix.call("#{transformer}.attention.wqkv.weight", q_width * 2 + kv_width * 2, config.dim)
      add_matrix.call("#{transformer}.attention.wo.weight", config.dim, q_width)
      add_vector.call("#{transformer}.attention.q_norm.weight", config.head_dim)
      add_vector.call("#{transformer}.attention.k_norm.weight", config.head_dim)
      add_matrix.call("#{transformer}.feed_forward.w1.weight", config.intermediate_size, config.dim)
      add_matrix.call("#{transformer}.feed_forward.w3.weight", config.intermediate_size, config.dim)
      add_matrix.call("#{transformer}.feed_forward.w2.weight", config.dim, config.intermediate_size)

      if index >= config.shared_enrichment_layers
        add_matrix.call("tower.codebook_heads.#{index - config.shared_enrichment_layers}.weight", config.vocab_size, config.dim)
      end
    end
    WeightStore.new(tensors)
  end

  describe "DRAMA-3 tower dense pair fusion" do
    it "matches two tree matvecs bitwise across widths, row counts, dtypes, offsets, and the original cancellation order" do
      drama3_tower_pair_with_device do
        MetalBackend.reset!
        ML::Metal::PipelineCache.clear
        context = Resident::Context.new
        receipt = context.upload(Array(Float32).new(6, 0_f32))
        fixtures = [] of Drama3TowerPairFixture
        begin
          [31, 32, 33, 63, 64, 65, 2048].each_with_index do |cols, index|
            first = drama3_tower_pair_dense_fixture(0, 5, cols, "F32", 21)
            second = drama3_tower_pair_dense_fixture(1, 3, cols, "BF16", 23)
            fixtures.concat([first, second])
            delta = index == 0 ? 1 : 0
            drama3_tower_pair_compare_case(
              context, {first[:matrix], second[:matrix]}, cols, receipt,
              expected_pipeline_delta: delta,
              output_offsets: cols == 65,
              cancellation: cols == 65,
            )
          end

          half = drama3_tower_pair_dense_fixture(0, 7, 65, "F16", 25)
          single = drama3_tower_pair_dense_fixture(1, 2, 65, "F32", 27)
          fixtures.concat([half, single])
          drama3_tower_pair_compare_case(context, {half[:matrix], single[:matrix]}, 65, receipt)

          simd0 = drama3_tower_pair_dense_fixture(0, 4, 65, "F32", 21)
          simd1 = drama3_tower_pair_dense_fixture(1, 6, 65, "BF16", 23)
          fixtures.concat([simd0, simd1])
          ML::Metal::PipelineCache.clear
          drama3_tower_pair_compare_case(
            context, {simd0[:matrix], simd1[:matrix]}, 65, receipt,
            reduction: "simd", expect_fused: false,
          )

          quantized = drama3_tower_pair_q4_fixture(7, 64)
          dense = drama3_tower_pair_dense_fixture(1, 7, 64, "F32", 21)
          fixtures.concat([quantized, dense])
          ML::Metal::PipelineCache.clear
          drama3_tower_pair_compare_case(
            context, {quantized[:matrix], dense[:matrix]}, 64, receipt,
            expect_fused: false,
          )

          context.dispatch("dr3_tower_pair_negative_control", DRAMA3_TOWER_PAIR_RECEIPT_SOURCE,
            [receipt], [] of UInt64, 1, 1)
          context.readbacks.should be_empty
          context.graph_stats["completion_waits"].should eq(0_i64)
          counts = context.read(receipt, "terminal").map(&.unsafe_as(UInt32))
          counts.should eq([0_u32, 0_u32, 2_u32, 1_u32, 0_u32, 0_u32])
          (counts[0] == 0 && counts[1] == 0 && counts[4] == 0 && counts[5] == 0).should be_true
          counts[2].should eq(2_u32)
          counts[3].should eq(1_u32)
          context.readbacks.keys.should eq(["terminal"])
          context.readbacks["terminal"].should eq(24_i64)
          graph = context.graph_stats
          graph["submitted"].should eq(1_i64)
          graph["completed"].should eq(1_i64)
          graph["completion_waits"].should eq(1_i64)
          graph["pending"].should eq(0_i64)
        ensure
          context.finish
          MetalBackend.reset!
          ML::Metal::PipelineCache.clear
          fixtures.each { |fixture| File.delete(fixture[:path]) if File.exists?(fixture[:path]) }
        end
      end
    end

    it "selects the aligned BF16 row-pair branch and keeps generic fallback for unsafe pairs" do
      drama3_tower_pair_with_device do
        MetalBackend.reset!
        ML::Metal::PipelineCache.clear
        context = Resident::Context.new
        receipt = context.upload(Array(Float32).new(6, 0_f32))
        fixtures = [] of Drama3TowerPairFixture
        begin
          rows, cols = 3, 65
          file_bytes = 1_048_576
          payload_bytes = rows * cols * 2
          exact_end_offset = file_bytes - payload_bytes
          exact_end_offset.even?.should be_true
          aligned0 = drama3_tower_pair_dense_fixture(0, rows, cols, "BF16", exact_end_offset)
          aligned1 = drama3_tower_pair_dense_fixture(1, rows, cols, "BF16", exact_end_offset)
          fixtures.concat([aligned0, aligned1])
          {aligned0, aligned1}.each do |fixture|
            weight = MetalBackend.tensor_buffer(fixture[:tensor], true)
            raw_size = fixture[:tensor].raw_bytes.not_nil!.size.to_i64
            (weight.offset + raw_size).should eq(weight.buffer.size)
          end
          drama3_tower_pair_compare_case(
            context, {aligned0[:matrix], aligned1[:matrix]}, cols, receipt,
            expected_pipeline_delta: 1,
            expected_pipeline_name: "dr3_dense_pair_bf16_aligned",
            output_offsets: true,
            cancellation: true,
          )

          [31, 32, 33, 63, 64, 2048].each do |width|
            first = drama3_tower_pair_dense_fixture(0, rows, width, "BF16", 24)
            second = drama3_tower_pair_dense_fixture(1, rows, width, "BF16", 26)
            fixtures.concat([first, second])
            drama3_tower_pair_compare_case(
              context, {first[:matrix], second[:matrix]}, width, receipt,
              expected_pipeline_delta: 1,
              expected_pipeline_name: "dr3_dense_pair_bf16_aligned",
              output_offsets: true,
            )
          end

          odd0 = drama3_tower_pair_dense_fixture(0, rows, cols, "BF16", 23)
          odd1 = drama3_tower_pair_dense_fixture(1, rows, cols, "BF16", 25)
          fixtures.concat([odd0, odd1])
          drama3_tower_pair_compare_case(
            context, {odd0[:matrix], odd1[:matrix]}, cols, receipt,
            expected_pipeline_delta: 1,
            expected_pipeline_name: "dr3_dense_pair_tree",
          )

          mixed0 = drama3_tower_pair_dense_fixture(0, rows, cols, "F32", 20)
          mixed1 = drama3_tower_pair_dense_fixture(1, rows, cols, "BF16", 24)
          fixtures.concat([mixed0, mixed1])
          drama3_tower_pair_compare_case(
            context, {mixed0[:matrix], mixed1[:matrix]}, cols, receipt,
            expected_pipeline_delta: 1,
            expected_pipeline_name: "dr3_dense_pair_tree",
          )

          unequal0 = drama3_tower_pair_dense_fixture(0, 2, cols, "BF16", 24)
          unequal1 = drama3_tower_pair_dense_fixture(1, rows, cols, "BF16", 26)
          fixtures.concat([unequal0, unequal1])
          drama3_tower_pair_compare_case(
            context, {unequal0[:matrix], unequal1[:matrix]}, cols, receipt,
            expected_pipeline_delta: 1,
            expected_pipeline_name: "dr3_dense_pair_tree",
          )

          counts = context.read(receipt, "terminal").map(&.unsafe_as(UInt32))
          counts[0].should eq(0_u32)
          counts[1].should eq(0_u32)
          counts[4].should eq(0_u32)
          counts[5].should eq(0_u32)
          context.readbacks.keys.should eq(["terminal"])
        ensure
          context.finish
          MetalBackend.reset!
          ML::Metal::PipelineCache.clear
          fixtures.each { |fixture| File.delete(fixture[:path]) if File.exists?(fixture[:path]) }
        end
      end
    end

    it "preserves BF16 zero, subnormal, infinity, and NaN bit patterns against the original GPU tree" do
      drama3_tower_pair_with_device do
        MetalBackend.reset!
        ML::Metal::PipelineCache.clear
        context = Resident::Context.new
        receipt = context.upload(Array(Float32).new(6, 0_f32))
        fixtures = [] of Drama3TowerPairFixture
        begin
          rows, cols = 10, 65
          patterns = [0x0000_u16, 0x8000_u16, 0x0001_u16, 0x8001_u16, 0x0080_u16,
                      0x7f7f_u16, 0x7f80_u16, 0xff80_u16, 0x7fc1_u16, 0xffc2_u16]
          2.times do |projection|
            values = Array(Float32).new(rows * cols) do |index|
              bits = patterns[(index // cols + projection) % rows].to_u32 << 16
              bits.unsafe_as(Float32)
            end
            fixtures << drama3_tower_pair_dense_fixture(projection, rows, cols, "BF16", 24 + projection * 2, values)
          end
          drama3_tower_pair_compare_case(
            context, {fixtures[0][:matrix], fixtures[1][:matrix]}, cols, receipt,
            expected_pipeline_delta: 1,
            expected_pipeline_name: "dr3_dense_pair_bf16_aligned",
            output_offsets: true,
          )
          counts = context.read(receipt, "terminal").map(&.unsafe_as(UInt32))
          counts[0].should eq(0_u32)
          counts[1].should be > 0_u32
          counts[4].should eq(0_u32)
          context.readbacks.keys.should eq(["terminal"])
        ensure
          context.finish
          MetalBackend.reset!
          ML::Metal::PipelineCache.clear
          fixtures.each { |fixture| File.delete(fixture[:path]) if File.exists?(fixture[:path]) }
        end
      end
    end

    it "validates owners and every width before allocating or resolving a pipeline" do
      drama3_tower_pair_with_device do
        ML::Metal::PipelineCache.clear
        context = Resident::Context.new
        other = Resident::Context.new
        matrices = {
          Matrix.new(2, 4, Array(Float32).new(8, 1_f32)),
          Matrix.new(3, 4, Array(Float32).new(12, 1_f32)),
        }
        foreign = other.upload(Array(Float32).new(4, 1_f32))
        before_allocations = context.allocation_stats[:allocations]
        before_graph = context.graph_stats.dup
        before_pipelines = ML::Metal::PipelineCache.entry_count
        expect_raises(ArgumentError, /buffer belongs to a different resident context/) do
          context.matvec_pair(matrices, foreign, fused: true)
        end
        context.allocation_stats[:allocations].should eq(before_allocations)
        context.graph_stats.should eq(before_graph)
        ML::Metal::PipelineCache.entry_count.should eq(before_pipelines)

        wrong_width = context.upload(Array(Float32).new(3, 1_f32))
        before_allocations = context.allocation_stats[:allocations]
        before_graph = context.graph_stats.dup
        before_pipelines = ML::Metal::PipelineCache.entry_count
        expect_raises(ArgumentError, /resident matvec width/) do
          context.matvec_pair(matrices, wrong_width, fused: true)
        end
        context.allocation_stats[:allocations].should eq(before_allocations)
        context.graph_stats.should eq(before_graph)
        ML::Metal::PipelineCache.entry_count.should eq(before_pipelines)

        bad_second = {matrices[0], Matrix.new(1, 3, Array(Float32).new(3, 1_f32))}
        before_allocations = context.allocation_stats[:allocations]
        before_graph = context.graph_stats.dup
        before_pipelines = ML::Metal::PipelineCache.entry_count
        expect_raises(ArgumentError, /resident matvec width/) do
          context.matvec_pair(bad_second, context.upload(Array(Float32).new(4, 1_f32)), fused: true)
        end
        # The explicit input upload is the only allowed allocation in this case.
        context.allocation_stats[:allocations].should eq(before_allocations + 1)
        before_graph.should eq(context.graph_stats)
        ML::Metal::PipelineCache.entry_count.should eq(before_pipelines)
        context.readbacks.should be_empty
      ensure
        MetalBackend.reset!
        ML::Metal::PipelineCache.clear
      end
    end

    it "freezes the strict selector and lets buffer_trace observe buffers without reading activations" do
      drama3_tower_pair_with_device do
        MetalBackend.reset!
        ML::Metal::PipelineCache.clear
        config = drama3_tower_pair_tiny_config
        store = drama3_tower_pair_tiny_store(config)
        context = Resident::Context.new
        receipt = context.upload(Array(Float32).new(6, 0_f32))
        ENV.delete("DRAMA3_TOWER_FFN_PAIR_FUSION")
        baseline = TowerResident.new(config, store, context, 2)
        baseline.ffn_pair_fusion_enabled.should be_false
        baseline.buffer_trace.should be_nil

        hidden = context.upload([0.1_f32, -0.2_f32, 0.3_f32, 0.4_f32])
        previous = context.upload([11_f32, 1_f32, 2_f32, 3_f32])
        current = context.upload([12_f32, 2_f32, 1_f32, 0_f32])
        expected = baseline.teacher_force(hidden, previous, current)
        ENV["DRAMA3_TOWER_FFN_PAIR_FUSION"] = "1"
        baseline.ffn_pair_fusion_enabled.should be_false

        fused = TowerResident.new(config, store, context, 2)
        fused.ffn_pair_fusion_enabled.should be_true
        fused.buffer_trace.should be_nil
        observed = {} of String => Tuple(Array(Int32), Resident::Buffer)
        fused.buffer_trace = ->(name : String, shape : Array(Int32), buffer : Resident::Buffer) do
          observed[name] = {shape.dup, buffer}
          nil
        end
        before_fused = ML::Metal::PipelineCache.entry_count
        actual = fused.teacher_force(hidden, previous, current)
        (ML::Metal::PipelineCache.entry_count - before_fused).should eq(1)
        ENV["DRAMA3_TOWER_FFN_PAIR_FUSION"] = "0"
        fused.ffn_pair_fusion_enabled.should be_true
        observed.keys.any?(&.ends_with?(".w1")).should be_true
        observed.keys.any?(&.ends_with?(".w3")).should be_true
        {".w1", ".w3"}.each do |suffix|
          name = observed.keys.find(&.ends_with?(suffix)).not_nil!
          shape, buffer = observed[name]
          shape.should eq([1_i32, 1_i32, config.intermediate_size])
          buffer.count.should eq(config.intermediate_size)
        end

        context.readbacks.should be_empty
        context.graph_stats["completion_waits"].should eq(0_i64)
        outputs = [{expected.hidden, actual.hidden}] of Tuple(Resident::Buffer, Resident::Buffer)
        expected.head_logits.each_with_index { |logits, index| outputs << {logits, actual.head_logits[index]} }
        sentinel = -81.5_f32
        outputs.each do |reference, candidate|
          drama3_tower_pair_compare(
            context,
            drama3_tower_pair_guarded_copy(context, reference, sentinel),
            drama3_tower_pair_guarded_copy(context, candidate, sentinel),
            receipt,
            sentinel,
          )
        end

        invalid_context = Resident::Context.new
        before_allocations = invalid_context.allocation_stats[:allocations]
        before_graph = invalid_context.graph_stats.dup
        before_pipelines = ML::Metal::PipelineCache.entry_count
        ENV["DRAMA3_TOWER_FFN_PAIR_FUSION"] = "true"
        expect_raises(ArgumentError, /DRAMA3_TOWER_FFN_PAIR_FUSION must be 0 or 1/) do
          TowerResident.new(config, store, invalid_context, 2)
        end
        invalid_context.allocation_stats[:allocations].should eq(before_allocations)
        invalid_context.graph_stats.should eq(before_graph)
        ML::Metal::PipelineCache.entry_count.should eq(before_pipelines)
        invalid_context.readbacks.should be_empty

        counts = context.read(receipt, "terminal").map(&.unsafe_as(UInt32))
        counts.should eq([0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32])
        # Context#read also consumes the existing finite-status word.
        context.readbacks.keys.sort.should eq(["control", "terminal"])
        context.readbacks["control"].should eq(4_i64)
        context.readbacks["terminal"].should eq(24_i64)
        graph = context.graph_stats
        graph["submitted"].should eq(1_i64)
        graph["completed"].should eq(1_i64)
        graph["completion_waits"].should eq(1_i64)
        graph["pending"].should eq(0_i64)
      ensure
        context.try(&.finish)
        MetalBackend.reset!
        ML::Metal::PipelineCache.clear
      end
    end
  end
{% end %}
