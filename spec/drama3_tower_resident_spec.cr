require "spec"
require "../src/ml/audio/drama3/tower_resident"

include ML::Audio::Drama3

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private def drama3_resident_tower_config : TowerConfig
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

  private def drama3_resident_tower_store(config : TowerConfig) : WeightStore
    tensors = {} of String => Tensor
    seed = 1
    add_matrix = ->(name : String, rows : Int32, cols : Int32) do
      values = Array(Float32).new(rows * cols) do |i|
        (((i + seed * 3) % 11) - 5).to_f32 * 0.017_f32
      end
      tensors[name] = Tensor.new([rows, cols], values)
      seed += 1
    end
    add_vector = ->(name : String, size : Int32) do
      values = Array(Float32).new(size) do |i|
        (((i + seed * 2) % 7) - 3).to_f32 * 0.006_f32
      end
      tensors[name] = Tensor.new([size], values)
      seed += 1
    end

    add_matrix.call("tower.codebook_embeddings.weight", config.vocab_size * config.num_codebooks, config.dim)
    layer_count = config.shared_enrichment_layers + config.num_codebooks - 1
    layer_count.times do |index|
      layer_prefix = if index < config.shared_enrichment_layers
                       "tower.shared_layers.#{index}"
                     else
                       "tower.residual_layers.#{index - config.shared_enrichment_layers}"
                     end
      embedding_name = index < config.shared_enrichment_layers ? "semantic_embedding" : "codebook_embedding"
      add_matrix.call("#{layer_prefix}.#{embedding_name}.weight", config.vocab_size, config.dim)
      {"enorm", "hnorm", "final_layernorm"}.each do |name|
        add_vector.call("#{layer_prefix}.#{name}.weight", config.dim)
      end
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
        head = index - config.shared_enrichment_layers
        add_matrix.call("tower.codebook_heads.#{head}.weight", config.vocab_size, config.dim)
      end
    end
    WeightStore.new(tensors)
  end

  private def drama3_resident_tower_capture : Tuple(Hash(String, Tuple(Array(Int32), Vector)), TowerTraceCallback)
    traces = {} of String => Tuple(Array(Int32), Vector)
    capture = ->(name : String, shape : Array(Int32), values : Vector) do
      traces[name] = {shape, values.dup}
    end
    {traces, capture}
  end

  private def drama3_resident_tower_argmax(logits : Vector) : Int32
    best_index = 0
    1.upto(logits.size - 1) do |index|
      best_index = index if logits[index] > logits[best_index]
    end
    best_index.to_i32
  end

  private def drama3_resident_tower_assert_traces_match(
    expected : Hash(String, Tuple(Array(Int32), Vector)),
    actual : Hash(String, Tuple(Array(Int32), Vector)),
  ) : Nil
    actual.keys.sort.should eq(expected.keys.sort)
    expected.each do |name, (shape, values)|
      actual_shape, actual_values = actual[name]
      actual_shape.should eq(shape)
      actual_values.size.should eq(values.size)
      actual_values.zip(values).each do |a, b|
        (a - b).abs.should be < 2.0e-4_f32
      end
    end
  end

  private def drama3_embedding_word(dtype : String, row : Int32, column : Int32) : UInt32
    # The three rows selected by [-1, 7, 2] after clipping are all negative
    # zero. This makes a fused accumulator initialized from the first row fail
    # the exact-bit comparison against the legacy +0 accumulator.
    return dtype == "F32" ? 0x8000_0000_u32 : 0x0000_8000_u32 if row == 0 || row == 7 || row == 10

    case (row * 4 + column) % 8
    when 0 then 0x0000_0001_u32
    when 1 then dtype == "F32" ? 0x8000_0001_u32 : 0x0000_8001_u32
    when 2 then dtype == "F32" ? 0x3f00_0000_u32 : (dtype == "BF16" ? 0x0000_3f00_u32 : 0x0000_3800_u32)
    when 3 then dtype == "F32" ? 0xbf80_0000_u32 : (dtype == "BF16" ? 0x0000_bf80_u32 : 0x0000_bc00_u32)
    when 4 then 0_u32
    when 5 then dtype == "F32" ? 0x8000_0000_u32 : 0x0000_8000_u32
    when 6 then dtype == "F32" ? 0x3fc0_0000_u32 : (dtype == "BF16" ? 0x0000_3fc0_u32 : 0x0000_3e00_u32)
    else        dtype == "F32" ? 0xbe80_0000_u32 : (dtype == "BF16" ? 0x0000_be80_u32 : 0x0000_b400_u32)
    end
  end

  private def drama3_embedding_table_fixture(dtype : String)
    raise ArgumentError.new("unsupported embedding fixture dtype") unless dtype.in?("F32", "BF16", "F16")
    rows, columns = 12, 4
    bytes_per_element = dtype == "F32" ? 4 : 2
    byte_count = rows * columns * bytes_per_element
    header = {"embedding" => {"dtype" => dtype, "shape" => [rows, columns], "data_offsets" => [0_i64, byte_count.to_i64]}}.to_json
    header += " " * ((8 - header.bytesize % 8) % 8)
    file = File.tempfile("drama3-codebook-", ".safetensors") do |stream|
      stream.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
      stream << header
      rows.times do |row|
        columns.times do |column|
          word = drama3_embedding_word(dtype, row, column)
          if dtype == "F32"
            stream.write_bytes(word, IO::ByteFormat::LittleEndian)
          else
            stream.write_bytes(word.to_u16, IO::ByteFormat::LittleEndian)
          end
        end
      end
    end
    owner = MappedFile.new(file.path)
    header_size = IO::ByteFormat::LittleEndian.decode(UInt64, owner.slice(0_i64, 8_i64)).to_i64
    tensor = Tensor.new([rows, columns], dtype, owner.slice(8_i64 + header_size, byte_count.to_i64), owner)
    {file: file, owner: owner, matrix: Matrix.new(tensor)}
  end

  private def drama3_codebook_cases : Array(Tuple(Array(Float32), Int32, Int32))
    cases = [] of Tuple(Array(Float32), Int32, Int32)
    clipped_negative_zero = [11_f32, -1_f32, 7_f32, 2_f32]
    signed_subnormal_and_negative = [12_f32, 3_f32, 0_f32, 1_f32]
    nonsemantic = [9_f32, -1_f32, 7_f32, 2_f32]
    cases << {clipped_negative_zero, 0, 3}
    cases << {signed_subnormal_and_negative, 0, 3}
    cases << {nonsemantic, 0, 3}
    cases << {signed_subnormal_and_negative, 0, 1}
    cases << {signed_subnormal_and_negative, 1, 2}
    cases << {signed_subnormal_and_negative, 2, 3}
    cases << {signed_subnormal_and_negative, 1, 1}
    cases << {signed_subnormal_and_negative, 3, 3}
    cases
  end

  describe "DRAMA-3 resident tower" do
    it "matches fused codebook sums bitwise for mapped F32/BF16/F16 tables and queued depths 0 and 2" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      previous_fusion = ENV["DRAMA3_TOWER_EMBEDDING_FUSION"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?
        cases = drama3_codebook_cases
        {"F32", "BF16", "F16"}.each do |dtype|
          fixture = drama3_embedding_table_fixture(dtype)
          begin
            table = fixture[:matrix]
            table_buffer = MetalBackend.tensor_buffer(table.tensor, true)
            table_buffer.offset.should be > 0
            {"0", "2"}.each do |depth|
              ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth
              ENV["DRAMA3_TOWER_EMBEDDING_FUSION"] = "0"
              context = Resident::Context.new
              legacy_outputs = [] of Resident::Buffer
              legacy_allocations = [] of Int64
              fused_outputs = [] of Resident::Buffer
              fused_allocations = [] of Int64
              begin
                cases.each do |values, first_codebook, end_codebook|
                  frame = context.upload(values)
                  before = context.allocation_stats[:allocations]
                  legacy_outputs << context.codebook_sum(table, frame, first_codebook, end_codebook,
                    4, 10, 13, RESIDENT_TOWER_SOURCE)
                  legacy_allocations << context.allocation_stats[:allocations] - before
                end
                graph_before = context.graph_stats
                reads_before = context.readbacks.dup
                context.submit
                after_legacy = context.graph_stats
                context.readbacks.should eq(reads_before)
                if depth == "0"
                  after_legacy["completion_waits"].should eq(graph_before["completion_waits"] + 1)
                else
                  after_legacy["completion_waits"].should eq(graph_before["completion_waits"])
                  after_legacy["pending"].should eq(1)
                end

                cases.each do |values, first_codebook, end_codebook|
                  frame = context.upload(values)
                  before = context.allocation_stats[:allocations]
                  fused_outputs << context.codebook_sum(table, frame, first_codebook, end_codebook,
                    4, 10, 13, RESIDENT_TOWER_SOURCE, true)
                  fused_allocations << context.allocation_stats[:allocations] - before
                end
                context.readbacks.should eq(reads_before)
                context.submit
                queued = context.graph_stats
                if depth == "0"
                  queued["completion_waits"].should eq(graph_before["completion_waits"] + 2)
                  queued["pending"].should eq(0)
                else
                  queued["completion_waits"].should eq(graph_before["completion_waits"])
                  queued["pending"].should eq(2)
                end
                context.finish
                drained = context.graph_stats
                drained["submitted"].should eq(graph_before["submitted"] + 2)
                drained["completed"].should eq(graph_before["completed"] + 2)
                drained["completion_waits"].should eq(graph_before["completion_waits"] + 2)
                drained["pending"].should eq(0)

                cases.each_with_index do |(_, first_codebook, end_codebook), index|
                  codebooks = end_codebook - first_codebook
                  legacy_allocations[index].should eq(codebooks == 0 ? 1_i64 : (3 * codebooks + 2).to_i64)
                  fused_allocations[index].should eq(1_i64)
                  legacy = context.read(legacy_outputs[index], "diagnostic")
                  fused = context.read(fused_outputs[index], "diagnostic")
                  fused.map(&.unsafe_as(UInt32)).should eq(legacy.map(&.unsafe_as(UInt32)))
                end
                context.graph_stats.should eq(drained)
                context.readbacks["diagnostic"].should eq((cases.size * 2 * 4 * 4).to_i64)
                context.readbacks["terminal"].should eq(0_i64)
                context.readbacks["control"].should eq(0_i64)
              ensure
                context.finish
                MetalBackend.reset!
              end
            end
          ensure
            fixture[:file].delete
          end
        end
      ensure
        MetalBackend.reset!
        {"DRAMA3_BACKEND"                 => previous_backend,
         "DRAMA3_COGNIGRAPH_MAX_INFLIGHT" => previous_depth,
         "DRAMA3_TOWER_EMBEDDING_FUSION"  => previous_fusion}.each do |name, value|
          if value
            ENV[name] = value
          else
            ENV.delete(name)
          end
        end
      end
    end

    it "keeps affine-quantized codebook tables on the legacy path when fusion is requested" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "0"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?
        fixture = drama3_embedding_table_fixture("F32")
        begin
          owner = fixture[:owner]
          packed = Tensor.new([12, 1], "U32", owner.slice(8_i64 + IO::ByteFormat::LittleEndian.decode(UInt64, owner.slice(0_i64, 8_i64)).to_i64, 12_i64 * 4), owner)
          scales = Tensor.new([12, 1], Array(Float32).new(12) { |index| (index + 1).to_f32 / 17_f32 })
          biases = Tensor.new([12, 1], Array(Float32).new(12) { |index| (index - 5).to_f32 / 23_f32 })
          quantized = Matrix.new(packed, scales, biases, 4, 8)
          cases = drama3_codebook_cases
          legacy_outputs = [] of Resident::Buffer
          requested_outputs = [] of Resident::Buffer
          context = Resident::Context.new
          begin
            cases.each do |values, first_codebook, end_codebook|
              frame = context.upload(values)
              legacy_outputs << context.codebook_sum(quantized, frame, first_codebook, end_codebook,
                4, 10, 13, RESIDENT_TOWER_SOURCE)
            end
            waits_before = context.graph_stats["completion_waits"]
            context.submit
            cases.each do |values, first_codebook, end_codebook|
              frame = context.upload(values)
              before = context.allocation_stats[:allocations]
              requested_outputs << context.codebook_sum(quantized, frame, first_codebook, end_codebook,
                4, 10, 13, RESIDENT_TOWER_SOURCE, true)
              allocations = context.allocation_stats[:allocations] - before
              codebooks = end_codebook - first_codebook
              allocations.should eq(codebooks == 0 ? 1_i64 : (3 * codebooks + 2).to_i64)
            end
            context.readbacks["terminal"].should eq(0_i64)
            context.readbacks["diagnostic"].should eq(0_i64)
            context.submit
            context.graph_stats["completion_waits"].should eq(waits_before + 2)
            context.finish
            cases.each_with_index do |_, index|
              context.read(requested_outputs[index], "diagnostic").map(&.unsafe_as(UInt32))
                .should eq(context.read(legacy_outputs[index], "diagnostic").map(&.unsafe_as(UInt32)))
            end
            context.readbacks["control"].should eq(0_i64)
          ensure
            context.finish
            MetalBackend.reset!
          end
        ensure
          fixture[:file].delete
        end
      ensure
        MetalBackend.reset!
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

    it "defaults to legacy codebook dispatch and opt-in fusion preserves output bits" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      previous_fusion = ENV["DRAMA3_TOWER_EMBEDDING_FUSION"]?
      contexts = [] of Resident::Context
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "0"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?
        config = drama3_resident_tower_config
        store = drama3_resident_tower_store(config)
        outputs = [] of Vector
        allocations = [] of Int64
        {"0", "1"}.each do |fusion|
          ENV["DRAMA3_TOWER_EMBEDDING_FUSION"] = fusion
          context = Resident::Context.new
          contexts << context
          resident = TowerResident.new(config, store, context, 2)
          frame = resident.upload_frame([12, -1, 7, 2])
          text = context.upload([0.1_f32, -0.2_f32, 0.3_f32, 0.4_f32])
          before = context.allocation_stats[:allocations]
          output = resident.embed_backbone(frame, text)
          allocations << context.allocation_stats[:allocations] - before
          outputs << context.read(output, "terminal")
          context.graph_stats["completion_waits"].should eq(1)
          context.readbacks["diagnostic"].should eq(0_i64)
        end
        allocations.should eq([6_i64, 2_i64])
        outputs[1].map(&.unsafe_as(UInt32)).should eq(outputs[0].map(&.unsafe_as(UInt32)))
      ensure
        contexts.each(&.finish)
        MetalBackend.reset!
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
        if previous_fusion
          ENV["DRAMA3_TOWER_EMBEDDING_FUSION"] = previous_fusion
        else
          ENV.delete("DRAMA3_TOWER_EMBEDDING_FUSION")
        end
      end
    end

    it "matches tiny CPU reference traces through backbone embedding, teacher force, and sampled frame" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        config = drama3_resident_tower_config
        store = drama3_resident_tower_store(config)
        cpu = Tower.new(config, store)
        context = Resident::Context.new
        resident = TowerResident.new(config, store, context, 2)

        frame = [12_i32, 2_i32, 1_i32, 0_i32]
        text = [0.1_f32, -0.2_f32, 0.3_f32, 0.4_f32]
        expected_embed, expected_embed_trace = drama3_resident_tower_capture
        expected_text = cpu.embed_backbone([frame], [text], expected_embed_trace)
        actual_embed, actual_embed_trace = drama3_resident_tower_capture
        actual_text = context.read(
          resident.embed_backbone(resident.upload_frame(frame), context.upload(text), actual_embed_trace),
          "terminal",
        )
        drama3_resident_tower_assert_traces_match(expected_embed, actual_embed)
        actual_text.zip(expected_text[0]).each { |a, b| (a - b).abs.should be < 2.0e-4_f32 }

        hidden = [0.1_f32, -0.2_f32, 0.3_f32, 0.4_f32]
        previous = [11_i32, 1_i32, 2_i32, 3_i32]
        current = [12_i32, 2_i32, 1_i32, 0_i32]
        expected_prefill, expected_prefill_trace = drama3_resident_tower_capture
        expected_output = cpu.teacher_force(hidden, previous, current, cpu.make_cache, expected_prefill_trace, "tower.prefill.row0")
        actual_prefill, actual_prefill_trace = drama3_resident_tower_capture
        actual_output = resident.teacher_force(
          resident.upload_hidden(hidden), resident.upload_frame(previous), resident.upload_frame(current),
          actual_prefill_trace, "tower.prefill.row0",
        )
        drama3_resident_tower_assert_traces_match(expected_prefill, actual_prefill)
        context.read(actual_output.hidden, "terminal").zip(expected_output.hidden).each do |a, b|
          (a - b).abs.should be < 2.0e-4_f32
        end
        actual_output.head_logits.each_with_index do |logits, index|
          context.read(logits, "terminal").zip(expected_output.head_logits[index]).each do |a, b|
            (a - b).abs.should be < 2.0e-4_f32
          end
        end

        next_hidden = [0.4_f32, 0.1_f32, -0.3_f32, 0.2_f32]
        decode_previous = current
        decode_main = 13_i32
        expected_decode, expected_decode_trace = drama3_resident_tower_capture
        # Reuse the cache populated by teacher forcing in both implementations.
        cpu_cached = Tower.new(config, store)
        cpu_cached_cache = cpu_cached.make_cache
        cpu_cached.teacher_force(hidden, previous, current, cpu_cached_cache)
        expected_sample = cpu_cached.sample_frame(decode_main, next_hidden, decode_previous, cpu_cached_cache, expected_decode_trace, "tower.decode.row0") do |logits|
          drama3_resident_tower_argmax(logits)
        end

        actual_decode, actual_decode_trace = drama3_resident_tower_capture
        draw = context.upload([0.0_f32])
        actual_frame = resident.sample_frame(
          context.upload([decode_main.to_f32]), resident.upload_hidden(next_hidden), resident.upload_frame(decode_previous),
          actual_decode_trace, "tower.decode.row0",
        ) do |logits|
          context.sample(logits, draw, 1.0, 1, 1.0)
        end
        drama3_resident_tower_assert_traces_match(expected_decode, actual_decode)
        actual_codes = context.read(actual_frame, "terminal").map(&.to_i32)
        actual_codes.should eq([decode_main] + expected_sample.codes)
        resident.offset.should eq(2)
        context.readbacks.keys.all? { |purpose| purpose.in?("terminal", "control", "diagnostic") }.should be_true
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
