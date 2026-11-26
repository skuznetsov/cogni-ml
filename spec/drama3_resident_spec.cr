require "spec"
require "../src/ml/audio/drama3/resident"
require "../src/ml/audio/drama3/backbone_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private def drama3_backbone_attention_values(count : Int32, seed : Int32) : Array(Float32)
    Array(Float32).new(count) do |index|
      (((index * 19 + seed * 23 + index // 7 * 11) % 61) - 30).to_f32 / 64_f32
    end
  end

  private def drama3_backbone_attention_oracle(q : Array(Float32), k : Array(Float32),
                                               v : Array(Float32), gate : Array(Float32),
                                               heads : Int32, kv_heads : Int32, dim : Int32,
                                               length : Int32) : Array(Float32)
    stride = kv_heads * dim
    scale = Math.sqrt(dim.to_f64)
    output = Array(Float32).new(heads * dim, 0_f32)
    heads.times do |head|
      kv_head = head // (heads // kv_heads)
      scores = Array(Float32).new(length, 0_f32)
      maximum = -Float32::INFINITY
      length.times do |time|
        dot = 0_f32
        dim.times do |channel|
          dot += q[head * dim + channel] * k[time * stride + kv_head * dim + channel]
        end
        score = dot / scale.to_f32
        scores[time] = score
        maximum = score if score > maximum
      end
      denominator = 0_f32
      length.times do |time|
        scores[time] = Math.exp((scores[time] - maximum).to_f64).to_f32
        denominator += scores[time]
      end
      dim.times do |channel|
        sum = 0_f32
        length.times do |time|
          sum += (scores[time] / denominator) * v[time * stride + kv_head * dim + channel]
        end
        value = gate[head * dim + channel]
        sigmoid = if value >= 0
                    1_f32 / (1_f32 + Math.exp(-value.to_f64).to_f32)
                  else
                    Math.exp(value.to_f64).to_f32 / (1_f32 + Math.exp(value.to_f64).to_f32)
                  end
        output[head * dim + channel] = sum * sigmoid
      end
    end
    output
  end

  private def drama3_backbone_attention_enqueue(context : ML::Audio::Drama3::Resident::Context,
                                                kernel : String,
                                                q : ML::Audio::Drama3::Resident::Buffer,
                                                k : ML::Audio::Drama3::Resident::Buffer,
                                                v : ML::Audio::Drama3::Resident::Buffer,
                                                gate : ML::Audio::Drama3::Resident::Buffer,
                                                scratch : ML::Audio::Drama3::Resident::Buffer?,
                                                heads : Int32, kv_heads : Int32, dim : Int32,
                                                length : Int32, capacity : Int32) : ML::Audio::Drama3::Resident::Buffer
    output = context.alloc(heads * dim)
    encoder = context.encoder
    encoder.set_pipeline(ML::Metal::PipelineCache.get_or_compile(kernel, ML::Audio::Drama3::BackboneResident::SOURCE))
    [q, k, v, gate, output].each_with_index do |buffer, index|
      encoder.set_buffer(buffer.buffer, index, offset: buffer.offset)
    end
    params = [heads.to_u64, kv_heads.to_u64, dim.to_u64, length.to_u64]
    if kernel == "dr3_full_attention_wide"
      score_buffer = scratch.not_nil!
      encoder.set_buffer(score_buffer.buffer, 5, offset: score_buffer.offset)
      params << capacity.to_u64
    end
    encoder.set_bytes(params.to_unsafe.as(Void*), (params.size * 8).to_i32, params.size == 4 ? 5 : 6)
    encoder.dispatch_threadgroups({heads, 1, 1}, {64, 1, 1})
    output
  end

  private DRAMA3_EXPERT_PACKED_ROWS_RECEIPT_SOURCE = ML::Audio::Drama3::Resident::SOURCE + <<-METAL
    inline void dr3_resident_packed_rows_record(device atomic_uint* receipt,
                                                uint bit_slot, uint finite_slot,
                                                float reference, float actual) {
        if (as_type<uint>(reference) != as_type<uint>(actual)) {
            atomic_fetch_add_explicit(&receipt[bit_slot], 1u, memory_order_relaxed);
        }
        if (!isfinite(reference) || !isfinite(actual)) {
            atomic_fetch_add_explicit(&receipt[finite_slot], 1u, memory_order_relaxed);
        }
    }

    kernel void dr3_resident_packed_rows_compare(
        device const float* reference [[buffer(0)]], device const float* actual [[buffer(1)]],
        device atomic_uint* receipt [[buffer(2)]], constant ulong* params [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
        if (ulong(index) >= params[0]) return;
        dr3_resident_packed_rows_record(receipt, 0, 1, reference[index], actual[index]);
    }

    kernel void dr3_resident_packed_rows_negative_control(
        device atomic_uint* receipt [[buffer(0)]], uint index [[thread_position_in_grid]]) {
        if (index != 0) return;
        float one = 1.0f;
        float one_ulp_up = as_type<float>(as_type<uint>(one) + 1u);
        dr3_resident_packed_rows_record(receipt, 2, 3, one, one_ulp_up);
        dr3_resident_packed_rows_record(receipt, 2, 3, NAN, 0.0f);
    }
    METAL

  private def drama3_packed_rows_with_device(&block : -> T) : T forall T
    keys = ["DRAMA3_BACKEND", "DRAMA3_COGNIGRAPH_MAX_INFLIGHT", "DRAMA3_METAL_TIMELINE",
            "DRAMA3_EXPERT_THREADS", "DRAMA3_EXPERT_PACKED_ROWS", "DRAMA3_PROFILE"]
    previous = keys.to_h { |key| {key, ENV[key]?} }
    begin
      ENV["DRAMA3_BACKEND"] = "metal"
      ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "0"
      ENV["DRAMA3_METAL_TIMELINE"] = "0"
      ENV["DRAMA3_EXPERT_THREADS"] = "64"
      ENV["DRAMA3_EXPERT_PACKED_ROWS"] = "0"
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

  private def drama3_packed_rows_fixture(path : String, rows : Int32, groups : Int32, bits : Int32,
                                         experts : Int32 = 3)
    group_size = 64
    cols = groups * group_size
    values_per_word = 32 // bits
    words = experts * rows * cols // values_per_word
    io = IO::Memory.new
    io.write(Bytes.new(22, 0_u8))
    words.times do |index|
      word = 0_u32
      values_per_word.times do |lane|
        value = (index * 13 + lane * 7 + index // 11) % (1 << bits)
        word |= value.to_u32 << (lane * bits)
      end
      io.write_bytes(word, IO::ByteFormat::LittleEndian)
    end
    File.write(path, io.to_slice)

    owner = ML::Audio::Drama3::MappedFile.new(path)
    packed = ML::Audio::Drama3::Tensor.new([experts, rows, cols // values_per_word], "U32",
      owner.slice(22_i64, words.to_i64 * 4_i64), owner)
    affine_count = experts * rows * groups
    scales = ML::Audio::Drama3::Tensor.new([experts, rows, groups],
      Array(Float32).new(affine_count) { |index| ((index % 7 + 1).to_f32 * 0.003_f32) })
    biases = ML::Audio::Drama3::Tensor.new([experts, rows, groups],
      Array(Float32).new(affine_count) { |index| ((index % 9 - 4).to_f32 * 0.002_f32) })
    {matrix: ML::Audio::Drama3::Matrix.new(packed, scales, biases, bits, group_size), owner: owner}
  end

  private def drama3_packed_rows_input(cols : Int32, ranks : Int32, per_rank : Bool) : Array(Float32)
    count = cols * (per_rank ? ranks : 1)
    Array(Float32).new(count) { |index| ((index * 17 + index // 5) % 37 - 18).to_f32 / 23_f32 }
  end

  private def drama3_packed_rows_receipt_accepts?(bit_mismatches : UInt32, nonfinite : UInt32,
                                                  expected_nonfinite : UInt32 = 0_u32) : Bool
    bit_mismatches == 0_u32 && nonfinite == expected_nonfinite
  end

  # The old and opted-in Contexts have distinct buffer owners. The reference
  # Context is drained first; this test-only encoder then compares its still-
  # live Metal output directly with the candidate output, reading only four
  # compact counters after the candidate's existing terminal drain.
  private def drama3_packed_rows_gpu_receipt(context : ML::Audio::Drama3::Resident::Context,
                                             reference : ML::Audio::Drama3::Resident::Buffer,
                                             actual : ML::Audio::Drama3::Resident::Buffer,
                                             expected_nonfinite : UInt32 = 0_u32) : Nil
    reference.count.should eq(actual.count)
    receipt = context.upload(Array(Float32).new(4, 0_f32))
    encoder = context.encoder
    encoder.set_pipeline(ML::Metal::PipelineCache.get_or_compile(
      "dr3_resident_packed_rows_compare", DRAMA3_EXPERT_PACKED_ROWS_RECEIPT_SOURCE,
    ))
    encoder.set_buffer(reference.buffer, 0, offset: reference.offset)
    encoder.set_buffer(actual.buffer, 1, offset: actual.offset)
    encoder.set_buffer(receipt.buffer, 2, offset: receipt.offset)
    params = [actual.count.to_u64]
    encoder.set_bytes(params.to_unsafe.as(Void*), 8, 3)
    encoder.dispatch_1d(actual.count, 256)

    encoder.set_pipeline(ML::Metal::PipelineCache.get_or_compile(
      "dr3_resident_packed_rows_negative_control", DRAMA3_EXPERT_PACKED_ROWS_RECEIPT_SOURCE,
    ))
    encoder.set_buffer(receipt.buffer, 0, offset: receipt.offset)
    encoder.dispatch_1d(1, 1)

    counts = context.read(receipt, "terminal").map(&.unsafe_as(UInt32))
    counts.should eq([0_u32, expected_nonfinite, 2_u32, 1_u32])
    drama3_packed_rows_receipt_accepts?(counts[0], counts[1], expected_nonfinite).should be_true
    # The injected one-ULP mismatch and NaN must be rejected by the same
    # compact acceptance predicate used for real old/new result pairs.
    drama3_packed_rows_receipt_accepts?(counts[2], counts[3]).should be_false
    context.readbacks.keys.should eq(["terminal"])
    context.readbacks["terminal"].should eq(16_i64)
    graph = context.graph_stats
    graph["submitted"].should eq(1_i64)
    graph["completed"].should eq(1_i64)
    graph["completion_waits"].should eq(1_i64)
    graph["pending"].should eq(0_i64)
  end

  private def drama3_packed_rows_compare_case(matrix : ML::Audio::Drama3::Matrix,
                                              inputs : Array(Float32), ids : Array(Float32),
                                              per_rank_input : Bool, requested_threads : String = "64",
                                              reduction : String = "native", batched : Bool = true,
                                              expect_packed_rows : Bool = false,
                                              expected_nonfinite : UInt32 = 0_u32) : Nil
    baseline_context : ML::Audio::Drama3::Resident::Context? = nil
    candidate_context : ML::Audio::Drama3::Resident::Context? = nil
    begin
      ENV["DRAMA3_EXPERT_THREADS"] = requested_threads
      ENV["DRAMA3_EXPERT_PACKED_ROWS"] = "0"
      baseline_context = ML::Audio::Drama3::Resident::Context.new
      baseline_context.not_nil!.expert_packed_rows_enabled.should be_false
      baseline_ids = baseline_context.not_nil!.upload(ids)
      baseline_input = baseline_context.not_nil!.upload(inputs)
      baseline_pipeline_count = nil.as(Int32?)
      if batched && reduction == "native"
        ML::Metal::PipelineCache.get_or_compile("dr3_experts", ML::Audio::Drama3::Resident::SOURCE)
        baseline_pipeline_count = ML::Metal::PipelineCache.entry_count
      end
      # The Context's selector is frozen at construction, even if the process
      # environment changes before its first operator dispatch.
      ENV["DRAMA3_EXPERT_PACKED_ROWS"] = "1"
      baseline = if batched
                   baseline_context.not_nil!.experts(matrix, baseline_input, baseline_ids,
                     per_rank_input: per_rank_input, reduction: reduction)
                 else
                   baseline_context.not_nil!.expert(matrix, baseline_input, baseline_ids, 1,
                     reduction: reduction)
                 end
      if count = baseline_pipeline_count
        ML::Metal::PipelineCache.entry_count.should eq(count)
      end
      baseline_context.not_nil!.finish
      baseline_context.not_nil!.readbacks.should be_empty
      baseline_context.not_nil!.graph_stats["submitted"].should eq(1_i64)
      baseline_context.not_nil!.graph_stats["completion_waits"].should eq(1_i64)

      ENV["DRAMA3_EXPERT_PACKED_ROWS"] = "1"
      candidate_context = ML::Audio::Drama3::Resident::Context.new
      candidate_context.not_nil!.expert_packed_rows_enabled.should be_true
      candidate_context.not_nil!.expert_threads.should eq(requested_threads.to_i)
      candidate_ids = candidate_context.not_nil!.upload(ids)
      candidate_input = candidate_context.not_nil!.upload(inputs)
      candidate_pipeline_count = ML::Metal::PipelineCache.entry_count
      ENV["DRAMA3_EXPERT_PACKED_ROWS"] = "0"
      candidate = if batched
                    candidate_context.not_nil!.experts(matrix, candidate_input, candidate_ids,
                      per_rank_input: per_rank_input, reduction: reduction)
                  else
                    candidate_context.not_nil!.expert(matrix, candidate_input, candidate_ids, 1,
                      reduction: reduction)
                  end
      # The baseline has already populated only its selected operator pipeline.
      # A single new cache entry therefore proves the candidate dispatch chose
      # the packed kernel; zero proves this case stayed on the existing path.
      ML::Metal::PipelineCache.entry_count.should eq(candidate_pipeline_count + (expect_packed_rows ? 1 : 0))
      drama3_packed_rows_gpu_receipt(candidate_context.not_nil!, baseline, candidate, expected_nonfinite)
    ensure
      candidate_context.try(&.abort)
      baseline_context.try(&.abort)
      ML::Audio::Drama3::MetalBackend.reset!
    end
  end

  private def drama3_packed_rows_compare_geometry(rows : Int32, groups : Int32, bits : Int32,
                                                  per_rank_input : Bool = false,
                                                  requested_threads : String = "64",
                                                  reduction : String = "native",
                                                  batched : Bool = true,
                                                  expect_packed_rows : Bool = false,
                                                  invalid_ids : Bool = false) : Nil
    path = File.tempname("drama3-packed-rows", ".bin")
    begin
      # Isolate the selected operator pipeline so the cache delta below tests
      # dispatch choice, including fallback cases, rather than output alone.
      ML::Metal::PipelineCache.clear
      fixture = drama3_packed_rows_fixture(path, rows, groups, bits)
      cols = groups * 64
      inputs = drama3_packed_rows_input(cols, 3, per_rank_input)
      ids = invalid_ids ? [-1_f32, 3_f32, 0.5_f32] : [2_f32, 0_f32, 2_f32]
      drama3_packed_rows_compare_case(fixture[:matrix], inputs, ids, per_rank_input,
        requested_threads, reduction, batched, expect_packed_rows, invalid_ids ? 15_u32 : 0_u32)
    ensure
      ML::Audio::Drama3::MetalBackend.reset!
      ML::Metal::PipelineCache.clear
      File.delete(path) if File.exists?(path)
    end
  end

  describe ML::Audio::Drama3::Resident do
    it "freezes and strictly validates the packed-row selector without submitting GPU work" do
      drama3_packed_rows_with_device do
        ENV.delete("DRAMA3_EXPERT_PACKED_ROWS")
        default_context = ML::Audio::Drama3::Resident::Context.new
        default_context.expert_packed_rows_enabled.should be_false
        ENV["DRAMA3_EXPERT_PACKED_ROWS"] = "1"
        default_context.expert_packed_rows_enabled.should be_false
        default_context.graph_stats["submitted"].should eq(0_i64)
        default_context.readbacks.should be_empty

        enabled_context = ML::Audio::Drama3::Resident::Context.new
        enabled_context.expert_packed_rows_enabled.should be_true
        ENV["DRAMA3_EXPERT_PACKED_ROWS"] = "0"
        enabled_context.expert_packed_rows_enabled.should be_true
        enabled_context.graph_stats["submitted"].should eq(0_i64)
        enabled_context.readbacks.should be_empty

        ["true", ""].each do |invalid|
          ENV["DRAMA3_EXPERT_PACKED_ROWS"] = invalid
          expect_raises(ArgumentError, /DRAMA3_EXPERT_PACKED_ROWS must be 0 or 1/) do
            ML::Audio::Drama3::Resident::Context.new
          end
        end
      end
    end

    it "matches batched native expert output through packed rows across supported and fallback geometry" do
      drama3_packed_rows_with_device do
        # Odd final row slots are exercised for both row packing layouts:
        # groups=32 shares the input across ranks; groups=8 models down
        # projection's per-rank input. Both Q4 and Q8 are covered.
        [4, 8].each do |bits|
          drama3_packed_rows_compare_geometry(5, 32, bits, expect_packed_rows: true)
          drama3_packed_rows_compare_geometry(5, 8, bits, per_rank_input: true, expect_packed_rows: true)
        end

        drama3_packed_rows_compare_geometry(5, 1, 4, expect_packed_rows: true)

        # Every unsupported group boundary has an exact old/new GPU receipt;
        # Q4/Q8 alternate so both packed encodings traverse the fallback guard.
        {3 => 4, 33 => 8, 65 => 4}.each do |groups, bits|
          drama3_packed_rows_compare_geometry(5, groups, bits)
        end

        # A user-requested 32-thread launch remains on the established native
        # kernel even when the selector is enabled and groups would otherwise pack.
        drama3_packed_rows_compare_geometry(5, 32, 8, requested_threads: "32")

        # Invalid routed IDs must remain exact NaNs in the eligible packed
        # kernel, including the odd final row slots and per-rank input layout.
        drama3_packed_rows_compare_geometry(5, 8, 4, per_rank_input: true,
          expect_packed_rows: true, invalid_ids: true)
      end
    end

    it "keeps scalar and MLX Q4 expert reductions outside the packed-row dispatch" do
      drama3_packed_rows_with_device do
        # Scalar expert is an unchanged control under the opted-in selector.
        drama3_packed_rows_compare_geometry(5, 32, 4, batched: false)

        # The batched MLX-Q4 early return must precede the native packed-row
        # selection. Its aligned geometry satisfies the separate MLX kernel.
        drama3_packed_rows_compare_geometry(8, 8, 4, reduction: "mlx_q4")
      end
    end

    it "batches affine expert ranks exactly with shared or per-rank inputs and unaligned weights" do
      path = File.tempname("drama3-resident-experts", ".bin")
      begin
        [4, 8].each do |bits|
          [1, 3, 65].each do |groups|
            rows, experts, group = 5, 3, 64
            cols, per = groups * group, 32 // bits
            io = IO::Memory.new
            io.write(Bytes.new(22, 0_u8))
            words = experts * rows * cols // per
            words.times do |i|
              word = 0_u32
              per.times { |lane| word |= (((i * 13 + lane * 7) % (1 << bits)).to_u32 << (lane * bits)) }
              io.write_bytes(word, IO::ByteFormat::LittleEndian)
            end
            File.write(path, io.to_slice)
            owner = ML::Audio::Drama3::MappedFile.new(path)
            packed = ML::Audio::Drama3::Tensor.new([experts, rows, cols // per], "U32", owner.slice(22_i64, words.to_i64 * 4), owner)
            scales = ML::Audio::Drama3::Tensor.new([experts, rows, groups], Array(Float32).new(experts * rows * groups) { |i| (i % 9 + 1).to_f32 * 0.00073_f32 })
            biases = ML::Audio::Drama3::Tensor.new([experts, rows, groups], Array(Float32).new(experts * rows * groups) { |i| (i % 11 - 5).to_f32 * 0.017_f32 })
            matrix = ML::Audio::Drama3::Matrix.new(packed, scales, biases, bits, group)
            [false, true].each do |per_rank|
              ctx = ML::Audio::Drama3::Resident::Context.new
              ids = ctx.upload([2_f32, 0_f32, 2_f32])
              input = ctx.upload(Array(Float32).new(cols * (per_rank ? 3 : 1)) { |i| (i % 17 - 8).to_f32 / 7_f32 })
              batched = ctx.experts(matrix, input, ids, per_rank)
              3.times do |rank|
                x = per_rank ? input.slice(rank * cols, cols) : input
                expected = ctx.read(ctx.expert(matrix, x, ids, rank), "terminal")
                actual = ctx.read(batched.slice(rank * rows, rows), "terminal")
                actual.map(&.unsafe_as(UInt32)).should eq(expected.map(&.unsafe_as(UInt32)))
              end
              bad = ctx.experts(matrix, input, ctx.upload([-1_f32, 3_f32, 0.5_f32]), per_rank)
              expect_raises(ArgumentError, /non-finite resident terminal/) { ctx.read(bad, "terminal") }
              expect_raises(ArgumentError, /input width/) { ctx.experts(matrix, ctx.upload([1_f32]), ids, per_rank) }
            end
            # All commands have completed; release mapped views before the
            # next case rewrites this temporary backing file.
            ML::Audio::Drama3::MetalBackend.reset!
          end
        end
      ensure
        ML::Audio::Drama3::MetalBackend.reset!
        File.delete(path) if File.exists?(path)
      end
    end

    it "keeps native affine expert group 32/33 boundaries exact for either selector" do
      previous_threads = ENV["DRAMA3_EXPERT_THREADS"]?
      path = File.tempname("drama3-resident-expert-width", ".bin")
      begin
        ["32", "64"].each do |requested_threads|
          ENV["DRAMA3_EXPERT_THREADS"] = requested_threads
          [4, 8].each do |bits|
            [32, 33].each do |groups|
              rows, experts, group = 1, 2, 64
              cols, per = groups * group, 32 // bits
              words = Array(UInt32).new(experts * rows * cols // per, 0_u32)
              scale_values = Array(Float32).new(experts * rows * groups, 0_f32)
              experts.times do |expert|
                word_index = expert * rows * (cols // per) + (groups - 1) * (group // per)
                words[word_index] = 1_u32
                scale_values[expert * groups + groups - 1] = expert == 0 ? 2_f32 : 4_f32
              end

              io = IO::Memory.new
              io.write(Bytes.new(22, 0_u8))
              words.each { |word| io.write_bytes(word, IO::ByteFormat::LittleEndian) }
              File.write(path, io.to_slice)
              owner = ML::Audio::Drama3::MappedFile.new(path)
              packed = ML::Audio::Drama3::Tensor.new([experts, rows, cols // per], "U32",
                owner.slice(22_i64, words.size.to_i64 * 4_i64), owner)
              scales = ML::Audio::Drama3::Tensor.new([experts, rows, groups], scale_values)
              biases = ML::Audio::Drama3::Tensor.new([experts, rows, groups], Array(Float32).new(scale_values.size, 0_f32))
              matrix = ML::Audio::Drama3::Matrix.new(packed, scales, biases, bits, group)

              ctx = ML::Audio::Drama3::Resident::Context.new
              ctx.expert_threads.should eq(requested_threads.to_i)
              ids = ctx.upload([1_f32, 0_f32])
              last_col = (groups - 1) * group

              shared_values = Array(Float32).new(cols, 0_f32)
              shared_values[last_col] = 3_f32
              shared = ctx.upload(shared_values)
              shared_batch = ctx.experts(matrix, shared, ids)
              shared_scalar = ctx.expert(matrix, shared, ids, 0)
              ctx.read(shared_scalar, "terminal").map(&.unsafe_as(UInt32)).should eq([12_f32].map(&.unsafe_as(UInt32)))
              ctx.read(shared_batch, "terminal").map(&.unsafe_as(UInt32)).should eq([12_f32, 6_f32].map(&.unsafe_as(UInt32)))

              per_rank_values = Array(Float32).new(cols * 2, 0_f32)
              per_rank_values[last_col] = 3_f32
              per_rank_values[cols + last_col] = 5_f32
              per_rank = ctx.upload(per_rank_values)
              per_rank_batch = ctx.experts(matrix, per_rank, ids, true)
              per_rank_scalar_input = ctx.upload(Array(Float32).new(cols) { |index| per_rank_values[index] })
              per_rank_scalar = ctx.expert(matrix, per_rank_scalar_input, ids, 0)
              ctx.read(per_rank_scalar, "terminal").map(&.unsafe_as(UInt32)).should eq([12_f32].map(&.unsafe_as(UInt32)))
              ctx.read(per_rank_batch, "terminal").map(&.unsafe_as(UInt32)).should eq([12_f32, 10_f32].map(&.unsafe_as(UInt32)))

              # Complete GPU use before rewriting the mapped fixture for the next case.
              ML::Audio::Drama3::MetalBackend.reset!
            end
          end
        end
      ensure
        ML::Audio::Drama3::MetalBackend.reset!
        if previous_threads
          ENV["DRAMA3_EXPERT_THREADS"] = previous_threads
        else
          ENV.delete("DRAMA3_EXPERT_THREADS")
        end
        File.delete(path) if File.exists?(path)
      end
    end

    it "preserves rank-ordered expert accumulation under cancellation" do
      ctx = ML::Audio::Drama3::Resident::Context.new
      ranks, width = 8, 3
      terms = [1e20_f32, -1e20_f32, 3_f32, 2_f32, -7_f32, 9_f32, -2_f32, 1_f32]
      values = ctx.upload(Array(Float32).new(ranks * width) { |i| terms[i // width] * (i % width + 1).to_f32 })
      routing = ctx.upload(Array(Float32).new(ranks, 0_f32) + Array(Float32).new(ranks, 1_f32))
      serial = ctx.zeros(width)
      ranks.times do |rank|
        ctx.dispatch("dr3_weighted_add", ML::Audio::Drama3::BackboneResident::SOURCE,
          [serial, values.slice(rank * width, width), routing.slice(ranks + rank, 1)], [width.to_u64, 0_u64], width)
      end
      batched = ctx.alloc(width)
      ctx.dispatch("dr3_expert_sum", ML::Audio::Drama3::BackboneResident::SOURCE,
        [values, routing, batched], [width.to_u64, ranks.to_u64], width)
      expected = ctx.read(serial, "terminal")
      expected[0].should eq(6_f32)
      ctx.read(batched, "terminal").map(&.unsafe_as(UInt32)).should eq(expected.map(&.unsafe_as(UInt32)))
    end

    it "preserves exact Delta outputs and every recurrent state value with transposed storage" do
      # Unequal, non-SIMD-aligned dimensions catch head/key/value index swaps.
      [{6, 7, 2, 5}, {32, 128, 16, 128}].each do |shape|
        vh, vd, kh, kd = shape
        ctx = ML::Audio::Drama3::Resident::Context.new
        sequence = ->(n : Int32, phase : Int32) { Array(Float32).new(n) { |i| (((i * 13 + phase) % 31) - 15).to_f32 / 128_f32 } }
        initial = sequence.call(vh * vd * kd, 4)
        transposed = Array(Float32).new(initial.size) do |i|
          h, rem = i // (kd * vd), i % (kd * vd)
          initial[(h * vd + rem % vd) * kd + rem // vd]
        end
        states = [ctx.upload(initial), ctx.upload(transposed)]
        names = ["dr3_delta_step", "dr3_delta_step_transposed"]
        4.times do |row|
          q, k = ctx.upload(sequence.call(kh * kd, row)), ctx.upload(sequence.call(kh * kd, row + 7))
          v = ctx.upload(sequence.call(vh * vd, row + 11))
          parameters = [0, 3, 9, 15].map { |phase| ctx.upload(sequence.call(vh, phase)) }
          outputs = names.map_with_index do |name, index|
            output = ctx.alloc(vh * vd)
            ctx.dispatch(name, ML::Audio::Drama3::BackboneResident::SOURCE, [q, k, v] + parameters + [states[index], output], shape.to_a.map(&.to_u64), output.count)
            output
          end
          ctx.read(outputs[1], "terminal").map(&.unsafe_as(UInt32)).should eq(ctx.read(outputs[0], "terminal").map(&.unsafe_as(UInt32)))
          legacy = ctx.read(states[0], "terminal")
          current = ctx.read(states[1], "terminal")
          current.each_with_index do |value, i|
            h, rem = i // (kd * vd), i % (kd * vd)
            value.unsafe_as(UInt32).should eq(legacy[(h * vd + rem % vd) * kd + rem // vd].unsafe_as(UInt32))
          end
        end
      end
    end

    it "preserves recurrent history across ring wrap and growing full-attention caches" do
      path = File.join(__DIR__, "support/drama3/reference_fixture")
      config = ML::Audio::Drama3::BackboneConfig.new(JSON.parse(File.read(File.join(path, "config.json"))))
      store = ML::Audio::Drama3::WeightStore.new(path)
      cpu = ML::Audio::Drama3::Backbone.new(config, store)
      ctx = ML::Audio::Drama3::Resident::Context.new
      gpu = ML::Audio::Drama3::BackboneResident.new(config, store, ctx, 8)
      rows = JSON.parse(File.read(File.join(path, "inputs.json")))["input_embeddings"].as_a
      8.times do |index|
        input = rows[index % rows.size].as_a.map { |v| v.as_f.to_f32 }
        expected = cpu.step(input)
        actual = ctx.read(gpu.step(ctx.upload(input)), "terminal")
        actual.zip(expected).each { |a, b| a.should be_close(b, 3e-5) }
      end
      gpu.position.should eq(8)
      expect_raises(ArgumentError, /capacity exceeded/) { gpu.step(ctx.upload(Array(Float32).new(config.hidden, 0_f32))) }
      ctx.readbacks["diagnostic"].should eq(0_i64)
    end

    it "preserves the legacy attention path through 1024 and uses bounded scratch through 4096" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        fixture = File.join(__DIR__, "support/drama3/reference_fixture")
        config = ML::Audio::Drama3::BackboneConfig.new(JSON.parse(File.read(File.join(fixture, "config.json"))))
        store = ML::Audio::Drama3::WeightStore.new(fixture)
        # Constructor admission validates the wide cache and score allocation;
        # zero and 4097 must fail before allocating any resident buffers.
        expect_raises(ArgumentError, /capacity must be 1..4096/) do
          ML::Audio::Drama3::BackboneResident.new(config, store, ML::Audio::Drama3::Resident::Context.new, 0)
        end
        ML::Audio::Drama3::BackboneResident.new(config, store, ML::Audio::Drama3::Resident::Context.new, 4096)
        expect_raises(ArgumentError, /capacity must be 1..4096/) do
          ML::Audio::Drama3::BackboneResident.new(config, store, ML::Audio::Drama3::Resident::Context.new, 4097)
        end
        overflow_json = JSON.parse(File.read(File.join(fixture, "config.json")))
        overflow_json.as_h["text_config"].as_h["num_attention_heads"] = JSON::Any.new(1_000_000_i64)
        overflow_config = ML::Audio::Drama3::BackboneConfig.new(overflow_json)
        expect_raises(ArgumentError, /exceeds Int32 element capacity/) do
          ML::Audio::Drama3::BackboneResident.new(overflow_config, store, ML::Audio::Drama3::Resident::Context.new, 4096)
        end

        [0, 2].each do |depth|
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth.to_s
          context = ML::Audio::Drama3::Resident::Context.new
          heads, kv_heads, dim, capacity = 4, 2, 4, 4096
          [1024, 1025, 2048, 4096].each do |length|
            q_values = drama3_backbone_attention_values(heads * dim, 1)
            k_values = drama3_backbone_attention_values(length * kv_heads * dim, 2)
            v_values = drama3_backbone_attention_values(length * kv_heads * dim, 3)
            gate_values = drama3_backbone_attention_values(heads * dim, 4)
            q = context.upload(q_values)
            k = context.upload(k_values)
            v = context.upload(v_values)
            gate = context.upload(gate_values)
            scratch = context.alloc(heads * capacity)
            wide = drama3_backbone_attention_enqueue(context, "dr3_full_attention_wide", q, k, v, gate, scratch,
              heads, kv_heads, dim, length, capacity)

            wide_again = nil.as(ML::Audio::Drama3::Resident::Buffer?)
            legacy = nil.as(ML::Audio::Drama3::Resident::Buffer?)
            if length == 1024
              # Reuse the same score slab in two submissions at each depth;
              # depth two permits both wide dispatches to remain in flight.
              context.submit
              wide_again = drama3_backbone_attention_enqueue(context, "dr3_full_attention_wide", q, k, v, gate, scratch,
                heads, kv_heads, dim, length, capacity)
              context.submit
              expected_pending = depth == 2 ? 2_i64 : 0_i64
              context.graph_stats["max_pending"].should eq(expected_pending)
              legacy = drama3_backbone_attention_enqueue(context, "dr3_full_attention", q, k, v, gate, nil,
                heads, kv_heads, dim, length, capacity)
              context.submit
            else
              context.submit
            end

            actual_wide = context.read(wide, "terminal")
            expected = drama3_backbone_attention_oracle(q_values, k_values, v_values, gate_values,
              heads, kv_heads, dim, length)
            actual_wide.zip(expected).each do |actual, expected_value|
              actual.should be_close(expected_value, 3e-5)
            end
            if second_output = wide_again
              actual_second = context.read(second_output, "terminal")
              actual_second.zip(expected).each do |actual, expected_value|
                actual.should be_close(expected_value, 3e-5)
              end
              actual_wide.map(&.unsafe_as(UInt32)).should eq(actual_second.map(&.unsafe_as(UInt32)))
            end
            if legacy_output = legacy
              actual_legacy = context.read(legacy_output, "terminal")
              actual_wide.map(&.unsafe_as(UInt32)).should eq(actual_legacy.map(&.unsafe_as(UInt32)))
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

    it "matches upstream backbone boundaries across recurrent and full attention cache updates" do
      path = File.join(__DIR__, "support/drama3/reference_fixture")
      expected = {} of String => ML::Audio::Drama3::Vector
      JSON.parse(File.read(File.join(path, "reference_traces.json")))["traces"].as_a.each do |record|
        name = record["name"].as_s
        expected[name] = record["values"].as_a.map { |v| v.as_f.to_f32 } if name.starts_with?("backbone.")
      end
      ctx = ML::Audio::Drama3::Resident::Context.new
      model = ML::Audio::Drama3::BackboneResident.new(ML::Audio::Drama3::BackboneConfig.new(JSON.parse(File.read(File.join(path, "config.json")))), ML::Audio::Drama3::WeightStore.new(path), ctx, 4)
      row = 0
      observed = [] of String
      model.trace = ->(name : String, values : ML::Audio::Drama3::Vector) {
        key = name.sub("backbone.", "backbone.row#{row}.")
        values.zip(expected[key]).each { |a, b| a.should be_close(b, 1e-5) }
        observed << key
        nil
      }
      JSON.parse(File.read(File.join(path, "inputs.json")))["input_embeddings"].as_a.each_with_index do |input, index|
        row = index
        model.step(ctx.upload(input.as_a.map { |v| v.as_f.to_f32 }))
      end
      observed.sort.should eq(expected.keys.sort)
      model.position.should eq(3)
      ctx.readbacks.keys.sort.should eq(["control", "diagnostic"])
    end
    it "keeps a dependent norm/projection chain resident and rejects unlabelled readback" do
      ctx = ML::Audio::Drama3::Resident::Context.new
      x = ctx.upload([1_f32, -2_f32, 3_f32, 0.5_f32])
      w = ctx.upload([0.1_f32, -0.2_f32, 0.3_f32, 0.4_f32])
      matrix = ML::Audio::Drama3::Matrix.new(2, 4, [1_f32, 2_f32, 3_f32, 4_f32, -1_f32, 0_f32, 2_f32, 0.5_f32])
      y = ctx.matvec(matrix, ctx.rms(x, w, 1e-6, true))
      expect_raises(ArgumentError, /readback purpose/) { ctx.read(y, "activation") }
      ctx.readbacks.empty?.should be_true
      actual = ctx.read(y, "terminal")
      expected = matrix.matvec(ML::Audio::Drama3::Ops.rms_norm([1_f32, -2_f32, 3_f32, 0.5_f32], [0.1_f32, -0.2_f32, 0.3_f32, 0.4_f32], 1e-6, true))
      actual.zip(expected).each { |a, b| (a - b).abs.should be < 1e-5_f32 }
      ctx.readbacks["terminal"].should eq(8_i64)
    end

    it "rejects a non-finite intermediate even when the terminal tensor is finite" do
      ctx = ML::Audio::Drama3::Resident::Context.new
      ctx.guard(ctx.upload([Float32::NAN]))
      terminal = ctx.upload([1_f32])
      expect_raises(ArgumentError, /non-finite resident intermediate/) { ctx.read(terminal, "terminal") }
      ctx.readbacks["control"].should eq(4_i64)
      ctx.readbacks["terminal"].should eq(0_i64)
    end

    it "rejects an out-of-range device expert id before reading weight storage" do
      store = ML::Audio::Drama3::WeightStore.new(File.join(__DIR__, "support/drama3/reference_fixture"))
      matrix = store.matrix("affine_fixture.q4.weight")
      ctx = ML::Audio::Drama3::Resident::Context.new
      input = ctx.upload(Array(Float32).new(matrix.cols, 1_f32))
      output = ctx.expert(matrix, input, ctx.upload([999_f32]))
      expect_raises(ArgumentError, /non-finite resident terminal/) { ctx.read(output, "terminal") }
      expect_raises(ArgumentError, /expert rank/) { ctx.expert(matrix, input, ctx.upload([0_f32]), 1) }
    end
  end
{% end %}
