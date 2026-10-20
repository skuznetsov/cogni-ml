require "spec"
require "json"
require "digest/sha256"
require "../src/ml/video/metal_attention"
require "../src/ml/video/metal_attention_mps"
require "./support/video_allocation_counter"

private def mps_attention_fixture_tensor(arena : ML::Video::Arena,
                                         name : String, record : JSON::Any,
                                         count : Int32) : ML::MetalBuffer
  bytes = count.to_i64 * sizeof(Float32)
  path = record["path"].as_s
  record["bytes"].as_i64.should eq(bytes)
  File.size(path).should eq(bytes)
  buffer = arena.alloc(name, count)
  raw = Bytes.new(buffer.contents.as(UInt8*), bytes.to_i32)
  File.open(path, "rb") { |io| io.read_fully(raw) }
  Digest::SHA256.hexdigest(raw).should eq(record["sha256"].as_s)
  buffer
end

private def mps_attention_copy_equal?(left : ML::MetalBuffer,
                                      right : ML::MetalBuffer,
                                      count : Int32) : Bool
  a = left.contents.as(Pointer(Float32))
  b = right.contents.as(Pointer(Float32))
  count.times do |i|
    return false unless a[i].unsafe_as(UInt32) == b[i].unsafe_as(UInt32)
  end
  true
end

private def mps_attention_repeat_rows!(source : ML::MetalBuffer,
                                       destination : ML::MetalBuffer,
                                       source_rows : Int32,
                                       destination_rows : Int32,
                                       channels : Int32,
                                       chunk_rows : Int32,
                                       chunk_shift : Int32) : Nil
  source_ptr = source.contents.as(Pointer(Float32))
  destination_ptr = destination.contents.as(Pointer(Float32))
  destination_rows.times do |row|
    source_row = (row + (row // chunk_rows) * chunk_shift) % source_rows
    source_offset = source_row * channels
    destination_offset = row * channels
    channels.times do |channel|
      destination_ptr[destination_offset + channel] = source_ptr[source_offset + channel]
    end
  end
end

private def mps_attention_fill_nan!(buffer : ML::MetalBuffer, count : Int32) : Nil
  ptr = buffer.contents.as(Pointer(Float32))
  count.times { |i| ptr[i] = Float32::NAN }
end

describe "Opt-in prepared MPSGraph SDPA" do
  it "rejects aliases and overflow before preparing a native graph" do
    arena = ML::Video::Arena.new
    chain : ML::Metal::MPSCommandChain? = nil
    begin
      q = arena.alloc("q", 64)
      k = arena.alloc("k", 64)
      v = arena.alloc("v", 64)
      output = arena.alloc("output", 64)
      scratch_q = arena.alloc("scratch_q", 64)
      scratch_a = arena.alloc("scratch_a", 64)
      scratch_b = arena.alloc("scratch_b", 64)
      expect_raises(ArgumentError) do
        ML::Video::WanAttentionMPSPlan.new(1, 1, 1, 1, 64,
          q, k, v, q, output)
      end
      expect_raises(ArgumentError) do
        ML::Video::WanAttentionMPSPlan.new(Int32::MAX, Int32::MAX, 1,
          Int32::MAX, Int32::MAX, q, k, v, output, arena.alloc("other", 1))
      end
      expect_raises(ArgumentError, "B=1") do
        ML::Video::WanAttentionMPSChunkWorkspace.new(2, 2, 2, 1, 64, 1,
          q, k, v, output, scratch_q, scratch_a, scratch_b)
      end
      expect_raises(ArgumentError, "1..1024") do
        ML::Video::WanAttentionMPSChunkWorkspace.new(1, 2, 2, 1, 64, 1025,
          q, k, v, output, scratch_q, scratch_a, scratch_b)
      end
      expect_raises(ArgumentError, "pairwise disjoint") do
        ML::Video::WanAttentionMPSChunkWorkspace.new(1, 1, 1, 1, 64, 1,
          q, k, v, output, scratch_q, scratch_a, scratch_a)
      end

      chain = ML::Metal::MPSCommandChain.new
      expect_raises(ArgumentError, "aligned") do
        ML::Video::WanAttentionMPSPlan.copy(chain.not_nil!, q, k, 4_i64,
          1_i64, 0_i64)
      end
      expect_raises(ArgumentError, "aligned") do
        ML::Video::WanAttentionMPSPlan.copy(chain.not_nil!, q, k, 3_i64)
      end
      expect_raises(ArgumentError, "in range") do
        ML::Video::WanAttentionMPSPlan.copy(chain.not_nil!, q, k, 4_i64,
          0_i64, 256_i64)
      end
      expect_raises(ArgumentError, "active ranges") do
        ML::Video::WanAttentionMPSPlan.copy(chain.not_nil!, q, q, 4_i64)
      end
    ensure
      chain.try do |active|
        active.commit_and_wait unless active.completed?
        active.release
      end
      arena.release
    end
  end

  it "matches all pinned outputs and preserves native/Graph order in one chain" do
    fixture_path = ENV["WAN_ATTENTION_MPS_FIXTURE"]?
    pending! "Set WAN_ATTENTION_MPS_FIXTURE to the qualified MPS fixture manifest" unless fixture_path
    fixture = JSON.parse(File.read(fixture_path))
    fixture["torch"].as_s.should eq("2.8.0")
    fixture["device"].as_s.should eq("mps")
    fixture["fallback_disabled"].as_bool.should be_true
    cases = fixture["cases"].as_a
    cases.map { |c| c["name"].as_s }.sort.should eq([
      "b1-q64-k1024-h40-d128", "b1-q64-k36608-h40-d128",
      "b1-q9-k65-h2-d64-near-tie", "b1-q9-k67-h2-d128-sharp",
      "b2-q1-k1-h1-d64-single", "b2-q13-k97-h3-d64",
    ])
    expected_geometry = {
      "b1-q64-k1024-h40-d128"     => {1, 64, 1024, 40, 128},
      "b1-q64-k36608-h40-d128"    => {1, 64, 36608, 40, 128},
      "b1-q9-k65-h2-d64-near-tie" => {1, 9, 65, 2, 64},
      "b1-q9-k67-h2-d128-sharp"   => {1, 9, 67, 2, 128},
      "b2-q1-k1-h1-d64-single"    => {2, 1, 1, 1, 64},
      "b2-q13-k97-h3-d64"         => {2, 13, 97, 3, 64},
    }

    # Compile the diagnostic mutation kernel before measuring encode-time
    # MetalBuffer allocations. Only the small near-tie fixture gets the
    # changed-input A→B→A path; all six cases still exercise prepared SDPA.
    ML::Video::MetalAttentionMPS.prepare!
    changed_query_cases = 0
    cases.each do |c|
      arena = ML::Video::Arena.new
      chain : ML::Metal::MPSCommandChain? = nil
      plan : ML::Video::WanAttentionMPSPlan? = nil
      begin
        batches, query_rows, key_rows, heads, dim = {
          c["batches"].as_i, c["query_rows"].as_i, c["key_rows"].as_i,
          c["heads"].as_i, c["dim"].as_i,
        }
        expected_geometry[c["name"].as_s]?.should eq({batches, query_rows,
                                                      key_rows, heads, dim})
        query_count = batches * query_rows * heads * dim
        kv_count = batches * key_rows * heads * dim
        q = mps_attention_fixture_tensor(arena, "q", c["q"], query_count)
        k = mps_attention_fixture_tensor(arena, "k", c["k"], kv_count)
        v = mps_attention_fixture_tensor(arena, "v", c["v"], kv_count)
        expected = mps_attention_fixture_tensor(arena, "expected", c["expected"], query_count)
        output_a = arena.alloc("output_a", query_count)
        output_b = arena.alloc("output_b", query_count)
        q_copy = arena.alloc("q_copy", query_count)
        first_a = arena.alloc("first_a", query_count)
        final_a = arena.alloc("final_a", query_count)

        plan = ML::Video::WanAttentionMPSPlan.new(batches, query_rows,
          key_rows, heads, dim, q, k, v, output_a, output_b)
        chain = ML::Metal::MPSCommandChain.new
        device = ML::Metal::Device.instance
        allocated_before = device.current_allocated_size
        recommended = device.recommended_working_set_size
        allocations_before = ML::MetalBuffer.video_spec_allocations

        copy_bytes = query_count.to_i64 * sizeof(Float32)
        changed_query = c["name"].as_s == "b1-q9-k65-h2-d64-near-tie"
        changed_query_cases += 1 if changed_query
        ML::Video::WanAttentionMPSPlan.copy(chain, q, q_copy, copy_bytes)
        chain.pause!
        chain.continue_async!
        chain.resume!
        plan.encode(chain, 0)
        ML::Video::WanAttentionMPSPlan.copy(chain, output_a, first_a, copy_bytes)
        if changed_query
          # Keep the same fixed Q backing in the prepared graph. The saved
          # copy is restored before the final A invocation; K/V are untouched.
          ML::Video::MetalAttentionMPS.negate_q!(chain, q, query_count)
          ML::Video::MetalOps.attention(chain.encoder, q, k, v, final_a,
            query_rows, key_rows, heads, dim, batches)
          plan.encode(chain, 1)
          ML::Video::WanAttentionMPSPlan.copy(chain, q_copy, q, copy_bytes)
          plan.encode(chain, 0)
        else
          plan.encode(chain, 1)
          plan.encode(chain, 0)
          ML::Video::WanAttentionMPSPlan.copy(chain, output_a, final_a, copy_bytes)
        end
        ML::MetalBuffer.video_spec_allocations.should eq(allocations_before),
          "#{c["name"].as_s}: encode allocated MetalBuffers"
        chain.wait_count.should eq(0)
        chain.commit_and_wait
        chain.wait_count.should eq(1)
        chain.segment_count.should be >= 2

        mps_attention_copy_equal?(q, q_copy, query_count).should be_true,
          "#{c["name"].as_s}: Q was not restored from its pre-Graph copy"
        if changed_query
          mps_attention_copy_equal?(first_a, output_a, query_count).should be_true,
            "#{c["name"].as_s}: changed-Q A→B→A reuse changed the restored A output"
          mps_attention_copy_equal?(output_b, first_a, query_count).should be_false,
            "#{c["name"].as_s}: changed Q did not change the SDPA output"
        else
          mps_attention_copy_equal?(first_a, final_a, query_count).should be_true,
            "#{c["name"].as_s}: A→B→A reuse changed the A output"
        end

        expected_ptr = expected.contents.as(Pointer(Float32))
        output_ptrs = [output_a.contents.as(Pointer(Float32)),
                       first_a.contents.as(Pointer(Float32))]
        unless changed_query
          output_ptrs << output_b.contents.as(Pointer(Float32))
          output_ptrs << final_a.contents.as(Pointer(Float32))
        end
        violations = 0
        worst_ratio = 0.0_f64
        query_count.times do |i|
          ref = expected_ptr[i]
          output_ptrs.each do |actual_ptr|
            actual = actual_ptr[i]
            ratio = (actual - ref).abs.to_f64 / (2e-5 + 2e-5 * ref.abs)
            violations += 1 unless actual.finite? && ref.finite? && ratio <= 1.0
            worst_ratio = Math.max(worst_ratio, ratio)
          end
        end
        violations.should eq(0),
          "#{c["name"].as_s}: strict SDPA mismatches=#{violations}, worst_ratio=#{worst_ratio}"

        if changed_query
          scalar_ptr = final_a.contents.as(Pointer(Float32))
          changed_ptr = output_b.contents.as(Pointer(Float32))
          changed_violations = 0
          changed_worst_ratio = 0.0_f64
          query_count.times do |i|
            ref = scalar_ptr[i]
            actual = changed_ptr[i]
            ratio = (actual - ref).abs.to_f64 / (2e-5 + 2e-5 * ref.abs)
            changed_violations += 1 unless actual.finite? && ref.finite? && ratio <= 1.0
            changed_worst_ratio = Math.max(changed_worst_ratio, ratio)
          end
          changed_violations.should eq(0),
            "#{c["name"].as_s}: changed-Q SDPA vs native attention mismatches=" +
            "#{changed_violations}, worst_ratio=#{changed_worst_ratio}"
        end

        allocated_after = device.current_allocated_size
        gpu_seconds = chain.gpu_elapsed_seconds
        gpu_seconds.finite?.should be_true
        gpu_seconds.should be >= 0.0
        puts "WAN_MPS_SDPA_COST case=#{c["name"].as_s} gpu_seconds=#{gpu_seconds} " +
             "current_allocated_before=#{allocated_before} current_allocated_after=#{allocated_after} " +
             "recommended_working_set=#{recommended} peak_allocated=unavailable " +
             "segments=#{chain.segment_count}"
      ensure
        chain.try do |active|
          active.commit_and_wait unless active.completed?
          active.release
        end
        plan.try(&.release)
        arena.release
      end
    end
    changed_query_cases.should eq(1), "expected one changed-Q A→B→A discriminator"

    # Exercise global row offsets, non-full tail padding, and prepared-plan
    # reuse on the long-key upstream case. Chunk-dependent source-row rotation
    # makes a broken offset-zero copy observably wrong in every later span.
    long_case = cases.find { |c| c["name"].as_s == "b1-q64-k36608-h40-d128" }
    raise "missing pinned long-key attention case" unless long_case
    source_rows = 64
    [512, 1024].each do |chunk_rows|
      full_rows = chunk_rows + 265
      channels = long_case["heads"].as_i * long_case["dim"].as_i
      source_count = source_rows * channels
      full_count = full_rows * channels
      kv_count = long_case["key_rows"].as_i * channels
      arena = ML::Video::Arena.new
      chain : ML::Metal::MPSCommandChain? = nil
      workspace : ML::Video::WanAttentionMPSChunkWorkspace? = nil
      begin
        q_source = mps_attention_fixture_tensor(arena, "chunk_q_source",
          long_case["q"], source_count)
        expected_source = mps_attention_fixture_tensor(arena, "chunk_expected_source",
          long_case["expected"], source_count)
        k = mps_attention_fixture_tensor(arena, "chunk_k", long_case["k"], kv_count)
        v = mps_attention_fixture_tensor(arena, "chunk_v", long_case["v"], kv_count)
        q = arena.alloc("chunk_q", full_count)
        expected = arena.alloc("chunk_expected", full_count)
        output = arena.alloc("chunk_output", full_count)
        first_output = arena.alloc("chunk_first_output", full_count)
        chunk_q = arena.alloc("chunk_q_scratch", chunk_rows * channels)
        chunk_output_a = arena.alloc("chunk_output_a", chunk_rows * channels)
        chunk_output_b = arena.alloc("chunk_output_b", chunk_rows * channels)
        mps_attention_repeat_rows!(q_source, q, source_rows, full_rows, channels,
          chunk_rows, 7)
        mps_attention_repeat_rows!(expected_source, expected, source_rows,
          full_rows, channels, chunk_rows, 7)
        mps_attention_fill_nan!(output, full_count)
        mps_attention_fill_nan!(first_output, full_count)
        workspace = ML::Video::WanAttentionMPSChunkWorkspace.new(1, full_rows,
          long_case["key_rows"].as_i, long_case["heads"].as_i,
          long_case["dim"].as_i, chunk_rows, q, k, v, output, chunk_q,
          chunk_output_a, chunk_output_b)
        workspace.scratch_bytes.should eq(chunk_rows.to_i64 * channels * sizeof(Float32) * 3)
        chain = ML::Metal::MPSCommandChain.new
        allocations_before = ML::MetalBuffer.video_spec_allocations
        device = ML::Metal::Device.instance
        current_before = device.current_allocated_size
        recommended = device.recommended_working_set_size
        full_bytes = full_count.to_i64 * sizeof(Float32)

        workspace.encode(chain.not_nil!)
        ML::Video::WanAttentionMPSPlan.copy(chain.not_nil!, output,
          first_output, full_bytes)
        workspace.encode(chain.not_nil!)
        ML::MetalBuffer.video_spec_allocations.should eq(allocations_before),
          "chunked MPS encode allocated MetalBuffers"
        chain.not_nil!.wait_count.should eq(0)
        chain.not_nil!.commit_and_wait
        chain.not_nil!.wait_count.should eq(1)

        output_ptr = output.contents.as(Pointer(Float32))
        expected_ptr = expected.contents.as(Pointer(Float32))
        violations = 0
        worst_ratio = 0.0_f64
        max_abs = 0.0_f64
        full_count.times do |i|
          actual = output_ptr[i]
          reference = expected_ptr[i]
          delta = (actual - reference).abs.to_f64
          ratio = delta / (2e-5 + 2e-5 * reference.abs)
          violations += 1 unless actual.finite? && reference.finite? && ratio <= 1.0
          worst_ratio = Math.max(worst_ratio, ratio)
          max_abs = Math.max(max_abs, delta)
        end
        violations.should eq(0),
          "chunked Q#{full_rows}/K#{long_case["key_rows"].as_i} strict mismatches=#{violations}, " +
          "worst_ratio=#{worst_ratio}, max_abs=#{max_abs}"
        mps_attention_copy_equal?(output, first_output, full_count).should be_true,
          "reusing one chunk workspace changed output bits"
        puts "WAN_MPS_SDPA_CHUNK_COST query_rows=#{full_rows} chunk_rows=#{chunk_rows} " +
             "chunks=#{(full_rows + chunk_rows - 1) // chunk_rows} " +
             "gpu_seconds=#{chain.not_nil!.gpu_elapsed_seconds} " +
             "current_allocated_before=#{current_before} " +
             "current_allocated_after=#{device.current_allocated_size} " +
             "recommended_working_set=#{recommended} peak_allocated=unavailable " +
             "segments=#{chain.not_nil!.segment_count} waits=#{chain.not_nil!.wait_count}"
      ensure
        chain.try do |active|
          active.commit_and_wait unless active.completed?
          active.release
        end
        workspace.try(&.release)
        arena.release
      end
    end

    # On a small case, mutate the same caller Q backing on GPU between two
    # chunked calls, compare B against the native kernel, then restore A and
    # require exact replay from the same prepared plan and scratch.
    small_case = cases.find { |c| c["name"].as_s == "b1-q9-k65-h2-d64-near-tie" }
    raise "missing pinned changed-Q attention case" unless small_case
    small_rows = small_case["query_rows"].as_i
    small_channels = small_case["heads"].as_i * small_case["dim"].as_i
    small_q_count = small_rows * small_channels
    small_kv_count = small_case["key_rows"].as_i * small_channels
    small_chunk_rows = 4
    arena = ML::Video::Arena.new
    chain = nil
    workspace = nil
    begin
      q = mps_attention_fixture_tensor(arena, "reuse_q", small_case["q"], small_q_count)
      k = mps_attention_fixture_tensor(arena, "reuse_k", small_case["k"], small_kv_count)
      v = mps_attention_fixture_tensor(arena, "reuse_v", small_case["v"], small_kv_count)
      expected_a = mps_attention_fixture_tensor(arena, "reuse_expected_a",
        small_case["expected"], small_q_count)
      output = arena.alloc("reuse_output", small_q_count)
      q_saved = arena.alloc("reuse_q_saved", small_q_count)
      output_a_saved = arena.alloc("reuse_output_a_saved", small_q_count)
      output_b = arena.alloc("reuse_output_b", small_q_count)
      reference_b = arena.alloc("reuse_reference_b", small_q_count)
      chunk_count = small_chunk_rows * small_channels
      chunk_q = arena.alloc("reuse_chunk_q", chunk_count)
      chunk_output_a = arena.alloc("reuse_chunk_output_a", chunk_count)
      chunk_output_b = arena.alloc("reuse_chunk_output_b", chunk_count)
      mps_attention_fill_nan!(output, small_q_count)
      workspace = ML::Video::WanAttentionMPSChunkWorkspace.new(1, small_rows,
        small_case["key_rows"].as_i, small_case["heads"].as_i,
        small_case["dim"].as_i, small_chunk_rows, q, k, v, output, chunk_q,
        chunk_output_a, chunk_output_b)
      workspace.scratch_bytes.should eq(small_chunk_rows.to_i64 * small_channels * sizeof(Float32) * 3)
      ML::Video::MetalOps.prepare!
      chain = ML::Metal::MPSCommandChain.new
      allocations_before = ML::MetalBuffer.video_spec_allocations
      small_bytes = small_q_count.to_i64 * sizeof(Float32)

      ML::Video::WanAttentionMPSPlan.copy(chain.not_nil!, q, q_saved, small_bytes)
      workspace.encode(chain.not_nil!)
      ML::Video::WanAttentionMPSPlan.copy(chain.not_nil!, output, output_a_saved,
        small_bytes)
      ML::Video::MetalAttentionMPS.negate_q!(chain.not_nil!, q, small_q_count)
      ML::Video::MetalOps.attention(chain.not_nil!.encoder, q, k, v,
        reference_b, small_rows, small_case["key_rows"].as_i,
        small_case["heads"].as_i, small_case["dim"].as_i)
      workspace.encode(chain.not_nil!)
      ML::Video::WanAttentionMPSPlan.copy(chain.not_nil!, output, output_b,
        small_bytes)
      ML::Video::MetalAttentionMPS.negate_q!(chain.not_nil!, q, small_q_count)
      workspace.encode(chain.not_nil!)
      ML::MetalBuffer.video_spec_allocations.should eq(allocations_before),
        "changed-input chunked encode allocated MetalBuffers"
      chain.not_nil!.wait_count.should eq(0)
      chain.not_nil!.commit_and_wait
      chain.not_nil!.wait_count.should eq(1)

      mps_attention_copy_equal?(q, q_saved, small_q_count).should be_true,
        "changed-input chunk workspace did not restore Q"
      mps_attention_copy_equal?(output, output_a_saved, small_q_count).should be_true,
        "chunk workspace A→B→A changed the restored A output"
      mps_attention_copy_equal?(output_a_saved, output_b, small_q_count).should be_false,
        "changed Q did not change the chunked attention output"

      expected_a_ptr = expected_a.contents.as(Pointer(Float32))
      actual_a_ptr = output.contents.as(Pointer(Float32))
      reference_b_ptr = reference_b.contents.as(Pointer(Float32))
      actual_b_ptr = output_b.contents.as(Pointer(Float32))
      violations = 0
      worst_ratio = 0.0_f64
      small_q_count.times do |i|
        [{actual_a_ptr[i], expected_a_ptr[i]},
         {actual_b_ptr[i], reference_b_ptr[i]}].each do |pair|
          pair_actual, pair_reference = pair
          ratio = (pair_actual - pair_reference).abs.to_f64 / (2e-5 + 2e-5 * pair_reference.abs)
          violations += 1 unless pair_actual.finite? && pair_reference.finite? && ratio <= 1.0
          worst_ratio = Math.max(worst_ratio, ratio)
        end
      end
      violations.should eq(0),
        "changed-input chunked A/B strict mismatches=#{violations}, worst_ratio=#{worst_ratio}"

      # Retain the prepared workspace across requests. Once request one has
      # completed, the same bindings and scratch may be used on a fresh chain
      # with changed Q; the prior native B result remains the independent check.
      chain.not_nil!.release
      chain = ML::Metal::MPSCommandChain.new
      ML::Video::MetalAttentionMPS.negate_q!(chain.not_nil!, q, small_q_count)
      workspace.encode(chain.not_nil!)
      ML::Video::WanAttentionMPSPlan.copy(chain.not_nil!, output, output_b,
        small_bytes)
      ML::Video::MetalAttentionMPS.negate_q!(chain.not_nil!, q, small_q_count)
      ML::MetalBuffer.video_spec_allocations.should eq(allocations_before),
        "retained workspace allocated MetalBuffers on a fresh command chain"
      chain.not_nil!.wait_count.should eq(0)
      chain.not_nil!.commit_and_wait
      chain.not_nil!.wait_count.should eq(1)
      mps_attention_copy_equal?(q, q_saved, small_q_count).should be_true,
        "fresh-chain workspace request did not restore Q"
      mps_attention_copy_equal?(output, output_a_saved, small_q_count).should be_false,
        "changed Q on a fresh chain did not change the output"
      violations = 0
      worst_ratio = 0.0_f64
      small_q_count.times do |i|
        actual = output_b.contents.as(Pointer(Float32))[i]
        reference = reference_b.contents.as(Pointer(Float32))[i]
        ratio = (actual - reference).abs.to_f64 / (2e-5 + 2e-5 * reference.abs)
        violations += 1 unless actual.finite? && reference.finite? && ratio <= 1.0
        worst_ratio = Math.max(worst_ratio, ratio)
      end
      violations.should eq(0),
        "fresh-chain changed-Q strict mismatches=#{violations}, worst_ratio=#{worst_ratio}"
    ensure
      chain.try do |active|
        active.commit_and_wait unless active.completed?
        active.release
      end
      workspace.try(&.release)
      arena.release
    end
  end
end
