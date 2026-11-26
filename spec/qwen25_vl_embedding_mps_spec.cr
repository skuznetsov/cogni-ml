require "spec"
require "../src/ml/gguf/qwen25_vl_embedding"
require "./support/video_allocation_counter"

describe "Qwen2.5-VL native BF16 embedding gather" do
  it "gathers exact BF16 words asynchronously on one reusable command chain" do
    {% if flag?(:cpu_only) %}
      pending!("requires a Metal-enabled build")
    {% else %}
      pending!("set COGNI_QWEN25_EMBEDDING_GPU=1 for the serial Metal qualification") unless ENV["COGNI_QWEN25_EMBEDDING_GPU"]? == "1"
      device = ML::Metal::Device.instance
      pending!("Metal device with unified memory is unavailable") unless device.available? && device.has_unified_memory?

      config = ML::GGUF::Qwen25VLConfig.new(
        dim: 4,
        ffn_dim: 8,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        layers: 3,
        max_positions: 16,
      )
      tokens = 6
      unique_rows = 3
      row_indices = [2, 0, 2, 1, 0, 2]
      source_words = UInt16[
        0x3f80_u16, 0x8000_u16, 0x0001_u16, 0xbf80_u16,
        0x4000_u16, 0x3f81_u16, 0x3f00_u16, 0x0000_u16,
        0xc000_u16, 0x4040_u16, 0x3f7f_u16, 0x0080_u16,
      ]
      expected_words = UInt16[
        0xc000_u16, 0x4040_u16, 0x3f7f_u16, 0x0080_u16,
        0x3f80_u16, 0x8000_u16, 0x0001_u16, 0xbf80_u16,
        0xc000_u16, 0x4040_u16, 0x3f7f_u16, 0x0080_u16,
        0x4000_u16, 0x3f81_u16, 0x3f00_u16, 0x0000_u16,
        0x3f80_u16, 0x8000_u16, 0x0001_u16, 0xbf80_u16,
        0xc000_u16, 0x4040_u16, 0x3f7f_u16, 0x0080_u16,
      ]
      named = ML::GGUF::Qwen25VLEmbeddingPlan.resident_bytes(config, tokens, unique_rows, 5)

      compact_rows = ML::MetalBuffer.new(source_words.size.to_i64 * 2)
      owned = [compact_rows]
      chains = [] of ML::Metal::MPSCommandChain
      plan : ML::GGUF::Qwen25VLEmbeddingPlan? = nil
      begin
        compact_rows.write_bytes(source_words.to_unsafe.as(UInt8*), source_words.size * 2)

        # A bad remap must fail in CPU preflight without allocating its owned buffers.
        allocations = ML::MetalBuffer.video_spec_allocations
        live_bytes = ML::MetalBuffer.stats[:live_bytes]
        expect_raises(ArgumentError, /row index/) do
          ML::GGUF::Qwen25VLEmbeddingPlan.new(config, tokens, compact_rows,
            [2, 0, 3, 1, 0, 2], named.total, vocab_size: 5)
        end
        ML::MetalBuffer.video_spec_allocations.should eq(allocations)
        ML::MetalBuffer.stats[:live_bytes].should eq(live_bytes)

        # The one-byte-short admission failure must also precede allocation.
        expect_raises(ArgumentError, /budget/) do
          ML::GGUF::Qwen25VLEmbeddingPlan.new(config, tokens, compact_rows,
            row_indices, named.total - 1, vocab_size: 5)
        end
        ML::MetalBuffer.video_spec_allocations.should eq(allocations)
        ML::MetalBuffer.stats[:live_bytes].should eq(live_bytes)

        # The native boundary rejects a byte-aligned alias. The public plan
        # always allocates these buffers separately, so exercise its FFI guard
        # directly with row-index and output handles referring to one buffer.
        alias_table = ML::MetalBuffer.new(8_i64)
        owned << alias_table
        alias_words = UInt16[0x3f80_u16, 0xbf80_u16, 0x4000_u16, 0xc000_u16]
        alias_table.write_bytes(alias_words.to_unsafe.as(UInt8*), alias_words.size * 2)
        alias_indices_output = ML::MetalBuffer.new(12_i64)
        owned << alias_indices_output
        alias_indices = Int32[0, 0, 0]
        alias_indices_output.write_bytes(alias_indices.to_unsafe.as(UInt8*), alias_indices.size * 4)
        alias_error = Bytes.new(512, 0_u8)
        alias_plan = Qwen25VLEmbeddingFFI.create(device.device_handle,
          3, 2, 2, alias_table.handle, alias_indices_output.handle,
          alias_indices_output.handle, alias_error.to_unsafe, alias_error.size)
        begin
          alias_plan.null?.should be_true
          String.new(alias_error.to_unsafe).should contain("disjoint")
        ensure
          Qwen25VLEmbeddingFFI.destroy(alias_plan) unless alias_plan.null?
        end

        before_plan = ML::MetalBuffer.video_spec_allocations
        plan = ML::GGUF::Qwen25VLEmbeddingPlan.new(config, tokens,
          compact_rows, row_indices, named.total, vocab_size: 5)
        ML::MetalBuffer.video_spec_allocations.should eq(before_plan + 2)
        plan.not_nil!.unique_row_count.should eq(unique_rows)
        plan.not_nil!.token_count.should eq(tokens)
        plan.not_nil!.named_bytes.should eq(named)
        plan.not_nil!.output.size.should eq(expected_words.size.to_i64 * 2)

        first_chain = ML::Metal::MPSCommandChain.new
        chains << first_chain
        before_encode = ML::MetalBuffer.video_spec_allocations
        plan.not_nil!.encode(first_chain)
        ML::MetalBuffer.video_spec_allocations.should eq(before_encode)
        first_chain.wait_count.should eq(0)
        expect_raises(ArgumentError, /terminal wait/) { plan.not_nil!.release }

        second_chain = ML::Metal::MPSCommandChain.new
        chains << second_chain
        before_rejected_encode = ML::MetalBuffer.video_spec_allocations
        expect_raises(ArgumentError, /in-flight chain/) { plan.not_nil!.encode(second_chain) }
        ML::MetalBuffer.video_spec_allocations.should eq(before_rejected_encode)
        first_chain.wait_count.should eq(0)

        # The original chain gets exactly one terminal wait. Once drained, the
        # same prepared plan may be encoded on a new chain with no new buffers.
        first_chain.commit_and_wait
        first_chain.wait_count.should eq(1)
        before_reuse = ML::MetalBuffer.video_spec_allocations
        plan.not_nil!.encode(second_chain)
        ML::MetalBuffer.video_spec_allocations.should eq(before_reuse)
        second_chain.wait_count.should eq(0)
        expect_raises(ArgumentError, /terminal wait/) { plan.not_nil!.release }
        second_chain.commit_and_wait
        second_chain.wait_count.should eq(1)

        actual_words = Slice(UInt16).new(expected_words.size)
        plan.not_nil!.output.read_bytes(actual_words.to_unsafe.as(UInt8*), actual_words.size * 2)
        actual_words.to_a.should eq(expected_words)

        plan.not_nil!.release
        plan = nil
      ensure
        chains.each do |chain|
          begin
            chain.commit_and_wait unless chain.completed?
          rescue
            # A failed GPU command still needs the native owner retained until
            # its terminal-drain path has been attempted.
          end
        end
        if current = plan
          begin
            current.release
          rescue
          end
        end
        chains.each do |chain|
          begin
            chain.release
          rescue
          end
        end
        owned.reverse_each(&.release)
      end
    {% end %}
  end
end
