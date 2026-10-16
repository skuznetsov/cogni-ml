require "spec"
require "../src/ml/audio/drama3/resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private def drama3_wrapper_fixture(&)
    path = File.tempname("drama3-wrapper", ".bin")
    io = IO::Memory.new
    io.write(Bytes.new(22, 0_u8))
    [1_f32, 2_f32, 3_f32, 4_f32].each { |v| io.write_bytes(v, IO::ByteFormat::LittleEndian) }
    File.write(path, io.to_slice)
    owner = ML::Audio::Drama3::MappedFile.new(path)
    tensor = ML::Audio::Drama3::Tensor.new([2, 2], "F32", owner.slice(22_i64, 16_i64), owner)
    begin
      yield owner, tensor
    ensure
      # Unlink only: never rewrite an inode with live mmap owners.
      File.delete(path) if File.exists?(path)
    end
  end

  describe "DRAMA-3 opt-in mapped weight wrapper probe" do
    it "retires a whole shard before recreating views over identical immutable bytes" do
      drama3_wrapper_fixture do |owner, tensor|
        bank = ML::Audio::Drama3::MetalBackend::WeightBufferBank.new
        first = bank.tensor_buffer(tensor)
        second_tensor = ML::Audio::Drama3::Tensor.new([2], "F32", owner.slice(30_i64, 8_i64), owner)
        second = bank.tensor_buffer(second_tensor)
        first.buffer.should be(second.buffer)
        first.offset.should eq(22)
        second.offset.should eq(30)
        old = first.buffer
        old_contents, old_size, old_mode = old.contents, old.size, old.storage_mode
        unchanged = bank.prepare(false)
        unchanged["mappings"].should eq(1)
        unchanged["constructed"].should eq(0)
        bank.tensor_buffer(tensor).should be(first)
        sham = bank.prepare(false, clear_views: true)
        sham["generation"].should eq(0)
        sham["view_generation"].should eq(1)
        sham["tensor_views_cleared"].should eq(2)
        sham["constructed"].should eq(0)
        sham["retired"].should eq(0)
        sham["factory_calls"].should eq(unchanged["factory_calls"])
        rebuilt = bank.tensor_buffer(tensor)
        rebuilt.should_not be(first)
        rebuilt.buffer.should be(old)
        rebuilt.buffer.valid?.should be_true
        rebuilt.buffer.contents.should eq(old_contents)
        rebuilt.buffer.size.should eq(old_size)
        rebuilt.buffer.storage_mode.should eq(old_mode)
        rebuilt.offset.should eq(first.offset)
        changed = bank.prepare(true)
        changed["generation"].should eq(1)
        changed["view_generation"].should eq(2)
        changed["retired"].should eq(1)
        changed["constructed"].should eq(1)
        changed["same_backing"].should eq(1)
        old.valid?.should be_false
        fresh = bank.tensor_buffer(tensor)
        fresh.buffer.should_not be(old)
        fresh.buffer.valid?.should be_true
        fresh.buffer.contents.should eq(old_contents)
        fresh.buffer.size.should eq(old_size)
        fresh.buffer.storage_mode.should eq(old_mode)
        fresh.offset.should eq(first.offset)
        bank.tensor_buffer(second_tensor).buffer.should be(fresh.buffer)
        # A foreign owner must never silently substitute another shard.
        other = ML::Audio::Drama3::MetalBackend::MappedBuffer.new(owner)
        drama3_wrapper_fixture do |_, foreign|
          expect_raises(ArgumentError, /owner/) do
            ML::Audio::Drama3::MetalBackend::TensorBuffer.new(foreign, true, other)
          end
        end
        other.buffer.release
      end
    end

    it "keeps GPU results and wait/read ledgers exact across wrapper rotation" do
      previous = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ["0", "2"].each do |depth|
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth
          drama3_wrapper_fixture do |_, tensor|
            context = ML::Audio::Drama3::Resident::Context.new
            context.enable_weight_wrapper_probe!
            context.prepare_weight_wrappers(false)["mappings"].should eq(0)
            matrix = ML::Audio::Drama3::Matrix.new(tensor)
            expected = [5_f32, 11_f32]
            3.times do |step|
              before = context.graph_stats
              reads = context.readbacks.dup
              stats = context.prepare_weight_wrappers(step == 2, clear_views: step == 1)
              context.graph_stats.should eq(before)
              context.readbacks.should eq(reads)
              stats["constructed"].should eq(step == 2 ? 1 : 0)
              context.begin_weight_wrapper_job!
              x = context.upload([1_f32, 2_f32])
              result = context.read(context.matvec(matrix, x), "terminal")
              result.map(&.unsafe_as(UInt32)).should eq(expected.map(&.unsafe_as(UInt32)))
              graph = context.graph_stats
              graph["submitted"].should eq(before["submitted"] + 1)
              graph["completion_waits"].should eq(before["completion_waits"] + 1)
              graph["pending"].should eq(0)
              context.readbacks["terminal"].should eq((reads["terminal"]? || 0) + 8)
              bindings = context.weight_wrapper_first_command_bindings
              bindings[:command_index].should eq(before["submitted"])
              bindings[:observed].should be_true
              bindings[:submitted].should be_true
              bindings[:raw_weight_bindings].should eq(1)
              bindings[:shards].size.should eq(1)
              bindings[:shards][0][:bindings].should eq(1)
            end
          end
        end
      ensure
        if previous
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
      end
    end

    it "preserves packed expert bindings and cached GPU F32 conversions" do
      path = File.tempname("drama3-wrapper-experts", ".bin")
      begin
        io = IO::Memory.new
        io.write(Bytes.new(22, 0_u8))
        32.times { io.write_bytes(0x11111111_u32, IO::ByteFormat::LittleEndian) }
        [1_f32, 2_f32, 3_f32, 4_f32].each { |v| io.write_bytes(v, IO::ByteFormat::LittleEndian) }
        4.times { io.write_bytes(-0.125_f32, IO::ByteFormat::LittleEndian) }
        File.write(path, io.to_slice)
        owner = ML::Audio::Drama3::MappedFile.new(path)
        packed = ML::Audio::Drama3::Tensor.new([2, 2, 8], "U32", owner.slice(22_i64, 128_i64), owner)
        scales = ML::Audio::Drama3::Tensor.new([2, 2, 1], "F32", owner.slice(150_i64, 16_i64), owner)
        biases = ML::Audio::Drama3::Tensor.new([2, 2, 1], "F32", owner.slice(166_i64, 16_i64), owner)
        matrix = ML::Audio::Drama3::Matrix.new(packed, scales, biases, 4, 64)
        context = ML::Audio::Drama3::Resident::Context.new
        context.enable_weight_wrapper_probe!
        converted = context.tensor(scales)
        context.read(converted, "terminal").should eq([1_f32, 2_f32, 3_f32, 4_f32])
        3.times do |step|
          context.prepare_weight_wrappers(step == 2, clear_views: step == 1)
          before_cache = context.converted_weight_cache_signature
          context.tensor(scales).should be(converted)
          input = context.upload(Array(Float32).new(64, 0.5_f32))
          ids = context.upload([1_f32, 0_f32])
          actual = context.read(context.experts(matrix, input, ids), "terminal")
          actual.map(&.unsafe_as(UInt32)).should eq([92_f32, 124_f32, 28_f32, 60_f32].map(&.unsafe_as(UInt32)))
          context.converted_weight_cache_signature.should eq(before_cache)
        end
      ensure
        File.delete(path) if File.exists?(path)
      end
    end

    it "counts successful first-command bindings rather than view lookups or later commands" do
      drama3_wrapper_fixture do |_, tensor|
        drama3_wrapper_fixture do |_, unused|
          bank = ML::Audio::Drama3::MetalBackend::WeightBufferBank.new
          raw = bank.tensor_buffer(tensor)
          bank.tensor_buffer(unused)
          bank.start_first_command(17_i64)
          # Lookup and reconstruction alone do not expose a shard to Metal.
          bank.tensor_buffer(unused)
          activation = ML::MetalBuffer.from_array([1_f32, 2_f32])
          command = ML::Metal::CommandBuffer.new
          bank.first_command_started!
          encoder = ML::Audio::Drama3::Resident::HostEncoder.new(command, ML::Audio::Drama3::HostProfile.new(false), bank)
          encoder.set_buffer(raw.buffer, 0)
          encoder.set_buffer(raw.buffer, 1)
          encoder.set_buffer(activation, 2)
          encoder.end_encoding
          command.commit_and_wait
          bank.first_command_submitted!
          snapshot = bank.first_command_bindings
          snapshot[:command_index].should eq(17)
          snapshot[:buffer_bindings].should eq(3)
          snapshot[:raw_weight_bindings].should eq(2)
          snapshot[:shards].map { |s| s[:bindings] }.should eq([2_i64, 0_i64])
          # A later bind must not mutate the sealed first-command result.
          bank.record_binding(raw.buffer)
          bank.first_command_bindings.should eq(snapshot)
          activation.release
        end
      end
    end

    it "excludes cached converted weights and refuses unpublished or aborted snapshots" do
      drama3_wrapper_fixture do |_, tensor|
        context = ML::Audio::Drama3::Resident::Context.new
        context.enable_weight_wrapper_probe!
        cached = context.tensor(tensor)
        context.read(cached, "terminal")
        context.begin_weight_wrapper_job!
        expect_raises(ArgumentError, /submitted/) { context.weight_wrapper_first_command_bindings }
        context.tensor(tensor).should be(cached)
        context.read(context.add(cached, context.zeros(4)), "terminal")
        snapshot = context.weight_wrapper_first_command_bindings
        snapshot[:buffer_bindings].should be > 0
        snapshot[:raw_weight_bindings].should eq(0)
        snapshot[:shards].map { |s| s[:bindings] }.should eq([0_i64])
        context.read(context.zeros(1), "terminal")
        context.weight_wrapper_first_command_bindings.should eq(snapshot)
        context.begin_weight_wrapper_job!
        context.matvec(ML::Audio::Drama3::Matrix.new(tensor), context.upload([1_f32, 2_f32]))
        before = context.graph_stats
        expect_raises(ArgumentError, /terminal/) { context.begin_weight_wrapper_job! }
        expect_raises(ArgumentError, /terminal/) { context.weight_wrapper_first_command_bindings }
        context.graph_stats.should eq(before)
        context.abort
        expect_raises(ArgumentError, /aborted/) { context.weight_wrapper_first_command_bindings }
      end
    end

    it "refuses changes during encoded, queued, workspace, late or aborted use without draining" do
      context = ML::Audio::Drama3::Resident::Context.new
      expect_raises(ArgumentError, /enabled/) { context.prepare_weight_wrappers(true) }
      context.enable_weight_wrapper_probe!
      context.zeros(4)
      before = context.graph_stats
      expect_raises(ArgumentError, /terminal/) { context.prepare_weight_wrappers(true) }
      context.graph_stats.should eq(before)
      context.submit
      before = context.graph_stats
      if context.graph_enabled?
        expect_raises(ArgumentError, /terminal/) { context.prepare_weight_wrappers(true) }
        context.graph_stats.should eq(before)
      end
      context.finish
      workspace = context.new_workspace
      context.with_workspace(workspace) do
        expect_raises(ArgumentError, /terminal/) { context.prepare_weight_wrappers(true) }
      end
      context.release_workspace(workspace)
      context.prepare_weight_wrappers(true)
      expect_raises(ArgumentError, /before/) { context.enable_weight_wrapper_probe! }
      context.abort
      before = context.graph_stats
      expect_raises(ArgumentError, /aborted/) { context.prepare_weight_wrappers(true) }
      context.graph_stats.should eq(before)
      late = ML::Audio::Drama3::Resident::Context.new
      late.read(late.zeros(1), "terminal")
      expect_raises(ArgumentError, /before/) { late.enable_weight_wrapper_probe! }
    end
  end
{% end %}
