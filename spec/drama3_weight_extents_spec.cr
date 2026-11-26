require "spec"
require "../src/ml/audio/drama3/resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private def drama3_extent_fixture(&)
    page = ML::Audio::Drama3::MetalBackend::Platform.getpagesize.to_i64
    path = File.tempname("drama3-extents", ".bin")
    bytes = Bytes.new((4 * page + 37).to_i32, 0_u8)
    [1_f32, 2_f32, 3_f32, 4_f32].each_with_index do |v, i|
      IO::ByteFormat::LittleEndian.encode(v, bytes[(page - 8 + i * 4).to_i32, 4])
      IO::ByteFormat::LittleEndian.encode(v, bytes[(2 * page + 150 + i * 4).to_i32, 4])
      IO::ByteFormat::LittleEndian.encode(-0.125_f32, bytes[(3 * page + 166 + i * 4).to_i32, 4])
    end
    32.times do |i|
      IO::ByteFormat::LittleEndian.encode(0x11111111_u32, bytes[(page + 22 + i * 4).to_i32, 4])
    end
    File.write(path, bytes)
    owner = ML::Audio::Drama3::MappedFile.new(path)
    dense = ML::Audio::Drama3::Tensor.new([2, 2], "F32", owner.slice(page - 8, 16_i64), owner)
    packed = ML::Audio::Drama3::Tensor.new([2, 2, 8], "U32", owner.slice(page + 22, 128_i64), owner)
    scales = ML::Audio::Drama3::Tensor.new([2, 2, 1], "F32", owner.slice(2 * page + 150, 16_i64), owner)
    biases = ML::Audio::Drama3::Tensor.new([2, 2, 1], "F32", owner.slice(3 * page + 166, 16_i64), owner)
    begin
      yield page, owner, dense, packed, scales, biases
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  describe "DRAMA-3 diagnostic mapped resource extents" do
    it "rounds and rebases full tensor intervals and rejects an undersized or foreign view" do
      drama3_extent_fixture do |page, owner, dense, packed, _, _|
        view = ML::Audio::Drama3::MetalBackend::MappedBuffer.new(owner, page + 22, 128_i64)
        view.file_offset.should eq(page)
        view.buffer.size.should eq(page)
        view.buffer.contents.should eq((owner.pointer + page).as(Void*))
        raw = ML::Audio::Drama3::MetalBackend::TensorBuffer.new(packed, true, view)
        raw.offset.should eq(22)
        raw.buffer.should be(view.buffer)
        expect_raises(ArgumentError, /range/) do
          ML::Audio::Drama3::MetalBackend::TensorBuffer.new(dense, true, view)
        end
        drama3_extent_fixture do |_, _, _, foreign, _, _|
          expect_raises(ArgumentError, /owner/) do
            ML::Audio::Drama3::MetalBackend::TensorBuffer.new(foreign, true, view)
          end
        end
        view.buffer.release
        [page - 1, page, page + 1].each do |start|
          [1_i64, page - 1, page, page + 1].each do |count|
            mapped = ML::Audio::Drama3::MetalBackend::MappedBuffer.new(owner, start, count)
            mapped.file_offset.should eq(start // page * page)
            mapped.buffer.size.should eq(((start + count + page - 1) // page) * page - mapped.file_offset)
            mapped.buffer.release
          end
        end
        tail = ML::Audio::Drama3::MetalBackend::MappedBuffer.new(owner, owner.size - 1, 1_i64)
        tail.buffer.size.should eq(page)
        tail.buffer.release
        [{-1_i64, 1_i64}, {0_i64, 0_i64}, {0_i64, -1_i64},
         {owner.size, 1_i64}, {0_i64, Int64::MAX}].each do |start, count|
          expect_raises(ArgumentError, /range/) do
            ML::Audio::Drama3::MetalBackend::MappedBuffer.new(owner, start, count)
          end
        end
      end
    end

    it "deduplicates equal envelopes and reports overlapping pages separately from unique coverage" do
      drama3_extent_fixture do |page, owner, dense, packed, scales, biases|
        bank = ML::Audio::Drama3::MetalBackend::WeightBufferBank.new
        whole = bank.tensor_buffer(dense).buffer
        [packed, scales, biases].each { |tensor| bank.tensor_buffer(tensor) }
        stats = bank.prepare_extents(true)
        whole.valid?.should be_false
        stats["retired"].should eq(1)
        stats["constructed"].should eq(4)
        stats["same_backing"].should eq(1)
        stats["narrow"].should eq(1)
        stats["layout_changed"].should eq(1)
        stats["mapped_bytes"].should eq(5 * page)
        stats["unique_mapped_bytes"].should eq(4 * page)
        stats["overlapping_bytes"].should eq(page)
        alias_tensor = ML::Audio::Drama3::Tensor.new([2], "F32", owner.slice(page + 30, 8_i64), owner)
        bank.tensor_buffer(alias_tensor).buffer.should be(bank.tensor_buffer(packed).buffer)
        bank.tensor_buffer(dense).offset.should eq(page - 8)
        bank.tensor_buffer(scales).offset.should eq(150)
        narrow = bank.tensor_buffer(packed).buffer
        sham = bank.prepare_extents(true, clear_views: true)
        sham["constructed"].should eq(0)
        sham["retired"].should eq(0)
        sham["layout_changed"].should eq(0)
        bank.tensor_buffer(packed).buffer.should be(narrow)
        restored = bank.prepare_extents(false)
        restored["retired"].should eq(4)
        restored["constructed"].should eq(1)
        restored["mapped_bytes"].should eq(5 * page)
        restored["overlapping_bytes"].should eq(0)
        narrow.valid?.should be_false
        bank.tensor_buffer(packed).offset.should eq(page + 22)
      end
    end

    it "preserves dense, dynamic expert and conversion results and ledgers at graph depths zero and two" do
      previous = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ["0", "2"].each do |depth|
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth
          drama3_extent_fixture do |_, _, dense, packed, scales, biases|
            context = ML::Audio::Drama3::Resident::Context.new
            context.enable_weight_wrapper_probe!
            cached = context.tensor(scales)
            context.read(cached, "terminal").should eq([1_f32, 2_f32, 3_f32, 4_f32])
            matrix = ML::Audio::Drama3::Matrix.new(packed, scales, biases, 4, 64)
            [false, true, true, false].each_with_index do |narrow, index|
              graph = context.graph_stats
              reads = context.readbacks.dup
              signature = context.converted_weight_cache_signature
              context.prepare_weight_extents(narrow, clear_views: index == 2)
              context.graph_stats.should eq(graph)
              context.readbacks.should eq(reads)
              context.converted_weight_cache_signature.should eq(signature)
              context.tensor(scales).should be(cached)
              context.begin_weight_wrapper_job!
              input = context.upload(Array(Float32).new(64, 0.5_f32))
              ids = context.upload([1_f32, 0_f32])
              result = context.read(context.experts(matrix, input, ids), "terminal")
              result.map(&.unsafe_as(UInt32)).should eq([92_f32, 124_f32, 28_f32, 60_f32].map(&.unsafe_as(UInt32)))
              binding = context.weight_wrapper_first_command_bindings
              binding[:raw_weight_bindings].should eq(3)
              binding[:resources].select { |r| r[:bindings] > 0 }.size.should eq(narrow ? 3 : 1)
              result = context.read(context.matvec(ML::Audio::Drama3::Matrix.new(dense), context.upload([1_f32, 2_f32])), "terminal")
              result.map(&.unsafe_as(UInt32)).should eq([5_f32, 11_f32].map(&.unsafe_as(UInt32)))
              context.graph_stats["submitted"].should eq(graph["submitted"] + 2)
              context.graph_stats["completion_waits"].should eq(graph["completion_waits"] + 2)
              context.graph_stats["pending"].should eq(0)
              context.readbacks["terminal"].should eq((reads["terminal"]? || 0) + 24)
              context.converted_weight_cache_signature.should eq(signature)
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

    it "converts a tensor whose page view starts after the beginning of its shard" do
      drama3_extent_fixture do |_, _, _, _, scales, _|
        context = ML::Audio::Drama3::Resident::Context.new
        context.enable_weight_wrapper_probe!
        context.prepare_weight_extents(true)
        context.read(context.tensor(scales), "terminal").should eq([1_f32, 2_f32, 3_f32, 4_f32])
        context.graph_stats["submitted"].should eq(1)
        context.graph_stats["completion_waits"].should eq(1)
        context.readbacks.should eq({"terminal" => 16_i64})
      end
    end

    it "invalidates a submitted binding receipt only when its Metal resources change" do
      drama3_extent_fixture do |_, _, dense, _, _, _|
        context = ML::Audio::Drama3::Resident::Context.new
        context.enable_weight_wrapper_probe!
        matrix = ML::Audio::Drama3::Matrix.new(dense)
        [false, true].each do |narrow|
          context.prepare_weight_extents(narrow)
          context.begin_weight_wrapper_job!
          context.read(context.matvec(matrix, context.upload([1_f32, 2_f32])), "terminal").should eq([5_f32, 11_f32])
          receipt = context.weight_wrapper_first_command_bindings
          context.prepare_weight_extents(narrow, clear_views: true)
          context.weight_wrapper_first_command_bindings.should eq(receipt)
          if narrow
            context.prepare_weight_wrappers(true)
          else
            context.prepare_weight_extents(true)
          end
          expect_raises(ArgumentError, /not been submitted/) { context.weight_wrapper_first_command_bindings }
        end
      end
    end

    it "keeps multi-page scales and biases complete for GPU-selected experts" do
      page = ML::Audio::Drama3::MetalBackend::Platform.getpagesize.to_i64
      groups = (page // 16 + 1).to_i32
      packed_size = groups.to_i64 * 128
      affine_size = groups.to_i64 * 16
      packed_start = page + 20
      scales_start = packed_start + packed_size + page + 148
      biases_start = scales_start + affine_size + page + 164
      bytes = Bytes.new((biases_start + affine_size + 37).to_i32, 0_u8)
      (packed_size // 4).times do |i|
        IO::ByteFormat::LittleEndian.encode(0x11111111_u32, bytes[(packed_start + i * 4).to_i32, 4])
      end
      (groups * 4).times do |i|
        IO::ByteFormat::LittleEndian.encode((i // groups + 1).to_f32, bytes[(scales_start + i * 4).to_i32, 4])
        IO::ByteFormat::LittleEndian.encode(-0.125_f32, bytes[(biases_start + i * 4).to_i32, 4])
      end
      path = File.tempname("drama3-extents-affine", ".bin")
      begin
        File.write(path, bytes)
        owner = ML::Audio::Drama3::MappedFile.new(path)
        packed = ML::Audio::Drama3::Tensor.new([2, 2, groups * 8], "U32", owner.slice(packed_start, packed_size), owner)
        scales = ML::Audio::Drama3::Tensor.new([2, 2, groups], "F32", owner.slice(scales_start, affine_size), owner)
        biases = ML::Audio::Drama3::Tensor.new([2, 2, groups], "F32", owner.slice(biases_start, affine_size), owner)
        context = ML::Audio::Drama3::Resident::Context.new
        context.enable_weight_wrapper_probe!
        matrix = ML::Audio::Drama3::Matrix.new(packed, scales, biases, 4, 64)
        [false, true].each do |narrow|
          context.prepare_weight_extents(narrow)
          context.begin_weight_wrapper_job!
          output = context.experts(matrix, context.upload(Array(Float32).new(groups * 64, 0.5_f32)), context.upload([1_f32, 0_f32]))
          context.read(output, "terminal").should eq([92_f32, 124_f32, 28_f32, 60_f32].map { |v| v * groups })
          if narrow
            resources = context.weight_wrapper_first_command_bindings[:resources]
            resources.size.should eq(3)
            resources.all? { |r| r[:mapped_bytes] > page && r[:bindings] == 1 }.should be_true
          end
        end
      ensure
        File.delete(path) if File.exists?(path)
      end
    end

    it "rejects extent changes during active encoding, queued work, workspace and aborted use without draining" do
      previous = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ["0", "2"].each do |depth|
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth
          context = ML::Audio::Drama3::Resident::Context.new
          expect_raises(ArgumentError, /enabled/) { context.prepare_weight_extents(true) }
          context.enable_weight_wrapper_probe!
          context.zeros(4)
          graph = context.graph_stats
          expect_raises(ArgumentError, /terminal/) { context.prepare_weight_extents(true) }
          context.graph_stats.should eq(graph)
          context.submit
          graph = context.graph_stats
          if context.graph_enabled?
            expect_raises(ArgumentError, /terminal/) { context.prepare_weight_extents(true) }
            context.graph_stats.should eq(graph)
          end
          context.finish
          workspace = context.new_workspace
          context.with_workspace(workspace) do
            expect_raises(ArgumentError, /terminal/) { context.prepare_weight_extents(true) }
          end
          context.release_workspace(workspace)
          context.prepare_weight_extents(true)
          context.abort
          graph = context.graph_stats
          expect_raises(ArgumentError, /aborted/) { context.prepare_weight_extents(false) }
          context.graph_stats.should eq(graph)
        end
      ensure
        if previous
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
      end
    end
  end
{% end %}
