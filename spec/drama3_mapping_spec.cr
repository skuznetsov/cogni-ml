require "spec"
require "../src/ml/audio/drama3/native"
require "../src/ml/audio/drama3/resident"

include ML::Audio::Drama3

private def drama3_mapping_fixture
  File.tempfile("drama3-mapping-", ".safetensors") do |file|
    [1_f32, 2_f32, 3_f32, 4_f32].each { |value| file.write_bytes(value, IO::ByteFormat::LittleEndian) }
  end
end

# Return only the tensor: its owner must survive after this caller root is gone.
private def drama3_mapping_tensor(path : String) : Tensor
  owner = MappedFile.new(path)
  Tensor.new([2, 2], "F32", owner.slice(0, 16), owner)
end

describe "DRAMA-3 read-only checkpoint mapping" do
  it "counts shared mapping pages once and only cached F32 conversions separately" do
    file = drama3_mapping_fixture
    begin
      owner = MappedFile.new(file.path)
      matrix = Tensor.new([2, 2], "F32", owner.slice(0, 16), owner)
      tensors = {
        "backbone.proj.weight"                        => matrix,
        "codec.proj.weight"                           => matrix,
        "backbone.norm.weight"                        => Tensor.new([2], "F32", owner.slice(0, 8), owner),
        "backbone.layers.0.linear_attn.conv1d.weight" => Tensor.new([1, 2, 2], "F32", owner.slice(0, 16), owner),
        "packed.weight"                               => Tensor.new([2, 2], "U32", owner.slice(0, 16), owner),
      }
      inventory = WeightStore.new(tensors).resident_memory_inventory
      page = LibC.sysconf(LibC::SC_PAGESIZE).to_i64
      inventory[:mapped_bytes].should eq(page)
      inventory[:largest_checkpoint_mapped_buffer_bytes].should eq(page)
      inventory[:converted_static_bytes].should eq(40_i64)
      inventory[:largest_converted_buffer_bytes].should eq(16_i64)
    ensure
      file.delete
    end
  end

  it "preserves mapped values and rejects escaping ranges in both modes" do
    previous = ENV["DRAMA3_MMAP_SHARED"]?
    file = drama3_mapping_fixture
    begin
      {"0", "1"}.each do |mode|
        ENV["DRAMA3_MMAP_SHARED"] = mode
        owner = MappedFile.new(file.path)
        tensor = Tensor.new([2, 2], "F32", owner.slice(0, 16), owner)
        tensor.values.should eq([1_f32, 2_f32, 3_f32, 4_f32])
        expect_raises(ArgumentError, /outside file/) { owner.slice(-1, 4) }
        expect_raises(ArgumentError, /outside file/) { owner.slice(12, 8) }
      end
    ensure
      if previous
        ENV["DRAMA3_MMAP_SHARED"] = previous
      else
        ENV.delete("DRAMA3_MMAP_SHARED")
      end
      file.delete
    end
  end

  it "rejects invalid mapping mode configuration" do
    previous = ENV["DRAMA3_MMAP_SHARED"]?
    file = drama3_mapping_fixture
    begin
      ENV["DRAMA3_MMAP_SHARED"] = "yes"
      expect_raises(ArgumentError, /must be 0 or 1/) { MappedFile.new(file.path) }
    ensure
      if previous
        ENV["DRAMA3_MMAP_SHARED"] = previous
      else
        ENV.delete("DRAMA3_MMAP_SHARED")
      end
      file.delete
    end
  end

  {% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
    it "retains read-only weights through queued GPU use and GC without changing the file" do
      previous = ENV["DRAMA3_MMAP_SHARED"]?
      depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      file = drama3_mapping_fixture
      before = File.read(file.path)
      ctx = nil.as(Resident::Context?)
      begin
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
        context = Resident::Context.new
        ctx = context
        input = context.upload([5_f32, 7_f32])
        outputs = {"0", "1"}.map do |mode|
          ENV["DRAMA3_MMAP_SHARED"] = mode
          matrix = Matrix.new(drama3_mapping_tensor(file.path))
          output = context.matvec(matrix, input)
          context.submit
          output
        end
        context.graph_stats["pending"].should eq(2)
        GC.collect
        outputs.each { |output| context.read(output, "terminal").should eq([19_f32, 43_f32]) }
        context.graph_stats["pending"].should eq(0)
        context.graph_stats["retained_bytes"].should eq(0)
        File.read(file.path).should eq(before)
      ensure
        ctx.try(&.abort)
        # All referencing commands are terminal before releasing weight views.
        MetalBackend.reset!
        if previous
          ENV["DRAMA3_MMAP_SHARED"] = previous
        else
          ENV.delete("DRAMA3_MMAP_SHARED")
        end
        if depth
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
        file.delete
      end
    end
  {% end %}
end
