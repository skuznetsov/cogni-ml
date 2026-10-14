require "spec"
require "../src/ml/audio/drama3/native"
require "../src/ml/audio/drama3/metal"

include ML::Audio::Drama3

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  describe "DRAMA-3 Metal affine projections" do
    it "preserves the CPU affine group reduction for Q4 and Q8 mapped tensors" do
      path = File.join(__DIR__, "support/drama3/reference_fixture")
      store = WeightStore.new(path)
      expected = JSON.parse(File.read(File.join(path, "quantization_expected.json"))).as_h
      previous = ENV["DRAMA3_BACKEND"]?
      begin
        expected.each do |name, fixture|
          matrix = store.matrix(name + ".weight")
          input = fixture["input"].as_a.map { |v| v.as_f.to_f32 }
          ENV["DRAMA3_BACKEND"] = "cpu"
          cpu = matrix.matvec(input)
          ENV["DRAMA3_BACKEND"] = "metal"
          MetalBackend.enabled?.should be_true
          actual = MetalBackend.affine(matrix, input)
          actual.should eq(cpu)
          reference = fixture["matvec"].as_a.map { |v| v.as_f.to_f32 }
          actual.zip(reference).each { |a, b| a.should be_close(b, 2e-6) }
        end
      ensure
        MetalBackend.reset!
        if previous
          ENV["DRAMA3_BACKEND"] = previous
        else
          ENV.delete("DRAMA3_BACKEND")
        end
      end
    end

    it "matches cancellation across 65 groups, expert offsets, mmap tails, and reset" do
      previous = ENV["DRAMA3_BACKEND"]?
      path = File.tempname("drama3-affine", ".bin")
      begin
        [4, 8].each do |bits|
          rows, experts, groups, group_size = 3, 2, 65, 64
          cols = groups * group_size
          per_word = 32 // bits
          count = experts * rows * (cols // per_word)
          io = IO::Memory.new
          io.write(Bytes.new(22, 0_u8))
          count.times do |i|
            word = 0_u32
            per_word.times { |lane| word |= (((i * 13 + lane * 7) % (1 << bits)).to_u32 << (lane * bits)) }
            io.write_bytes(word, IO::ByteFormat::LittleEndian)
          end
          File.write(path, io.to_slice)
          owner = MappedFile.new(path)
          packed = Tensor.new([experts, rows, cols // per_word], "U32", owner.slice(22, count.to_i64 * 4), owner)
          scales = Tensor.new([experts, rows, groups], Array(Float32).new(experts * rows * groups) { |i| (i % 9 + 1).to_f32 * 0.00073_f32 })
          biases = Tensor.new([experts, rows, groups], Array(Float32).new(experts * rows * groups) { |i| (i % 11 - 5).to_f32 * 0.017_f32 })
          input = Array(Float32).new(cols) { |i| (i % 17 - 8).to_f32 / 7_f32 }
          matrix = Matrix.new(packed, scales, biases, bits, group_size, expert: 1)
          ENV["DRAMA3_BACKEND"] = "cpu"
          cpu = matrix.matvec(input)
          ENV["DRAMA3_BACKEND"] = "metal"
          matrix.matvec(input).should eq(cpu)
          MetalBackend.reset!
          matrix.matvec(input).should eq(cpu)
        end
      ensure
        MetalBackend.reset!
        if previous
          ENV["DRAMA3_BACKEND"] = previous
        else
          ENV.delete("DRAMA3_BACKEND")
        end
        File.delete(path) if File.exists?(path)
      end
    end

    it "matches CPU affine output for raw BF16/F16 scales and biases at unaligned offsets" do
      cases = [
        {
          scale_dtype: "BF16", bias_dtype: "BF16",
          scale_words: [0x3f00_u16, 0x3f80_u16, 0x3f80_u16, 0x4000_u16, 0x3e80_u16, 0x3f00_u16, 0x4000_u16, 0x3f00_u16],
          bias_words: [0xbf00_u16, 0x3f00_u16, 0x0000_u16, 0x3d00_u16, 0x3e00_u16, 0xbe00_u16, 0x3f00_u16, 0x0000_u16],
        },
        {
          scale_dtype: "F16", bias_dtype: "F16",
          scale_words: [0x0001_u16, 0x8001_u16, 0x03ff_u16, 0x83ff_u16, 0x3c00_u16, 0x3800_u16, 0x4000_u16, 0x0000_u16],
          bias_words: [0x8001_u16, 0x0001_u16, 0x3c00_u16, 0xbc00_u16, 0x03ff_u16, 0x83ff_u16, 0x3800_u16, 0x0000_u16],
        },
        {
          scale_dtype: "BF16", bias_dtype: "F16",
          scale_words: [0x3f00_u16, 0x3f80_u16, 0x3f80_u16, 0x4000_u16, 0x3e80_u16, 0x3f00_u16, 0x4000_u16, 0x3f00_u16],
          bias_words: [0x0001_u16, 0x8001_u16, 0x3c00_u16, 0xbc00_u16, 0x03ff_u16, 0x83ff_u16, 0x3800_u16, 0x0000_u16],
        },
      ]
      previous = ENV["DRAMA3_BACKEND"]?
      begin
        cases.each do |scenario|
          path = File.tempname("drama3-affine-raw", ".bin")
          begin
            rows, experts, groups, group_size = 2, 2, 2, 64
            cols = groups * group_size
            per_word = 8
            count = experts * rows * groups
            word_count = experts * rows * (cols // per_word)
            weight_offset = 22_i64
            scales_offset = weight_offset + word_count.to_i64 * 4 + 1
            biases_offset = scales_offset + count.to_i64 * 2

            io = IO::Memory.new
            io.write(Bytes.new(weight_offset.to_i, 0_u8))
            word_count.times do |i|
              word = 0_u32
              per_word.times { |lane| word |= (((i * 13 + lane * 7) % 16).to_u32 << (lane * 4)) }
              io.write_bytes(word, IO::ByteFormat::LittleEndian)
            end
            io.write(Bytes.new(1, 0xa5_u8))
            scenario[:scale_words].each { |word| io.write_bytes(word, IO::ByteFormat::LittleEndian) }
            scenario[:bias_words].each { |word| io.write_bytes(word, IO::ByteFormat::LittleEndian) }
            File.write(path, io.to_slice)

            owner = MappedFile.new(path)
            packed = Tensor.new([experts, rows, cols // per_word], "U32", owner.slice(weight_offset, word_count.to_i64 * 4), owner)
            scales = Tensor.new([experts, rows, groups], scenario[:scale_dtype], owner.slice(scales_offset, count.to_i64 * 2), owner)
            biases = Tensor.new([experts, rows, groups], scenario[:bias_dtype], owner.slice(biases_offset, count.to_i64 * 2), owner)
            input = Array(Float32).new(cols) { |i| ((i % 5) - 2).to_f32 * 1_048_576_f32 }

            MetalBackend.tensor_buffer(packed, true).offset.should eq(weight_offset)
            MetalBackend.tensor_buffer(scales, true).offset.should eq(scales_offset)
            MetalBackend.tensor_buffer(biases, true).offset.should eq(biases_offset)

            experts.times do |expert|
              matrix = Matrix.new(packed, scales, biases, 4, group_size, expert: expert)
              ENV["DRAMA3_BACKEND"] = "cpu"
              expected = matrix.matvec(input)
              ENV["DRAMA3_BACKEND"] = "metal"
              matrix.matvec(input).should eq(expected)
            end
          ensure
            MetalBackend.reset!
            File.delete(path) if File.exists?(path)
          end
        end
      ensure
        MetalBackend.reset!
        if previous
          ENV["DRAMA3_BACKEND"] = previous
        else
          ENV.delete("DRAMA3_BACKEND")
        end
      end
    end

    it "batches distinct expert matvecs with one command and rejects invalid counts and widths" do
      previous = ENV["DRAMA3_BACKEND"]?
      previous_profile = ENV["DRAMA3_PROFILE"]?
      path = File.tempname("drama3-affine-batch", ".bin")
      profile_path = File.tempname("drama3-affine-batch-profile", ".json")
      begin
        rows, experts, cols, group_size = 2, 2, 128, 64
        q4_words = experts * rows * (cols // 8)
        q8_words = experts * rows * (cols // 4)
        q4_offset = 22_i64
        q8_offset = q4_offset + q4_words.to_i64 * 4 + 1

        io = IO::Memory.new
        io.write(Bytes.new(q4_offset.to_i, 0_u8))
        q4_words.times do |i|
          word = 0_u32
          8.times { |lane| word |= (((i * 7 + lane * 3) % 16).to_u32 << (lane * 4)) }
          io.write_bytes(word, IO::ByteFormat::LittleEndian)
        end
        io.write(Bytes.new(1, 0xa5_u8))
        q8_words.times do |i|
          word = 0_u32
          4.times { |lane| word |= (((i * 11 + lane * 31) % 256).to_u32 << (lane * 8)) }
          io.write_bytes(word, IO::ByteFormat::LittleEndian)
        end
        File.write(path, io.to_slice)

        owner = MappedFile.new(path)
        q4_tensor = Tensor.new([experts, rows, cols // 8], "U32", owner.slice(q4_offset, q4_words.to_i64 * 4), owner)
        q8_tensor = Tensor.new([experts, rows, cols // 4], "U32", owner.slice(q8_offset, q8_words.to_i64 * 4), owner)
        groups = cols // group_size
        count = experts * rows * groups
        q4_scales = Tensor.new([experts, rows, groups], Array(Float32).new(count) { |i| (i % 7 + 1).to_f32 * 0.00073_f32 })
        q4_biases = Tensor.new([experts, rows, groups], Array(Float32).new(count) { |i| (i % 9 - 4).to_f32 * 0.013_f32 })
        q8_scales = Tensor.new([experts, rows, groups], Array(Float32).new(count) { |i| (i % 5 + 1).to_f32 * 0.00031_f32 })
        q8_biases = Tensor.new([experts, rows, groups], Array(Float32).new(count) { |i| (i % 11 - 5).to_f32 * 0.007_f32 })
        matrices = [
          Matrix.new(q4_tensor, q4_scales, q4_biases, 4, group_size, expert: 1),
          Matrix.new(q8_tensor, q8_scales, q8_biases, 8, group_size, expert: 0),
        ]
        inputs = [
          Array(Float32).new(cols) { |i| ((i % 11) - 5).to_f32 / 11_f32 },
          Array(Float32).new(cols) { |i| ((i % 13) - 6).to_f32 * 0.02_f32 },
        ]

        ENV["DRAMA3_BACKEND"] = "cpu"
        expected = Array(Vector).new(matrices.size) { |i| matrices[i].matvec(inputs[i]) }
        ENV["DRAMA3_BACKEND"] = "metal"
        individual = Array(Vector).new(matrices.size) { |i| matrices[i].matvec(inputs[i]) }
        ENV["DRAMA3_PROFILE"] = profile_path
        Profile.configure_from_env!
        batched = MetalBackend.affine_batch(matrices, inputs)
        batched.should eq(individual)
        batched.should eq(expected)
        submits = Profile.snapshot.reduce(0_i64) do |sum, stat|
          sum + (stat.path.ends_with?("metal.submit_wait") ? stat.calls : 0_i64)
        end
        submits.should eq(1_i64)

        expect_raises(ArgumentError, /count mismatch/) { MetalBackend.affine_batch(matrices, [inputs[0]]) }
        short_input = Array(Float32).new(cols - 1, 0_f32)
        expect_raises(ArgumentError, /input width/) { MetalBackend.affine_batch(matrices, [inputs[0], short_input]) }
      ensure
        MetalBackend.reset!
        if previous
          ENV["DRAMA3_BACKEND"] = previous
        else
          ENV.delete("DRAMA3_BACKEND")
        end
        if previous_profile
          ENV["DRAMA3_PROFILE"] = previous_profile
        else
          ENV.delete("DRAMA3_PROFILE")
        end
        Profile.configure_from_env!
        File.delete(path) if File.exists?(path)
        File.delete(profile_path) if File.exists?(profile_path)
      end
    end
  end
{% end %}

describe "DRAMA-3 backend selection" do
  it "rejects unknown backends before inference" do
    previous = ENV["DRAMA3_BACKEND"]?
    begin
      ENV["DRAMA3_BACKEND"] = "metla"
      expect_raises(ArgumentError, /cpu or metal/) { MetalBackend.enabled? }
    ensure
      if previous
        ENV["DRAMA3_BACKEND"] = previous
      else
        ENV.delete("DRAMA3_BACKEND")
      end
    end
  end
end
