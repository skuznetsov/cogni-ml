require "spec"
require "../src/ml/audio/drama3/resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private def drama3_dense_with_device(&block : -> T) : T forall T
    previous_backend = ENV["DRAMA3_BACKEND"]?
    begin
      ENV["DRAMA3_BACKEND"] = "metal"
      device = ML::Metal::Device.instance
      pending!("Metal is unavailable or lacks unified memory") unless device.available? && device.has_unified_memory?
      ML::Audio::Drama3::MetalBackend.enabled?.should be_true
      yield
    ensure
      if previous_backend
        ENV["DRAMA3_BACKEND"] = previous_backend
      else
        ENV.delete("DRAMA3_BACKEND")
      end
    end
  end

  private def drama3_dense_values(rows : Int32, cols : Int32) : Array(Float32)
    Array(Float32).new(rows * cols) do |index|
      row, col = index // cols, index % cols
      if row == 1 && cols >= 33
        case col
        when  0 then 2048_f32
        when  1 then 0.25_f32
        when 32 then -2048_f32
        else         0_f32
        end
      elsif row == 2 && col == cols - 1
        -0.75_f32
      else
        (((index * 13 + 5) % 17) - 8).to_f32 / 8_f32
      end
    end
  end

  private def drama3_dense_input(cols : Int32) : Array(Float32)
    values = Array(Float32).new(cols) { |index| (((index * 7 + 3) % 19) - 9).to_f32 / 8_f32 }
    values[0] = 8192_f32
    values[1] = 1_f32
    values[32] = 8192_f32 if cols >= 33
    values
  end

  # Fixtures use finite, exactly representable values (plus zero). This simple
  # conversion keeps their raw F16 bytes independent of Metal code.
  private def drama3_dense_f16_bits(value : Float32) : UInt16
    return 0_u16 if value == 0_f32
    bits = value.unsafe_as(UInt32)
    exponent = ((bits >> 23) & 0xff).to_i32 - 127 + 15
    raise ArgumentError.new("DRAMA-3 dense F16 fixture must be a normal value") unless exponent > 0 && exponent < 31
    sign = (bits >> 16) & 0x8000
    fraction = (bits & 0x7fffff) >> 13
    (sign | (exponent.to_u32 << 10) | fraction).to_u16
  end

  private def drama3_dense_write_value(io : IO, dtype : String, value : Float32) : Nil
    case dtype
    when "F32"
      io.write_bytes(value, IO::ByteFormat::LittleEndian)
    when "BF16"
      io.write_bytes((value.unsafe_as(UInt32) >> 16).to_u16, IO::ByteFormat::LittleEndian)
    when "F16"
      io.write_bytes(drama3_dense_f16_bits(value), IO::ByteFormat::LittleEndian)
    else
      raise ArgumentError.new("unsupported DRAMA-3 dense fixture dtype #{dtype}")
    end
  end

  describe "DRAMA-3 resident dense reduction" do
    it "matches the original tree bitwise for mapped raw dtypes, tails, and cancellation" do
      previous_reduction = ENV["DRAMA3_DENSE_REDUCTION"]?
      begin
        ["F32", "BF16", "F16"].each do |dtype|
          [31, 32, 33, 63, 64, 65, 2048, 2049].each do |cols|
            offsets = dtype == "BF16" ? [21, 22] : [22]
            offsets.each do |weight_offset|
              rows = 3
              values = drama3_dense_values(rows, cols)
              path = File.tempname("drama3-dense", ".bin")
              begin
                io = IO::Memory.new
                io.write(Bytes.new(weight_offset, 0_u8))
                values.each { |value| drama3_dense_write_value(io, dtype, value) }
                File.write(path, io.to_slice)

                owner = ML::Audio::Drama3::MappedFile.new(path)
                byte_width = dtype == "F32" ? 4 : 2
                tensor = ML::Audio::Drama3::Tensor.new(
                  [rows, cols], dtype, owner.slice(weight_offset.to_i64, (values.size * byte_width).to_i64), owner,
                )
                matrix = ML::Audio::Drama3::Matrix.new(tensor)

                ENV["DRAMA3_DENSE_REDUCTION"] = "tree"
                context = ML::Audio::Drama3::Resident::Context.new
                context.dense_reduction.should eq("tree")
                ENV["DRAMA3_DENSE_REDUCTION"] = "invalid-after-init"
                input_values = drama3_dense_input(cols)
                guarded_input = Array(Float32).new(cols + 2, -37_f32)
                input_values.each_with_index { |value, index| guarded_input[index + 1] = value }
                input = context.upload(guarded_input).slice(1, cols)
                ML::Audio::Drama3::MetalBackend.tensor_buffer(tensor, true).offset.should eq(weight_offset.to_i64)
                baseline = context.matvec(matrix, input)
                candidate = context.matvec(matrix, input, dense_reduction: "simd")
                expected = context.read(baseline, "terminal")
                actual = context.read(candidate, "terminal")
                actual.map(&.unsafe_as(UInt32)).should eq(expected.map(&.unsafe_as(UInt32)))

                if cols >= 33
                  expected[1].should eq(0.25_f32)
                  left_fold = 0_f32
                  cols.times do |column|
                    left_fold += tensor[(cols + column).to_i32] * input_values[column]
                  end
                  left_fold.should eq(0_f32)
                  left_fold.unsafe_as(UInt32).should_not eq(expected[1].unsafe_as(UInt32))
                end
              ensure
                ML::Audio::Drama3::MetalBackend.reset!
                File.delete(path) if File.exists?(path)
              end
            end
          end
        end
      ensure
        if previous_reduction
          ENV["DRAMA3_DENSE_REDUCTION"] = previous_reduction
        else
          ENV.delete("DRAMA3_DENSE_REDUCTION")
        end
      end
    end

    it "rejects invalid reduction selectors and buffers owned by another context" do
      previous_reduction = ENV["DRAMA3_DENSE_REDUCTION"]?
      begin
        ENV["DRAMA3_DENSE_REDUCTION"] = "tree"
        drama3_dense_with_device do
          context = ML::Audio::Drama3::Resident::Context.new
          other = ML::Audio::Drama3::Resident::Context.new
          matrix = ML::Audio::Drama3::Matrix.new(1, 1, [1_f32])
          input = context.upload([1_f32])

          expect_raises(ArgumentError, /DRAMA3_DENSE_REDUCTION must be tree or simd/) do
            context.matvec(matrix, input, dense_reduction: "shuffle")
          end
          expect_raises(ArgumentError, /buffer belongs to a different resident context/) do
            other.matvec(matrix, input)
          end
        ensure
          ML::Audio::Drama3::MetalBackend.reset!
        end
      ensure
        if previous_reduction
          ENV["DRAMA3_DENSE_REDUCTION"] = previous_reduction
        else
          ENV.delete("DRAMA3_DENSE_REDUCTION")
        end
      end
    end
  end
{% end %}
