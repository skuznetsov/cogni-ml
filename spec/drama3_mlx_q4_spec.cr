require "spec"
require "../src/ml/audio/drama3/backbone_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  include ML::Audio::Drama3

  private def drama3_mlx_q4_with_device(&block : -> T) : T forall T
    keys = ["DRAMA3_BACKEND", "DRAMA3_COGNIGRAPH_MAX_INFLIGHT", "DRAMA3_METAL_TIMELINE", "DRAMA3_PROFILE"]
    previous = keys.to_h { |key| {key, ENV[key]?} }
    begin
      ENV["DRAMA3_BACKEND"] = "metal"
      ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
      ENV["DRAMA3_METAL_TIMELINE"] = "0"
      ENV.delete("DRAMA3_PROFILE")
      Profile.configure_from_env!
      device = ML::Metal::Device.instance
      pending!("Metal is unavailable or lacks unified memory") unless device.available? && device.has_unified_memory?
      MetalBackend.enabled?.should be_true
      yield
    ensure
      MetalBackend.reset!
      keys.each do |key|
        if value = previous[key]
          ENV[key] = value
        else
          ENV.delete(key)
        end
      end
      Profile.configure_from_env!
    end
  end

  private def drama3_mlx_q4_f16_bits(value : Float32) : UInt16
    return 0_u16 if value == 0_f32
    bits = value.unsafe_as(UInt32)
    exponent = ((bits >> 23) & 0xff).to_i32 - 127 + 15
    raise ArgumentError.new("Q4 fixture F16 value must be a normal number") unless exponent > 0 && exponent < 31
    sign = (bits >> 16) & 0x8000
    fraction = (bits & 0x7fffff) >> 13
    (sign | (exponent.to_u32 << 10) | fraction).to_u16
  end

  private def drama3_mlx_q4_write_float(io : IO, dtype : String, value : Float32) : Nil
    case dtype
    when "F32"
      io.write_bytes(value, IO::ByteFormat::LittleEndian)
    when "BF16"
      io.write_bytes((value.unsafe_as(UInt32) >> 16).to_u16, IO::ByteFormat::LittleEndian)
    when "F16"
      io.write_bytes(drama3_mlx_q4_f16_bits(value), IO::ByteFormat::LittleEndian)
    else
      raise ArgumentError.new("unsupported Q4 fixture dtype #{dtype}")
    end
  end

  private def drama3_mlx_q4_fixture(rows : Int32 = 8, cols : Int32 = 512, experts : Int32 = 109,
                                    bits : Int32 = 4, group : Int32 = 64,
                                    scale_dtype : String = "F32", bias_dtype : String = "F32")
    per_word = 32 // bits
    groups = cols // group
    word_count = experts * rows * (cols // per_word)
    affine_count = experts * rows * groups
    weight_offset = 22_i64
    weight_bytes = word_count.to_i64 * 4
    scale_bytes = affine_count.to_i64 * (scale_dtype == "F32" ? 4 : 2)
    scale_offset = weight_offset + weight_bytes + 1
    bias_offset = scale_offset + scale_bytes + 2
    path = File.tempname("drama3-mlx-q4", ".bin")

    io = IO::Memory.new
    io.write(Bytes.new(weight_offset.to_i, 0_u8))
    word_count.times do |index|
      word = 0_u32
      per_word.times do |lane|
        q = (index * 3 + (index // 16) * 7 + lane * 5 + 7) % (1 << bits)
        word |= q.to_u32 << (lane * bits)
      end
      io.write_bytes(word, IO::ByteFormat::LittleEndian)
    end
    io.write(Bytes.new(1, 0xa5_u8))
    affine_count.times do |index|
      scale = [0.5_f32, 1_f32, 1.5_f32, 2_f32][index % 4]
      drama3_mlx_q4_write_float(io, scale_dtype, scale)
    end
    io.write(Bytes.new(2, 0x5a_u8))
    affine_count.times do |index|
      bias = [-1_f32, -0.5_f32, 0_f32, 0.5_f32, 1_f32][index % 5]
      drama3_mlx_q4_write_float(io, bias_dtype, bias)
    end
    File.write(path, io.to_slice)

    owner = MappedFile.new(path)
    packed = Tensor.new([experts, rows, cols // per_word], "U32", owner.slice(weight_offset, weight_bytes), owner)
    scales = Tensor.new([experts, rows, groups], scale_dtype, owner.slice(scale_offset, scale_bytes), owner)
    bias_bytes = affine_count.to_i64 * (bias_dtype == "F32" ? 4 : 2)
    biases = Tensor.new([experts, rows, groups], bias_dtype, owner.slice(bias_offset, bias_bytes), owner)
    matrix = Matrix.new(packed, scales, biases, bits, group)
    {path: path, owner: owner, matrix: matrix, weight_offset: weight_offset,
     scale_offset: scale_offset, bias_offset: bias_offset}
  end

  private def drama3_mlx_q4_input(rank : Int32, cols : Int32) : Array(Float32)
    Array(Float32).new(cols) do |column|
      case (column * 7 + rank * 3) % 5
      when 0 then -1_f32
      when 1 then -0.5_f32
      when 2 then 0_f32
      when 3 then 0.5_f32
      else        1_f32
      end
    end
  end

  # Decode each quantized weight and accumulate the ordinary affine dot product.
  # This does not reuse the kernel's K tiling, packed-lane scaling, or SIMD sum.
  private def drama3_mlx_q4_reference(matrix : Matrix, expert : Int32, input : Array(Float32)) : Array(Float32)
    groups = matrix.cols // matrix.group_size
    Array(Float32).new(matrix.rows) do |row|
      total = 0_f32
      groups.times do |group|
        parameter = (expert * matrix.rows + row) * groups + group
        scale = matrix.scales.not_nil![parameter]
        bias = matrix.biases.not_nil![parameter]
        subtotal = 0_f32
        matrix.group_size.times do |offset|
          column = group * matrix.group_size + offset
          word_index = (expert * matrix.rows + row) * (matrix.cols // 8) + column // 8
          q = ((matrix.tensor.word(word_index) >> ((column % 8) * 4)) & 0xf_u32).to_f32
          subtotal += (scale * q + bias) * input[column]
        end
        total += subtotal
      end
      total
    end
  end

  private def drama3_mlx_q4_bits(values : Array(Float32)) : Array(UInt32)
    values.map(&.unsafe_as(UInt32))
  end

  describe "DRAMA-3 opt-in MLX Q4 expert reduction" do
    it "matches independent affine arithmetic for shuffled expert IDs and per-rank inputs, including K=1024" do
      drama3_mlx_q4_with_device do
        [{512, "F32", "F32"}, {512, "BF16", "F16"}, {512, "F16", "BF16"}, {1024, "F16", "BF16"}].each do |case_data|
          cols, scale_dtype, bias_dtype = case_data
          fixture = drama3_mlx_q4_fixture(cols: cols, scale_dtype: scale_dtype, bias_dtype: bias_dtype)
          ctx = Resident::Context.new
          begin
            MetalBackend.tensor_buffer(fixture[:matrix].tensor, true).offset.should eq(fixture[:weight_offset])
            MetalBackend.tensor_buffer(fixture[:matrix].scales.not_nil!, true).offset.should eq(fixture[:scale_offset])
            MetalBackend.tensor_buffer(fixture[:matrix].biases.not_nil!, true).offset.should eq(fixture[:bias_offset])
            fixture[:scale_offset].odd?.should be_true
            fixture[:bias_offset].odd?.should be_true

            ids = [108, 7, 29, 3, 77, 0, 51, 42]
            ids_buffer = ctx.upload(ids.map(&.to_f32))
            inputs = Array(Array(Float32)).new(ids.size) { |rank| drama3_mlx_q4_input(rank, cols) }
            input_buffer = ctx.upload(inputs.flatten)
            batched = ctx.experts(fixture[:matrix], input_buffer, ids_buffer,
              per_rank_input: true, reduction: "mlx_q4")
            scalar = Array(Resident::Buffer).new(ids.size) do |rank|
              ctx.expert(fixture[:matrix], input_buffer.slice(rank * cols, cols), ids_buffer, rank,
                reduction: "mlx_q4")
            end
            expected = ids.zip(inputs).flat_map do |expert, input|
              drama3_mlx_q4_reference(fixture[:matrix], expert, input)
            end

            default_native = nil.as(Resident::Buffer?)
            explicit_native = nil.as(Resident::Buffer?)
            if cols == 512 && scale_dtype == "F32" && bias_dtype == "F32"
              default_native = ctx.expert(fixture[:matrix], input_buffer.slice(0, cols), ids_buffer, 0)
              explicit_native = ctx.expert(fixture[:matrix], input_buffer.slice(0, cols), ids_buffer, 0,
                reduction: "native")
            end

            ctx.readbacks.should be_empty
            ctx.graph_stats["submitted"].should eq(0_i64)
            ctx.graph_stats["completion_waits"].should eq(0_i64)

            actual_batched = ctx.read(batched, "terminal")
            drama3_mlx_q4_bits(actual_batched).should eq(drama3_mlx_q4_bits(expected))
            scalar.zip(ids.zip(inputs)).each do |output, pair|
              actual = ctx.read(output, "terminal")
              reference = drama3_mlx_q4_reference(fixture[:matrix], pair[0], pair[1])
              drama3_mlx_q4_bits(actual).should eq(drama3_mlx_q4_bits(reference))
            end
            if default_output = default_native
              explicit_output = explicit_native.not_nil!
              drama3_mlx_q4_bits(ctx.read(default_output, "terminal")).should eq(
                drama3_mlx_q4_bits(ctx.read(explicit_output, "terminal")),
              )
            end
            ctx.graph_stats["submitted"].should eq(1_i64)
            ctx.graph_stats["completed"].should eq(1_i64)
            ctx.graph_stats["completion_waits"].should eq(1_i64)
            ctx.graph_stats["pending"].should eq(0_i64)
            expected_readback = (expected.size + ids.size * fixture[:matrix].rows + (default_native ? 2 * fixture[:matrix].rows : 0)) * 4
            ctx.readbacks["terminal"].should eq(expected_readback.to_i64)
          ensure
            ctx.abort
            MetalBackend.reset!
            File.delete(fixture[:path]) if File.exists?(fixture[:path])
          end
        end
      end
    end

    it "rejects unsupported affine metadata and foreign buffers before submission or readback" do
      drama3_mlx_q4_with_device do
        fixtures = [] of NamedTuple(name: String, path: String, matrix: Matrix, owner: MappedFile)
        begin
          [{"Q8", 8, 512, 8, 64}, {"Q4 group 32", 4, 512, 8, 32},
           {"Q4 K tail", 4, 576, 8, 64}, {"Q4 N tail", 4, 512, 9, 64}].each do |case_data|
            name, bits, cols, rows, group = case_data
            fixture = drama3_mlx_q4_fixture(rows: rows, cols: cols, bits: bits, group: group)
            fixtures << {name: name, path: fixture[:path], matrix: fixture[:matrix], owner: fixture[:owner]}
          end

          ctx = Resident::Context.new
          foreign_ctx = Resident::Context.new
          begin
            ids = ctx.upload([108_f32, 7_f32, 29_f32, 3_f32, 77_f32, 0_f32, 51_f32, 42_f32])
            fixtures.each do |fixture|
              matrix = fixture[:matrix]
              before = ctx.graph_stats.dup
              batched_input = ctx.upload(Array(Float32).new(matrix.cols * 8, 0.5_f32))
              expect_raises(ArgumentError) do
                ctx.experts(matrix, batched_input, ids, per_rank_input: true, reduction: "mlx_q4")
              end
              ctx.graph_stats["submitted"].should eq(before["submitted"])
              ctx.graph_stats["completion_waits"].should eq(before["completion_waits"])
              ctx.readbacks.should be_empty

              scalar_input = ctx.upload(Array(Float32).new(matrix.cols, 0.5_f32))
              expect_raises(ArgumentError) { ctx.expert(matrix, scalar_input, ids, 0, reduction: "mlx_q4") }
              ctx.graph_stats["submitted"].should eq(before["submitted"])
              ctx.graph_stats["completion_waits"].should eq(before["completion_waits"])
              ctx.readbacks.should be_empty
            end

            dense = Matrix.new(8, 512, Array(Float32).new(8 * 512, 1_f32))
            dense_input = ctx.upload(Array(Float32).new(512 * 8, 1_f32))
            before = ctx.graph_stats.dup
            expect_raises(ArgumentError) do
              ctx.experts(dense, dense_input, ids, per_rank_input: true, reduction: "mlx_q4")
            end
            expect_raises(ArgumentError) do
              ctx.experts(fixtures[1][:matrix], dense_input, ids, per_rank_input: true, reduction: "unsupported")
            end

            valid = drama3_mlx_q4_fixture
            foreign = foreign_ctx.upload(Array(Float32).new(valid[:matrix].cols, 1_f32))
            expect_raises(ArgumentError, /different resident context/) do
              ctx.expert(valid[:matrix], foreign, ids, 0, reduction: "mlx_q4")
            end
            ctx.graph_stats["submitted"].should eq(before["submitted"])
            ctx.graph_stats["completion_waits"].should eq(before["completion_waits"])
            ctx.readbacks.should be_empty
            ctx.abort
            foreign_ctx.abort
            MetalBackend.reset!
            File.delete(valid[:path]) if File.exists?(valid[:path])
          ensure
            ctx.abort
            foreign_ctx.abort
          end
        ensure
          MetalBackend.reset!
          fixtures.each { |fixture| File.delete(fixture[:path]) if File.exists?(fixture[:path]) }
        end
      end
    end

    it "freezes the backbone selector and lets buffer tracing observe without a host boundary" do
      drama3_mlx_q4_with_device do
        previous = ENV["DRAMA3_EXPERT_DOWN_REDUCTION"]?
        fixture_path = File.join(__DIR__, "support/drama3/reference_fixture")
        contexts = [] of Resident::Context
        begin
          ENV.delete("DRAMA3_EXPERT_DOWN_REDUCTION")
          config = BackboneConfig.new(JSON.parse(File.read(File.join(fixture_path, "config.json"))))
          store = WeightStore.new(fixture_path)
          ctx = Resident::Context.new
          contexts << ctx
          model = BackboneResident.new(config, store, ctx, 4)
          model.expert_down_reduction.should eq("native")

          ENV["DRAMA3_EXPERT_DOWN_REDUCTION"] = "mlx_q4"
          model.expert_down_reduction.should eq("native")
          before_invalid = ctx.graph_stats.dup
          ENV["DRAMA3_EXPERT_DOWN_REDUCTION"] = "bad-value"
          expect_raises(ArgumentError, /DRAMA3_EXPERT_DOWN_REDUCTION/) do
            BackboneResident.new(config, store, ctx, 4)
          end
          ctx.graph_stats["submitted"].should eq(before_invalid["submitted"])
          ctx.graph_stats["completion_waits"].should eq(before_invalid["completion_waits"])
          ctx.readbacks.should be_empty

          # The existing model must stay on its captured native path after the
          # process environment changes to the opt-in value.
          ENV["DRAMA3_EXPERT_DOWN_REDUCTION"] = "mlx_q4"
          observed = {} of String => Int32
          model.buffer_trace = ->(name : String, buffer : Resident::Buffer) {
            observed[name] = buffer.count
            nil
          }
          input = JSON.parse(File.read(File.join(fixture_path, "inputs.json")))["input_embeddings"][0].as_a.map do |value|
            value.as_f.to_f32
          end
          output = model.step(ctx.upload(input))
          observed.has_key?("backbone.layers.0.attention_residual").should be_true
          observed.has_key?("backbone.layers.0.hidden").should be_true
          observed.has_key?("backbone.hidden").should be_true
          observed.values.all? { |count| count > 0 }.should be_true
          ctx.readbacks.should be_empty
          ctx.submissions.should eq(0_i64)
          ctx.graph_stats["completion_waits"].should eq(0_i64)
          ctx.read(output, "terminal").size.should eq(config.hidden)
          ctx.readbacks["terminal"].should eq(config.hidden.to_i64 * 4_i64)
          ctx.readbacks["control"].should eq(4_i64)
          ctx.readbacks["diagnostic"].should eq(0_i64)
          ctx.graph_stats["completion_waits"].should eq(1_i64)

          opt_in_ctx = Resident::Context.new
          contexts << opt_in_ctx
          opt_in = BackboneResident.new(config, store, opt_in_ctx, 4)
          opt_in.expert_down_reduction.should eq("mlx_q4")
        ensure
          contexts.each(&.abort)
          MetalBackend.reset!
          if previous
            ENV["DRAMA3_EXPERT_DOWN_REDUCTION"] = previous
          else
            ENV.delete("DRAMA3_EXPERT_DOWN_REDUCTION")
          end
        end
      end
    end
  end
{% end %}
