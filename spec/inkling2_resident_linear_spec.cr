require "spec"
require "../src/ml/gguf/inkling2_engine"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  module Inkling2ResidentLinearSpecSupport
    extend self

    EXPERT_WIDTH = 96

    record Model, path : String, offsets : Hash(String, Int64)

    def f32_bytes(values : Array(Float32)) : Bytes
      Bytes.new(values.to_unsafe.as(UInt8*), values.size * sizeof(Float32), read_only: true)
    end

    def q8_bytes(out_dim : Int32, in_dim : Int32, expert : Int32 = 0) : Bytes
      raise ArgumentError.new("Q8 fixture width must be divisible by 32") unless in_dim % 32 == 0
      blocks = in_dim // 32
      bytes = Bytes.new(out_dim * blocks * 34, 0_u8)
      out_dim.times do |row|
        blocks.times do |block|
          offset = (row * blocks + block) * 34
          # Q8_0 scale = 0.125.
          bytes[offset] = 0x00_u8
          bytes[offset + 1] = 0x30_u8
          32.times do |column|
            code = ((expert * 11 + row * 3 + block * 7 + column * 5) % 17) - 8
            bytes[offset + 2 + column] = code.to_i8.unsafe_as(UInt8)
          end
        end
      end
      bytes
    end

    def iq4_bytes(out_dim : Int32, in_dim : Int32, expert : Int32 = 0) : Bytes
      raise ArgumentError.new("IQ4_NL fixture width must be divisible by 32") unless in_dim % 32 == 0
      blocks = in_dim // 32
      bytes = Bytes.new(out_dim * blocks * 18, 0_u8)
      out_dim.times do |row|
        blocks.times do |block|
          offset = (row * blocks + block) * 18
          # IQ4_NL scale = 0.0625. Low/high nibbles encode separate 16-value halves.
          bytes[offset] = 0x00_u8
          bytes[offset + 1] = 0x2C_u8
          16.times do |column|
            low = (expert + row * 3 + block * 5 + column) % 16
            high = (expert * 3 + row + block * 7 + column * 5) % 16
            bytes[offset + 2 + column] = (low | (high << 4)).to_u8
          end
        end
      end
      bytes
    end

    def write_model : Model
      path = File.join(Dir.tempdir, "inkling2-resident-linear-#{Random.rand(1_000_000_000)}.gguf")
      offsets = {} of String => Int64
      File.open(path, "wb") do |io|
        io.write("GGUF".to_slice)
        io.write_bytes(3_u32, IO::ByteFormat::LittleEndian)
        io.write_bytes(0_i64, IO::ByteFormat::LittleEndian) # tensor count
        io.write_bytes(0_i64, IO::ByteFormat::LittleEndian) # metadata count
        io.write(Bytes.new(8, 0_u8))                        # align empty tensor table to 32 bytes

        f32_projection = Array(Float32).new(15) do |index|
          row, column = index // 3, index % 3
          (row - 2).to_f32 * 0.125_f32 + (column - 1).to_f32 * 0.03125_f32
        end
        offsets["f32_projection"] = io.pos
        io.write(f32_bytes(f32_projection))

        q8_projection = q8_bytes(7, 32)
        offsets["q8_projection"] = io.pos
        io.write(q8_projection)

        offsets["q8_tiled_projection"] = io.pos
        io.write(q8_bytes(39, 64))

        router = Array(Float32).new(3 * 32, 0_f32)
        router[0] = 16_f32
        router[32] = -16_f32
        router[65] = 16_f32
        io.write(Bytes.new((-io.pos % sizeof(Float32)).to_i, 0_u8))
        offsets["router"] = io.pos
        io.write(f32_bytes(router))

        gate_up = Bytes.new(3 * 2 * EXPERT_WIDTH * 34, 0_u8)
        3.times do |expert|
          matrix = q8_bytes(2 * EXPERT_WIDTH, 32, expert)
          gate_up[expert * matrix.size, matrix.size].copy_from(matrix)
        end
        offsets["gate_up"] = io.pos
        io.write(gate_up)

        down = Bytes.new(3 * EXPERT_WIDTH * 34, 0_u8)
        3.times do |expert|
          matrix = q8_bytes(32, EXPERT_WIDTH, expert + 3)
          down[expert * matrix.size, matrix.size].copy_from(matrix)
        end
        offsets["down"] = io.pos
        io.write(down)

        iq4_gate_up = Bytes.new(3 * 2 * EXPERT_WIDTH * 18, 0_u8)
        iq4_down = Bytes.new(3 * EXPERT_WIDTH * 18, 0_u8)
        3.times do |expert|
          matrix = iq4_bytes(2 * EXPERT_WIDTH, 32, expert + 1)
          iq4_gate_up[expert * matrix.size, matrix.size].copy_from(matrix)
          matrix = iq4_bytes(32, EXPERT_WIDTH, expert + 4)
          iq4_down[expert * matrix.size, matrix.size].copy_from(matrix)
        end
        offsets["iq4_gate_up"] = io.pos
        io.write(iq4_gate_up)
        offsets["iq4_down"] = io.pos
        io.write(iq4_down)

        io.write(Bytes.new(65_536 - io.pos.to_i, 0_u8))
      end
      Model.new(path, offsets)
    end

    def mapped_bytes(gguf : ML::GGUF::GGUFFile, offset : Int64, size : Int32) : Bytes
      base, file_size = gguf.mmap_region || raise "GGUF test model did not mmap"
      raise "test tensor is outside fixture" if offset < 0 || offset + size > file_size
      Bytes.new(base + offset, size, read_only: true)
    end

    def read(view : ML::GGUF::Inkling2View, elements : Int32) : Array(Float32)
      offset = (view.offset // sizeof(Float32)).to_i32
      view.buffer.read(offset + elements)[offset, elements].to_a
    end

    def float_bits(values : Array(Float32)) : Array(UInt32)
      values.map(&.unsafe_as(UInt32))
    end

    def float_view(values : Array(Float32), buffers : Array(ML::MetalBuffer)) : ML::GGUF::Inkling2View
      buffer = ML::MetalBuffer.from_array(values)
      buffers << buffer
      ML::GGUF::Inkling2View.new(buffer)
    end

    def projection_expected(weight : ML::GGUF::QuantWeight, input : Array(Float32), rows : Int32) : Array(Float32)
      ML::GGUF::QuantMatmul.matmul_add(input, rows, weight.in_dim, weight.raw, weight.type,
        weight.out_dim, Array(Float32).new(weight.out_dim, 0_f32))
    end

    def moe_expected(layer : ML::GGUF::Inkling2LayerWeights, config : ML::GGUF::Inkling2Config,
                     input : Array(Float32), rows : Int32) : Array(Float32)
      logits = ML::GGUF::QuantMatmul.matmul_add(input, rows, config.n_embd, layer.router.raw,
        layer.router.type, layer.router.out_dim, Array(Float32).new(layer.router.out_dim, 0_f32))
      output = Array(Float32).new(rows * config.n_embd, 0_f32)

      rows.times do |row|
        row_input = input[row * config.n_embd, config.n_embd].to_a
        routed = ML::GGUF::Inkling2Math.route(
          logits[row * config.n_expert, config.n_expert].to_a,
          layer.router_bias,
          config.n_expert_used,
          config.expert_scale,
        )
        routed.each do |expert_id, routing_weight|
          gate_up_weight = layer.gate_up.expert(expert_id)
          gate_up = ML::GGUF::QuantMatmul.matmul_add(row_input, 1, config.n_embd,
            gate_up_weight.raw, gate_up_weight.type, gate_up_weight.out_dim,
            Array(Float32).new(gate_up_weight.out_dim, 0_f32))
          gated = Array(Float32).new(config.expert_dim) do |column|
            ML::GGUF::Inkling2Math.situ(gate_up[column], gate_up[config.expert_dim + column],
              config.situ_beta, config.situ_linear_beta)
          end
          down_weight = layer.down.expert(expert_id)
          down = ML::GGUF::QuantMatmul.matmul_add(gated, 1, config.expert_dim,
            down_weight.raw, down_weight.type, down_weight.out_dim,
            Array(Float32).new(down_weight.out_dim, 0_f32))
          config.n_embd.times do |column|
            output[row * config.n_embd + column] += routing_weight * down[column]
          end
        end
      end
      output
    end
  end

  alias Inkling2LinearSpecSupport = Inkling2ResidentLinearSpecSupport

  describe ML::GGUF::Inkling2ResidentLinear do
    it "encodes F32 and Q8_0 single-row and batched projections with batched MoE in one command buffer" do
      pending! "Metal unavailable" unless ML::Metal::Device.init!

      fixture = Inkling2LinearSpecSupport.write_model
      gguf = ML::GGUF::GGUFFile.new(fixture.path)
      linear = ML::GGUF::Inkling2ResidentLinear.new(gguf)
      scratch = ML::GGUF::Inkling2ResidentScratch.new
      buffers = [] of ML::MetalBuffer
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)

      begin
        f32_projection = ML::GGUF::QuantWeight.new(
          Inkling2LinearSpecSupport.mapped_bytes(gguf, fixture.offsets["f32_projection"], 15 * sizeof(Float32)),
          ML::GGUF::TensorType::F32, 5_i32, 3_i32,
        )
        q8_projection = ML::GGUF::QuantWeight.new(
          Inkling2LinearSpecSupport.mapped_bytes(gguf, fixture.offsets["q8_projection"], 7 * 34),
          ML::GGUF::TensorType::Q8_0, 7_i32, 32_i32,
        )
        q8_tiled_projection = ML::GGUF::QuantWeight.new(
          Inkling2LinearSpecSupport.mapped_bytes(gguf, fixture.offsets["q8_tiled_projection"], 39 * 2 * 34),
          ML::GGUF::TensorType::Q8_0, 39_i32, 64_i32,
        )
        router = ML::GGUF::QuantWeight.new(
          Inkling2LinearSpecSupport.mapped_bytes(gguf, fixture.offsets["router"], 3 * 32 * sizeof(Float32)),
          ML::GGUF::TensorType::F32, 3_i32, 32_i32,
        )
        # More than one 32-feature gate/up tile, including both halves of each tile.
        expert_width = Inkling2LinearSpecSupport::EXPERT_WIDTH
        gate_up_raw = Inkling2LinearSpecSupport.mapped_bytes(gguf, fixture.offsets["gate_up"], 3 * 2 * expert_width * 34)
        down_raw = Inkling2LinearSpecSupport.mapped_bytes(gguf, fixture.offsets["down"], 3 * expert_width * 34)
        iq4_gate_up_raw = Inkling2LinearSpecSupport.mapped_bytes(gguf, fixture.offsets["iq4_gate_up"], 3 * 2 * expert_width * 18)
        iq4_down_raw = Inkling2LinearSpecSupport.mapped_bytes(gguf, fixture.offsets["iq4_down"], 3 * expert_width * 18)
        gate_up = ML::GGUF::Inkling2ExpertWeights.new(gate_up_raw, ML::GGUF::TensorType::Q8_0, 32_i32, 2 * expert_width, 3_i32)
        down = ML::GGUF::Inkling2ExpertWeights.new(down_raw, ML::GGUF::TensorType::Q8_0, expert_width, 32_i32, 3_i32)
        iq4_gate_up = ML::GGUF::Inkling2ExpertWeights.new(iq4_gate_up_raw, ML::GGUF::TensorType::IQ4_NL, 32_i32, 2 * expert_width, 3_i32)
        iq4_down = ML::GGUF::Inkling2ExpertWeights.new(iq4_down_raw, ML::GGUF::TensorType::IQ4_NL, expert_width, 32_i32, 3_i32)
        attention = ML::GGUF::Inkling2AttentionWeights.new(
          router, router, router, router, router, router, router,
          [] of Float32, [] of Float32, [] of Float32, [] of Float32, [] of Float32, [] of Float32,
          nil, nil,
        )
        layer = ML::GGUF::Inkling2LayerWeights.new(
          attention, [] of Float32, [] of Float32, [] of Float32, nil, [] of Float32, [] of Float32,
          [] of Float32, router, [0_f32, 0_f32, -1_f32], gate_up, down, nil,
        )
        iq4_layer = ML::GGUF::Inkling2LayerWeights.new(
          attention, [] of Float32, [] of Float32, [] of Float32, nil, [] of Float32, [] of Float32,
          [] of Float32, router, [0_f32, 0_f32, -0.125_f32], iq4_gate_up, iq4_down, nil,
        )
        config = ML::GGUF::Inkling2Config.new(
          1_i32, 32_i32, 2_i32, 1_i32, 2_i32, 2_i32, 2_i32, 16_i32, 4_i32, 2_i32,
          1_f32, 1_f32, 1_f32, 1e-5_f32, 3_i32, 2_i32, expert_width, 1.25_f32, 4_f32,
          25_f32, 1_i32, 8_i32, 8_i32, 32_i32, 0_i32, 1_i32, 3_i32, 2_i32,
          30_f32, false, [true],
        )

        ML::GGUF::Inkling2ResidentLinear.projection_supported?(f32_projection).should be_true
        ML::GGUF::Inkling2ResidentLinear.projection_supported?(q8_projection).should be_true
        ML::GGUF::Inkling2ResidentLinear.moe_supported?(layer, config).should be_true
        ML::GGUF::Inkling2ResidentLinear.moe_supported?(iq4_layer, config).should be_true

        f32_input = [0.25_f32, -0.5_f32, 0.75_f32, -0.125_f32, 0.375_f32, 0.625_f32]
        f32_view = Inkling2LinearSpecSupport.float_view(f32_input, buffers)
        f32_output = Inkling2LinearSpecSupport.float_view(Array(Float32).new(10, 0_f32), buffers)
        linear.project(encoder, f32_projection, f32_view, f32_output, 2_i32)

        q8_input = Array(Float32).new(3 * 32) { |index| ((index % 19) - 9).to_f32 * 0.025_f32 }
        q8_view = Inkling2LinearSpecSupport.float_view(q8_input, buffers)
        q8_batch_output = Inkling2LinearSpecSupport.float_view(Array(Float32).new(3 * 7, 0_f32), buffers)
        q8_single_output = Inkling2LinearSpecSupport.float_view(Array(Float32).new(7, 0_f32), buffers)
        linear.project(encoder, q8_projection, q8_view, q8_batch_output, 3_i32)
        linear.project(encoder, q8_projection, q8_view, q8_single_output, 1_i32)

        # Two K blocks, partial batch and output tiles, and a four-byte input offset.
        q8_tiled_input = Array(Float32).new(35 * 64) { |index| ((index % 23) - 11).to_f32 * 0.013_f32 }
        q8_tiled_view = Inkling2LinearSpecSupport.float_view([0_f32] + q8_tiled_input, buffers).slice(1)
        q8_tiled_output = Inkling2LinearSpecSupport.float_view(Array(Float32).new(35 * 39, 0_f32), buffers)
        linear.project(encoder, q8_tiled_projection, q8_tiled_view, q8_tiled_output, 35_i32)

        # The decode projection tile must produce the same bits as four
        # independent one-row GEMVs, including a two-block Q8_0 weight row.
        f32_decode_input = Array(Float32).new(4 * 3) do |index|
          ((index % 7) - 3).to_f32 * 0.0625_f32
        end
        f32_decode_view = Inkling2LinearSpecSupport.float_view(f32_decode_input, buffers)
        f32_decode_output = Inkling2LinearSpecSupport.float_view(Array(Float32).new(4 * 5, 0_f32), buffers)
        f32_rowwise_output = Inkling2LinearSpecSupport.float_view(Array(Float32).new(4 * 5, 0_f32), buffers)
        linear.project(encoder, f32_projection, f32_decode_view, f32_decode_output, 4_i32, decode_block: true)
        4.times do |row|
          linear.project(encoder, f32_projection, f32_decode_view.slice(row * 3), f32_rowwise_output.slice(row * 5), 1_i32)
        end

        q8_decode_input = Array(Float32).new(4 * 64) do |index|
          ((index % 29) - 14).to_f32 * 0.00625_f32
        end
        q8_decode_view = Inkling2LinearSpecSupport.float_view(q8_decode_input, buffers)
        q8_decode_output = Inkling2LinearSpecSupport.float_view(Array(Float32).new(4 * 39, 0_f32), buffers)
        q8_rowwise_output = Inkling2LinearSpecSupport.float_view(Array(Float32).new(4 * 39, 0_f32), buffers)
        linear.project(encoder, q8_tiled_projection, q8_decode_view, q8_decode_output, 4_i32, decode_block: true)
        4.times do |row|
          linear.project(encoder, q8_tiled_projection, q8_decode_view.slice(row * 64), q8_rowwise_output.slice(row * 39), 1_i32)
        end

        moe_input = Array(Float32).new(3 * 32) { |index| ((index % 13) - 6).to_f32 * 0.015_f32 }
        moe_input_view = Inkling2LinearSpecSupport.float_view(moe_input, buffers)
        bias_view = Inkling2LinearSpecSupport.float_view([0_f32, 0_f32, -1_f32], buffers)
        moe_output = linear.moe(encoder, scratch, layer, config, moe_input_view, bias_view, 3_i32)
        moe_readback = Inkling2LinearSpecSupport.float_view(Array(Float32).new(3 * 32, 0_f32), buffers)
        ML::GGUF::Inkling2ResidentMath.copy(encoder, moe_output, moe_readback, 3 * 32)
        iq4_bias_view = Inkling2LinearSpecSupport.float_view(iq4_layer.router_bias.not_nil!, buffers)
        iq4_output = linear.moe(encoder, scratch, iq4_layer, config, moe_input_view, iq4_bias_view, 3_i32)
        iq4_readback = Inkling2LinearSpecSupport.float_view(Array(Float32).new(3 * 32, 0_f32), buffers)
        ML::GGUF::Inkling2ResidentMath.copy(encoder, iq4_output, iq4_readback, 3 * 32)

        # All four rows select experts 0 and 1, so each grouped exact matvec
        # reuses the expert weights across routes. The rowwise baseline also
        # exercises route-rank scatter before ordered top-k combine.
        decode_moe_input = Array(Float32).new(4 * 32) do |index|
          row, column = index.divmod(32)
          case column
          when 0 then 0.25_f32
          when 1 then 0_f32
          else        ((row * 7 + column) % 13 - 6).to_f32 * 0.01_f32
          end
        end
        decode_moe_view = Inkling2LinearSpecSupport.float_view(decode_moe_input, buffers)
        decode_moe_cases = {layer, iq4_layer}.zip({bias_view, iq4_bias_view}).map do |weights, bias|
          rowwise = linear.moe(encoder, scratch, weights, config, decode_moe_view, bias, 4_i32)
          rowwise_readback = Inkling2LinearSpecSupport.float_view(Array(Float32).new(4 * 32, 0_f32), buffers)
          ML::GGUF::Inkling2ResidentMath.copy(encoder, rowwise, rowwise_readback, 4 * 32)
          grouped = linear.moe(encoder, scratch, weights, config, decode_moe_view, bias, 4_i32, decode_block: true)
          grouped_readback = Inkling2LinearSpecSupport.float_view(Array(Float32).new(4 * 32, 0_f32), buffers)
          ML::GGUF::Inkling2ResidentMath.copy(encoder, grouped, grouped_readback, 4 * 32)
          {rowwise_readback, grouped_readback}
        end

        expect_raises(ArgumentError) do
          linear.project(encoder, f32_projection, f32_view, f32_output, 5_i32, decode_block: true)
        end
        expect_raises(ArgumentError) do
          linear.moe(encoder, scratch, layer, config, moe_input_view, bias_view, 5_i32, decode_block: true)
        end

        # Alternate the rank of experts 0 and 1. Expert 2 stays empty
        # for Q8_0 and receives only the last token for IQ4_NL.
        grouped_rows = 19_i32
        grouped_input = Array(Float32).new(grouped_rows * 32) do |index|
          row, column = index.divmod(32)
          case column
          when 0 then row.even? ? 0.25_f32 : -0.25_f32
          when 1 then row == grouped_rows - 1 ? 0.25_f32 : -0.25_f32
          else        ((index % 17) - 8).to_f32 * 0.01_f32
          end
        end
        grouped_view = Inkling2LinearSpecSupport.float_view(grouped_input, buffers)
        grouped_cases = {layer, iq4_layer}.zip({bias_view, iq4_bias_view}).map do |weights, bias|
          result = linear.moe(encoder, scratch, weights, config, grouped_view, bias, grouped_rows)
          readback = Inkling2LinearSpecSupport.float_view(Array(Float32).new(grouped_rows * 32, 0_f32), buffers)
          ML::GGUF::Inkling2ResidentMath.copy(encoder, result, readback, grouped_rows * 32)
          {weights, readback}
        end

        encoder.end_encoding
        command.commit_and_wait

        Inkling2LinearSpecSupport.read(f32_output, 10_i32).zip(Inkling2LinearSpecSupport.projection_expected(f32_projection, f32_input, 2_i32)).each do |actual, expected|
          actual.should be_close(expected, 2e-6_f32)
        end
        q8_expected = Inkling2LinearSpecSupport.projection_expected(q8_projection, q8_input, 3_i32)
        Inkling2LinearSpecSupport.read(q8_batch_output, 21_i32).zip(q8_expected).each do |actual, expected|
          actual.should be_close(expected, 2e-5_f32)
        end
        Inkling2LinearSpecSupport.read(q8_single_output, 7_i32).zip(q8_expected[0, 7]).each do |actual, expected|
          actual.should be_close(expected, 2e-5_f32)
        end
        q8_tiled_expected = Inkling2LinearSpecSupport.projection_expected(q8_tiled_projection, q8_tiled_input, 35_i32)
        Inkling2LinearSpecSupport.read(q8_tiled_output, 35 * 39).zip(q8_tiled_expected).each do |actual, expected|
          actual.should be_close(expected, 2e-5_f32)
        end
        Inkling2LinearSpecSupport.float_bits(Inkling2LinearSpecSupport.read(f32_decode_output, 20_i32)).should eq(
          Inkling2LinearSpecSupport.float_bits(Inkling2LinearSpecSupport.read(f32_rowwise_output, 20_i32)))
        Inkling2LinearSpecSupport.float_bits(Inkling2LinearSpecSupport.read(q8_decode_output, 4 * 39)).should eq(
          Inkling2LinearSpecSupport.float_bits(Inkling2LinearSpecSupport.read(q8_rowwise_output, 4 * 39)))
        expected_moe = Inkling2LinearSpecSupport.moe_expected(layer, config, moe_input, 3_i32)
        Inkling2LinearSpecSupport.read(moe_readback, 96_i32).zip(expected_moe).each do |actual, expected|
          actual.should be_close(expected, 2e-3_f32)
        end
        expected_iq4 = Inkling2LinearSpecSupport.moe_expected(iq4_layer, config, moe_input, 3_i32)
        Inkling2LinearSpecSupport.read(iq4_readback, 96_i32).zip(expected_iq4).each do |actual, expected|
          actual.should be_close(expected, 2e-2_f32)
        end
        decode_moe_cases.each do |rowwise_readback, grouped_readback|
          Inkling2LinearSpecSupport.float_bits(Inkling2LinearSpecSupport.read(grouped_readback, 4 * 32)).should eq(
            Inkling2LinearSpecSupport.float_bits(Inkling2LinearSpecSupport.read(rowwise_readback, 4 * 32)))
        end
        grouped_cases.each do |weights, readback|
          expected = Inkling2LinearSpecSupport.moe_expected(weights, config, grouped_input, grouped_rows)
          tolerance = weights.gate_up.type.iq4_nl? ? 2e-2_f32 : 2e-3_f32
          Inkling2LinearSpecSupport.read(readback, grouped_rows * 32).zip(expected).each do |actual, reference|
            actual.should be_close(reference, tolerance)
          end
        end
      ensure
        encoder.end_encoding
        buffers.each(&.release)
        scratch.close
        linear.close
        gguf.close
        File.delete(fixture.path) if File.exists?(fixture.path)
      end
    end
  end
{% end %}
