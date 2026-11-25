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

        3.times do |matrix|
          offsets["f32_triplet_#{matrix}"] = io.pos
          values = Array(Float32).new(37 * 7) do |index|
            ((index * 13 + matrix * 19) % 41 - 20).to_f32 * 0.017_f32
          end
          io.write(f32_bytes(values))
        end

        2.times do |matrix|
          offsets["control_#{matrix}"] = io.pos
          values = Array(Float32).new(64 * 35) do |index|
            ((index * 17 + matrix * 11) % 71 - 35).to_f32 * 0.0091_f32
          end
          io.write(f32_bytes(values))
        end

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

        io.write(Bytes.new((-io.pos % 16_384).to_i, 0_u8))
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

    def moe_route_counts(layer : ML::GGUF::Inkling2LayerWeights, config : ML::GGUF::Inkling2Config,
                         input : Array(Float32), rows : Int32) : Array(Int32)
      logits = ML::GGUF::QuantMatmul.matmul_add(input, rows, config.n_embd, layer.router.raw,
        layer.router.type, layer.router.out_dim, Array(Float32).new(rows * config.n_expert, 0_f32))
      counts = Array(Int32).new(config.n_expert, 0_i32)
      rows.times do |row|
        ML::GGUF::Inkling2Math.route(logits[row * config.n_expert, config.n_expert].to_a,
          layer.router_bias, config.n_expert_used, config.expert_scale).each do |expert_id, _|
          counts[expert_id] += 1
        end
      end
      counts
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

  class ML::Metal::PipelineCache
    def self.resident_linear_spec_contains?(name : String) : Bool
      @@cache.has_key?(name)
    end
  end

  class Inkling2UnfusedResidentLinearSpec < ML::GGUF::Inkling2ResidentLinear
    private def decode_fused_iq4_gate_up?(_decode_block : Bool) : Bool
      false
    end
  end

  class Inkling2UntiledControlPairSpec < ML::GGUF::Inkling2ResidentLinear
    private def f32_control_pair_tiled?(_rank_dim : Int32, _rows : Int32) : Bool
      false
    end
  end

  describe ML::GGUF::Inkling2ResidentLinear do
    it "keeps FP32 controller bits across partial row and channel tiles" do
      pending! "Metal unavailable" unless ML::Metal::Device.init!
      {% unless flag?(:aarch64) %}
        pending! "Rank-64 tiling requires Apple Silicon"
      {% end %}

      fixture = Inkling2LinearSpecSupport.write_model
      gguf = ML::GGUF::GGUFFile.new(fixture.path)
      reference = Inkling2UntiledControlPairSpec.new(gguf)
      tiled = ML::GGUF::Inkling2ResidentLinear.new(gguf)
      buffers = [] of ML::MetalBuffer
      begin
        in_dim, out_dim, rows = 64, 35, 19
        weights = {0, 1}.map do |matrix|
          raw = Inkling2LinearSpecSupport.mapped_bytes(gguf, fixture.offsets["control_#{matrix}"],
            in_dim * out_dim * sizeof(Float32))
          ML::GGUF::QuantWeight.new(raw, ML::GGUF::TensorType::F32, out_dim, in_dim)
        end
        elements = rows * out_dim
        rank = Inkling2LinearSpecSupport.float_view(Array(Float32).new(rows * in_dim + 1) do |i|
          ((i * 7) % 331 - 165).to_f32 * 0.028_f32
        end, buffers).slice(1)
        input = Inkling2LinearSpecSupport.float_view(Array(Float32).new(elements + 1) do |i|
          ((i * 13) % 67 - 33).to_f32 * 0.035_f32
        end, buffers).slice(1)
        message = Inkling2LinearSpecSupport.float_view(Array(Float32).new(elements + 2) do |i|
          ((i * 29) % 41 - 20).to_f32 * 0.081_f32
        end, buffers).slice(2)
        guards = Array(Float32).new(elements + 2, 123.0_f32)
        outputs = {reference, tiled}.map do |linear|
          output = Inkling2LinearSpecSupport.float_view(guards, buffers)
          ML::Metal::Dispatch.execute_sequence do |command|
            encoder = ML::Metal::ComputeEncoder.new(command)
            linear.project_f32_control_pair(encoder, weights, rank, input, message,
              output.slice(1), out_dim, rows: rows).should be_true
            encoder.end_encoding
          end
          values = Inkling2LinearSpecSupport.read(output, guards.size)
          values.first.should eq(123.0_f32)
          values.last.should eq(123.0_f32)
          values[1, elements].should_not contain(123.0_f32)
          Inkling2LinearSpecSupport.float_bits(values)
        end
        outputs[1].should eq(outputs[0])
        ML::Metal::PipelineCache.resident_linear_spec_contains?("inkling2_resident_f32_control_pair_coalesced8").should be_true
      ensure
        buffers.each(&.release)
        reference.close
        tiled.close
        gguf.close
        File.delete?(fixture.path)
      end
    end

    it "shares a dispatch across three independent F32 projections" do
      pending! "Metal unavailable" unless ML::Metal::Device.init!

      fixture = Inkling2LinearSpecSupport.write_model
      gguf = ML::GGUF::GGUFFile.new(fixture.path)
      linear = ML::GGUF::Inkling2ResidentLinear.new(gguf)
      buffers = [] of ML::MetalBuffer
      begin
        in_dim = 37
        out_dim = 7
        weights = {0, 1, 2}.map do |index|
          raw = Inkling2LinearSpecSupport.mapped_bytes(gguf, fixture.offsets["f32_triplet_#{index}"],
            in_dim * out_dim * sizeof(Float32))
          ML::GGUF::QuantWeight.new(raw, ML::GGUF::TensorType::F32, out_dim, in_dim)
        end
        # Distinct inputs and nonzero buffer offsets catch cross-talk and writes
        # outside a projection. Thirty-seven columns span two lane iterations.
        values = Array(Float32).new(3 * in_dim) { |index| ((index * 7) % 31 - 15).to_f32 * 0.019_f32 }
        input = Inkling2LinearSpecSupport.float_view([123_f32] + values, buffers).slice(1)
        reference = Inkling2LinearSpecSupport.float_view(Array(Float32).new(3 * out_dim, 0_f32), buffers)
        guarded = Inkling2LinearSpecSupport.float_view(Array(Float32).new(3 * out_dim + 2, 123_f32), buffers)
        command = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(command)
        weights.each_with_index do |weight, index|
          linear.project(encoder, weight, input.slice(index * in_dim), reference.slice(index * out_dim), 1)
        end
        linear.project_f32_triplet(encoder, weights, input, guarded.slice(1))
        encoder.end_encoding
        command.commit_and_wait

        data = Inkling2LinearSpecSupport.read(guarded, 3 * out_dim + 2)
        data.first.should eq(123_f32)
        data.last.should eq(123_f32)
        bits = Inkling2LinearSpecSupport.float_bits(Inkling2LinearSpecSupport.read(reference, 3 * out_dim))
        Inkling2LinearSpecSupport.float_bits(data[1, 3 * out_dim]).should eq(bits)
      ensure
        buffers.each(&.release)
        linear.close
        gguf.close
        File.delete?(fixture.path)
      end
    end

    it "keeps separate Q8_0 reductions and output ranges in a projection group" do
      pending! "Metal unavailable" unless ML::Metal::Device.init!

      fixture = Inkling2LinearSpecSupport.write_model
      gguf = ML::GGUF::GGUFFile.new(fixture.path)
      linear = ML::GGUF::Inkling2ResidentLinear.new(gguf)
      buffers = [] of ML::MetalBuffer
      begin
        first_raw = Inkling2LinearSpecSupport.mapped_bytes(gguf, fixture.offsets["q8_projection"], 7 * 34)
        second_raw = Inkling2LinearSpecSupport.mapped_bytes(gguf, fixture.offsets["gate_up"], 8 * 34)
        first = ML::GGUF::QuantWeight.new(first_raw[0, 4 * 34], ML::GGUF::TensorType::Q8_0, 4, 32)
        second = ML::GGUF::QuantWeight.new(second_raw, ML::GGUF::TensorType::Q8_0, 8, 32)
        unaligned = ML::GGUF::QuantWeight.new(first_raw, ML::GGUF::TensorType::Q8_0, 7, 32)

        {1, 2, 3, 4}.each do |rows|
          values = Array(Float32).new(2 * rows * 32 + 1) { |index| ((index * 7) % 31 - 15).to_f32 * 0.019_f32 }
          input = Inkling2LinearSpecSupport.float_view(values, buffers)
          inputs = {input.slice(1), input, input.slice(1 + rows * 32)}
          reference = Inkling2LinearSpecSupport.float_view(Array(Float32).new(12 * rows, 0_f32), buffers)
          guarded = Inkling2LinearSpecSupport.float_view(Array(Float32).new(12 * rows + 3, 123_f32), buffers)
          outputs = {guarded.slice(1), nil, guarded.slice(2 + 4 * rows)}

          command = ML::Metal::CommandBuffer.new
          encoder = ML::Metal::ComputeEncoder.new(command)
          linear.project(encoder, first, inputs[0], reference, rows)
          linear.project(encoder, second, inputs[2], reference.slice(4 * rows), rows)
          linear.project_q8_group(encoder, {unaligned, nil, second}, inputs, outputs, rows).should be_false
          linear.project_q8_group(encoder, {first, nil, second}, inputs, outputs, rows).should be_true
          encoder.end_encoding
          command.commit_and_wait

          data = Inkling2LinearSpecSupport.read(guarded, 12 * rows + 3)
          {data.first, data[1 + 4 * rows], data.last}.should eq({123_f32, 123_f32, 123_f32})
          actual = data[1, 4 * rows] + data[2 + 4 * rows, 8 * rows]
          expected = Inkling2LinearSpecSupport.read(reference, 12 * rows)
          Inkling2LinearSpecSupport.float_bits(actual).should eq(Inkling2LinearSpecSupport.float_bits(expected))
        end
      ensure
        buffers.each(&.release)
        linear.close
        gguf.close
        File.delete?(fixture.path)
      end
    end

    {% if flag?(:aarch64) %}
      it "shares the Q8_0 prefill input across projections with independent output buffers" do
        pending! "Metal unavailable" unless ML::Metal::Device.init!

        fixture = Inkling2LinearSpecSupport.write_model
        gguf = ML::GGUF::GGUFFile.new(fixture.path)
        linear = ML::GGUF::Inkling2ResidentLinear.new(gguf)
        buffers = [] of ML::MetalBuffer
        begin
          rows, in_dim, out_dim = 35, 64, 39
          elements = rows * out_dim
          bytes = out_dim * (in_dim // 32) * 34
          weights = {"q8_tiled_projection", "gate_up"}.map do |name|
            raw = Inkling2LinearSpecSupport.mapped_bytes(gguf, fixture.offsets[name], bytes)
            ML::GGUF::QuantWeight.new(raw, ML::GGUF::TensorType::Q8_0, out_dim, in_dim)
          end
          input = Inkling2LinearSpecSupport.float_view(
            Array(Float32).new(rows * in_dim + 1) { |index| ((index * 13) % 71 - 35).to_f32 * 0.019_f32 },
            buffers
          ).slice(1)
          guarded = weights.map do
            Inkling2LinearSpecSupport.float_view(Array(Float32).new(elements + 2, 123_f32), buffers)
          end
          outputs = guarded.map(&.slice(1))
          reference = Inkling2LinearSpecSupport.float_view(Array(Float32).new(2 * elements, 0_f32), buffers)

          command = ML::Metal::CommandBuffer.new
          encoder = ML::Metal::ComputeEncoder.new(command)
          weights.each_with_index do |weight, i|
            linear.project(encoder, weight, input, reference.slice(i * elements), rows)
          end
          linear.project_q8_prefill_pair(encoder, weights, input, outputs, 15).should be_false
          linear.project_q8_prefill_pair(encoder, weights, input, outputs, rows).should be_true
          encoder.end_encoding
          command.commit_and_wait

          guarded.each_with_index do |view, i|
            data = Inkling2LinearSpecSupport.read(view, elements + 2)
            {data.first, data.last}.should eq({123_f32, 123_f32})
            expected = Inkling2LinearSpecSupport.read(reference.slice(i * elements), elements)
            Inkling2LinearSpecSupport.float_bits(data[1, elements]).should eq(Inkling2LinearSpecSupport.float_bits(expected))
          end
        ensure
          buffers.each(&.release)
          linear.close
          gguf.close
          File.delete?(fixture.path)
        end
      end
    {% end %}

    it "keeps raw Q8_0 projection history across post-convolution wrap and rewind" do
      pending! "Metal unavailable" unless ML::Metal::Device.init!

      fixture = Inkling2LinearSpecSupport.write_model
      gguf = ML::GGUF::GGUFFile.new(fixture.path)
      linear = ML::GGUF::Inkling2ResidentLinear.new(gguf)
      buffers = [] of ML::MetalBuffer
      scratch = ML::GGUF::Inkling2ResidentScratch.new
      begin
        channels = 39_i32
        input_width = 64_i32
        tap_count = 3_i32
        raw = Inkling2LinearSpecSupport.mapped_bytes(gguf, fixture.offsets["q8_tiled_projection"], channels * 2 * 34)
        weight = ML::GGUF::QuantWeight.new(raw, ML::GGUF::TensorType::Q8_0, channels, input_width)
        input = Inkling2LinearSpecSupport.float_view(
          Array(Float32).new(14 * input_width) { |i| ((i * 23) % 97 - 48).to_f32 * 0.013_f32 }, buffers)
        taps = Inkling2LinearSpecSupport.float_view(
          Array(Float32).new(channels * tap_count) { |i| ((i * 3) % 17 - 8).to_f32 * 0.027_f32 }, buffers)

        {0, 4}.each do |rollback_rows|
          reference = ML::GGUF::Inkling2ResidentConvolution.new(channels, tap_count, rollback_rows: rollback_rows)
          fused = ML::GGUF::Inkling2ResidentConvolution.new(channels, tap_count, rollback_rows: rollback_rows)
          steps = rollback_rows == 0 ? [{0, 4, 0}, {4, 1, 4}, {5, 4, 5}, {9, 1, 9}] :
            [{0, 4, 0}, {4, 4, 4}, {5, 1, 8}, {6, 4, 9}, {10, 1, 13}]
          begin
            steps.each_with_index do |(start, rows, input_row), index|
              if rollback_rows > 0 && index == 2
                reference.rewind(start)
                fused.rewind(start)
              end
              elements = rows * channels
              result_view = Inkling2LinearSpecSupport.float_view(Array(Float32).new(2 * elements, 0_f32), buffers)
              activation = input.slice(input_row * input_width)
              scratch.reset
              command = ML::Metal::CommandBuffer.new
              encoder = ML::Metal::ComputeEncoder.new(command)
              begin
                projected = scratch.take(elements)
                linear.project(encoder, weight, activation, projected, rows, decode_block: rows > 1)
                separate = reference.encode(encoder, scratch, start, rows, projected, taps)
                combined = if rows == 4
                             fused.encode_q8_block(encoder, scratch, start, rows, linear, weight, activation, taps).not_nil!
                           else
                             fused.encode(encoder, scratch, start, rows, projected, taps)
                           end
                ML::GGUF::Inkling2ResidentMath.copy(encoder, separate, result_view, elements)
                ML::GGUF::Inkling2ResidentMath.copy(encoder, combined, result_view.slice(elements), elements)
              ensure
                encoder.end_encoding
              end
              command.commit_and_wait
              bits = Inkling2LinearSpecSupport.float_bits(Inkling2LinearSpecSupport.read(result_view, 2 * elements))
              bits[0, elements].should eq(bits[elements, elements])
            end
          ensure
            reference.close
            fused.close
          end
        end
      ensure
        scratch.close
        buffers.each(&.release)
        linear.close
        gguf.close
        File.delete?(fixture.path)
      end
    end

    it "encodes F32 and Q8_0 single-row and batched projections with batched MoE in one command buffer" do
      pending! "Metal unavailable" unless ML::Metal::Device.init!

      # Make cache keys a local witness of the pipelines selected below.
      ML::Metal::PipelineCache.clear
      m32_pipelines = {
        "inkling2_resident_moe_gemm_situ_q8_0_m32",
        "inkling2_resident_moe_gemm_q8_0_m32",
        "inkling2_resident_moe_gemm_situ_iq4_nl_m32",
        "inkling2_resident_moe_gemm_iq4_nl_m32",
      }
      m64_pipelines = {
        "inkling2_resident_moe_gemm_situ_q8_0_m64",
        "inkling2_resident_moe_gemm_q8_0_m64",
        "inkling2_resident_moe_gemm_situ_iq4_nl_m64",
        "inkling2_resident_moe_gemm_iq4_nl_m64",
      }
      fixture = Inkling2LinearSpecSupport.write_model
      gguf = ML::GGUF::GGUFFile.new(fixture.path)
      linear = ML::GGUF::Inkling2ResidentLinear.new(gguf)
      linear_indirect = ML::GGUF::Inkling2ResidentLinear.new(gguf, prefill_indirect_dispatch: true)
      linear_m32 = ML::GGUF::Inkling2ResidentLinear.new(gguf, prefill_expert_tile_routes: 32_i32)
      linear_m64 = ML::GGUF::Inkling2ResidentLinear.new(gguf, prefill_expert_tile_routes: 64_i32)
      scratch = ML::GGUF::Inkling2ResidentScratch.new
      scratch_indirect = ML::GGUF::Inkling2ResidentScratch.new
      scratch_m32 = ML::GGUF::Inkling2ResidentScratch.new
      scratch_m64 = ML::GGUF::Inkling2ResidentScratch.new
      buffers = [] of ML::MetalBuffer
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      encoding_ended = false

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

        # Cover every small Q8_0 decode width and every partial final tile.
        q8_decode_input = Array(Float32).new(7 * 64) do |index|
          ((index % 29) - 14).to_f32 * 0.00625_f32
        end
        q8_decode_view = Inkling2LinearSpecSupport.float_view(q8_decode_input, buffers)
        q8_rowwise_output = Inkling2LinearSpecSupport.float_view(Array(Float32).new(7 * 39, 0_f32), buffers)
        7.times do |row|
          linear.project(encoder, q8_tiled_projection, q8_decode_view.slice(row * 64), q8_rowwise_output.slice(row * 39), 1_i32)
        end
        q8_decode_batches = (2..7).map do |rows|
          guarded_output = Inkling2LinearSpecSupport.float_view(Array(Float32).new(rows * 39 + 2, 123_f32), buffers)
          linear.project(encoder, q8_tiled_projection, q8_decode_view, guarded_output.slice(1), rows,
            decode_block: rows <= 4)
          {rows, guarded_output}
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
          grouped_m32 = linear_m32.moe(encoder, scratch_m32, weights, config, decode_moe_view, bias, 4_i32, decode_block: true)
          grouped_m32_readback = Inkling2LinearSpecSupport.float_view(Array(Float32).new(4 * 32, 0_f32), buffers)
          ML::GGUF::Inkling2ResidentMath.copy(encoder, grouped_m32, grouped_m32_readback, 4 * 32)
          grouped_indirect = linear_indirect.moe(encoder, scratch_indirect, weights, config, decode_moe_view, bias, 4_i32, decode_block: true)
          grouped_indirect_readback = Inkling2LinearSpecSupport.float_view(Array(Float32).new(4 * 32, 0_f32), buffers)
          ML::GGUF::Inkling2ResidentMath.copy(encoder, grouped_indirect, grouped_indirect_readback, 4 * 32)
          {rowwise_readback, grouped_readback, grouped_m32_readback, grouped_indirect_readback}
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

        {% if flag?(:aarch64) %}
          # Four-row decode stays on its shared decode kernels in all
          # prefill modes.
          ML::Metal::PipelineCache.resident_linear_spec_contains?("inkling2_resident_moe_group_routes_indirect").should be_false
          m32_pipelines.each do |pipeline|
            ML::Metal::PipelineCache.resident_linear_spec_contains?(pipeline).should be_false
          end
          ML::Metal::PipelineCache.resident_linear_spec_contains?("inkling2_resident_moe_group_routes_split").should be_false
        {% end %}

        # Alternating the winning expert and enabling expert 2 on eight rows
        # keeps every IQ4 group at or below M16, leaving the split wide map empty.
        empty_wide_rows = 17_i32
        empty_wide_input = Array(Float32).new(empty_wide_rows * 32) do |index|
          row, column = index.divmod(32)
          case column
          when 0 then row.even? ? 0.25_f32 : -0.25_f32
          when 1 then row < empty_wide_rows // 2 ? 0.25_f32 : -0.25_f32
          else        ((index % 17) - 8).to_f32 * 0.01_f32
          end
        end
        Inkling2LinearSpecSupport.moe_route_counts(iq4_layer, config, empty_wide_input, empty_wide_rows).should eq([13, 13, 8])
        empty_wide_input_view = Inkling2LinearSpecSupport.float_view(empty_wide_input, buffers)
        empty_wide_m16 = linear.moe(encoder, scratch, iq4_layer, config, empty_wide_input_view, iq4_bias_view, empty_wide_rows)
        empty_wide_m16_readback = Inkling2LinearSpecSupport.float_view(Array(Float32).new(empty_wide_rows * 32, 0_f32), buffers)
        ML::GGUF::Inkling2ResidentMath.copy(encoder, empty_wide_m16, empty_wide_m16_readback, empty_wide_rows * 32)
        empty_wide_m32 = linear_m32.moe(encoder, scratch_m32, iq4_layer, config, empty_wide_input_view, iq4_bias_view, empty_wide_rows)
        empty_wide_m32_readback = Inkling2LinearSpecSupport.float_view(Array(Float32).new(empty_wide_rows * 32, 0_f32), buffers)
        ML::GGUF::Inkling2ResidentMath.copy(encoder, empty_wide_m32, empty_wide_m32_readback, empty_wide_rows * 32)
        {% if flag?(:aarch64) %}
          ML::Metal::PipelineCache.resident_linear_spec_contains?("inkling2_resident_moe_group_routes_split").should be_true
        {% end %}

        # Exercise both sides of the M64 boundary, mixed-size tails and
        # multiple full M64 tiles. Expert 2 is empty or receives one route.
        prefill_moe_cases = {16, 17, 25, 33, 63, 64, 65, 97, 145}.flat_map do |rows|
          input = Array(Float32).new(rows * 32) do |index|
            row, column = index.divmod(32)
            case column
            when 0 then 0.25_f32
            when 1 then row == rows - 1 ? 0.25_f32 : -0.25_f32
            else        ((index % 17) - 8).to_f32 * 0.01_f32
            end
          end
          Inkling2LinearSpecSupport.moe_route_counts(layer, config, input, rows).should eq([rows, rows, 0])
          Inkling2LinearSpecSupport.moe_route_counts(iq4_layer, config, input, rows).should eq([rows, rows - 1, 1])
          input_view = Inkling2LinearSpecSupport.float_view(input, buffers)
          {layer, iq4_layer}.zip({bias_view, iq4_bias_view}).map do |weights, bias|
            m16 = linear.moe(encoder, scratch, weights, config, input_view, bias, rows)
            m16_readback = Inkling2LinearSpecSupport.float_view(Array(Float32).new(rows * 32, 0_f32), buffers)
            ML::GGUF::Inkling2ResidentMath.copy(encoder, m16, m16_readback, rows * 32)
            indirect = linear_indirect.moe(encoder, scratch_indirect, weights, config, input_view, bias, rows)
            indirect_readback = Inkling2LinearSpecSupport.float_view(Array(Float32).new(rows * 32, 0_f32), buffers)
            ML::GGUF::Inkling2ResidentMath.copy(encoder, indirect, indirect_readback, rows * 32)
            m32 = linear_m32.moe(encoder, scratch_m32, weights, config, input_view, bias, rows)
            m32_readback = Inkling2LinearSpecSupport.float_view(Array(Float32).new(rows * 32, 0_f32), buffers)
            ML::GGUF::Inkling2ResidentMath.copy(encoder, m32, m32_readback, rows * 32)
            m64 = linear_m64.moe(encoder, scratch_m64, weights, config, input_view, bias, rows)
            m64_readback = Inkling2LinearSpecSupport.float_view(Array(Float32).new(rows * 32, 0_f32), buffers)
            ML::GGUF::Inkling2ResidentMath.copy(encoder, m64, m64_readback, rows * 32)
            {rows, m16_readback, indirect_readback, m32_readback, m64_readback}
          end
        end

        encoder.end_encoding
        encoding_ended = true
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
        q8_decode_batches.each do |rows, guarded_output|
          entries = rows * 39
          data = Inkling2LinearSpecSupport.read(guarded_output, entries + 2)
          {data.first, data.last}.should eq({123_f32, 123_f32})
          Inkling2LinearSpecSupport.float_bits(data[1, entries]).should eq(
            Inkling2LinearSpecSupport.float_bits(Inkling2LinearSpecSupport.read(q8_rowwise_output, entries)))
        end
        expected_moe = Inkling2LinearSpecSupport.moe_expected(layer, config, moe_input, 3_i32)
        Inkling2LinearSpecSupport.read(moe_readback, 96_i32).zip(expected_moe).each do |actual, expected|
          actual.should be_close(expected, 2e-3_f32)
        end
        expected_iq4 = Inkling2LinearSpecSupport.moe_expected(iq4_layer, config, moe_input, 3_i32)
        Inkling2LinearSpecSupport.read(iq4_readback, 96_i32).zip(expected_iq4).each do |actual, expected|
          actual.should be_close(expected, 2e-2_f32)
        end
        decode_moe_cases.each do |rowwise_readback, grouped_readback, grouped_m32_readback, grouped_indirect_readback|
          expected = Inkling2LinearSpecSupport.float_bits(Inkling2LinearSpecSupport.read(rowwise_readback, 4 * 32))
          {grouped_readback, grouped_m32_readback, grouped_indirect_readback}.each do |readback|
            Inkling2LinearSpecSupport.float_bits(Inkling2LinearSpecSupport.read(readback, 4 * 32)).should eq(expected)
          end
        end
        grouped_cases.each do |weights, readback|
          expected = Inkling2LinearSpecSupport.moe_expected(weights, config, grouped_input, grouped_rows)
          tolerance = weights.gate_up.type.iq4_nl? ? 2e-2_f32 : 2e-3_f32
          Inkling2LinearSpecSupport.read(readback, grouped_rows * 32).zip(expected).each do |actual, reference|
            actual.should be_close(reference, tolerance)
          end
        end
        prefill_moe_cases.each do |rows, m16_readback, indirect_readback, m32_readback, m64_readback|
          expected = Inkling2LinearSpecSupport.float_bits(Inkling2LinearSpecSupport.read(m16_readback, rows * 32))
          {indirect_readback, m32_readback, m64_readback}.each do |readback|
            Inkling2LinearSpecSupport.float_bits(Inkling2LinearSpecSupport.read(readback, rows * 32)).should eq(expected)
          end
        end
        Inkling2LinearSpecSupport.float_bits(Inkling2LinearSpecSupport.read(empty_wide_m32_readback, empty_wide_rows * 32)).should eq(
          Inkling2LinearSpecSupport.float_bits(Inkling2LinearSpecSupport.read(empty_wide_m16_readback, empty_wide_rows * 32)))
        {% if flag?(:aarch64) %}
          ML::Metal::PipelineCache.resident_linear_spec_contains?("inkling2_resident_moe_group_routes_indirect").should be_true
          m32_pipelines.each do |pipeline|
            ML::Metal::PipelineCache.resident_linear_spec_contains?(pipeline).should be_true
          end
          m64_pipelines.each do |pipeline|
            ML::Metal::PipelineCache.resident_linear_spec_contains?(pipeline).should be_true
          end
        {% end %}
      ensure
        encoder.end_encoding unless encoding_ended
        buffers.each(&.release)
        scratch.close
        scratch_indirect.close
        scratch_m32.close
        scratch_m64.close
        linear.close
        linear_indirect.close
        linear_m32.close
        linear_m64.close
        gguf.close
        File.delete(fixture.path) if File.exists?(fixture.path)
      end
    end

    it "keeps fused IQ4 SiTU MoE bit-identical to the unfused scalar and decode paths" do
      pending! "Metal unavailable" unless ML::Metal::Device.init!

      ML::Metal::PipelineCache.clear
      fixture = Inkling2LinearSpecSupport.write_model
      gguf = ML::GGUF::GGUFFile.new(fixture.path)
      fused = ML::GGUF::Inkling2ResidentLinear.new(gguf)
      unfused = Inkling2UnfusedResidentLinearSpec.new(gguf)
      fused_scratch = ML::GGUF::Inkling2ResidentScratch.new
      unfused_scratch = ML::GGUF::Inkling2ResidentScratch.new
      buffers = [] of ML::MetalBuffer
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      encoding_ended = false

      begin
        expert_width = Inkling2LinearSpecSupport::EXPERT_WIDTH
        router = ML::GGUF::QuantWeight.new(
          Inkling2LinearSpecSupport.mapped_bytes(gguf, fixture.offsets["router"], 3 * 32 * sizeof(Float32)),
          ML::GGUF::TensorType::F32, 3_i32, 32_i32,
        )
        gate_up = ML::GGUF::Inkling2ExpertWeights.new(
          Inkling2LinearSpecSupport.mapped_bytes(gguf, fixture.offsets["iq4_gate_up"], 3 * 2 * expert_width * 18),
          ML::GGUF::TensorType::IQ4_NL, 32_i32, 2 * expert_width, 3_i32,
        )
        down = ML::GGUF::Inkling2ExpertWeights.new(
          Inkling2LinearSpecSupport.mapped_bytes(gguf, fixture.offsets["iq4_down"], 3 * expert_width * 18),
          ML::GGUF::TensorType::IQ4_NL, expert_width, 32_i32, 3_i32,
        )
        attention = ML::GGUF::Inkling2AttentionWeights.new(
          router, router, router, router, router, router, router,
          [] of Float32, [] of Float32, [] of Float32, [] of Float32, [] of Float32, [] of Float32,
          nil, nil,
        )
        iq4_layer = ML::GGUF::Inkling2LayerWeights.new(
          attention, [] of Float32, [] of Float32, [] of Float32, nil, [] of Float32, [] of Float32,
          [] of Float32, router, [0_f32, 0_f32, -0.125_f32], gate_up, down, nil,
        )
        config = ML::GGUF::Inkling2Config.new(
          1_i32, 32_i32, 2_i32, 1_i32, 2_i32, 2_i32, 2_i32, 16_i32, 4_i32, 2_i32,
          1_f32, 1_f32, 1_f32, 1e-5_f32, 3_i32, 2_i32, expert_width, 1.25_f32, 4_f32,
          25_f32, 1_i32, 8_i32, 8_i32, 32_i32, 0_i32, 1_i32, 3_i32, 2_i32,
          30_f32, false, [true],
        )
        bias = Inkling2LinearSpecSupport.float_view(iq4_layer.router_bias.not_nil!, buffers)

        # These rows cover repeated expert use and different expert pairs in
        # the same block, including both route ranks for the grouped kernel.
        decode_input = Array(Float32).new(4 * 32) do |index|
          row, column = index.divmod(32)
          case column
          when 0 then row.even? ? 0.25_f32 : -0.25_f32
          when 1 then row < 2 ? 0_f32 : -0.25_f32
          else        ((row * 7 + column) % 13 - 6).to_f32 * 0.01_f32
          end
        end
        Inkling2LinearSpecSupport.moe_route_counts(iq4_layer, config, decode_input, 4_i32).should eq([3, 3, 2])

        scalar_input = Inkling2LinearSpecSupport.float_view(decode_input[0, 32].to_a, buffers)
        scalar_unfused = unfused.moe(encoder, unfused_scratch, iq4_layer, config, scalar_input, bias, 1_i32)
        scalar_unfused_readback = Inkling2LinearSpecSupport.float_view(Array(Float32).new(32, 0_f32), buffers)
        ML::GGUF::Inkling2ResidentMath.copy(encoder, scalar_unfused, scalar_unfused_readback, 32)
        ML::Metal::PipelineCache.resident_linear_spec_contains?("inkling2_resident_moe_matvec_iq4_nl").should be_true
        ML::Metal::PipelineCache.resident_linear_spec_contains?("inkling2_resident_moe_matvec_situ_iq4_nl").should be_false

        scalar_fused = fused.moe(encoder, fused_scratch, iq4_layer, config, scalar_input, bias, 1_i32)
        scalar_fused_readback = Inkling2LinearSpecSupport.float_view(Array(Float32).new(32, 0_f32), buffers)
        ML::GGUF::Inkling2ResidentMath.copy(encoder, scalar_fused, scalar_fused_readback, 32)
        ML::Metal::PipelineCache.resident_linear_spec_contains?("inkling2_resident_moe_matvec_situ_iq4_nl").should be_true

        decode_view = Inkling2LinearSpecSupport.float_view(decode_input, buffers)
        decode_unfused = unfused.moe(encoder, unfused_scratch, iq4_layer, config, decode_view, bias, 4_i32,
          decode_block: true)
        decode_unfused_readback = Inkling2LinearSpecSupport.float_view(Array(Float32).new(4 * 32, 0_f32), buffers)
        ML::GGUF::Inkling2ResidentMath.copy(encoder, decode_unfused, decode_unfused_readback, 4 * 32)
        ML::Metal::PipelineCache.resident_linear_spec_contains?("inkling2_resident_moe_decode_grouped_iq4_nl").should be_true
        ML::Metal::PipelineCache.resident_linear_spec_contains?("inkling2_resident_moe_decode_grouped_situ_iq4_nl").should be_false

        decode_fused = fused.moe(encoder, fused_scratch, iq4_layer, config, decode_view, bias, 4_i32,
          decode_block: true)
        decode_fused_readback = Inkling2LinearSpecSupport.float_view(Array(Float32).new(4 * 32, 0_f32), buffers)
        ML::GGUF::Inkling2ResidentMath.copy(encoder, decode_fused, decode_fused_readback, 4 * 32)
        ML::Metal::PipelineCache.resident_linear_spec_contains?("inkling2_resident_moe_decode_grouped_situ_iq4_nl").should be_true

        encoder.end_encoding
        encoding_ended = true
        command.commit_and_wait

        Inkling2LinearSpecSupport.float_bits(Inkling2LinearSpecSupport.read(scalar_fused_readback, 32)).should eq(
          Inkling2LinearSpecSupport.float_bits(Inkling2LinearSpecSupport.read(scalar_unfused_readback, 32)))
        Inkling2LinearSpecSupport.float_bits(Inkling2LinearSpecSupport.read(decode_fused_readback, 4 * 32)).should eq(
          Inkling2LinearSpecSupport.float_bits(Inkling2LinearSpecSupport.read(decode_unfused_readback, 4 * 32)))
      ensure
        encoder.end_encoding unless encoding_ended
        buffers.each(&.release)
        fused_scratch.close
        unfused_scratch.close
        fused.close
        unfused.close
        gguf.close
        File.delete(fixture.path) if File.exists?(fixture.path)
      end
    end
  end
{% end %}
