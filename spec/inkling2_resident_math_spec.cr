require "spec"
require "../src/ml/gguf/inkling2_engine"
require "../src/ml/gguf/inkling2_resident_math"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  module Inkling2ResidentMathSpecSupport
    extend self

    PREFIX = 2

    def float_view(values : Array(Float32), buffers : Array(ML::MetalBuffer)) : ML::GGUF::Inkling2View
      data = Array(Float32).new(PREFIX, -777.0_f32) + values
      buffer = ML::MetalBuffer.from_array(data)
      buffers << buffer
      ML::GGUF::Inkling2View.new(buffer, PREFIX.to_i64 * sizeof(Float32))
    end

    def byte_view(bytes : Bytes, buffers : Array(ML::MetalBuffer)) : ML::GGUF::Inkling2View
      buffer = ML::MetalBuffer.new(bytes.size.to_i64)
      buffer.write_bytes(bytes.to_unsafe, bytes.size)
      buffers << buffer
      ML::GGUF::Inkling2View.new(buffer, (PREFIX * sizeof(Int32)).to_i64)
    end

    def token_view(ids : Array(Int32), buffers : Array(ML::MetalBuffer)) : ML::GGUF::Inkling2View
      buffer = ML::MetalBuffer.new(((PREFIX + ids.size) * sizeof(Int32)).to_i64)
      pointer = buffer.contents.as(Int32*)
      PREFIX.times { |index| pointer[index] = -777 }
      ids.each_with_index { |id, index| pointer[PREFIX + index] = id }
      buffers << buffer
      ML::GGUF::Inkling2View.new(buffer, PREFIX.to_i64 * sizeof(Int32))
    end

    def read(view : ML::GGUF::Inkling2View, count : Int32) : Array(Float32)
      offset = (view.offset // sizeof(Float32)).to_i32
      view.buffer.read(offset + count)[offset, count].to_a
    end
  end

  describe ML::GGUF::Inkling2ResidentMath do
    it "encodes offset-aware row and element math in one command buffer" do
      pending! "Metal unavailable" unless ML::Metal::Device.init!

      buffers = [] of ML::MetalBuffer
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      encoder_ended = false
      begin
        width = 3_i32
        rows = 2_i32
        eps = 1e-6_f32
        x = [0.25_f32, -0.75_f32, 1.5_f32, 2.0_f32, 0.5_f32, -1.0_f32]
        y = [-0.5_f32, 1.0_f32, 0.25_f32, 0.75_f32, -2.0_f32, 0.1_f32]
        weights = [1.25_f32, 0.5_f32, -0.75_f32]
        x_view = Inkling2ResidentMathSpecSupport.float_view(x, buffers)
        y_view = Inkling2ResidentMathSpecSupport.float_view(y, buffers)
        weight_view = Inkling2ResidentMathSpecSupport.float_view(weights, buffers)
        rms_view = Inkling2ResidentMathSpecSupport.float_view(Array(Float32).new(x.size, 0_f32), buffers)
        concat_view = Inkling2ResidentMathSpecSupport.float_view(Array(Float32).new(x.size * 2, 0_f32), buffers)

        ML::GGUF::Inkling2ResidentMath.rms(encoder, x_view, rms_view, weight_view, width, rows, eps)
        ML::GGUF::Inkling2ResidentMath.rms_concat(encoder, x_view, y_view, concat_view, width, rows, eps)

        depth_bank = [
          0.4_f32, -0.3_f32, 0.8_f32, 0.1_f32, 0.9_f32, -0.6_f32,
          -0.7_f32, 0.2_f32, 0.5_f32, 0.3_f32, -0.4_f32, 0.7_f32,
        ]
        current = [0.2_f32, 0.6_f32, -0.1_f32, -0.5_f32, 0.8_f32, 0.4_f32]
        queries = [0.7_f32, -0.2_f32, 0.3_f32, -0.4_f32, 0.1_f32, 0.9_f32]
        bank_view = Inkling2ResidentMathSpecSupport.float_view(depth_bank, buffers)
        current_view = Inkling2ResidentMathSpecSupport.float_view(current, buffers)
        query_view = Inkling2ResidentMathSpecSupport.float_view(queries, buffers)
        depth_view = Inkling2ResidentMathSpecSupport.float_view(Array(Float32).new(rows * 2 * width, 0_f32), buffers)
        past_view = Inkling2ResidentMathSpecSupport.float_view(Array(Float32).new(rows * 2 * width, 0_f32), buffers)
        ML::GGUF::Inkling2ResidentMath.depth_mix(encoder, bank_view, 2_i32, current_view, query_view, depth_view, width, rows, 2_i32)
        ML::GGUF::Inkling2ResidentMath.depth_mix(encoder, bank_view, 2_i32, nil, query_view, past_view, width, rows, 2_i32)

        sum_view = Inkling2ResidentMathSpecSupport.float_view(Array(Float32).new(x.size, 0_f32), buffers)
        copy_view = Inkling2ResidentMathSpecSupport.float_view(Array(Float32).new(x.size, 0_f32), buffers)
        ML::GGUF::Inkling2ResidentMath.add(encoder, x_view, y_view, sum_view, x.size.to_i32)
        ML::GGUF::Inkling2ResidentMath.copy(encoder, x_view, copy_view, x.size.to_i32)

        gain = [0.2_f32, -0.3_f32, 0.6_f32, -0.1_f32, 0.7_f32, -0.5_f32]
        forget = [-0.4_f32, 0.1_f32, 0.3_f32, 0.5_f32, -0.2_f32, 0.8_f32]
        control_view = Inkling2ResidentMathSpecSupport.float_view(Array(Float32).new(x.size, 0_f32), buffers)
        gain_view = Inkling2ResidentMathSpecSupport.float_view(gain, buffers)
        forget_view = Inkling2ResidentMathSpecSupport.float_view(forget, buffers)
        ML::GGUF::Inkling2ResidentMath.controls(encoder, x_view, y_view, gain_view, forget_view, control_view, x.size.to_i32)

        embedding = [
          1.0_f32, 0.0_f32, -1.0_f32,
          0.25_f32, -0.5_f32, 0.75_f32,
          -0.2_f32, 0.4_f32, 0.6_f32,
        ]
        embedding_view = Inkling2ResidentMathSpecSupport.float_view(embedding, buffers)
        ids_view = Inkling2ResidentMathSpecSupport.token_view([2_i32, 1_i32], buffers)
        embed_view = Inkling2ResidentMathSpecSupport.float_view(Array(Float32).new(x.size, 0_f32), buffers)
        ML::GGUF::Inkling2ResidentMath.embed(encoder, embedding_view, ML::GGUF::TensorType::F32, ids_view, embed_view, width, rows).should be_true

        logits = [1.2_f32, -0.4_f32, 0.2_f32, 4.0_f32]
        logits_view = Inkling2ResidentMathSpecSupport.float_view(logits, buffers)
        softcap_view = Inkling2ResidentMathSpecSupport.float_view(Array(Float32).new(logits.size, 0_f32), buffers)
        ML::GGUF::Inkling2ResidentMath.softcap(encoder, logits_view, softcap_view, 0.7_f32, logits.size.to_i32)

        encoder.end_encoding
        encoder_ended = true
        command.commit_and_wait

        expected_rms = Array(Float32).new(x.size) do |index|
          row = index // width
          column = index % width
          ML::GGUF::Inkling2Math.rms_norm(x[row * width, width].to_a, eps, weights)[column]
        end
        Inkling2ResidentMathSpecSupport.read(rms_view, x.size.to_i32).zip(expected_rms).each do |actual, expected|
          actual.should be_close(expected, 2e-5_f32)
        end

        expected_concat = Array(Float32).new
        rows.times do |row|
          expected_concat.concat(
            ML::GGUF::Inkling2Math.rms_norm(x[row * width, width].to_a, eps) +
            ML::GGUF::Inkling2Math.rms_norm(y[row * width, width].to_a, eps)
          )
        end
        Inkling2ResidentMathSpecSupport.read(concat_view, (x.size * 2).to_i32).zip(expected_concat).each do |actual, expected|
          actual.should be_close(expected, 2e-5_f32)
        end

        expected_depth = Array(Float32).new(rows * 2 * width, 0_f32)
        expected_past = Array(Float32).new(rows * 2 * width, 0_f32)
        rows.times do |row|
          bank_states = Array(Array(Float32)).new(2) do |slot|
            start = slot * rows * width + row * width
            depth_bank[start, width].to_a
          end
          with_current = bank_states + [current[row * width, width].to_a]
          2.times do |query_id|
            query = queries[query_id * width, width].to_a
            mixed = ML::GGUF::Inkling2Math.depth_mix(with_current, query).first
            mixed_past = ML::GGUF::Inkling2Math.depth_mix(bank_states, query).first
            target = (query_id * rows + row) * width
            width.times do |column|
              expected_depth[target + column] = mixed[column]
              expected_past[target + column] = mixed_past[column]
            end
          end
        end
        Inkling2ResidentMathSpecSupport.read(depth_view, expected_depth.size.to_i32).zip(expected_depth).each do |actual, expected|
          actual.should be_close(expected, 2e-5_f32)
        end
        Inkling2ResidentMathSpecSupport.read(past_view, expected_past.size.to_i32).zip(expected_past).each do |actual, expected|
          actual.should be_close(expected, 2e-5_f32)
        end

        Inkling2ResidentMathSpecSupport.read(sum_view, x.size.to_i32).should eq(x.zip(y).map { |a, b| a + b })
        Inkling2ResidentMathSpecSupport.read(copy_view, x.size.to_i32).should eq(x)
        expected_control = x.each_index.map do |index|
          x[index] - Math.tanh(forget[index]) * x[index] + Math.tanh(gain[index]) * y[index]
        end
        Inkling2ResidentMathSpecSupport.read(control_view, x.size.to_i32).zip(expected_control).each do |actual, expected|
          actual.should be_close(expected, 2e-6_f32)
        end
        Inkling2ResidentMathSpecSupport.read(embed_view, x.size.to_i32).should eq(embedding[6, 3].to_a + embedding[3, 3].to_a)
        Inkling2ResidentMathSpecSupport.read(softcap_view, logits.size.to_i32).zip(logits).each do |actual, value|
          actual.should be_close(0.7_f32 * Math.tanh(value / 0.7_f32), 2e-6_f32)
        end
      ensure
        encoder.end_encoding unless encoder_ended
        buffers.each(&.release)
      end
    end

    it "returns the first finite maximum after the same Float32 softcap" do
      pending! "Metal unavailable" unless ML::Metal::Device.init!

      vocab, stride, rows = 2051, 2064, 4
      raw = Array(Float32).new(stride * rows, Float32::NAN)
      rows.times { |row| raw[row * stride + vocab] = 1000_f32 }
      # Distinct raw scores that round to the same score after softcap.
      raw[5] = 990.000061_f32
      raw[1029] = 990.000122_f32
      raw[stride + 1000] = Float32::INFINITY
      raw[stride + 2010] = 10_f32
      raw[2 * stride + 2] = -Float32::INFINITY
      raw[3 * stride + 6] = -0.0_f32
      raw[3 * stride + 1024] = 0.0_f32
      raw[3 * stride + 2049] = 1e-44_f32

      buffers = [] of ML::MetalBuffer
      scratch = ML::GGUF::Inkling2ResidentScratch.new
      begin
        input = Inkling2ResidentMathSpecSupport.float_view(raw, buffers)
        {0_f32, 30_f32}.each do |cap|
          capped = Inkling2ResidentMathSpecSupport.float_view(Array(Float32).new(vocab * rows, 0_f32), buffers)
          ids = Inkling2ResidentMathSpecSupport.token_view(Array(Int32).new(rows + 1, -888), buffers)
          scratch.reset
          command = ML::Metal::CommandBuffer.new
          encoder = ML::Metal::ComputeEncoder.new(command)
          begin
            rows.times do |row|
              ML::GGUF::Inkling2ResidentMath.softcap(encoder, input.slice(row * stride), capped.slice(row * vocab), cap, vocab)
            end
            ML::GGUF::Inkling2ResidentMath.greedy_ids(encoder, scratch, input, ids, cap, vocab, stride, rows)
          ensure
            encoder.end_encoding
          end
          command.commit_and_wait

          scores = Inkling2ResidentMathSpecSupport.read(capped, vocab * rows)
          expected = Array(Int32).new(rows) do |row|
            logits = scores[row * vocab, vocab]
            logits.any?(&.finite?) ? ML::GGUF::Inkling2Engine.greedy_token_id(logits) : -1
          end
          if cap.zero?
            expected.should eq([1029, 2010, -1, 2049])
          else
            expected.first.should eq(5)
          end
          data = (ids.buffer.contents + ids.offset).as(Int32*)
          Slice.new(data, rows + 1).to_a.should eq(expected + [-888])
        end
      ensure
        scratch.close
        buffers.each(&.release)
      end
    end

    it "decodes offset Q8_0 and IQ4_NL embedding rows" do
      pending! "Metal unavailable" unless ML::Metal::Device.init!

      buffers = [] of ML::MetalBuffer
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      encoder_ended = false
      begin
        q8_bytes = Bytes.new(Inkling2ResidentMathSpecSupport::PREFIX * sizeof(Int32) + 2 * 34, 0_u8)
        q8_bytes[8] = 0x00_u8
        q8_bytes[9] = 0x3C_u8 # half 1.0
        32.times { |i| q8_bytes[10 + i] = i.to_i8.unsafe_as(UInt8) }
        q8_bytes[42] = 0x00_u8
        q8_bytes[43] = 0x38_u8 # half 0.5
        32.times { |i| q8_bytes[44 + i] = (31 - i).to_i8.unsafe_as(UInt8) }
        q8_weight = Inkling2ResidentMathSpecSupport.byte_view(q8_bytes, buffers)
        q8_ids = Inkling2ResidentMathSpecSupport.token_view([1_i32], buffers)
        q8_output = Inkling2ResidentMathSpecSupport.float_view(Array(Float32).new(32, 0_f32), buffers)
        ML::GGUF::Inkling2ResidentMath.embed(encoder, q8_weight, ML::GGUF::TensorType::Q8_0, q8_ids, q8_output, 32_i32, 1_i32).should be_true

        iq4_bytes = Bytes.new(Inkling2ResidentMathSpecSupport::PREFIX * sizeof(Int32) + 2 * 18, 0_u8)
        iq4_bytes[8] = 0x00_u8
        iq4_bytes[9] = 0x3C_u8 # half 1.0
        16.times { |i| iq4_bytes[10 + i] = ((i << 4) | i).to_u8 }
        iq4_bytes[26] = 0x00_u8
        iq4_bytes[27] = 0x3C_u8
        16.times { |i| iq4_bytes[28 + i] = ((15 - i) << 4 | (15 - i)).to_u8 }
        iq4_weight = Inkling2ResidentMathSpecSupport.byte_view(iq4_bytes, buffers)
        iq4_ids = Inkling2ResidentMathSpecSupport.token_view([1_i32], buffers)
        iq4_output = Inkling2ResidentMathSpecSupport.float_view(Array(Float32).new(32, 0_f32), buffers)
        ML::GGUF::Inkling2ResidentMath.embed(encoder, iq4_weight, ML::GGUF::TensorType::IQ4_NL, iq4_ids, iq4_output, 32_i32, 1_i32).should be_true

        encoder.end_encoding
        encoder_ended = true
        command.commit_and_wait

        expected_q8 = Array(Float32).new(32) { |index| (31 - index).to_f32 * 0.5_f32 }
        Inkling2ResidentMathSpecSupport.read(q8_output, 32_i32).should eq(expected_q8)
        iq4_codes = ML::GGUF::Dequant::IQ4_NL_VALUES
        expected_iq4 = Array(Float32).new(32) do |index|
          code = index < 16 ? 15 - index : 15 - (index - 16)
          iq4_codes[code].to_f32
        end
        Inkling2ResidentMathSpecSupport.read(iq4_output, 32_i32).should eq(expected_iq4)
      ensure
        encoder.end_encoding unless encoder_ended
        buffers.each(&.release)
      end
    end
  end
{% end %}
