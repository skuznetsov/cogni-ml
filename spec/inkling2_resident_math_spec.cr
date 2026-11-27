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

    def guarded_float_view(values : Array(Float32), suffix : Int32,
                           buffers : Array(ML::MetalBuffer)) : ML::GGUF::Inkling2View
      data = Array(Float32).new(PREFIX, -777.0_f32) + values + Array(Float32).new(suffix, -999.0_f32)
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

    def raw_bits(view : ML::GGUF::Inkling2View, count : Int32) : Array(UInt32)
      view.buffer.read(count).map(&.unsafe_as(UInt32))
    end
  end

  describe ML::GGUF::Inkling2ResidentMath do
    it "checks triplet scratch extents before Int32 allocation" do
      max_width = Int32::MAX // 6
      ML::GGUF::Inkling2ResidentMath.rms_concat_triplet_elements(max_width).should eq(max_width * 6)
      expect_raises(ArgumentError) do
        ML::GGUF::Inkling2ResidentMath.rms_concat_triplet_elements(max_width + 1)
      end
      expect_raises(ArgumentError) do
        ML::GGUF::Inkling2ResidentMath.triplet_elements(Int32::MAX, 3)
      end
    end

    it "batches three offset RMS-concat rows without changing FP32 bits or guards" do
      pending! "Metal unavailable" unless ML::Metal::Device.init!

      buffers = [] of ML::MetalBuffer
      width = 513_i32
      eps = 1e-6_f32
      suffix = 5_i32
      guard_count = Inkling2ResidentMathSpecSupport::PREFIX + suffix
      x_values = Array(Float32).new(3 * width) do |index|
        (((index * 17) % 127) - 63).to_f32 * 0.00390625_f32
      end
      message_values = Array(Float32).new(3 * width) do |index|
        (((index * 29) % 113) - 56).to_f32 * 0.005859375_f32
      end
      alias_values = x_values[0, width].to_a
      x_view = Inkling2ResidentMathSpecSupport.guarded_float_view(x_values, suffix, buffers)
      alias_view = Inkling2ResidentMathSpecSupport.guarded_float_view(alias_values, suffix, buffers)
      message_view = Inkling2ResidentMathSpecSupport.guarded_float_view(message_values, suffix, buffers)
      x_before = x_view.buffer.read(guard_count + x_values.size.to_i32).map(&.unsafe_as(UInt32))
      alias_before = alias_view.buffer.read(guard_count + alias_values.size.to_i32).map(&.unsafe_as(UInt32))
      message_before = message_view.buffer.read(guard_count + message_values.size.to_i32).map(&.unsafe_as(UInt32))

      input_sets = [
        {x_view, x_view.slice(width), x_view.slice(2 * width)},
        {alias_view, alias_view, alias_view},
      ]
      expected_views = Array(Array(ML::GGUF::Inkling2View)).new(2) do
        Array(ML::GGUF::Inkling2View).new(3) do
          Inkling2ResidentMathSpecSupport.guarded_float_view(
            Array(Float32).new(2 * width, 123.5_f32), suffix, buffers
          )
        end
      end
      output_views = Array(ML::GGUF::Inkling2View).new(2) do
        Inkling2ResidentMathSpecSupport.guarded_float_view(
          Array(Float32).new(6 * width, 321.25_f32), suffix, buffers
        )
      end

      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      encoder_ended = false
      begin
        input_sets.each_with_index do |inputs, case_index|
          3.times do |index|
            ML::GGUF::Inkling2ResidentMath.rms_concat(
              encoder, inputs[index], message_view.slice(index * width), expected_views[case_index][index],
              width, 1_i32, eps
            )
          end
          ML::GGUF::Inkling2ResidentMath.rms_concat_triplet(
            encoder, inputs, message_view, output_views[case_index], width, eps
          )
        end
        encoder.end_encoding
        encoder_ended = true
        command.commit_and_wait

        input_sets.size.times do |case_index|
          expected = expected_views[case_index].flat_map do |view|
            Inkling2ResidentMathSpecSupport.read(view, 2 * width)
          end
          actual_view = output_views[case_index]
          actual = Inkling2ResidentMathSpecSupport.read(actual_view, 6 * width)
          actual.map(&.unsafe_as(UInt32)).should eq(expected.map(&.unsafe_as(UInt32)))

          full_output = actual_view.buffer.read(guard_count + 6 * width)
          full_output[0, Inkling2ResidentMathSpecSupport::PREFIX].map(&.unsafe_as(UInt32)).should eq(
            Array(Float32).new(Inkling2ResidentMathSpecSupport::PREFIX, -777.0_f32).map(&.unsafe_as(UInt32))
          )
          full_output[Inkling2ResidentMathSpecSupport::PREFIX + 6 * width, suffix].map(&.unsafe_as(UInt32)).should eq(
            Array(Float32).new(suffix, -999.0_f32).map(&.unsafe_as(UInt32))
          )

          expected_views[case_index].each do |view|
            guarded = view.buffer.read(guard_count + 2 * width)
            guarded[0, Inkling2ResidentMathSpecSupport::PREFIX].map(&.unsafe_as(UInt32)).should eq(
              Array(Float32).new(Inkling2ResidentMathSpecSupport::PREFIX, -777.0_f32).map(&.unsafe_as(UInt32))
            )
            guarded[Inkling2ResidentMathSpecSupport::PREFIX + 2 * width, suffix].map(&.unsafe_as(UInt32)).should eq(
              Array(Float32).new(suffix, -999.0_f32).map(&.unsafe_as(UInt32))
            )
          end
        end

        x_view.buffer.read(guard_count + x_values.size.to_i32).map(&.unsafe_as(UInt32)).should eq(x_before)
        alias_view.buffer.read(guard_count + alias_values.size.to_i32).map(&.unsafe_as(UInt32)).should eq(alias_before)
        message_view.buffer.read(guard_count + message_values.size.to_i32).map(&.unsafe_as(UInt32)).should eq(message_before)
      ensure
        encoder.end_encoding unless encoder_ended
        buffers.each(&.release)
      end
    end

    it "matches three weighted RMS reductions with aliased inputs and offset weights" do
      pending! "Metal unavailable" unless ML::Metal::Device.init!

      support = Inkling2ResidentMathSpecSupport
      buffers = [] of ML::MetalBuffer
      width = 513
      eps = 1e-6_f32
      x = support.float_view(Array(Float32).new(2 * width) do |i|
        ((i * 19) % 157 - 78).to_f32 * 0.033_f32
      end, buffers)
      scales = support.float_view(Array(Float32).new(3 * width) do |i|
        ((i * 31) % 105 - 52).to_f32 * 0.021_f32
      end, buffers)
      inputs = {x, x.slice(width), x}
      weights = {scales, scales.slice(width), scales.slice(2 * width)}
      guard = -999.0_f32
      reference = support.guarded_float_view(Array(Float32).new(3 * width, 123.0_f32), 1, buffers)
      output = support.guarded_float_view(Array(Float32).new(3 * width, 321.0_f32), 1, buffers)
      math = ML::GGUF::Inkling2ResidentMath

      begin
        ML::Metal::Dispatch.execute_sequence do |command|
          encoder = ML::Metal::ComputeEncoder.new(command)
          begin
            expect_raises(ArgumentError, /overlaps/) do
              math.rms_triplet(encoder, inputs, weights, scales, width, eps)
            end
            inputs.each_with_index do |input, index|
              math.rms(encoder, input, reference.slice(index * width), weights[index], width, 1, eps)
            end
            math.rms_triplet(encoder, inputs, weights, output, width, eps)
          ensure
            encoder.end_encoding
          end
        end

        support.read(output, 3 * width + 1).map(&.unsafe_as(UInt32)).should eq(
          support.read(reference, 3 * width + 1).map(&.unsafe_as(UInt32))
        )
        prefix = Inkling2ResidentMathSpecSupport::PREFIX
        output.buffer.read(prefix).to_a.should eq(Array(Float32).new(prefix, -777.0_f32))
        support.read(output.slice(3 * width), 1).should eq([guard])
      ensure
        buffers.each(&.release)
      end
    end

    it "matches RMS-depth-mix-RMS for one query and preserves or appends the bank" do
      pending! "Metal unavailable" unless ML::Metal::Device.init!

      support = Inkling2ResidentMathSpecSupport
      math = ML::GGUF::Inkling2ResidentMath
      suffix = 3_i32
      prefix = Inkling2ResidentMathSpecSupport::PREFIX
      prefix_bits = Array(Float32).new(prefix, -777.0_f32).map(&.unsafe_as(UInt32))
      suffix_bits = Array(Float32).new(suffix, -999.0_f32).map(&.unsafe_as(UInt32))

      {2304_i32, 513_i32}.each do |width|
        {0_i32, 1_i32, 2_i32, 3_i32, 4_i32}.each do |count|
          buffers = [] of ML::MetalBuffer
          begin
            unit_eps = 0.00031_f32
            output_eps = 0.0021_f32
            bank_capacity = 5_i32
            bank_elements = bank_capacity * width
            input_values = Array(Float32).new(width) do |index|
              ((index * 37 % 251) - 125).to_f32 * 0.0078125_f32 +
                ((index * 11 % 19) - 9).to_f32 * 0.0009765625_f32
            end
            query_values = Array(Float32).new(width) do |index|
              ((index * 23 % 173) - 86).to_f32 * 0.0107421875_f32 +
                ((index * 7 % 13) - 6).to_f32 * 0.001953125_f32
            end
            bank_values = Array(Float32).new(bank_elements) do |index|
              ((index * 29 % 311) - 155).to_f32 * 0.005859375_f32 +
                ((index // width) - 2).to_f32 * 0.03125_f32
            end
            weight_values = Array(Float32).new(width) do |index|
              ((index * 13 % 47) - 23).to_f32 * 0.021484375_f32
            end

            input = support.guarded_float_view(input_values, suffix, buffers)
            query = support.guarded_float_view(query_values, suffix, buffers)
            weights = support.guarded_float_view(weight_values, suffix, buffers)
            reference_bank = support.guarded_float_view(bank_values, suffix, buffers)
            no_save_bank = support.guarded_float_view(bank_values, suffix, buffers)
            save_bank = support.guarded_float_view(bank_values, suffix, buffers)
            unit = support.guarded_float_view(Array(Float32).new(width, 0.0_f32), suffix, buffers)
            mixed = support.guarded_float_view(Array(Float32).new(width, 0.0_f32), suffix, buffers)
            expected = support.guarded_float_view(Array(Float32).new(width, 123.0_f32), suffix, buffers)
            no_save_output = support.guarded_float_view(Array(Float32).new(width, 321.25_f32), suffix, buffers)
            save_output = support.guarded_float_view(Array(Float32).new(width, 321.25_f32), suffix, buffers)

            input_before = support.raw_bits(input, prefix + width + suffix)
            query_before = support.raw_bits(query, prefix + width + suffix)
            weights_before = support.raw_bits(weights, prefix + width + suffix)
            bank_buffer_elements = prefix + bank_elements + suffix
            bank_before = support.raw_bits(no_save_bank, bank_buffer_elements)

            command = ML::Metal::CommandBuffer.new
            encoder = ML::Metal::ComputeEncoder.new(command)
            encoder_ended = false
            begin
              math.rms(encoder, input, unit, nil, width, 1_i32, unit_eps)
              math.depth_mix(encoder, reference_bank, count, unit, query, mixed, width, 1_i32, 1_i32)
              math.rms(encoder, mixed, expected, weights, width, 1_i32, output_eps)
              math.depth_mix_rms(
                encoder, input, no_save_bank, count, query, weights, no_save_output,
                width, unit_eps, output_eps
              )
              math.depth_mix_rms(
                encoder, input, save_bank, count, query, weights, save_output,
                width, unit_eps, output_eps, save_to_bank: true
              )
              encoder.end_encoding
              encoder_ended = true
              command.commit_and_wait

              expected_bits = support.read(expected, width).map(&.unsafe_as(UInt32))
              support.read(no_save_output, width).map(&.unsafe_as(UInt32)).should eq(expected_bits)
              support.read(save_output, width).map(&.unsafe_as(UInt32)).should eq(expected_bits)

              support.raw_bits(input, prefix + width + suffix).should eq(input_before)
              support.raw_bits(query, prefix + width + suffix).should eq(query_before)
              support.raw_bits(weights, prefix + width + suffix).should eq(weights_before)
              support.raw_bits(reference_bank, bank_buffer_elements).should eq(bank_before)
              support.raw_bits(no_save_bank, bank_buffer_elements).should eq(bank_before)

              expected_save_bank = bank_before.dup
              unit_bits = support.read(unit, width).map(&.unsafe_as(UInt32))
              unit_bits.each_with_index do |bits, index|
                expected_save_bank[prefix + count * width + index] = bits
              end
              support.raw_bits(save_bank, bank_buffer_elements).should eq(expected_save_bank)

              {expected, no_save_output, save_output}.each do |view|
                output_bits = support.raw_bits(view, prefix + width + suffix)
                output_bits[0, prefix].should eq(prefix_bits)
                output_bits[prefix + width, suffix].should eq(suffix_bits)
              end
              {reference_bank, no_save_bank, save_bank}.each do |view|
                bank_bits = support.raw_bits(view, bank_buffer_elements)
                bank_bits[0, prefix].should eq(prefix_bits)
                bank_bits[prefix + bank_elements, suffix].should eq(suffix_bits)
              end
            ensure
              encoder.end_encoding unless encoder_ended
            end
          ensure
            buffers.each(&.release)
          end
        end
      end
    end

    it "matches scalar controller features and weighted depth-mix triplets bitwise" do
      pending! "Metal unavailable" unless ML::Metal::Device.init!

      support = Inkling2ResidentMathSpecSupport
      math = ML::GGUF::Inkling2ResidentMath
      prefix = Inkling2ResidentMathSpecSupport::PREFIX
      suffix = 4_i32
      prefix_bits = Array(Float32).new(prefix, -777.0_f32).map(&.unsafe_as(UInt32))
      suffix_bits = Array(Float32).new(suffix, -999.0_f32).map(&.unsafe_as(UInt32))
      count = 4_i32
      bank_capacity = count + 1

      {513_i32, 2304_i32}.each do |width|
        buffers = [] of ML::MetalBuffer
        begin
          bank_elements = bank_capacity * width
          bank_buffer_elements = prefix + bank_elements + suffix
          input_values = Array(Float32).new(width) do |index|
            ((index * 37 % 251) - 125).to_f32 * 0.0078125_f32 +
              ((index * 11 % 19) - 9).to_f32 * 0.0009765625_f32
          end
          bank_values = Array(Float32).new(bank_elements) do |index|
            ((index * 29 % 311) - 155).to_f32 * 0.005859375_f32 +
              ((index // width) - 2).to_f32 * 0.03125_f32
          end
          query_values = Array(Float32).new(3 * width) do |index|
            feature = index // width
            column = index % width
            ((column * (23 + 2 * feature) + 31 * feature) % 173 - 86).to_f32 *
              (0.0107421875_f32 + feature.to_f32 * 0.001953125_f32)
          end
          message_values = Array(Float32).new(3 * width) do |index|
            feature = index // width
            column = index % width
            ((column * (29 + 4 * feature) + 17 * feature) % 113 - 56).to_f32 *
              (0.005859375_f32 + feature.to_f32 * 0.00146484375_f32)
          end

          concat_input = support.guarded_float_view(input_values, suffix, buffers)
          concat_reference_bank = support.guarded_float_view(bank_values, suffix, buffers)
          concat_no_save_bank = support.guarded_float_view(bank_values, suffix, buffers)
          concat_save_bank = support.guarded_float_view(bank_values, suffix, buffers)
          concat_queries = support.guarded_float_view(query_values, suffix, buffers)
          concat_messages = support.guarded_float_view(message_values, suffix, buffers)
          concat_unit = support.guarded_float_view(Array(Float32).new(width, 0.0_f32), suffix, buffers)
          concat_reference_mixed = support.guarded_float_view(Array(Float32).new(width, 0.0_f32), suffix, buffers)
          concat_reference_features = support.guarded_float_view(Array(Float32).new(6 * width, 123.0_f32), suffix, buffers)
          concat_no_save_mixed = support.guarded_float_view(Array(Float32).new(width, 321.0_f32), suffix, buffers)
          concat_no_save_features = support.guarded_float_view(Array(Float32).new(6 * width, 321.0_f32), suffix, buffers)
          concat_save_mixed = support.guarded_float_view(Array(Float32).new(width, 321.0_f32), suffix, buffers)
          concat_save_features = support.guarded_float_view(Array(Float32).new(6 * width, 321.0_f32), suffix, buffers)

          weighted_input_values = Array(Float32).new(width) do |index|
            ((index * 41 % 263) - 131).to_f32 * 0.0068359375_f32 +
              ((index * 5 % 23) - 11).to_f32 * 0.001220703125_f32
          end
          weighted_bank_values = Array(Float32).new(bank_elements) do |index|
            ((index * 31 % 337) - 168).to_f32 * 0.0048828125_f32 +
              ((index // width) - 1).to_f32 * 0.02734375_f32
          end
          weighted_query_values = Array(Float32).new(3 * width) do |index|
            feature = index // width
            column = index % width
            ((column * (19 + 6 * feature) + 13 * feature) % 191 - 95).to_f32 *
              (0.0087890625_f32 + feature.to_f32 * 0.00244140625_f32)
          end
          weight_values = Array(Array(Float32)).new(3) do |feature|
            Array(Float32).new(width) do |index|
              ((index * (13 + 2 * feature) + 7 * feature) % 47 - 23).to_f32 *
                (0.021484375_f32 + feature.to_f32 * 0.00390625_f32)
            end
          end
          weighted_input = support.guarded_float_view(weighted_input_values, suffix, buffers)
          weighted_reference_bank = support.guarded_float_view(weighted_bank_values, suffix, buffers)
          weighted_no_save_bank = support.guarded_float_view(weighted_bank_values, suffix, buffers)
          weighted_save_bank = support.guarded_float_view(weighted_bank_values, suffix, buffers)
          weighted_queries = support.guarded_float_view(weighted_query_values, suffix, buffers)
          weighted_unit = support.guarded_float_view(Array(Float32).new(width, 0.0_f32), suffix, buffers)
          weighted_weights = {
            support.guarded_float_view(weight_values[0], suffix, buffers),
            support.guarded_float_view(weight_values[1], suffix, buffers),
            support.guarded_float_view(weight_values[2], suffix, buffers),
          }
          weighted_reference_mixtures = support.guarded_float_view(Array(Float32).new(3 * width, 0.0_f32), suffix, buffers)
          weighted_reference_output = support.guarded_float_view(Array(Float32).new(3 * width, 123.0_f32), suffix, buffers)
          weighted_no_save_output = support.guarded_float_view(Array(Float32).new(3 * width, 321.0_f32), suffix, buffers)
          weighted_save_output = support.guarded_float_view(Array(Float32).new(3 * width, 321.0_f32), suffix, buffers)

          concat_input_before = support.raw_bits(concat_input, prefix + width + suffix)
          concat_queries_before = support.raw_bits(concat_queries, prefix + 3 * width + suffix)
          concat_messages_before = support.raw_bits(concat_messages, prefix + 3 * width + suffix)
          concat_bank_before = support.raw_bits(concat_reference_bank, bank_buffer_elements)
          weighted_input_before = support.raw_bits(weighted_input, prefix + width + suffix)
          weighted_queries_before = support.raw_bits(weighted_queries, prefix + 3 * width + suffix)
          weighted_bank_before = support.raw_bits(weighted_reference_bank, bank_buffer_elements)
          weighted_weights_before = weighted_weights.map do |weight|
            support.raw_bits(weight, prefix + width + suffix)
          end

          command = ML::Metal::CommandBuffer.new
          encoder = ML::Metal::ComputeEncoder.new(command)
          encoder_ended = false
          begin
            concat_eps = 0.00031_f32
            unit_eps = 0.00047_f32
            output_eps = 0.0021_f32

            # Independent normalization, raw depth mix, and three controller
            # feature pairs are the oracle for the fused scalar-query path.
            math.rms(encoder, concat_input, concat_unit, nil, width, 1_i32, concat_eps)
            math.depth_mix(encoder, concat_reference_bank, count, concat_unit, concat_queries,
              concat_reference_mixed, width, 1_i32, 1_i32)
            3.times do |feature|
              math.rms_concat(encoder, concat_reference_mixed, concat_messages.slice(feature * width),
                concat_reference_features.slice(feature * 2 * width), width, 1_i32, concat_eps)
            end
            math.depth_mix_concat(encoder, concat_input, concat_no_save_bank, count, concat_queries,
              concat_messages, concat_no_save_mixed, concat_no_save_features, width, concat_eps,
              1_i32, 3_i32)
            math.depth_mix_concat(encoder, concat_input, concat_save_bank, count, concat_queries,
              concat_messages, concat_save_mixed, concat_save_features, width, concat_eps,
              1_i32, 3_i32, save_to_bank: true)

            # Three independent query/weight dispatches are the oracle for
            # the fused weighted triplet; each query and scale is distinct.
            math.rms(encoder, weighted_input, weighted_unit, nil, width, 1_i32, unit_eps)
            3.times do |feature|
              offset = feature * width
              math.depth_mix(encoder, weighted_reference_bank, count, weighted_unit,
                weighted_queries.slice(offset), weighted_reference_mixtures.slice(offset),
                width, 1_i32, 1_i32)
              math.rms(encoder, weighted_reference_mixtures.slice(offset),
                weighted_reference_output.slice(offset), weighted_weights[feature],
                width, 1_i32, output_eps)
            end
            math.depth_mix_rms_triplet(encoder, weighted_input, weighted_no_save_bank, count,
              weighted_queries, weighted_weights, weighted_no_save_output, width, unit_eps, output_eps)
            math.depth_mix_rms_triplet(encoder, weighted_input, weighted_save_bank, count,
              weighted_queries, weighted_weights, weighted_save_output, width, unit_eps, output_eps,
              save_to_bank: true)

            encoder.end_encoding
            encoder_ended = true
            command.commit_and_wait

            concat_expected_mixed_bits = support.read(concat_reference_mixed, width).map(&.unsafe_as(UInt32))
            support.read(concat_no_save_mixed, width).map(&.unsafe_as(UInt32)).should eq(concat_expected_mixed_bits)
            support.read(concat_save_mixed, width).map(&.unsafe_as(UInt32)).should eq(concat_expected_mixed_bits)
            concat_expected_features_bits = support.read(concat_reference_features, 6 * width).map(&.unsafe_as(UInt32))
            support.read(concat_no_save_features, 6 * width).map(&.unsafe_as(UInt32)).should eq(concat_expected_features_bits)
            support.read(concat_save_features, 6 * width).map(&.unsafe_as(UInt32)).should eq(concat_expected_features_bits)

            weighted_expected_bits = support.read(weighted_reference_output, 3 * width).map(&.unsafe_as(UInt32))
            support.read(weighted_no_save_output, 3 * width).map(&.unsafe_as(UInt32)).should eq(weighted_expected_bits)
            support.read(weighted_save_output, 3 * width).map(&.unsafe_as(UInt32)).should eq(weighted_expected_bits)

            support.raw_bits(concat_input, prefix + width + suffix).should eq(concat_input_before)
            support.raw_bits(concat_queries, prefix + 3 * width + suffix).should eq(concat_queries_before)
            support.raw_bits(concat_messages, prefix + 3 * width + suffix).should eq(concat_messages_before)
            support.raw_bits(concat_reference_bank, bank_buffer_elements).should eq(concat_bank_before)
            support.raw_bits(concat_no_save_bank, bank_buffer_elements).should eq(concat_bank_before)
            concat_expected_save_bank = concat_bank_before.dup
            concat_unit_bits = support.read(concat_unit, width).map(&.unsafe_as(UInt32))
            concat_unit_bits.each_with_index do |bits, index|
              concat_expected_save_bank[prefix + count * width + index] = bits
            end
            support.raw_bits(concat_save_bank, bank_buffer_elements).should eq(concat_expected_save_bank)

            support.raw_bits(weighted_input, prefix + width + suffix).should eq(weighted_input_before)
            support.raw_bits(weighted_queries, prefix + 3 * width + suffix).should eq(weighted_queries_before)
            weighted_weights.each_with_index do |weight, index|
              support.raw_bits(weight, prefix + width + suffix).should eq(weighted_weights_before[index])
            end
            support.raw_bits(weighted_reference_bank, bank_buffer_elements).should eq(weighted_bank_before)
            support.raw_bits(weighted_no_save_bank, bank_buffer_elements).should eq(weighted_bank_before)
            weighted_expected_save_bank = weighted_bank_before.dup
            weighted_unit_bits = support.read(weighted_unit, width).map(&.unsafe_as(UInt32))
            weighted_unit_bits.each_with_index do |bits, index|
              weighted_expected_save_bank[prefix + count * width + index] = bits
            end
            support.raw_bits(weighted_save_bank, bank_buffer_elements).should eq(weighted_expected_save_bank)

            [
              {concat_unit, width}, {concat_reference_mixed, width},
              {concat_no_save_mixed, width}, {concat_save_mixed, width},
              {concat_reference_features, 6 * width}, {concat_no_save_features, 6 * width},
              {concat_save_features, 6 * width}, {weighted_unit, width},
              {weighted_reference_mixtures, 3 * width}, {weighted_reference_output, 3 * width},
              {weighted_no_save_output, 3 * width}, {weighted_save_output, 3 * width},
            ].each do |guarded|
              bits = support.raw_bits(guarded[0], prefix + guarded[1] + suffix)
              bits[0, prefix].should eq(prefix_bits)
              bits[prefix + guarded[1], suffix].should eq(suffix_bits)
            end
            {concat_reference_bank, concat_no_save_bank, concat_save_bank,
             weighted_reference_bank, weighted_no_save_bank, weighted_save_bank}.each do |bank|
              bits = support.raw_bits(bank, bank_buffer_elements)
              bits[0, prefix].should eq(prefix_bits)
              bits[prefix + bank_elements, suffix].should eq(suffix_bits)
            end
          ensure
            encoder.end_encoding unless encoder_ended
          end
        ensure
          buffers.each(&.release)
        end
      end
    end

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
