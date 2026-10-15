require "spec"
require "../src/ml/audio/drama3/codec_encoder_resident"

include ML::Audio::Drama3

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private def drama3_encoder_attention_fixture_values(rows : Int32, heads : Int32,
                                                      seed : Int32) : Array(Float32)
    Array(Float32).new(rows * heads * 64) do |index|
      row = index // (heads * 64)
      head = (index // 64) % heads
      channel = index % 64
      mixed = (row * 17 + head * 23 + channel * 13 + seed * 29 + row * channel * 3) % 101
      (mixed - 50).to_f32 * 0.005_f32
    end
  end

  private def drama3_encoder_attention_oracle(q : Array(Float32), k : Array(Float32),
                                              v : Array(Float32), rows : Int32,
                                              heads : Int32, window : Int32) : Array(Float32)
    channels = heads * 64
    scale = 1.0_f64 / Math.sqrt(64.0_f64)
    output = Array(Float32).new(rows * channels, 0_f32)

    rows.times do |row|
      first_key = row - window + 1
      first_key = 0 if first_key < 0
      heads.times do |head|
        scores = Array(Float64).new(row - first_key + 1, 0_f64)
        scores.size.times do |offset|
          key_row = first_key + offset
          dot = 0_f64
          64.times do |channel|
            q_index = row * channels + head * 64 + channel
            k_index = key_row * channels + head * 64 + channel
            dot += q[q_index].to_f64 * k[k_index].to_f64
          end
          scores[offset] = dot * scale
        end

        maximum = -Float64::MAX
        scores.each { |score| maximum = score if score > maximum }
        weights = scores.map { |score| Math.exp(score - maximum) }
        denominator = weights.sum
        64.times do |channel|
          weighted_value = 0_f64
          weights.each_with_index do |weight, offset|
            key_row = first_key + offset
            v_index = key_row * channels + head * 64 + channel
            weighted_value += weight * v[v_index].to_f64
          end
          output[row * channels + head * 64 + channel] = (weighted_value / denominator).to_f32
        end
      end
    end
    output
  end

  describe "DRAMA-3 resident encoder attention tails" do
    it "matches an independent F64 causal-window oracle across query and key tile edges" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        # Covers empty/partial 8-row query fragments, BK32 edges, a multi-BK
        # sequence whose first complete key tile is masked, and every window
        # regime used by the reference encoder.
        cases = [
          {rows: 1, window: 512},
          {rows: 7, window: 1},
          {rows: 8, window: 3},
          {rows: 9, window: 512},
          {rows: 31, window: 3},
          {rows: 32, window: 32},
          {rows: 33, window: 3},
          {rows: 65, window: 32},
        ]

        cases.each do |shape|
          rows = shape[:rows]
          heads = 2
          window = shape[:window]
          count = rows * heads * 64
          q = drama3_encoder_attention_fixture_values(rows, heads, 1)
          k = drama3_encoder_attention_fixture_values(rows, heads, 2)
          v = drama3_encoder_attention_fixture_values(rows, heads, 3)
          expected = drama3_encoder_attention_oracle(q, k, v, rows, heads, window)

          context = Resident::Context.new
          q_buffer = context.upload(q)
          k_buffer = context.upload(k)
          v_buffer = context.upload(v)
          output = context.alloc(count)
          context.readbacks.empty?.should be_true

          scale = (1.0_f32 / Math.sqrt(64.0_f32).to_f32).to_f32
          tiles = ((rows + 7) // 8) * heads
          context.dispatch("dr3e_attention_32", CodecEncoderResident::SOURCE,
            [q_buffer, k_buffer, v_buffer, output],
            [rows.to_u64, heads.to_u64, window.to_u64, scale.unsafe_as(UInt32).to_u64],
            tiles * 32, 32)

          context.finish
          actual = context.read(output, "terminal")
          actual.size.should eq(expected.size)
          actual.all?(&.finite?).should be_true
          max_error = 0_f32
          actual.zip(expected).each do |got, want|
            error = (got - want).abs
            max_error = error if error > max_error
          end
          max_error.should be < 2.0e-5_f32
          actual.should eq(v) if window == 1
          context.readbacks.should eq({"terminal" => (count * 4).to_i64})
        end
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
      end
    end
  end
{% end %}
