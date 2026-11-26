require "./pixelumm_block_probe"
require "./hy_nomas_probe_instrumentation"

lib PixelUMMProjectionFFI
  fun create = gs_pixelumm_projection_create(device : Void*, input : Void*, weight : Void*, output : Void*,
                                             rows : Int32, input_width : Int32, output_width : Int32, mode : Int32, product_type : UInt32*,
                                             error : UInt8*, capacity : LibC::SizeT) : Void*
  fun encode = gs_pixelumm_projection_encode(plan : Void*, command : Void*, error : UInt8*, capacity : LibC::SizeT) : Int32
  fun release = gs_pixelumm_projection_release(plan : Void*) : Void
end

# Projection-only diagnostics. No decoder path selects these candidates.
module PixelUMMFFNProjectionProbe
  extend self
  SOURCE = {{ read_file("#{__DIR__}/pixelumm_ffn_dense.metal") }}
  ARMS   = ["mps.bf16", "mps.f32", "mps.f16-rounded-f32", "native.f16-mma"]
  alias Metric = NamedTuple(failures: UInt32, different_words: UInt32, nonfinite: UInt32, elements: UInt32)

  def projection(name : String)
    case name
    when "gate", "up" then {input: "post_attn_norm", output: "#{name}_proj", weight: "mlp.#{name}_proj.weight", k: 4096, n: 12288}
    when "down"       then {input: "product", output: "down_proj", weight: "mlp.down_proj.weight", k: 12288, n: 4096}
    else                   raise ArgumentError.new("projection must be gate, up or down")
    end
  end

  def median(values : Array(Float64)) : Float64
    raise ArgumentError.new("empty timing set") if values.empty?
    sorted = values.sort
    (sorted[(sorted.size - 1) // 2] + sorted[sorted.size // 2]) / 2
  end

  def compare(expected : Bytes, actual : Bytes) : Metric
    raise ArgumentError.new("invalid BF16 comparison extent") unless expected.size == actual.size && expected.size > 0 && expected.size.even?
    failures = 0_u32
    different = 0_u32
    nonfinite = 0_u32
    (expected.size // 2).times do |i|
      a_word = IO::ByteFormat::LittleEndian.decode(UInt16, expected[i * 2, 2])
      b_word = IO::ByteFormat::LittleEndian.decode(UInt16, actual[i * 2, 2])
      a = (a_word.to_u32 << 16).unsafe_as(Float32)
      b = (b_word.to_u32 << 16).unsafe_as(Float32)
      different += 1 if a_word != b_word
      if !a.finite? || !b.finite?
        nonfinite += 1
        failures += 1
      elsif (a - b).abs > 2e-5_f32 + 2e-5_f32 * a.abs
        failures += 1
      end
    end
    {failures: failures, different_words: different, nonfinite: nonfinite, elements: (expected.size // 2).to_u32}
  end

  {% unless flag?(:cpu_only) %}
  class Owner
    getter named_bytes : Int64
    getter product_types = {} of String => UInt32
    getter resources = {} of String => Int64
    getter input_record : PixelUMMBlockProbe::Record
    getter weight_record : PixelUMMBlockProbe::Record
    getter expected_record : PixelUMMBlockProbe::Record
    @buffers = {} of String => ML::MetalBuffer
    @plans = {} of String => Pointer(Void)
    @pipelines = {} of String => ML::Metal::ComputePipeline
    @k : Int32
    @n : Int32
    @released = false

    def initialize(@fixture : PixelUMMBlockProbe::Fixture, name : String, @rows : Int32, @samples : Int32,
                   max_named_bytes : Int64)
      p = PixelUMMFFNProjectionProbe.projection(name)
      @k, @n = p[:k], p[:n]
      raise ArgumentError.new("projection probe requires saved J2048 fixture and J2048/J21120") unless @fixture.grid == [2, 32, 32] && {2048, 21120}.includes?(@rows)
      @input_record = @fixture.captures[p[:input]]
      @weight_record = @fixture.weights[p[:weight]]
      @expected_record = @fixture.captures[p[:output]]
      raise ArgumentError.new("projection records have unexpected geometry") unless @input_record.shape == [1, 2048, @k] && @weight_record.shape == [@n, @k] && @expected_record.shape == [1, 2048, @n] && {@input_record, @weight_record, @expected_record}.all? { |r| r.dtype == "BF16" }
      @resources = {"saved.input" => @input_record.nbytes, "weight" => @weight_record.nbytes,
                    "expected" => @expected_record.nbytes, "input" => @rows.to_i64 * @k * 2,
                    "metrics" => (@samples + 1).to_i64 * 4 * 4, "roundtrip" => 3_i64 * 3 * 4,
                    "roundtrip.positive" => 8_i64}
      ARMS.each { |arm| @resources[arm] = @rows.to_i64 * @n * 2 }
      @named_bytes = @resources.values.sum(0_i64)
      raise ArgumentError.new("projection named memory exceeds cap before allocation") unless @named_bytes <= max_named_bytes
      begin
        raise "Metal unavailable" unless ML::Metal::Device.init!
        @resources.each { |key, bytes| @buffers[key] = ML::MetalBuffer.new(bytes) }
        {"saved.input" => @input_record, "weight" => @weight_record, "expected" => @expected_record}.each do |key, record|
          bytes = @fixture.read(record)
          buffer(key).write_bytes(bytes.to_unsafe, bytes.size)
        end
        {"metrics", "roundtrip"}.each do |key|
          bytes = Bytes.new(buffer(key).size.to_i32, 0_u8)
          buffer(key).write_bytes(bytes.to_unsafe, bytes.size)
        end
        # 1, negative zero, 2^-30 and 65536: exactly one underflow and one
        # overflow under BF16->F16. This qualifies the operand counters.
        known = [0x3f80_u16, 0x8000_u16, 0x3080_u16, 0x4780_u16]
        buffer("roundtrip.positive").write_bytes(known.to_unsafe.as(UInt8*), 8)
        {"pixelumm_ffn_bf16_b64", "pixelumm_ffn_repeat_bf16", "pixelumm_ffn_f16_roundtrip", "pixelumm_ffn_compare_bf16"}.each do |kernel|
          @pipelines[kernel] = ML::Metal::ComputePipeline.new(kernel, SOURCE)
        end
        ARMS[0, 3].each_with_index do |arm, mode|
          error = Bytes.new(1024, 0_u8)
          product_type = 0_u32
          plan = PixelUMMProjectionFFI.create(ML::Metal::Device.instance.device_handle,
            buffer("input").handle, buffer("weight").handle, buffer(arm).handle,
            @rows, @k, @n, mode, pointerof(product_type), error.to_unsafe, error.size)
          raise "projection creation #{arm}: #{String.new(error.to_unsafe)}" if plan.null?
          @plans[arm] = plan
          @product_types[arm] = product_type
        end
      rescue ex
        release
        raise ex
      end
    end

    def buffer(name : String) : ML::MetalBuffer
      @buffers[name]
    end

    def setup(chain : ML::Metal::MPSCommandChain) : Nil
      e = chain.encoder
      e.set_pipeline(@pipelines["pixelumm_ffn_repeat_bf16"])
        .set_buffer(buffer("saved.input"), 0).set_buffer(buffer("input"), 1, ML::Metal::BufferAccess::Write)
        .set_value(2048_u32, 2).set_value(@rows.to_u32, 3).set_value(@k.to_u32, 4)
        .dispatch({@rows * @k, 1, 1}, {256, 1, 1})
      {"input", "weight", "roundtrip.positive"}.each_with_index do |name, i|
        e.set_pipeline(@pipelines["pixelumm_ffn_f16_roundtrip"])
          .set_buffer(buffer(name), 0).set_buffer(buffer("roundtrip"), 1, ML::Metal::BufferAccess::ReadWrite, i.to_i64 * 12)
          .set_value((buffer(name).size // 2).to_u32, 2)
          .dispatch_threadgroups({((buffer(name).size // 2 + 255) // 256).to_i32, 1, 1}, {256, 1, 1})
      end
    end

    def poison(chain : ML::Metal::MPSCommandChain, arm : String) : Nil
      chain.pause!
      blit = MetalDispatchFFI.create_blit_encoder(chain.root_handle)
      raise "projection poison blit unavailable" if blit.null?
      MetalDispatchFFI.blit_fill_buffer(blit, buffer(arm).handle, 255_u8, 0_i64, buffer(arm).size)
      MetalDispatchFFI.blit_end_encoding(blit)
      chain.resume!
    end

    def encode(chain : ML::Metal::MPSCommandChain, arm : String) : Nil
      if arm == "native.f16-mma"
        chain.encoder.set_pipeline(@pipelines["pixelumm_ffn_bf16_b64"])
          .set_buffer(buffer("weight"), 0).set_buffer(buffer("input"), 1)
          .set_buffer(buffer(arm), 2, ML::Metal::BufferAccess::Write)
          .set_value(@k.to_u32, 3).set_value(@n.to_u32, 4).set_value(@rows.to_u32, 5)
          .set_threadgroup_memory(16384, 0)
          .dispatch_threadgroups({@rows // 64, @n // 64, 1}, {256, 1, 1})
      else
        chain.pause!
        error = Bytes.new(1024, 0_u8)
        status = PixelUMMProjectionFFI.encode(@plans[arm], chain.handle, error.to_unsafe, error.size)
        raise "projection encode #{arm}: #{status}: #{String.new(error.to_unsafe)}" unless status == 0
        chain.resume!
      end
    end

    def check(chain : ML::Metal::MPSCommandChain, arm : String, sample : Int32, *, inject = false) : Nil
      chain.encoder.set_pipeline(@pipelines["pixelumm_ffn_compare_bf16"])
        .set_buffer(buffer(arm), 0).set_buffer(buffer("expected"), 1)
        .set_buffer(buffer("metrics"), 2, ML::Metal::BufferAccess::ReadWrite, sample.to_i64 * 16)
        .set_value((@rows * @n).to_u32, 3).set_value((2048 * @n).to_u32, 4).set_value(inject ? 1_u32 : 0_u32, 5)
        .dispatch_threadgroups({(@rows * @n + 255) // 256, 1, 1}, {256, 1, 1})
    end

    def metrics(chain : ML::Metal::MPSCommandChain) : Array(Metric)
      terminal!(chain)
      words = Array(UInt32).new((@samples + 1) * 4, 0_u32)
      buffer("metrics").read_bytes(words.to_unsafe.as(UInt8*), words.size * 4)
      (@samples + 1).times.map do |i|
        {failures: words[i * 4], different_words: words[i * 4 + 1], nonfinite: words[i * 4 + 2], elements: words[i * 4 + 3]}
      end.to_a
    end

    def roundtrip(chain : ML::Metal::MPSCommandChain) : Hash(String, Array(UInt32))
      terminal!(chain)
      words = Array(UInt32).new(9, 0_u32)
      buffer("roundtrip").read_bytes(words.to_unsafe.as(UInt8*), words.size * 4)
      {"input" => words[0, 3], "weight" => words[3, 3], "positive" => words[6, 3]}
    end

    def prefix_comparisons(chain : ML::Metal::MPSCommandChain)
      terminal!(chain)
      expected = @fixture.read(@expected_record)
      ARMS.map do |arm|
        actual = Bytes.new(expected.size)
        buffer(arm).read_bytes(actual.to_unsafe, actual.size)
        {arm: arm, metric: PixelUMMFFNProjectionProbe.compare(expected, actual),
         first_2048_rows_sha256: Digest::SHA256.hexdigest(actual)}
      end
    end

    def release : Nil
      return if @released
      @plans.each_value { |plan| PixelUMMProjectionFFI.release(plan) }
      @buffers.each_value(&.release)
      @released = true
    end

    private def terminal!(chain : ML::Metal::MPSCommandChain) : Nil
      raise "projection diagnostics require sole terminal wait" unless chain.completed? && chain.wait_count == 1
    end
  end
  {% end %}

  def next_root(chain : ML::Metal::MPSCommandChain) : Nil
    before = chain.observed_root_count
    chain.pause!
    chain.continue_async!
    chain.resume!
    raise "diagnostic continuation changed wait/count" unless chain.observed_root_count == before + 1 && chain.wait_count == 0
  end
end
