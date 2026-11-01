require "json"
require "option_parser"
require "digest/sha256"
require "../../src/ml/video/hy_nomas_continuation_history"
require "./hy_nomas_probe_instrumentation"

{% unless flag?(:hy_nomas_history_probe) %}
  {% raise "Compile this probe with -Dhy_nomas_history_probe" %}
{% end %}

module HyNomasContinuationHistoryProbe
  LAYERS              = 54_i32
  SEGMENTS            =  3_i32
  VIDEO_TOKENS        =  2_i32
  VALID_TEXT_TOKENS   =  1_i32
  STORAGE_TEXT_TOKENS =  2_i32
  QUERY_ROWS          = VIDEO_TOKENS + VALID_TEXT_TOKENS
  STORAGE_ROWS        = VIDEO_TOKENS + STORAGE_TEXT_TOKENS
  HEADS               =      2_i32
  HEAD_DIM            =      2_i32
  FORWARD_COUNT       =     14_i32
  ATOL                =     2.0e-5
  SENTINEL            = -19.25_f32
  COPY_KERNEL         = "hn_continuation_probe_copy_f32"
  COPY_SOURCE         = ML::Video::MetalOps::SOURCE + "\n" + {{ read_file("#{__DIR__}/hy_nomas_continuation_history_probe.metal") }}

  class CopyPlan
    @pipeline : ML::Metal::ComputePipeline

    def initialize
      @pipeline = ML::Metal::PipelineCache.get(COPY_KERNEL) do
        ML::Metal::ComputePipeline.new(COPY_KERNEL, COPY_SOURCE)
      end
    end

    def encode(encoder : ML::Metal::ComputeEncoder,
               source : ML::MetalBuffer, destination : ML::MetalBuffer,
               source_offset : Int32, destination_offset : Int32,
               count : Int32) : Nil
      unless source_offset >= 0 && destination_offset >= 0 && count > 0 &&
             source_offset.to_i64 + count <= Int32::MAX &&
             destination_offset.to_i64 + count <= Int32::MAX
        raise ArgumentError.new("Hy-Nomas continuation probe copy extent exceeds the native limit")
      end
      ML::Video::MetalOps.check(source, source_offset.to_i64 + count)
      ML::Video::MetalOps.check(destination, destination_offset.to_i64 + count)
      unless source.storage_mode.shared? && destination.storage_mode.shared?
        raise ArgumentError.new("Hy-Nomas continuation probe copies require shared F32 buffers")
      end
      ML::Video::MetalOps.distinct(source, destination)
      encoder.set_pipeline(@pipeline)
        .set_buffer(source, 0)
        .set_buffer(destination, 1, ML::Metal::BufferAccess::Write)
        .set_value(source_offset.to_u32, 2)
        .set_value(destination_offset.to_u32, 3)
        .set_value(count.to_u32, 4)
        .dispatch_1d(count)
    end
  end

  record InputBanks,
    query_buffer : ML::MetalBuffer,
    key_buffer : ML::MetalBuffer,
    value_buffer : ML::MetalBuffer,
    query_values : Array(Float32),
    key_values : Array(Float32),
    value_values : Array(Float32)

  def self.storage_elements : Int32
    STORAGE_ROWS * HEADS * HEAD_DIM
  end

  def self.query_elements : Int32
    QUERY_ROWS * HEADS * HEAD_DIM
  end

  def self.cache_snapshot_elements : Int32
    LAYERS * ((SEGMENTS - 1) * VIDEO_TOKENS + VALID_TEXT_TOKENS) * HEADS * HEAD_DIM
  end

  def self.forward_elements : Int32
    LAYERS * query_elements
  end

  def self.input_offset(forward : Int32, layer : Int32) : Int32
    (forward * LAYERS + layer) * storage_elements
  end

  def self.index(forward : Int32, layer : Int32, row : Int32,
                 head : Int32, feature : Int32) : Int32
    ((((forward * LAYERS + layer) * STORAGE_ROWS + row) * HEADS + head) * HEAD_DIM + feature)
  end

  def self.build_input_banks : InputBanks
    count = FORWARD_COUNT * LAYERS * storage_elements
    query_values = Array(Float32).new(count, 0.0_f32)
    key_values = Array(Float32).new(count, 0.0_f32)
    value_values = Array(Float32).new(count, 0.0_f32)
    FORWARD_COUNT.times do |forward|
      LAYERS.times do |layer|
        STORAGE_ROWS.times do |row|
          HEADS.times do |head|
            HEAD_DIM.times do |feature|
              offset = index(forward, layer, row, head, feature)
              if row >= QUERY_ROWS
                query_values[offset] = 100.0_f32 + layer + feature
                key_values[offset] = 200.0_f32 + forward + layer
                value_values[offset] = 300.0_f32 + forward + feature
              else
                query_values[offset] = (0.025 + forward * 0.004 + layer * 0.0007 +
                                        row * 0.003 + head * 0.002 + feature * 0.001).to_f32
                key_values[offset] = (-0.04 + forward * 0.003 + layer * 0.0002 +
                                      row * 0.002 + head * 0.001 + feature * 0.0015).to_f32
                value_values[offset] = (-0.06 + forward * 0.005 + layer * 0.0003 +
                                        row * 0.0035 + head * 0.001 + feature * 0.002).to_f32
              end
            end
          end
        end
      end
    end
    InputBanks.new(
      ML::MetalBuffer.from_array(query_values),
      ML::MetalBuffer.from_array(key_values),
      ML::MetalBuffer.from_array(value_values),
      query_values, key_values, value_values)
  end

  def self.input_bank_sha256(banks : InputBanks) : String
    digest = Digest::SHA256.new
    {banks.query_values, banks.key_values, banks.value_values}.each do |values|
      digest.update(Bytes.new(values.to_unsafe.as(UInt8*), values.size * sizeof(Float32)))
    end
    digest.hexfinal
  end

  def self.copy_current_inputs!(plan : CopyPlan, chain : ML::Metal::MPSCommandChain,
                                banks : InputBanks, forward : Int32, layer : Int32,
                                query : ML::MetalBuffer, key : ML::MetalBuffer,
                                value : ML::MetalBuffer) : Nil
    source_offset = input_offset(forward, layer)
    encoder = chain.encoder
    plan.encode(encoder, banks.query_buffer, query, source_offset, 0, storage_elements)
    plan.encode(encoder, banks.key_buffer, key, source_offset, 0, storage_elements)
    plan.encode(encoder, banks.value_buffer, value, source_offset, 0, storage_elements)
  end

  def self.run_forward!(provider : ML::Video::HyNomasContinuationHistory,
                        plan : CopyPlan, chain : ML::Metal::MPSCommandChain,
                        banks : InputBanks, forward : Int32,
                        query : ML::MetalBuffer, key : ML::MetalBuffer,
                        value : ML::MetalBuffer, output : ML::MetalBuffer,
                        capture_destination : ML::MetalBuffer? = nil,
                        capture_offset : Int32 = 0) : Nil
    LAYERS.times do |layer|
      copy_current_inputs!(plan, chain, banks, forward, layer, query, key, value)
      provider.encode(chain, layer)
      if destination = capture_destination
        plan.encode(chain.encoder, output, destination, 0,
          capture_offset + layer * query_elements, query_elements)
      end
    end
  end

  # Independent F64 reference over an explicitly ordered row list. Current
  # text is excluded only on later denoise calls; every writer attends it.
  def self.expected_output(banks : InputBanks, layer : Int32,
                           query_forward : Int32, segment : Int32, writer : Bool) : Array(Float32)
    sources = [] of Tuple(Int32, Int32)
    segment.times do |previous|
      rows = previous == 0 ? QUERY_ROWS : VIDEO_TOKENS
      rows.times { |row| sources << {previous * 5 + 4, row} }
    end
    current_rows = segment == 0 || writer ? QUERY_ROWS : VIDEO_TOKENS
    current_rows.times { |row| sources << {query_forward, row} }
    result = Array(Float32).new(query_elements, 0.0_f32)
    scale = 1.0 / Math.sqrt(HEAD_DIM.to_f64)
    QUERY_ROWS.times do |query_row|
      HEADS.times do |head|
        scores = sources.map do |source|
          source_forward, source_row = source
          dot = 0.0_f64
          HEAD_DIM.times do |feature|
            dot += banks.query_values[index(query_forward, layer, query_row, head, feature)].to_f64 *
                   banks.key_values[index(source_forward, layer, source_row, head, feature)].to_f64
          end
          dot * scale
        end
        maximum = scores.max
        weights = scores.map { |score| Math.exp(score - maximum) }
        denominator = weights.sum
        HEAD_DIM.times do |feature|
          total = 0.0_f64
          sources.each_with_index do |source, i|
            total += (weights[i] / denominator) *
                     banks.value_values[index(source[0], layer, source[1], head, feature)].to_f64
          end
          result[(query_row * HEADS + head) * HEAD_DIM + feature] = total.to_f32
        end
      end
    end
    result
  end

  def self.write_fresh_output(path : String, & : IO::FileDescriptor ->) : Nil
    fd = LibC.open(path, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
    raise IO::Error.new("probe output must be fresh and writable: #{path}") if fd < 0
    io = IO::FileDescriptor.new(fd)
    begin
      yield io
    ensure
      io.close
    end
  end

  def self.write_artifact(directory : String, name : String, values : Array(Float32), shape : Array(Int32))
    path = File.join(directory, name + ".f32le")
    write_fresh_output(path) { |io| io.write(Bytes.new(values.to_unsafe.as(UInt8*), values.size * sizeof(Float32))) }
    {path: File.basename(path), sha256: Digest::SHA256.hexdigest(File.read(path)), shape: shape}
  end

  def self.max_abs_error(actual : Array(Float32), expected : Array(Float32)) : Float64
    raise ArgumentError.new("probe comparison vector length mismatch") unless actual.size == expected.size
    error = 0.0_f64
    actual.each_with_index do |value, index|
      reference = expected[index]
      raise "non-finite continuation output" unless value.finite? && reference.finite?
      error = Math.max(error, (value.to_f64 - reference.to_f64).abs)
    end
    error
  end
end
