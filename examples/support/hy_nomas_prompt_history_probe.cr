require "digest/sha256"
require "json"
require "option_parser"
require "./hy_nomas_continuation_history_probe"

{% unless flag?(:hy_nomas_history_probe) %}
  {% raise "Compile this probe with -Dhy_nomas_history_probe" %}
{% end %}

module HyNomasPromptHistoryProbe
  LAYERS              = 54_i32
  SEGMENTS            =  3_i32
  VIDEO_TOKENS        =  2_i32
  VALID_TEXT_TOKENS   =  2_i32
  STORAGE_TEXT_TOKENS =  2_i32
  SEGMENT_TEXT_TOKENS = [1_i32, 2_i32, 1_i32]
  SEGMENT_CACHE_TEXT  = [true, true, true]
  QUERY_ROWS          = VIDEO_TOKENS + VALID_TEXT_TOKENS
  STORAGE_ROWS        = VIDEO_TOKENS + STORAGE_TEXT_TOKENS
  HEADS               = 2_i32
  HEAD_DIM            = 2_i32
  DENOISE_STEPS       = 4_i32
  WRITER_COUNT        = SEGMENTS - 1
  FORWARD_COUNT       = SEGMENTS * DENOISE_STEPS + WRITER_COUNT
  CACHE_ROWS          = (VIDEO_TOKENS + SEGMENT_TEXT_TOKENS[0]) +
                        (VIDEO_TOKENS + SEGMENT_TEXT_TOKENS[1])
  ATOL     =     2.0e-5
  SENTINEL = -19.25_f32
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
    LAYERS * CACHE_ROWS * HEADS * HEAD_DIM
  end

  def self.forward_elements : Int32
    LAYERS * query_elements
  end

  def self.forward_index(segment : Int32, denoise : Int32) : Int32
    segment * (DENOISE_STEPS + 1) + denoise
  end

  def self.writer_index(segment : Int32) : Int32
    forward_index(segment, DENOISE_STEPS)
  end

  def self.index(forward : Int32, layer : Int32, row : Int32,
                 head : Int32, feature : Int32) : Int32
    ((((forward * LAYERS + layer) * STORAGE_ROWS + row) * HEADS + head) * HEAD_DIM + feature)
  end

  def self.active_rows(segment : Int32) : Int32
    VIDEO_TOKENS + SEGMENT_TEXT_TOKENS[segment]
  end

  def self.build_input_banks : InputBanks
    count = FORWARD_COUNT * LAYERS * storage_elements
    query_values = Array(Float32).new(count, 0.0_f32)
    key_values = Array(Float32).new(count, 0.0_f32)
    value_values = Array(Float32).new(count, 0.0_f32)
    FORWARD_COUNT.times do |forward|
      segment = forward < 5 ? 0 : (forward < 10 ? 1 : 2)
      active = active_rows(segment)
      LAYERS.times do |layer|
        STORAGE_ROWS.times do |row|
          HEADS.times do |head|
            HEAD_DIM.times do |feature|
              offset = index(forward, layer, row, head, feature)
              if row >= active
                # Poison padded rows so an over-wide K/V copy is observable.
                query_values[offset] = 101.0_f32 + layer + feature
                key_values[offset] = 211.0_f32 + forward + layer
                value_values[offset] = 307.0_f32 + forward + feature
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

  def self.copy_current_inputs!(plan : HyNomasContinuationHistoryProbe::CopyPlan,
                                chain : ML::Metal::MPSCommandChain,
                                banks : InputBanks, forward : Int32, layer : Int32,
                                query : ML::MetalBuffer, key : ML::MetalBuffer,
                                value : ML::MetalBuffer) : Nil
    source_offset = (forward * LAYERS + layer) * storage_elements
    encoder = chain.encoder
    plan.encode(encoder, banks.query_buffer, query, source_offset, 0, storage_elements)
    plan.encode(encoder, banks.key_buffer, key, source_offset, 0, storage_elements)
    plan.encode(encoder, banks.value_buffer, value, source_offset, 0, storage_elements)
  end

  def self.run_forward!(provider : ML::Video::HyNomasContinuationHistory,
                        plan : HyNomasContinuationHistoryProbe::CopyPlan,
                        chain : ML::Metal::MPSCommandChain,
                        banks : InputBanks, forward : Int32,
                        query : ML::MetalBuffer, key : ML::MetalBuffer,
                        value : ML::MetalBuffer, output : ML::MetalBuffer,
                        capture_destination : ML::MetalBuffer,
                        capture_offset : Int32) : Nil
    LAYERS.times do |layer|
      copy_current_inputs!(plan, chain, banks, forward, layer, query, key, value)
      provider.encode(chain, layer)
      plan.encode(chain.encoder, output, capture_destination, 0,
        capture_offset + layer * query_elements, query_elements)
    end
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

  def self.write_artifact(directory : String, name : String,
                          values : Array(Float32), shape : Array(Int32))
    path = File.join(directory, name + ".f32le")
    write_fresh_output(path) do |io|
      io.write(Bytes.new(values.to_unsafe.as(UInt8*), values.size * sizeof(Float32)))
    end
    {path: File.basename(path), sha256: Digest::SHA256.hexdigest(File.read(path)), shape: shape}
  end

  def self.max_abs_error(actual : Array(Float32), expected : Array(Float32)) : Float64
    raise ArgumentError.new("probe comparison vector length mismatch") unless actual.size == expected.size
    error = 0.0_f64
    actual.each_with_index do |value, index|
      reference = expected[index]
      raise "non-finite prompt-history output" unless value.finite? && reference.finite?
      error = Math.max(error, (value.to_f64 - reference.to_f64).abs)
    end
    error
  end
end
