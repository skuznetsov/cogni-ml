require "./hy_nomas_reference_block0_setup"
require "./hy_nomas_reference_history_probe"

module HyNomasReferenceHistoryCapture
  extend self

  record CaptureRecord, name : String, dtype : String, shape : Array(Int32),
    nbytes : Int32, sha256 : String

  def sources(cache_plan : ML::Video::HyNomasReferenceKVCache::Plan,
              key_read : ML::MetalBuffer, value_read : ML::MetalBuffer,
              valid_text_tokens : Int32) : Array(HyNomasReferenceBlock0Probe::CaptureTensor)
    entries = HyNomasReferenceHistoryProbe.header_entries(valid_text_tokens)
    buffers = {
      "history.k_bf16"     => cache_plan.key_storage,
      "history.v_bf16"     => cache_plan.value_storage,
      "history.k_read_f32" => key_read,
      "history.v_read_f32" => value_read,
    }
    entries.map do |entry|
      buffer = buffers[entry.name]
      HyNomasReferenceBlock0Probe::CaptureTensor.new(entry.name, buffer,
        entry.shape, entry.byte_count, "first_t0_history_#{entry.dtype.downcase}")
    end
  end

  def write_safetensors!(artifact_path : String,
                         sources : Array(HyNomasReferenceBlock0Probe::CaptureTensor),
                         valid_text_tokens : Int32) : Tuple(String, Array(CaptureRecord), Int64)
    entries = HyNomasReferenceHistoryProbe.header_entries(valid_text_tokens)
    unless sources.map(&.name) == HyNomasReferenceHistoryProbe::TENSOR_NAMES
      raise ArgumentError.new("reference history capture inventory/order mismatch")
    end
    total_bytes = sources.zip(entries).sum(0_i64) do |source, entry|
      unless source.name == entry.name && source.shape == entry.shape && source.byte_count == entry.byte_count
        raise ArgumentError.new("reference history capture shape/extent mismatch: #{source.name}")
      end
      unless source.buffer.valid? && source.buffer.storage_mode.shared? &&
             source.buffer.size == entry.byte_count
        raise ArgumentError.new("reference history capture must use an exact-size shared buffer: #{source.name}")
      end
      entry.byte_count.to_i64
    end
    header = HyNomasReferenceHistoryProbe.safetensors_header(entries)
    header_length = Bytes.new(sizeof(UInt64))
    IO::ByteFormat::LittleEndian.encode(header.bytesize.to_u64, header_length)
    total_digest = Digest::SHA256.new
    total_digest.update(header_length)
    total_digest.update(header.to_slice)
    records = [] of CaptureRecord
    offset = 0_i64
    payload_io = HyNomasReferenceCacheProbe.open_fresh_output!(artifact_path)
    begin
      payload_io.write(header_length)
      payload_io.write(header.to_slice)
      sources.zip(entries).each do |source, entry|
        view = Bytes.new(entry.byte_count)
        source.buffer.read_bytes(view.to_unsafe, view.size)
        digest = Digest::SHA256.hexdigest(view)
        total_digest.update(view)
        payload_io.write(view)
        records << CaptureRecord.new(entry.name, entry.dtype, entry.shape,
          entry.byte_count, digest)
        offset += entry.byte_count
      end
      payload_io.flush
      payload_io.fsync
    ensure
      payload_io.close
    end
    unless offset == total_bytes
      raise "reference history payload size mismatch: wrote #{offset}, expected #{total_bytes}"
    end
    expected_file_bytes = HyNomasReferenceHistoryProbe.safetensors_file_bytes(header, total_bytes)
    actual_file_bytes = File.size(artifact_path)
    unless actual_file_bytes == expected_file_bytes
      raise "reference history safetensors file size mismatch: #{actual_file_bytes} != #{expected_file_bytes}"
    end
    {total_digest.hexfinal, records, actual_file_bytes}
  end
end
