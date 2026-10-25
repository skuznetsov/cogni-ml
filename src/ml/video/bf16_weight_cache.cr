require "digest/sha256"
require "json"
require "./checkpoint"

module ML::Video::BF16WeightCache
  extend self

  SCHEMA           = "ml-video-bf16-weight-cache"
  SCHEMA_VERSION   = "1"
  CONVERTER        = "ml-video-bf16-resident-byte-copy-v1"
  TEMP_PREFIX      = ".ml-video-bf16-cache-"
  TEMP_SUFFIX      = ".partial"
  HASH_CHUNK_BYTES = 1024 * 1024

  private record ResidentTensor, source : String, shape : Array(Int64),
    n_elements : Int64, target : String, buffer : ML::MetalBuffer

  # Authenticate an existing cache before Metal initialization. The caller
  # still checks that its names and shapes match the intended WeightPlan.
  def authenticate(path : String, expected_sha256 : String) : CheckpointFile
    validate_sha256!(expected_sha256)
    before = regular_file_info!(path)
    actual_sha256 = sha256_file(path)
    after_hash = regular_file_info!(path)
    unless before.same_file?(after_hash) && before.size == after_hash.size &&
           before.modification_time == after_hash.modification_time
      raise ArgumentError.new("BF16 cache changed while hashing")
    end
    unless actual_sha256 == expected_sha256
      raise ArgumentError.new("BF16 cache SHA-256 mismatch")
    end

    checkpoint = CheckpointFile.new(path)
    begin
      checkpoint.unchanged!
      after_parse = regular_file_info!(path)
      unless before.same_file?(after_parse) && before.size == after_parse.size &&
             before.modification_time == after_parse.modification_time
        raise ArgumentError.new("BF16 cache changed while parsing")
      end
      checkpoint.tensors.each_value do |tensor|
        unless tensor.dtype.bf16?
          raise ArgumentError.new("BF16 cache contains a non-BF16 tensor #{tensor.name}")
        end
      end
      checkpoint.unchanged!
      checkpoint
    rescue ex
      checkpoint.close
      raise ex
    end
  end

  # Persist BF16 resident buffers only after the caller's one terminal native
  # wait. Source/prepared/destination/chain roots remain caller-owned.
  def write_after_terminal!(path : String, plan : WeightPlan, arena : Arena,
                            chain : T, source_sha256 : String) : String forall T
    complete = chain.completed?
    waits = chain.wait_count
    unless complete && waits == 1
      raise ArgumentError.new("BF16 cache requires one completed terminal wait")
    end
    # completed? and wait_count establish that the native terminal wait ran;
    # this idempotent call validates its result without another native wait.
    chain.commit_and_wait

    validate_sha256!(source_sha256)
    ensure_unused_target!(path)
    residents = validate_residents!(plan, arena)
    header, payload_bytes = build_header(residents, source_sha256)
    padding = (8 - header.bytesize % 8) % 8
    header += " " * padding
    raise ArgumentError.new("BF16 cache header is too large") if header.bytesize > CheckpointFile::MAX_HEADER_BYTES

    directory = File.dirname(path)
    unless File.directory?(directory)
      raise ArgumentError.new("BF16 cache destination directory does not exist")
    end

    temporary = File.tempfile(TEMP_PREFIX, TEMP_SUFFIX, dir: directory)
    temporary_path = temporary.path
    begin
      temporary.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
      temporary.write(header.to_slice)
      residents.each do |resident|
        byte_count = resident.n_elements * WeightDType::BF16.bytes_per_element
        pointer = resident.buffer.contents.as(UInt8*)
        raise ArgumentError.new("BF16 cache buffer has null contents #{resident.target}") if pointer.null?
        # Bytes is a borrowed view over shared Metal storage. No F32 decode,
        # conversion, or whole-tensor host copy occurs here.
        temporary.write(Bytes.new(pointer, byte_count.to_i))
      end
      temporary.flush
      temporary.fsync
      temporary.close

      expected_file_bytes = 8_i64 + header.bytesize.to_i64 + payload_bytes
      unless File.info(temporary_path).size == expected_file_bytes
        raise ArgumentError.new("BF16 cache payload length mismatch")
      end

      file_sha256 = sha256_file(temporary_path)
      verified = authenticate(temporary_path, file_sha256)
      verified.close

      # A same-directory hard link is atomic and fails if the target exists.
      # It provides no-clobber publication without ever opening the target for
      # write or deleting a pre-existing path.
      File.link(temporary_path, path)
      file_sha256
    ensure
      temporary.close unless temporary.closed?
      File.delete(temporary_path) if File.exists?(temporary_path)
    end
  end

  private def validate_residents!(plan : WeightPlan, arena : Arena) : Array(ResidentTensor)
    unless plan.resident_dtype.bf16?
      raise ArgumentError.new("BF16 cache requires BF16 resident weights")
    end
    raise ArgumentError.new("BF16 cache cannot serialize pooled arena views") if arena.pooled_allocations?

    entries = plan.entries
    raise ArgumentError.new("BF16 cache plan has no source bindings") if entries.empty?

    file_sources = {} of UInt64 => {CheckpointFile, Set(String)}
    targets = Set(String).new
    sources = Set(String).new

    entries.each do |entry|
      binding = entry.binding
      info = entry.info
      unless !binding.target.empty? && targets.add?(binding.target)
        raise ArgumentError.new("BF16 cache has duplicate or empty destinations")
      end
      unless !binding.source.empty? && sources.add?(binding.source)
        raise ArgumentError.new("BF16 cache has duplicate or empty source names")
      end
      unless info.name == binding.source && info.shape == binding.shape
        raise ArgumentError.new("BF16 cache source binding does not match its checkpoint tensor")
      end
      unless info.dtype.bf16? || info.dtype.f32?
        raise ArgumentError.new("BF16 cache source must be F32 or BF16: #{binding.source}")
      end

      entry.file.validate_tensor!(info)
      key = entry.file.object_id
      if file_entry = file_sources[key]?
        file_entry[1] << binding.source
      else
        file_sources[key] = {entry.file, Set{binding.source}}
      end
    end

    # WeightPlan's constructor can optionally permit partial bindings. A cache
    # is only sound when every tensor in each represented source shard has one
    # output entry, independent of that constructor flag.
    file_sources.each_value do |file, bound_names|
      file.unchanged!
      inventory = file.tensors
      unless bound_names.size == inventory.size && inventory.size == inventory.keys.to_set.size
        raise ArgumentError.new("BF16 cache source bindings are not a complete inventory")
      end
      inventory.each do |name, info|
        unless bound_names.includes?(name)
          raise ArgumentError.new("BF16 cache source bindings omit #{name}")
        end
        unless info.dtype.bf16? || info.dtype.f32?
          raise ArgumentError.new("BF16 cache source must be F32 or BF16: #{name}")
        end
      end
    end

    # Do all source-plan checks before touching Arena. In particular, a
    # partial WeightPlan must fail without requiring a Metal device or reading
    # any resident pointer.
    residents = entries.map do |entry|
      binding = entry.binding
      info = entry.info
      current = arena.buffers[binding.target]?
      unless current
        raise ArgumentError.new("BF16 cache destination is missing #{binding.target}")
      end
      unless arena.dtype(binding.target).bf16?
        raise ArgumentError.new("BF16 cache destination is not BF16 #{binding.target}")
      end
      expected_bytes = info.n_elements * WeightDType::BF16.bytes_per_element
      unless current.valid? && current.storage_mode.shared? && current.size == expected_bytes
        raise ArgumentError.new("BF16 cache destination has invalid shared storage size #{binding.target}")
      end
      ResidentTensor.new(binding.source, info.shape.dup, info.n_elements,
        binding.target, current)
    end

    residents.sort_by!(&.source)
    residents.each_with_index do |resident, index|
      residents[0, index].each do |prior|
        if MetalOps.overlap?(prior.buffer, resident.buffer)
          raise ArgumentError.new("BF16 cache destination buffers overlap")
        end
      end
    end
    residents
  end

  private def build_header(residents : Array(ResidentTensor), source_sha256 : String) : {String, Int64}
    offset = 0_i64
    header = JSON.build do |json|
      json.object do
        json.field "__metadata__" do
          json.object do
            json.field "schema", SCHEMA
            json.field "schema_version", SCHEMA_VERSION
            json.field "source_sha256", source_sha256
            json.field "converter", CONVERTER
          end
        end
        residents.each do |resident|
          byte_count = resident.n_elements * WeightDType::BF16.bytes_per_element
          finish = offset + byte_count
          json.field resident.source do
            json.object do
              json.field "dtype", "BF16"
              json.field "shape", resident.shape
              json.field "data_offsets", [offset, finish]
            end
          end
          offset = finish
        end
      end
    end
    {header, offset}
  end

  private def ensure_unused_target!(path : String) : Nil
    if path.empty? || File.basename(path).empty? || File.basename(path) == "." || File.basename(path) == ".."
      raise ArgumentError.new("Invalid BF16 cache destination path")
    end
    if File.info?(path, false)
      raise ArgumentError.new("BF16 cache destination already exists")
    end
  end

  private def regular_file_info!(path : String) : File::Info
    info = File.info?(path, false)
    unless info && info.file? && !info.symlink?
      raise ArgumentError.new("BF16 cache must be a non-symlink regular file")
    end
    info
  end

  private def validate_sha256!(sha256 : String) : Nil
    unless sha256.matches?(/\A[0-9a-f]{64}\z/)
      raise ArgumentError.new("Invalid BF16 cache SHA-256 pin")
    end
  end

  private def sha256_file(path : String) : String
    digest = Digest::SHA256.new
    File.open(path, "rb") do |io|
      chunk = Bytes.new(HASH_CHUNK_BYTES)
      loop do
        count = io.read(chunk)
        break if count == 0
        digest.update(chunk[0, count])
      end
    end
    digest.hexfinal
  end
end
