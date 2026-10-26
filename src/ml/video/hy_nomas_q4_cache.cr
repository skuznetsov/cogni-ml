require "digest/sha256"
require "file/tempfile"
require "json"
require "set"
require "../metal/dispatch"
require "./checkpoint"
require "./hy_nomas_block"
require "./metal_mps_command_chain"
require "./metal_ops"
require "./q4k_range_weights"

module ML::Video::HyNomasQ4Cache
  extend self

  SCHEMA             = "hy-nomas-q4-block-cache-v1"
  SCHEMA_VERSION     = 1
  MATRIX_ENCODING    = "q4_k_8range"
  VECTOR_ENCODING    = "bf16-le"
  CACHE_PREFIX       = ".hy-nomas-q4-cache-"
  CACHE_SUFFIX       = ".partial"
  HASH_CHUNK_BYTES   = 1024 * 1024
  MAX_MANIFEST_BYTES = 1024 * 1024
  DIT_BLOCK_COUNT    = 54
  MATRIX_COUNT       = 14
  VECTOR_COUNT       = 18
  SOURCE_HEADER      = "hymm/models/dit/nomas_cpt.py"

  record Tensor, source : String, target : String, shape : Array(Int64),
    elements : Int64, source_dtype : String, encoding : String,
    offset_bytes : Int64, nbytes : Int64 do
    def snapshot : self
      self.class.new(source, target, shape.dup, elements, source_dtype,
        encoding, offset_bytes, nbytes)
    end
  end

  record PublishedCache, manifest_path : String, payload_path : String,
    manifest_sha256 : String, payload_sha256 : String,
    payload_bytes : Int64

  # Strict CPU-only authentication and GPU-resident access for one encoded
  # block. Construction hashes and validates both files before Metal is used.
  class Block
    getter manifest_path : String
    getter payload_path : String
    getter manifest_sha256 : String
    getter source_sha256 : String
    getter source_bytes : Int64
    getter payload_sha256 : String
    getter payload_bytes : Int64
    getter block_index : Int32
    getter geometry : Hash(String, Int32)

    @tensors : Array(Tensor)
    @manifest_info : File::Info
    @payload_info : File::Info
    @payload : ML::MetalBuffer? = nil
    @released = false

    def initialize(@manifest_path : String, expected_manifest_sha256 : String,
                   expected_source_sha256 : String,
                   config : HyNomasConfig, @block_index : Int32)
      config.validate!
      unless (0...DIT_BLOCK_COUNT).includes?(@block_index)
        raise ArgumentError.new("Hy-Nomas Q4 cache block index must be in 0...54")
      end
      Integrity.validate_sha256!(expected_manifest_sha256, "manifest")
      Integrity.validate_sha256!(expected_source_sha256, "source")

      @manifest_info = Integrity.regular_file_info!(@manifest_path, "manifest")
      unless @manifest_info.size > 0 && @manifest_info.size <= MAX_MANIFEST_BYTES
        raise ArgumentError.new("Hy-Nomas Q4 cache manifest size is invalid")
      end
      manifest_bytes = File.read(@manifest_path)
      @manifest_sha256 = Digest::SHA256.hexdigest(manifest_bytes)
      unless @manifest_sha256 == expected_manifest_sha256
        raise ArgumentError.new("Hy-Nomas Q4 cache manifest SHA-256 mismatch")
      end
      unless Integrity.same_snapshot?(@manifest_info,
               Integrity.regular_file_info!(@manifest_path, "manifest"))
        raise ArgumentError.new("Hy-Nomas Q4 cache manifest changed while reading")
      end

      document = JSON.parse(manifest_bytes).as_h
      Integrity.exact_keys!(document, Set{"schema", "schema_version", "source",
                                          "block_index", "geometry", "codec", "payload", "tensors"},
        "manifest")
      unless document["schema"].as_s == SCHEMA &&
             document["schema_version"].as_i == SCHEMA_VERSION
        raise ArgumentError.new("Hy-Nomas Q4 cache schema mismatch")
      end
      unless document["block_index"].as_i == @block_index
        raise ArgumentError.new("Hy-Nomas Q4 cache block identity mismatch")
      end

      source = document["source"].as_h
      Integrity.exact_keys!(source, Set{"sha256", "bytes"}, "source metadata")
      @source_sha256 = source["sha256"].as_s
      Integrity.validate_sha256!(@source_sha256, "source")
      unless @source_sha256 == expected_source_sha256
        raise ArgumentError.new("Hy-Nomas Q4 cache source checkpoint SHA-256 mismatch")
      end
      @source_bytes = source["bytes"].as_i64
      unless @source_bytes > 0
        raise ArgumentError.new("Hy-Nomas Q4 cache source byte size is invalid")
      end

      @geometry = parse_geometry!(document["geometry"], config)
      validate_codec!(document["codec"])
      payload_meta = document["payload"].as_h
      Integrity.exact_keys!(payload_meta, Set{"basename", "nbytes", "sha256"},
        "payload metadata")
      basename = payload_meta["basename"].as_s
      stem = Integrity.manifest_stem!(@manifest_path)
      expected_basename = "#{stem}.bin"
      unless basename == expected_basename && Integrity.safe_basename?(basename)
        raise ArgumentError.new("Hy-Nomas Q4 payload must be the manifest's safe sibling .bin file")
      end
      @payload_path = File.join(File.dirname(@manifest_path), basename)
      @payload_bytes = payload_meta["nbytes"].as_i64
      unless @payload_bytes > 0 && @payload_bytes <= Int32::MAX
        raise ArgumentError.new("Hy-Nomas Q4 payload size is outside the supported range")
      end
      @payload_sha256 = payload_meta["sha256"].as_s
      Integrity.validate_sha256!(@payload_sha256, "payload")

      bindings = HyNomasDualStreamBlock.checkpoint_bindings(config,
        block_index: @block_index)
      binding_by_source = bindings.to_h { |binding| {binding.source, binding} }
      tensor_doc = document["tensors"].as_h
      unless tensor_doc.keys.to_set == binding_by_source.keys.to_set
        raise ArgumentError.new("Hy-Nomas Q4 cache tensor inventory mismatch")
      end

      cursor = 0_i64
      @tensors = tensor_doc.keys.sort.map do |name|
        descriptor = tensor_doc[name].as_h
        Integrity.exact_keys!(descriptor,
          Set{"shape", "source_dtype", "resident_dtype", "encoding",
              "offset_bytes", "nbytes"}, "tensor #{name}")
        binding = binding_by_source[name]
        shape = descriptor["shape"].as_a.map(&.as_i64)
        unless shape == binding.shape
          raise ArgumentError.new("Hy-Nomas Q4 cache tensor shape mismatch #{name}")
        end
        source_dtype = descriptor["source_dtype"].as_s
        unless source_dtype == "F32" || source_dtype == "BF16"
          raise ArgumentError.new("Hy-Nomas Q4 source tensor dtype is unsupported #{name}")
        end
        unless descriptor["resident_dtype"].as_s == "BF16"
          raise ArgumentError.new("Hy-Nomas Q4 cache resident dtype must be BF16 #{name}")
        end
        encoding = descriptor["encoding"].as_s
        elements = Integrity.element_count!(shape, name)
        expected_encoding = shape.size == 2 ? MATRIX_ENCODING : VECTOR_ENCODING
        expected_bytes = if shape.size == 2
                           unless elements % Q4KRangeWeights::BLOCK_ELEMENTS == 0
                             raise ArgumentError.new("Hy-Nomas Q4 matrix is not Q4_K block aligned #{name}")
                           end
                           Q4KRangeWeights.payload_bytes(elements.to_i32)
                         else
                           unless shape.size == 1
                             raise ArgumentError.new("Hy-Nomas Q4 tensor rank is unsupported #{name}")
                           end
                           elements * WeightDType::BF16.bytes_per_element
                         end
        offset = descriptor["offset_bytes"].as_i64
        nbytes = descriptor["nbytes"].as_i64
        unless encoding == expected_encoding && offset == cursor && nbytes == expected_bytes
          raise ArgumentError.new("Hy-Nomas Q4 tensor encoding or payload layout mismatch #{name}")
        end
        cursor += nbytes
        Tensor.new(name, binding.target, shape, elements, source_dtype,
          encoding, offset, nbytes)
      end
      unless @tensors.size == MATRIX_COUNT + VECTOR_COUNT && cursor == @payload_bytes
        raise ArgumentError.new("Hy-Nomas Q4 payload length does not match its exact tensor inventory")
      end

      @payload_info = Integrity.regular_file_info!(@payload_path, "payload")
      unless @payload_info.size == @payload_bytes
        raise ArgumentError.new("Hy-Nomas Q4 payload file size mismatch")
      end
      actual_payload_sha = Integrity.sha256_file(@payload_path)
      unless actual_payload_sha == @payload_sha256
        raise ArgumentError.new("Hy-Nomas Q4 payload SHA-256 mismatch")
      end
      unless Integrity.same_snapshot?(@payload_info,
               Integrity.regular_file_info!(@payload_path, "payload"))
        raise ArgumentError.new("Hy-Nomas Q4 payload changed while hashing")
      end
    end

    def tensors : Array(Tensor)
      @tensors.map(&.snapshot)
    end

    # The stack uses this read-only identity for cross-block alias checks.
    # The Metal buffer itself remains owned and released by Block.
    def resident_payload : ML::MetalBuffer
      payload
    end

    # The caller must complete metadata/hash admission and aggregate-budget
    # checks for every block before loading any payload into Metal.
    def load! : Nil
      raise ArgumentError.new("Hy-Nomas Q4 cache block has been released") if @released
      return if @payload
      current = Integrity.regular_file_info!(@payload_path, "payload")
      unless Integrity.same_snapshot?(@payload_info, current)
        raise ArgumentError.new("Hy-Nomas Q4 payload changed after admission")
      end
      buffer = ML::MetalBuffer.new(@payload_bytes)
      begin
        pointer = buffer.contents.as(UInt8*)
        raise ArgumentError.new("Hy-Nomas Q4 payload buffer has null contents") if pointer.null?
        File.open(@payload_path, "rb") do |io|
          io.read_fully(Bytes.new(pointer, @payload_bytes.to_i32))
        end
        digest = Digest::SHA256.new
        digest.update(Bytes.new(pointer, @payload_bytes.to_i32))
        unless digest.hexfinal == @payload_sha256
          raise ArgumentError.new("Hy-Nomas Q4 payload changed while loading")
        end
        after = Integrity.regular_file_info!(@payload_path, "payload")
        unless Integrity.same_snapshot?(@payload_info, after)
          raise ArgumentError.new("Hy-Nomas Q4 payload changed while loading")
        end
        @payload = buffer
      rescue ex
        buffer.release
        raise ex
      end
    end

    # Validate every destination and alias before dispatching the first tensor.
    def validate_destinations!(arena : Arena) : Nil
      raise ArgumentError.new("Hy-Nomas Q4 cache payload is not loaded") unless payload
      raise ArgumentError.new("Hy-Nomas Q4 cache block has been released") if @released
      destinations = [] of ML::MetalBuffer
      @tensors.each do |tensor|
        buffer = arena.buffers[tensor.target]?
        unless buffer && arena.dtype(tensor.target).bf16? && buffer.valid? &&
               buffer.storage_mode.shared? &&
               buffer.size == tensor.elements * WeightDType::BF16.bytes_per_element
          raise ArgumentError.new("Hy-Nomas Q4 BF16 destination is missing or changed #{tensor.target}")
        end
        MetalOps.distinct(buffer, payload)
        destinations.each do |prior|
          MetalOps.distinct(buffer, prior)
        end
        destinations << buffer
      end
    end

    def decode_into(encoder : ML::Metal::ComputeEncoder, arena : Arena) : Nil
      validate_destinations!(arena)
      HyNomasQ4Cache.prepare_decode!
      source = payload
      @tensors.each do |tensor|
        destination = arena[tensor.target]
        count = tensor.elements.to_u32
        if tensor.encoding == MATRIX_ENCODING
          encoder.set_pipeline(HyNomasQ4Cache.q4_pipeline)
            .set_buffer(source, 0, offset: tensor.offset_bytes)
            .set_buffer(destination, 1, ML::Metal::BufferAccess::Write)
            .set_value(count, 2)
            .dispatch_1d(tensor.elements.to_i32)
        else
          encoder.set_pipeline(HyNomasQ4Cache.copy_pipeline)
            .set_buffer(source, 0, offset: tensor.offset_bytes)
            .set_buffer(destination, 1, ML::Metal::BufferAccess::Write)
            .set_value(count, 2)
            .dispatch_1d(tensor.elements.to_i32)
        end
      end
    end

    def release : Nil
      return if @released
      @payload.try(&.release)
      @payload = nil
      @released = true
    end

    private def payload : ML::MetalBuffer
      @payload || raise ArgumentError.new("Hy-Nomas Q4 cache payload is not loaded")
    end

    private def parse_geometry!(value : JSON::Any,
                                config : HyNomasConfig) : Hash(String, Int32)
      expected = HyNomasQ4Cache.geometry(config)
      actual = value.as_h
      unless actual.keys.to_set == expected.keys.to_set &&
             expected.all? { |name, number| actual[name].as_i == number }
        raise ArgumentError.new("Hy-Nomas Q4 cache geometry mismatch")
      end
      expected
    end

    private def validate_codec!(value : JSON::Any) : Nil
      codec = value.as_h
      Integrity.exact_keys!(codec, Set{"matrix", "ranges", "block_elements",
                                       "block_bytes", "vector"}, "codec")
      unless codec["matrix"].as_s == "q4_k" && codec["ranges"].as_i == 8 &&
             codec["block_elements"].as_i == Q4KRangeWeights::BLOCK_ELEMENTS &&
             codec["block_bytes"].as_i == Q4KRangeWeights::Q4_K_BLOCK_BYTES &&
             codec["vector"].as_s == VECTOR_ENCODING
        raise ArgumentError.new("Hy-Nomas Q4 codec metadata mismatch")
      end
    end
  end

  # A single-block Q4_K packer. The caller prepares BF16 resident weights,
  # encodes any F32->BF16 conversion into the same chain, then calls
  # encode_pack(chain). Disk writes are terminal-only and never decode neural
  # values on CPU.
  class PackBlock
    private record PackTensor, source : String, target : String,
      shape : Array(Int64), elements : Int64, source_dtype : String,
      encoding : String, offset_bytes : Int64, nbytes : Int64,
      source_buffer : ML::MetalBuffer, packed_buffer : ML::MetalBuffer? do
      def with_packed_buffer(buffer : ML::MetalBuffer) : self
        self.class.new(source, target, shape, elements, source_dtype,
          encoding, offset_bytes, nbytes, source_buffer, buffer)
      end
    end

    getter block_index : Int32
    getter payload_bytes : Int64
    getter source_sha256 : String
    getter source_bytes : Int64

    @config : HyNomasConfig
    @tensors : Array(PackTensor)
    @status : ML::MetalBuffer? = nil
    @source_arena : Arena
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @encoded = false
    @released = false

    def initialize(@config : HyNomasConfig, @block_index : Int32,
                   @source_sha256 : String, @source_bytes : Int64,
                   plan : WeightPlan, source_arena : Arena)
      @config.validate!
      @source_arena = source_arena
      unless (0...DIT_BLOCK_COUNT).includes?(@block_index)
        raise ArgumentError.new("Hy-Nomas Q4 pack block index must be in 0...54")
      end
      Integrity.validate_sha256!(@source_sha256, "source")
      unless @source_bytes > 0
        raise ArgumentError.new("Hy-Nomas Q4 source byte size is invalid")
      end
      required_reserve = self.class.resident_bytes_for(@config, @block_index)
      unless plan.reserve_bytes >= required_reserve
        raise ArgumentError.new("Hy-Nomas Q4 pack plan does not reserve all packed-output buffers")
      end
      @tensors = self.class.validate_plan!(plan, source_arena, @config, @block_index)
      @payload_bytes = @tensors.sum(&.nbytes)
      unless @payload_bytes > 0 && @payload_bytes <= Int32::MAX
        raise ArgumentError.new("Hy-Nomas Q4 block payload exceeds supported storage")
      end

      created = [] of ML::MetalBuffer
      begin
        @tensors = @tensors.map do |tensor|
          if tensor.encoding == MATRIX_ENCODING
            packed = ML::MetalBuffer.new(tensor.nbytes)
            created << packed
            tensor.with_packed_buffer(packed)
          else
            tensor
          end
        end
        status = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
        @status = status
        created << status
        status.contents.as(UInt32*).value = 0_u32
        @tensors.each do |tensor|
          next unless packed = tensor.packed_buffer
          MetalOps.distinct(packed, status)
          @tensors.each do |other|
            MetalOps.distinct(packed, other.source_buffer)
          end
        end
        @tensors.each_with_index do |tensor, index|
          if packed = tensor.packed_buffer
            @tensors[0...index].each do |prior|
              prior.packed_buffer.try { |prior_buffer| MetalOps.distinct(packed, prior_buffer) }
            end
          end
        end
      rescue ex
        created.each(&.release)
        raise ex
      end
    end

    # Exact additional named Metal residency created by this packer: one Q4_K
    # output per matrix and one sticky status word. Vectors reuse the existing
    # BF16 source Arena; the pack kernel has no separately allocated panels.
    # Call before constructing PreparedWeights so WeightPlan includes this
    # reservation in its aggregate peak check.
    def self.resident_bytes_for(config : HyNomasConfig,
                                block_index : Int32) : Int64
      config.validate!
      unless (0...DIT_BLOCK_COUNT).includes?(block_index)
        raise ArgumentError.new("Hy-Nomas Q4 pack block index must be in 0...54")
      end
      bindings = HyNomasDualStreamBlock.checkpoint_bindings(config,
        block_index: block_index)
      unless bindings.size == MATRIX_COUNT + VECTOR_COUNT
        raise ArgumentError.new("Hy-Nomas Q4 pack source inventory changed")
      end
      output_bytes = 0_i64
      matrix_count = 0
      vector_count = 0
      bindings.each do |binding|
        elements = Integrity.element_count!(binding.shape, binding.source)
        if binding.shape.size == 2
          matrix_count += 1
          unless elements % Q4KRangeWeights::BLOCK_ELEMENTS == 0
            raise ArgumentError.new("Hy-Nomas Q4 matrix is not block aligned #{binding.source}")
          end
          bytes = Q4KRangeWeights.payload_bytes(elements.to_i32)
          if bytes > Int64::MAX - output_bytes
            raise ArgumentError.new("Hy-Nomas Q4 pack reservation overflow")
          end
          output_bytes += bytes
        elsif binding.shape.size == 1
          vector_count += 1
        else
          raise ArgumentError.new("Hy-Nomas Q4 tensor rank is unsupported #{binding.source}")
        end
      end
      unless matrix_count == MATRIX_COUNT && vector_count == VECTOR_COUNT
        raise ArgumentError.new("Hy-Nomas Q4 pack source rank inventory changed")
      end
      status_bytes = sizeof(UInt32).to_i64
      raise ArgumentError.new("Hy-Nomas Q4 pack reservation overflow") if output_bytes > Int64::MAX - status_bytes
      output_bytes + status_bytes
    end

    # Structural preflight is independent of Metal; a bad plan fails before
    # this constructor allocates any cache payload or status buffers.
    def self.validate_plan!(plan : WeightPlan, source_arena : Arena,
                            config : HyNomasConfig, block_index : Int32) : Array(PackTensor)
      unless plan.resident_dtype.bf16?
        raise ArgumentError.new("Hy-Nomas Q4 pack requires BF16 resident weights")
      end
      entries = plan.entries
      bindings = HyNomasDualStreamBlock.checkpoint_bindings(config,
        block_index: block_index)
      binding_by_source = bindings.to_h { |binding| {binding.source, binding} }
      unless entries.size == MATRIX_COUNT + VECTOR_COUNT &&
             entries.map(&.binding.source).to_set == binding_by_source.keys.to_set
        raise ArgumentError.new("Hy-Nomas Q4 pack plan must bind the exact block tensor inventory")
      end
      first_file = entries.first.file
      targets = Set(String).new
      sources = Set(String).new
      result = entries.map do |entry|
        binding = entry.binding
        info = entry.info
        expected_binding = binding_by_source[binding.source]?
        unless expected_binding && binding.target == expected_binding.target &&
               binding.shape == expected_binding.shape && info.name == binding.source &&
               info.shape == binding.shape
          raise ArgumentError.new("Hy-Nomas Q4 pack binding mismatch #{binding.source}")
        end
        unless !sources.includes?(binding.source) && !targets.includes?(binding.target)
          raise ArgumentError.new("Hy-Nomas Q4 pack plan contains duplicate tensor bindings")
        end
        sources << binding.source
        targets << binding.target
        unless entry.file.same?(first_file)
          raise ArgumentError.new("Hy-Nomas Q4 block weights must come from one authenticated checkpoint")
        end
        entry.file.validate_tensor!(info)
        source_dtype = Integrity.dtype_name(info.dtype)
        unless source_dtype == "F32" || source_dtype == "BF16"
          raise ArgumentError.new("Hy-Nomas Q4 source tensor dtype is unsupported #{binding.source}")
        end
        if info.dtype.f32? && !plan.allow_f32_to_bf16
          raise ArgumentError.new("F32 Hy-Nomas Q4 source requires BF16 conversion admission")
        end
        buffer = source_arena.buffers[binding.target]?
        unless buffer && source_arena.dtype(binding.target).bf16? && buffer.valid? &&
               buffer.storage_mode.shared? &&
               buffer.size == info.n_elements * WeightDType::BF16.bytes_per_element
          raise ArgumentError.new("Hy-Nomas Q4 BF16 source buffer is missing or invalid #{binding.target}")
        end
        elements = info.n_elements
        encoding = if binding.shape.size == 2
                     unless elements % Q4KRangeWeights::BLOCK_ELEMENTS == 0
                       raise ArgumentError.new("Hy-Nomas Q4 source matrix is not block aligned #{binding.source}")
                     end
                     MATRIX_ENCODING
                   else
                     unless binding.shape.size == 1
                       raise ArgumentError.new("Hy-Nomas Q4 source tensor rank is unsupported #{binding.source}")
                     end
                     VECTOR_ENCODING
                   end
        nbytes = encoding == MATRIX_ENCODING ? Q4KRangeWeights.payload_bytes(elements.to_i32) : elements * WeightDType::BF16.bytes_per_element
        PackTensor.new(binding.source, binding.target, binding.shape.dup,
          elements, source_dtype, encoding, 0_i64, nbytes, buffer, nil)
      end
      result.sort_by!(&.source)
      result.each_with_index do |tensor, index|
        result[0...index].each do |prior|
          MetalOps.distinct(tensor.source_buffer, prior.source_buffer)
        end
      end
      cursor = 0_i64
      result.map do |tensor|
        updated = PackTensor.new(tensor.source, tensor.target, tensor.shape,
          tensor.elements, tensor.source_dtype, tensor.encoding, cursor,
          tensor.nbytes, tensor.source_buffer, tensor.packed_buffer)
        cursor += tensor.nbytes
        updated
      end
    end

    def encode_pack(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("Hy-Nomas Q4 packer has been released") if @released
      raise ArgumentError.new("Hy-Nomas Q4 packer can encode only once") if @encoded
      chain.encoder
      Q4KRangeWeights.prepare!
      @tensors.each do |tensor|
        unless tensor.source_buffer.valid? && tensor.source_buffer.storage_mode.shared?
          raise ArgumentError.new("Hy-Nomas Q4 source buffer changed #{tensor.target}")
        end
        if packed = tensor.packed_buffer
          Q4KRangeWeights.pack(chain.encoder, tensor.source_buffer, packed,
            status_buffer, tensor.elements.to_i32, 8)
        end
      end
      @last_chain = chain
      @encoded = true
    end

    # Persist one authenticated pair after a single successful terminal wait.
    # Existing outputs are never overwritten; the manifest is linked last and
    # acts as the atomic visibility marker for its sibling payload.
    def write_after_terminal!(manifest_path : String,
                              chain : ML::Metal::MPSCommandChain) : PublishedCache
      raise ArgumentError.new("Hy-Nomas Q4 packer has been released") if @released
      unless @encoded && @last_chain && @last_chain.not_nil!.same?(chain)
        raise ArgumentError.new("Hy-Nomas Q4 cache write does not match its encoded chain")
      end
      unless chain.completed? && chain.wait_count == 1
        raise ArgumentError.new("Hy-Nomas Q4 cache write requires one completed terminal wait")
      end
      chain.commit_and_wait
      unless status_buffer.contents.as(UInt32*).value == 0_u32
        raise ArgumentError.new("Hy-Nomas Q4 pack encountered non-finite or invalid values")
      end
      validate_source_buffers!

      manifest_basename = File.basename(manifest_path)
      unless manifest_basename.ends_with?(".json") && Integrity.safe_basename?(manifest_basename)
        raise ArgumentError.new("Hy-Nomas Q4 cache manifest path must end in .json")
      end
      stem = manifest_basename[0, manifest_basename.size - 5]
      payload_basename = "#{stem}.bin"
      directory = File.dirname(manifest_path)
      unless File.directory?(directory)
        raise ArgumentError.new("Hy-Nomas Q4 cache directory does not exist")
      end
      payload_path = File.join(directory, payload_basename)
      Integrity.ensure_unused_target!(manifest_path)
      Integrity.ensure_unused_target!(payload_path)

      temporary_payload = File.tempfile(CACHE_PREFIX, CACHE_SUFFIX, dir: directory)
      temporary_manifest = nil.as(File?)
      begin
        payload_digest = Digest::SHA256.new
        payload_cursor = 0_i64
        @tensors.each do |tensor|
          pointer = if tensor.encoding == MATRIX_ENCODING
                      packed = tensor.packed_buffer || raise ArgumentError.new("Missing Q4_K output buffer")
                      packed.contents.as(UInt8*)
                    else
                      tensor.source_buffer.contents.as(UInt8*)
                    end
          raise ArgumentError.new("Hy-Nomas Q4 cache tensor has null shared contents #{tensor.source}") if pointer.null?
          bytes = Bytes.new(pointer, tensor.nbytes.to_i32)
          temporary_payload.write(bytes)
          payload_digest.update(bytes)
          payload_cursor += tensor.nbytes
        end
        unless payload_cursor == @payload_bytes
          raise ArgumentError.new("Hy-Nomas Q4 payload serialization length mismatch")
        end
        temporary_payload.flush
        temporary_payload.fsync
        temporary_payload.close
        unless File.info(temporary_payload.path).size == @payload_bytes
          raise ArgumentError.new("Hy-Nomas Q4 payload file length mismatch")
        end
        payload_sha256 = payload_digest.hexfinal
        unless Integrity.sha256_file(temporary_payload.path) == payload_sha256
          raise ArgumentError.new("Hy-Nomas Q4 payload changed while writing")
        end

        manifest = build_manifest(payload_basename, payload_sha256)
        temporary_manifest = File.tempfile(CACHE_PREFIX, ".json.partial", dir: directory)
        temporary_manifest.not_nil!.write(manifest.to_slice)
        temporary_manifest.not_nil!.flush
        temporary_manifest.not_nil!.fsync
        temporary_manifest.not_nil!.close
        manifest_sha256 = Integrity.sha256_file(temporary_manifest.not_nil!.path)

        File.link(temporary_payload.path, payload_path)
        begin
          File.link(temporary_manifest.not_nil!.path, manifest_path)
        rescue ex
          # Remove only the payload hard link created by this invocation. Do
          # not remove a path that raced with or predated our publication.
          if published = File.info?(payload_path, false)
            if temporary_info = File.info?(temporary_payload.path, false)
              File.delete(payload_path) if published.same_file?(temporary_info)
            end
          end
          raise ex
        end
        PublishedCache.new(manifest_path, payload_path, manifest_sha256,
          payload_sha256, @payload_bytes)
      ensure
        temporary_payload.close unless temporary_payload.closed?
        temporary_manifest.try { |file| file.close unless file.closed? }
        File.delete(temporary_payload.path) if File.exists?(temporary_payload.path)
        if file = temporary_manifest
          File.delete(file.path) if File.exists?(file.path)
        end
      end
    end

    def release : Nil
      return if @released
      if chain = @last_chain
        raise ArgumentError.new("Complete Hy-Nomas Q4 pack chain before release") unless chain.completed?
      end
      @tensors.each { |tensor| tensor.packed_buffer.try(&.release) }
      @status.try(&.release)
      @status = nil
      @released = true
    end

    private def status_buffer : ML::MetalBuffer
      @status || raise ArgumentError.new("Hy-Nomas Q4 pack status buffer is unavailable")
    end

    private def validate_source_buffers! : Nil
      buffers = [] of ML::MetalBuffer
      @tensors.each do |tensor|
        buffer = tensor.source_buffer
        unless buffer.valid? && buffer.storage_mode.shared? &&
               buffer.size == tensor.elements * WeightDType::BF16.bytes_per_element
          raise ArgumentError.new("Hy-Nomas Q4 source BF16 buffer changed #{tensor.target}")
        end
        buffers.each { |prior| MetalOps.distinct(buffer, prior) }
        buffers << buffer
        tensor.packed_buffer.try { |packed| MetalOps.distinct(packed, buffer) }
      end
    end

    private def build_manifest(payload_basename : String,
                               payload_sha256 : String) : String
      JSON.build do |json|
        json.object do
          json.field "schema", SCHEMA
          json.field "schema_version", SCHEMA_VERSION
          json.field "source" do
            json.object do
              json.field "sha256", @source_sha256
              json.field "bytes", @source_bytes
            end
          end
          json.field "block_index", @block_index
          json.field "geometry" do
            json.object do
              HyNomasQ4Cache.geometry(@config).each do |name, value|
                json.field name, value
              end
            end
          end
          json.field "codec" do
            json.object do
              json.field "matrix", "q4_k"
              json.field "ranges", 8
              json.field "block_elements", Q4KRangeWeights::BLOCK_ELEMENTS
              json.field "block_bytes", Q4KRangeWeights::Q4_K_BLOCK_BYTES
              json.field "vector", VECTOR_ENCODING
            end
          end
          json.field "payload" do
            json.object do
              json.field "basename", payload_basename
              json.field "nbytes", @payload_bytes
              json.field "sha256", payload_sha256
            end
          end
          json.field "tensors" do
            json.object do
              @tensors.each do |tensor|
                json.field tensor.source do
                  json.object do
                    json.field "shape", tensor.shape
                    json.field "source_dtype", tensor.source_dtype
                    json.field "resident_dtype", "BF16"
                    json.field "encoding", tensor.encoding
                    json.field "offset_bytes", tensor.offset_bytes
                    json.field "nbytes", tensor.nbytes
                  end
                end
              end
            end
          end
        end
      end
    end
  end

  def self.geometry(config : HyNomasConfig) : Hash(String, Int32)
    {"batches"      => config.batches,
     "video_tokens" => config.video_tokens,
     "text_tokens"  => config.text_tokens,
     "dim"          => config.dim,
     "heads"        => config.heads,
     "head_dim"     => config.head_dim,
     "ffn_dim"      => config.ffn_dim}
  end

  def self.prepare_decode! : Nil
    @@q4_pipeline ||= ML::Metal::PipelineCache.get("hy_nomas_q4k_to_bf16") do
      ML::Metal::ComputePipeline.new("hy_nomas_q4k_to_bf16", SOURCE)
    end
    @@copy_pipeline ||= ML::Metal::PipelineCache.get("hy_nomas_copy_bf16") do
      ML::Metal::ComputePipeline.new("hy_nomas_copy_bf16", SOURCE)
    end
  end

  SOURCE = {{ read_file("#{__DIR__}/kernels/hy_nomas_q4_cache.metal") }}
  @@q4_pipeline : ML::Metal::ComputePipeline?
  @@copy_pipeline : ML::Metal::ComputePipeline?

  def self.q4_pipeline : ML::Metal::ComputePipeline
    @@q4_pipeline || raise ArgumentError.new("Call HyNomasQ4Cache.prepare_decode! before dispatch")
  end

  def self.copy_pipeline : ML::Metal::ComputePipeline
    @@copy_pipeline || raise ArgumentError.new("Call HyNomasQ4Cache.prepare_decode! before dispatch")
  end

  # Shared strict file/inventory routines used by cache admission and writing.
  module Integrity
    extend self

    def validate_sha256!(value : String, label : String) : Nil
      unless value.matches?(/\A[0-9a-f]{64}\z/)
        raise ArgumentError.new("Invalid Hy-Nomas Q4 #{label} SHA-256")
      end
    end

    def exact_keys!(actual : Hash(String, JSON::Any), expected : Set(String),
                    label : String) : Nil
      unless actual.keys.to_set == expected
        raise ArgumentError.new("Hy-Nomas Q4 #{label} fields are not exact")
      end
    end

    def dtype_name(dtype : ML::GGUF::SafeTensorDType) : String
      case dtype
      in .f32?  then "F32"
      in .bf16? then "BF16"
      in .f16?  then "F16"
      end
    end

    def element_count!(shape : Array(Int64), name : String) : Int64
      raise ArgumentError.new("Hy-Nomas Q4 tensor has empty shape #{name}") if shape.empty?
      count = 1_i64
      shape.each do |dimension|
        unless dimension > 0 && dimension <= Int32::MAX && count <= Int32::MAX // dimension
          raise ArgumentError.new("Hy-Nomas Q4 tensor shape is unsupported #{name}")
        end
        count *= dimension
      end
      count
    end

    def manifest_stem!(path : String) : String
      basename = File.basename(path)
      unless basename.ends_with?(".json") && safe_basename?(basename)
        raise ArgumentError.new("Hy-Nomas Q4 cache manifest must be a safe .json sibling")
      end
      basename[0, basename.size - 5]
    end

    def safe_basename?(name : String) : Bool
      !name.empty? && name != "." && name != ".." &&
        name == File.basename(name) && !name.includes?('\\') &&
        !name.includes?('\0')
    end

    def regular_file_info!(path : String, label : String) : File::Info
      info = File.info?(path, false)
      unless info && info.file? && !info.symlink?
        raise ArgumentError.new("Hy-Nomas Q4 #{label} must be a non-symlink regular file")
      end
      info
    end

    def same_snapshot?(left : File::Info, right : File::Info) : Bool
      left.same_file?(right) && left.size == right.size &&
        left.modification_time == right.modification_time
    end

    def sha256_file(path : String) : String
      before = regular_file_info!(path, "hashed file")
      digest = Digest::SHA256.new
      File.open(path, "rb") do |io|
        chunk = Bytes.new(HASH_CHUNK_BYTES)
        loop do
          count = io.read(chunk)
          break if count == 0
          digest.update(chunk[0, count])
        end
      end
      after = regular_file_info!(path, "hashed file")
      unless same_snapshot?(before, after)
        raise ArgumentError.new("Hy-Nomas Q4 file changed while hashing")
      end
      digest.hexfinal
    end

    def ensure_unused_target!(path : String) : Nil
      unless safe_basename?(File.basename(path)) && File.basename(path).size > 1
        raise ArgumentError.new("Hy-Nomas Q4 cache output path is invalid")
      end
      if File.info?(path, false)
        raise ArgumentError.new("Hy-Nomas Q4 cache output already exists")
      end
    end
  end
end
