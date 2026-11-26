require "digest/sha256"
require "json"
require "set"
require "../../src/ml/video/hy_nomas_dit_boundaries"
require "../../src/ml/video/hy_nomas_q4_stack"
require "../../src/ml/video/bf16_weight_cache"

module HyNomasQ4DiTIO
  extend self
  SOURCE_SHA   = "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70"
  SOURCE_BYTES = 34_213_991_280_i64
  HEADER_SHA   = "cbfa83dc81af315b425fd783e0ce42d8ff09df772952ef2a8aff48493e21b200"
  INDEX_SCHEMA = "hy-nomas-q4-dit-cache-index-v1"
  CONFIG       = ML::Video::HyNomasConfig.new(batches: 1, video_tokens: 5280,
    text_tokens: 1280, dim: 2048, heads: 16, head_dim: 128, ffn_dim: 8192)
  BLOCKS                   = 54
  BF16_FINITE_STATUS_BYTES = sizeof(UInt32).to_i64
  BF16_FINITE_SOURCE       = <<-METAL
    #include <metal_stdlib>
    using namespace metal;

    constant uint HY_NOMAS_BF16_NONFINITE = 1u;

    kernel void hy_nomas_q4_bf16_finite_status(
        device const uchar *source [[buffer(0)]],
        device atomic_uint *status [[buffer(1)]],
        constant uint &count [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
      if (gid >= count) return;
      const uint byte_offset = gid * 2u;
      const uint bits = uint(source[byte_offset]) |
                        (uint(source[byte_offset + 1u]) << 8u);
      // BF16 shares F32's exponent width. Every NaN and infinity has all
      // exponent bits set, including a finite F32 value that overflows when
      // the preceding conversion rounds it into BF16.
      if ((bits & 0x7f80u) == 0x7f80u) {
        atomic_fetch_or_explicit(status, HY_NOMAS_BF16_NONFINITE,
                                 memory_order_relaxed);
      }
    }
    METAL

  @@bf16_finite_pipeline : ML::Metal::ComputePipeline?
  GLOBAL_BINDINGS = ML::Video::HyNomasTextRefinerMPSPlan.checkpoint_bindings +
                    ML::Video::HyNomasPatchMPSPlan.checkpoint_bindings +
                    ML::Video::HyNomasFinalMPSPlan.checkpoint_bindings

  def bf16_finite_pipeline : ML::Metal::ComputePipeline
    @@bf16_finite_pipeline ||= ML::Metal::PipelineCache.get("hy_nomas_q4_bf16_finite_status") do
      ML::Metal::ComputePipeline.new("hy_nomas_q4_bf16_finite_status", BF16_FINITE_SOURCE)
    end
  end

  # Setup-owned scalar gate for the actual BF16 residents. The caller reserves
  # this one status word in WeightPlan before constructing PreparedWeights.
  # Each tensor scan joins the caller's existing command chain after any F32
  # conversion, and the status is read only after that chain's terminal wait.
  class BF16FiniteCheck
    private record Source, target : String, elements : Int32,
      buffer : ML::MetalBuffer

    @arena : ML::Video::Arena
    @sources : Array(Source)
    @status : ML::MetalBuffer?
    @pipeline : ML::Metal::ComputePipeline
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @encoded = false
    @released = false

    def initialize(plan : ML::Video::WeightPlan,
                   @arena : ML::Video::Arena)
      unless plan.resident_dtype.bf16? && plan.reserve_bytes >= BF16_FINITE_STATUS_BYTES
        raise ArgumentError.new("BF16 finiteness check requires BF16 residency and a reserved status word")
      end
      entries = plan.entries
      raise ArgumentError.new("BF16 finiteness check requires a nonempty plan") if entries.empty?

      targets = Set(String).new
      buffers = [] of ML::MetalBuffer
      @sources = entries.map do |entry|
        target = entry.binding.target
        unless targets.add?(target)
          raise ArgumentError.new("Duplicate BF16 finiteness destination #{target}")
        end
        buffer = @arena.buffers[target]?
        unless buffer && @arena.dtype(target).bf16? && buffer.valid? &&
               buffer.storage_mode.shared? &&
               buffer.size == entry.info.n_elements * ML::Video::WeightDType::BF16.bytes_per_element
          raise ArgumentError.new("BF16 finiteness destination is missing or changed #{target}")
        end
        unless entry.info.n_elements > 0 && entry.info.n_elements <= Int32::MAX
          raise ArgumentError.new("BF16 finiteness tensor count is unsupported #{target}")
        end
        buffers.each { |prior| ML::Video::MetalOps.distinct(buffer, prior) }
        buffers << buffer
        Source.new(target, entry.info.n_elements.to_i32, buffer)
      end

      pipeline = HyNomasQ4DiTIO.bf16_finite_pipeline
      @pipeline = pipeline
      status = ML::MetalBuffer.new(BF16_FINITE_STATUS_BYTES)
      @status = status
      begin
        status_pointer = status.contents.as(UInt32*)
        raise ArgumentError.new("BF16 finiteness status buffer has null contents") if status_pointer.null?
        status_pointer.value = 0_u32
        buffers.each { |buffer| ML::Video::MetalOps.distinct(buffer, status) }
      rescue ex
        status.release
        @status = nil
        raise ex
      end
    end

    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("BF16 finiteness checker has been released") if @released
      raise ArgumentError.new("BF16 finiteness checker can encode only once") if @encoded
      encoder = chain.encoder
      status = status_buffer
      @last_chain = chain
      @sources.each do |source|
        current = @arena.buffers[source.target]?
        unless current && current.same?(source.buffer) &&
               @arena.dtype(source.target).bf16? && source.buffer.valid?
          raise ArgumentError.new("BF16 finiteness source changed #{source.target}")
        end
        encoder.set_pipeline(@pipeline)
          .set_buffer(source.buffer, 0)
          .set_buffer(status, 1, ML::Metal::BufferAccess::ReadWrite)
          .set_value(source.elements.to_u32, 2)
          .dispatch_1d(source.elements)
      end
      @encoded = true
    end

    def validate_after_terminal!(chain : ML::Metal::MPSCommandChain) : Nil
      unless !@released && @encoded && @last_chain.try(&.same?(chain)) &&
             chain.completed? && chain.wait_count == 1
        raise ArgumentError.new("BF16 finiteness status requires its one completed terminal wait")
      end
      status = status_buffer.contents.as(UInt32*).value
      unless status == 0_u32
        raise ArgumentError.new("BF16 resident weights contain NaN or infinity")
      end
    end

    def release : Nil
      return if @released
      if chain = @last_chain
        raise ArgumentError.new("Complete BF16 finiteness check chain before release") unless chain.completed?
      end
      @status.try(&.release)
      @status = nil
      @sources.clear
      @released = true
    end

    private def status_buffer : ML::MetalBuffer
      @status || raise ArgumentError.new("BF16 finiteness status buffer is unavailable")
    end
  end

  def regular!(path : String) : File::Info
    info = File.info?(path, false)
    unless info && info.file? && !info.symlink?
      raise ArgumentError.new("Expected an immutable non-symlink regular file: #{path}")
    end
    info
  end

  def unchanged!(path : String, original : File::Info) : Nil
    current = regular!(path)
    unless current.same_file?(original) && current.size == original.size &&
           current.modification_time == original.modification_time
      raise ArgumentError.new("Artifact changed after authentication: #{path}")
    end
  end

  def sha256_file(path : String) : String
    info = regular!(path)
    digest = Digest::SHA256.new
    File.open(path, "rb") do |io|
      chunk = Bytes.new(1024 * 1024)
      loop do
        count = io.read(chunk)
        break if count == 0
        digest.update(chunk[0, count])
      end
    end
    unchanged!(path, info)
    digest.hexfinal
  end

  def authenticate_source!(path : String) : ML::Video::CheckpointFile
    info = regular!(path)
    unless info.size == SOURCE_BYTES && sha256_file(path) == SOURCE_SHA
      raise ArgumentError.new("Full Hy-Nomas checkpoint byte count/digest mismatch")
    end
    File.open(path, "rb") do |io|
      length = io.read_bytes(UInt64, IO::ByteFormat::LittleEndian)
      raise ArgumentError.new("Pinned Hy-Nomas header length mismatch") unless length == 252_904
      raw = Bytes.new(length.to_i32)
      io.read_fully(raw)
      raise ArgumentError.new("Pinned Hy-Nomas header digest mismatch") unless Digest::SHA256.hexdigest(raw) == HEADER_SHA
    end
    source = ML::Video::CheckpointFile.new(path)
    begin
      inventory = source.tensors
      unless inventory.size == 2268 && inventory.values.all?(&.dtype.f32?)
        raise ArgumentError.new("Full Hy-Nomas checkpoint tensor inventory mismatch")
      end
      bindings = GLOBAL_BINDINGS + (0...BLOCKS).flat_map { |i| ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(CONFIG, block_index: i) }
      bindings.each do |binding|
        tensor = inventory[binding.source]? || raise ArgumentError.new("Missing active DiT tensor #{binding.source}")
        raise ArgumentError.new("Active DiT tensor shape mismatch") unless tensor.shape == binding.shape
      end
      unless bindings.size == 1776 && bindings.map(&.source).uniq.size == bindings.size
        raise ArgumentError.new("Active DiT route must bind exactly 1776 distinct tensors")
      end
      unchanged!(path, info)
      source
    rescue ex
      source.close
      raise ex
    end
  end

  def write_new!(path : String, contents : String) : Nil
    raise ArgumentError.new("Refusing to replace artifact: #{path}") if File.info?(path, false)
    temp = File.tempfile(".hy-nomas-q4-", ".partial", dir: File.dirname(path))
    begin
      temp << contents
      temp.flush
      temp.fsync
      temp.close
      File.link(temp.path, path)
    ensure
      temp.close unless temp.closed?
      File.delete(temp.path) if File.info?(temp.path, false)
    end
  end

  private def binding_signature(bindings : Array(ML::Video::WeightBinding)) : Array(Tuple(String, String, Array(Int64)))
    bindings.map do |binding|
      {binding.source, binding.target, binding.shape.dup}
    end.sort_by { |signature| signature[0] }
  end

  private def selected_inventory?(bindings : Array(ML::Video::WeightBinding)) : Bool
    signature = binding_signature(bindings)
    return true if signature == binding_signature(GLOBAL_BINDINGS)

    (0...BLOCKS).any? do |block_index|
      block_bindings = ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(CONFIG,
        block_index: block_index)
      signature == binding_signature(block_bindings)
    end
  end

  # This producer deliberately selects a complete declared route from the
  # authenticated full checkpoint. The generic BF16WeightCache producer keeps
  # its stricter whole-source contract. No CPU float conversion is performed.
  def write_bf16_selection!(path : String, plan : ML::Video::WeightPlan,
                            arena : ML::Video::Arena,
                            chain : ML::Metal::MPSCommandChain) : String
    unless chain.completed? && chain.wait_count == 1 && plan.resident_dtype.bf16?
      raise ArgumentError.new("Selected BF16 persistence requires one successful terminal wait")
    end
    raise ArgumentError.new("Refusing to replace selected BF16 cache") if File.info?(path, false)
    entries = plan.entries.sort_by(&.binding.source)
    bindings = entries.map(&.binding)
    unless selected_inventory?(bindings)
      raise ArgumentError.new("Selected BF16 cache requires exact global or single-block bindings")
    end
    first_file = entries.first.file
    buffers = [] of ML::MetalBuffer
    entries.each do |entry|
      entry.file.validate_tensor!(entry.info)
      unless entry.file.same?(first_file) && entry.info.dtype.f32?
        raise ArgumentError.new("Selected BF16 source must be one F32 checkpoint inventory")
      end
      buffer = arena[entry.binding.target]
      unless arena.dtype(entry.binding.target).bf16? && buffer.valid? &&
             buffer.size == entry.info.n_elements * 2 && buffer.storage_mode.shared?
        raise ArgumentError.new("Selected BF16 resident binding changed")
      end
      buffers.each { |prior| ML::Video::MetalOps.distinct(buffer, prior) }
      buffers << buffer
    end
    # All binding and alias rejection happens before this idempotent terminal
    # validation and before any file bytes are written.
    chain.commit_and_wait
    offset = 0_i64
    header = JSON.build do |json|
      json.object do
        json.field "__metadata__", {"schema" => "hy-nomas-selected-bf16-cache-v1", "source_sha256" => SOURCE_SHA}
        entries.each do |entry|
          finish = offset + entry.info.n_elements * 2
          json.field entry.binding.source do
            json.object do
              json.field "dtype", "BF16"
              json.field "shape", entry.binding.shape
              json.field "data_offsets", [offset, finish]
            end
          end
          offset = finish
        end
      end
    end
    header += " " * ((8 - header.bytesize % 8) % 8)
    temp = File.tempfile(".hy-nomas-selected-bf16-", ".partial", dir: File.dirname(path))
    begin
      temp.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
      temp << header
      entries.each do |entry|
        buffer = arena[entry.binding.target]
        temp.write(Bytes.new(buffer.contents.as(UInt8*), buffer.size.to_i32))
      end
      temp.flush
      temp.fsync
      temp.close
      sha = sha256_file(temp.path)
      check = ML::Video::BF16WeightCache.authenticate(temp.path, sha)
      begin
        unless check.tensors.keys.to_set == entries.map(&.binding.source).to_set &&
               entries.all? { |entry| check.tensors[entry.binding.source].shape == entry.binding.shape }
          raise ArgumentError.new("Selected BF16 persisted inventory mismatch")
        end
      ensure
        check.close
      end
      entries.each { |entry| entry.file.unchanged! }
      File.link(temp.path, path)
      sha
    ensure
      temp.close unless temp.closed?
      File.delete(temp.path) if File.info?(temp.path, false)
    end
  end
end
