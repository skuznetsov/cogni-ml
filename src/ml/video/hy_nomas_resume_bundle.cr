require "digest/sha256"
require "file/tempfile"
require "json"
require "set"

module ML::Video
  # A CPU-only, single-file checkpoint for the exact byte state at a drained
  # Hy-Nomas boundary. The file reserves a bounded header region so payload
  # records can be streamed once, in order, without staging them in memory.
  class HyNomasResumeBundle
    SCHEMA           = "hy-nomas-resume-bundle-v1"
    SCHEMA_VERSION   = 1
    MAGIC            = "HNRSB01\0"
    MAX_HEADER_BYTES = 1024 * 1024
    PREFIX_BYTES     = MAGIC.bytesize + sizeof(UInt32)
    PAYLOAD_OFFSET   = PREFIX_BYTES + MAX_HEADER_BYTES
    IO_CHUNK_BYTES   = 64 * 1024

    record Entry, name : String, nbytes : Int64, offset : Int64,
      sha256 : String

    # Writes each named record to a bounded sink in sorted name order. The
    # callback must provide exactly the declared number of bytes. Returns the
    # complete file SHA-256, suitable for storage beside the bundle path.
    def self.write(path : String, contract : String,
                   descriptors : Hash(String, Int64),
                   &block : String, IO ->) : String
      validate_output_path!(path)
      validate_contract!(contract)
      entries, total_bytes = validate_inventory!(descriptors)

      directory = File.dirname(path)
      temporary = File.tempfile(".hy-nomas-resume-", ".partial", dir: directory)
      begin
        zeroes = Bytes.new(IO_CHUNK_BYTES, 0_u8)
        remaining_header = PREFIX_BYTES + MAX_HEADER_BYTES
        while remaining_header > 0
          count = Math.min(remaining_header, zeroes.size)
          temporary.write(zeroes[0, count])
          remaining_header -= count
        end

        cursor = 0_i64
        entries = entries.map do |descriptor|
          digest = Digest::SHA256.new
          sink = RecordSink.new(temporary, digest, descriptor.name, descriptor.nbytes)
          begin
            yield descriptor.name, sink
            record_sha256 = sink.finish
          ensure
            sink.deactivate
          end

          entry = Entry.new(descriptor.name, descriptor.nbytes, cursor, record_sha256)
          cursor = checked_add!(cursor, descriptor.nbytes, "payload")
          entry
        end
        unless cursor == total_bytes
          raise ArgumentError.new("Hy-Nomas resume bundle payload length mismatch")
        end

        header = build_header(contract, entries)
        if header.bytesize > MAX_HEADER_BYTES
          raise ArgumentError.new("Hy-Nomas resume bundle header size exceeds 1 MiB")
        end

        temporary.rewind
        temporary.write(MAGIC.to_slice)
        temporary.write_bytes(header.bytesize.to_u32, IO::ByteFormat::LittleEndian)
        temporary.write(header.to_slice)
        temporary.flush
        temporary.fsync

        expected_file_bytes = checked_add!(PAYLOAD_OFFSET.to_i64, total_bytes, "file")
        unless temporary.info.size == expected_file_bytes
          raise ArgumentError.new("Hy-Nomas resume bundle file length mismatch")
        end
        temporary.close

        digest = sha256_file(temporary.path)
        ensure_unused_target!(path)
        File.link(temporary.path, path)
        published_info = regular_file_info!(path, "published bundle")
        temporary_info = regular_file_info!(temporary.path, "temporary bundle")
        unless published_info.same_file?(temporary_info) && published_info.size == expected_file_bytes
          raise ArgumentError.new("Hy-Nomas resume bundle publication changed")
        end
        digest
      ensure
        temporary.close unless temporary.closed?
        File.delete(temporary.path) if File.exists?(temporary.path)
      end
    end

    # Hashes and validates the complete bundle, contract, and exact inventory
    # before returning a reader that can copy records into caller-owned bytes.
    def self.open(path : String, expected_sha256 : String, contract : String,
                  expected_inventory : Hash(String, Int64)) : Reader
      Reader.new(path, expected_sha256, contract, expected_inventory)
    end

    def self.validate_output_path!(path : String) : Nil
      raise ArgumentError.new("Hy-Nomas resume bundle path is empty") if path.empty?
      raise ArgumentError.new("Hy-Nomas resume bundle path contains NUL") if path.includes?('\0')
      basename = File.basename(path)
      unless basename != "." && basename != ".." && !basename.empty?
        raise ArgumentError.new("Hy-Nomas resume bundle output path is invalid")
      end
      directory = File.dirname(path)
      unless File.directory?(directory)
        raise ArgumentError.new("Hy-Nomas resume bundle directory does not exist")
      end
      ensure_unused_target!(path)
    end

    def self.ensure_unused_target!(path : String) : Nil
      if File.info?(path, false)
        raise ArgumentError.new("Hy-Nomas resume bundle output already exists")
      end
    end

    def self.validate_contract!(contract : String) : Nil
      unless !contract.empty? && contract.valid_encoding? && !contract.includes?('\0')
        raise ArgumentError.new("Hy-Nomas resume bundle contract is invalid")
      end
    end

    def self.validate_inventory!(inventory : Hash(String, Int64)) : Tuple(Array(Descriptor), Int64)
      raise ArgumentError.new("Hy-Nomas resume bundle inventory is empty") if inventory.empty?
      if inventory.size > MAX_HEADER_BYTES // 16
        raise ArgumentError.new("Hy-Nomas resume bundle header size exceeds 1 MiB")
      end
      total_bytes = 0_i64
      descriptors = inventory.map do |name, nbytes|
        unless !name.empty? && name.valid_encoding? && !name.includes?('\0')
          raise ArgumentError.new("Hy-Nomas resume bundle record name is invalid")
        end
        unless nbytes >= 0 && nbytes < Int32::MAX
          raise ArgumentError.new("Hy-Nomas resume bundle record size is outside the supported range")
        end
        total_bytes = checked_add!(total_bytes, nbytes, "payload")
        Descriptor.new(name, nbytes)
      end
      {descriptors.sort_by(&.name), total_bytes}
    end

    def self.checked_add!(left : Int64, right : Int64, label : String) : Int64
      if right < 0 || left > Int64::MAX - right
        raise ArgumentError.new("Hy-Nomas resume bundle #{label} length overflows Int64")
      end
      left + right
    end

    def self.build_header(contract : String, entries : Array(Entry)) : String
      output = BoundedHeaderIO.new(MAX_HEADER_BYTES)
      JSON.build(output) do |json|
        json.object do
          json.field "schema", SCHEMA
          json.field "schema_version", SCHEMA_VERSION
          json.field "contract", contract
          json.field "records" do
            json.array do
              entries.each do |entry|
                json.object do
                  json.field "name", entry.name
                  json.field "nbytes", entry.nbytes
                  json.field "offset", entry.offset
                  json.field "sha256", entry.sha256
                end
              end
            end
          end
        end
      end
      output.to_s
    end

    def self.regular_file_info!(path : String, label : String) : File::Info
      info = File.info?(path, false)
      unless info && info.file? && !info.symlink?
        raise ArgumentError.new("Hy-Nomas resume bundle #{label} must be a non-symlink regular file")
      end
      info
    end

    def self.same_snapshot?(left : File::Info, right : File::Info) : Bool
      left.same_file?(right) && left.size == right.size &&
        left.modification_time == right.modification_time
    end

    def self.validate_sha256!(sha256 : String, label : String) : Nil
      unless sha256.size == 64 && sha256.each_byte.all? { |byte| byte >= '0'.ord && byte <= '9'.ord || byte >= 'a'.ord && byte <= 'f'.ord }
        raise ArgumentError.new("Hy-Nomas resume bundle #{label} SHA-256 is invalid")
      end
    end

    def self.sha256_file(path : String) : String
      before = regular_file_info!(path, "hashed file")
      digest = Digest::SHA256.new
      File.open(path, "rb") do |io|
        chunk = Bytes.new(IO_CHUNK_BYTES)
        while (count = io.read(chunk)) > 0
          digest.update(chunk[0, count])
        end
      end
      after = regular_file_info!(path, "hashed file")
      unless same_snapshot?(before, after)
        raise ArgumentError.new("Hy-Nomas resume bundle changed while hashing")
      end
      digest.hexfinal
    end

    record Descriptor, name : String, nbytes : Int64

    private class RecordSink < IO
      getter bytes_written : Int64 = 0_i64

      def initialize(@target : IO, @digest : Digest::SHA256,
                     @name : String, @expected_bytes : Int64)
        @active = true
      end

      def read(slice : Bytes) : Int32
        raise IO::Error.new("Hy-Nomas resume bundle record sink is write-only")
      end

      def write(slice : Bytes) : Nil
        raise IO::Error.new("Hy-Nomas resume bundle record sink is closed") unless @active
        return if slice.empty?
        remaining = @expected_bytes - @bytes_written
        if slice.size.to_i64 > remaining
          raise ArgumentError.new("Hy-Nomas resume bundle record byte count exceeds descriptor for #{@name}")
        end
        @target.write(slice)
        @digest.update(slice)
        @bytes_written += slice.size
      end

      def finish : String
        @active = false
        unless @bytes_written == @expected_bytes
          raise ArgumentError.new("Hy-Nomas resume bundle record byte count does not match descriptor for #{@name}")
        end
        @digest.hexfinal
      end

      def deactivate : Nil
        @active = false
      end
    end

    private class BoundedHeaderIO < IO
      @memory : IO::Memory
      @written = 0

      def initialize(@limit : Int32)
        @memory = IO::Memory.new(Math.min(@limit, IO_CHUNK_BYTES))
      end

      def read(slice : Bytes) : Int32
        @memory.read(slice)
      end

      def write(slice : Bytes) : Nil
        if slice.size > @limit - @written
          raise ArgumentError.new("Hy-Nomas resume bundle header size exceeds 1 MiB")
        end
        @memory.write(slice)
        @written += slice.size
      end

      def to_s : String
        @memory.to_s
      end
    end

    class Reader
      getter path : String
      getter sha256 : String

      @file : File
      @initial_file_info : File::Info
      @initial_path_info : File::Info
      @records : Hash(String, Entry) = {} of String => Entry
      @contract = ""
      @inventory = {} of String => Int64
      @closed = false
      @copying = false

      def initialize(@path : String, expected_sha256 : String, expected_contract : String,
                     expected_inventory : Hash(String, Int64))
        HyNomasResumeBundle.validate_sha256!(expected_sha256, "expected full-file")
        HyNomasResumeBundle.validate_contract!(expected_contract)
        expected_entries, _ = HyNomasResumeBundle.validate_inventory!(expected_inventory)
        @sha256 = expected_sha256

        @initial_path_info = HyNomasResumeBundle.regular_file_info!(@path, "input")
        @file = File.open(@path, "rb")
        @initial_file_info = @file.info
        begin
          unless HyNomasResumeBundle.same_snapshot?(@initial_path_info, @initial_file_info)
            raise ArgumentError.new("Hy-Nomas resume bundle input changed while opening")
          end
          verify_full_sha256!
          contract, records, total_bytes = begin
            parse_header!
          rescue ex : ArgumentError
            raise ex
          rescue
            raise ArgumentError.new("Hy-Nomas resume bundle header is malformed")
          end
          unless contract == expected_contract
            raise ArgumentError.new("Hy-Nomas resume bundle contract mismatch")
          end
          expected_map = expected_entries.to_h { |entry| {entry.name, entry.nbytes} }
          actual_map = records.to_h { |entry| {entry.name, entry.nbytes} }
          unless actual_map == expected_map
            raise ArgumentError.new("Hy-Nomas resume bundle inventory mismatch")
          end
          expected_size = HyNomasResumeBundle.checked_add!(PAYLOAD_OFFSET.to_i64,
            total_bytes, "file")
          unless @initial_file_info.size == expected_size
            raise ArgumentError.new("Hy-Nomas resume bundle file length mismatch")
          end
          @contract = contract
          @inventory = actual_map
          @records = records.to_h { |entry| {entry.name, entry} }
          unchanged!
        rescue ex
          @file.close unless @file.closed?
          @closed = true
          raise ex
        end
      end

      def close : Nil
        return if @closed
        @closed = true
        @file.close unless @file.closed?
      end

      def finalize
        close
      end

      # Rechecks the caller's setup-pinned identity against the identity that
      # was authenticated when this reader was admitted, and confirms the file
      # still has its admitted snapshot before any further record copies.
      def validate_contract!(expected_contract : String,
                             expected_inventory : Hash(String, Int64)) : Nil
        check_open!
        if @copying
          raise ArgumentError.new("Hy-Nomas resume bundle reader cannot validate during a copy")
        end
        HyNomasResumeBundle.validate_contract!(expected_contract)
        expected_entries, _ = HyNomasResumeBundle.validate_inventory!(expected_inventory)
        expected_map = expected_entries.to_h { |entry| {entry.name, entry.nbytes} }
        unless expected_contract == @contract && expected_map == @inventory
          raise ArgumentError.new("Hy-Nomas resume bundle contract or inventory mismatch")
        end
        unchanged!
      end

      # Copies one authenticated record directly into caller-owned, exact-size
      # destination bytes. Discard the destination if this method raises after
      # a partial read.
      def copy_to!(name : String, destination : Bytes) : Nil
        check_open!
        if @copying
          raise ArgumentError.new("Hy-Nomas resume bundle reader does not allow concurrent copies")
        end
        validate_contract!(@contract, @inventory)
        entry = @records[name]?
        raise ArgumentError.new("Hy-Nomas resume bundle unknown record #{name}") unless entry
        unless destination.size.to_i64 == entry.nbytes
          raise ArgumentError.new("Hy-Nomas resume bundle destination size mismatch for #{name}")
        end

        unchanged!
        @copying = true
        begin
          @file.seek(HyNomasResumeBundle::PAYLOAD_OFFSET.to_i64 + entry.offset)
          digest = Digest::SHA256.new
          chunk = Bytes.new(IO_CHUNK_BYTES)
          written = 0
          while written.to_i64 < entry.nbytes
            wanted = Math.min(chunk.size.to_i64, entry.nbytes - written.to_i64).to_i
            count = @file.read(chunk[0, wanted])
            if count == 0
              raise ArgumentError.new("Truncated Hy-Nomas resume bundle record #{name}")
            end
            part = chunk[0, count]
            destination[written, count].copy_from(part)
            digest.update(part)
            written += count
          end
          unchanged!
          unless digest.hexfinal == entry.sha256
            raise ArgumentError.new("Hy-Nomas resume bundle record SHA-256 mismatch for #{name}")
          end
        ensure
          @copying = false
        end
      end

      private def verify_full_sha256! : Nil
        digest = Digest::SHA256.new
        @file.rewind
        chunk = Bytes.new(IO_CHUNK_BYTES)
        while (count = @file.read(chunk)) > 0
          digest.update(chunk[0, count])
        end
        unless digest.hexfinal == @sha256
          raise ArgumentError.new("Hy-Nomas resume bundle full-file SHA-256 mismatch")
        end
        unchanged!
      end

      private def parse_header! : Tuple(String, Array(Entry), Int64)
        @file.rewind
        magic = Bytes.new(MAGIC.bytesize)
        read_fully!(magic, "magic")
        unless String.new(magic) == MAGIC
          raise ArgumentError.new("Hy-Nomas resume bundle magic mismatch")
        end
        length_bytes = Bytes.new(sizeof(UInt32))
        read_fully!(length_bytes, "header length")
        header_length = IO::ByteFormat::LittleEndian.decode(UInt32, length_bytes).to_i
        unless header_length > 0 && header_length <= MAX_HEADER_BYTES
          raise ArgumentError.new("Hy-Nomas resume bundle header size is invalid")
        end
        if @initial_file_info.size < PAYLOAD_OFFSET
          raise ArgumentError.new("Truncated Hy-Nomas resume bundle header area")
        end

        header_bytes = Bytes.new(header_length)
        read_fully!(header_bytes, "header")
        raw_header = String.new(header_bytes)
        padding_remaining = MAX_HEADER_BYTES - header_length
        padding = Bytes.new(IO_CHUNK_BYTES)
        while padding_remaining > 0
          count = Math.min(padding_remaining, padding.size)
          part = padding[0, count]
          read_fully!(part, "header padding")
          unless part.all?(&.== 0_u8)
            raise ArgumentError.new("Hy-Nomas resume bundle header padding is not zero")
          end
          padding_remaining -= count
        end

        document = begin
          JSON.parse(raw_header).as_h
        rescue ex
          raise ArgumentError.new("Hy-Nomas resume bundle header JSON is invalid")
        end
        exact_keys!(document, Set{"schema", "schema_version", "contract", "records"}, "header")
        unless document["schema"].as_s == SCHEMA && document["schema_version"].as_i == SCHEMA_VERSION
          raise ArgumentError.new("Hy-Nomas resume bundle schema mismatch")
        end
        contract = document["contract"].as_s
        HyNomasResumeBundle.validate_contract!(contract)
        record_docs = document["records"].as_a
        raise ArgumentError.new("Hy-Nomas resume bundle inventory is empty") if record_docs.empty?

        entries = [] of Entry
        seen = Set(String).new
        cursor = 0_i64
        record_docs.each do |record_doc|
          fields = record_doc.as_h
          exact_keys!(fields, Set{"name", "nbytes", "offset", "sha256"}, "record")
          name = fields["name"].as_s
          unless !name.empty? && name.valid_encoding? && !name.includes?('\0')
            raise ArgumentError.new("Hy-Nomas resume bundle record name is invalid")
          end
          raise ArgumentError.new("Hy-Nomas resume bundle duplicate record #{name}") unless seen.add?(name)
          nbytes = fields["nbytes"].as_i64
          offset = fields["offset"].as_i64
          unless nbytes >= 0 && nbytes < Int32::MAX && offset == cursor
            raise ArgumentError.new("Hy-Nomas resume bundle record geometry mismatch for #{name}")
          end
          record_sha256 = fields["sha256"].as_s
          HyNomasResumeBundle.validate_sha256!(record_sha256, "record")
          entries << Entry.new(name, nbytes, offset, record_sha256)
          cursor = HyNomasResumeBundle.checked_add!(cursor, nbytes, "payload")
        end
        unless entries.map(&.name) == entries.map(&.name).sort
          raise ArgumentError.new("Hy-Nomas resume bundle records are not in canonical order")
        end
        unless HyNomasResumeBundle.build_header(contract, entries) == raw_header
          raise ArgumentError.new("Hy-Nomas resume bundle header is not canonical")
        end
        {contract, entries, cursor}
      end

      private def exact_keys!(document : Hash(String, JSON::Any),
                              expected : Set(String), label : String) : Nil
        unless document.keys.to_set == expected
          raise ArgumentError.new("Hy-Nomas resume bundle #{label} fields mismatch")
        end
      end

      private def read_fully!(bytes : Bytes, label : String) : Nil
        @file.read_fully(bytes)
      rescue IO::EOFError
        raise ArgumentError.new("Truncated Hy-Nomas resume bundle #{label}")
      end

      private def unchanged! : Nil
        check_open!
        current_file = @file.info
        current_path = HyNomasResumeBundle.regular_file_info!(@path, "input")
        unless HyNomasResumeBundle.same_snapshot?(@initial_file_info, current_file) &&
               HyNomasResumeBundle.same_snapshot?(@initial_path_info, current_path) &&
               current_file.same_file?(current_path)
          raise ArgumentError.new("Hy-Nomas resume bundle changed after admission")
        end
      end

      private def check_open! : Nil
        raise ArgumentError.new("Hy-Nomas resume bundle reader is closed") if @closed
      end
    end
  end
end
