require "http/client"

module ML::Energy
  # Minimal Prometheus Remote Write v1 sender. It intentionally supports only
  # loopback HTTP and the protobuf/Snappy subset needed by bounded experiments.
  class PrometheusRemoteWrite
    USER_AGENT = "cogni-ml/0.40.0"
    MAX_ATTEMPTS = 3
    ENDPOINT = "/api/v1/write"

    class WriteError < Exception
    end

    struct Sample
      getter timestamp_ms : Int64
      getter value : Float64

      def initialize(@timestamp_ms : Int64, value : Float64)
        @value = value
        raise ArgumentError.new("sample timestamp must be nonnegative") if @timestamp_ms < 0
        raise ArgumentError.new("sample values must be finite") unless @value.finite?
      end
    end

    class Series
      getter metric : String
      getter labels : Array(Tuple(String, String))
      getter samples : Array(Sample)

      def initialize(@metric : String, labels : Hash(String, String), samples : Array(Sample))
        unless PrometheusRemoteWrite.valid_metric_name?(@metric)
          raise ArgumentError.new("invalid Prometheus metric name")
        end
        raise ArgumentError.new("remote-write series must contain samples") if samples.empty?
        normalized = [] of Tuple(String, String)
        labels.each do |name, value|
          raise ArgumentError.new("__name__ is reserved for the metric name") if name == "__name__"
          unless PrometheusRemoteWrite.valid_label_name?(name)
            raise ArgumentError.new("invalid Prometheus label name")
          end
          if value.empty? || value.bytesize > 256
            raise ArgumentError.new("label values must be nonempty and at most 256 bytes")
          end
          normalized << {name, value}
        end
        normalized << {"__name__", @metric}
        @labels = normalized.sort_by(&.[0])
        @samples = samples.dup
        previous = nil.as(Int64?)
        @samples.each do |sample|
          if timestamp = previous
            raise ArgumentError.new("sample timestamps must be strictly increasing") unless sample.timestamp_ms > timestamp
          end
          previous = sample.timestamp_ms
        end
      end

      def selector : String
        label_text = @labels.map do |name, value|
          %(#{name}="#{PrometheusRemoteWrite.escape_label_value(value)}")
        end.join(",")
        "#{@metric}{#{label_text}}"
      end
    end

    struct DecodedSeries
      getter labels : Array(Tuple(String, String))
      getter samples : Array(Tuple(Int64, Float64))

      def initialize(@labels : Array(Tuple(String, String)), @samples : Array(Tuple(Int64, Float64)))
      end
    end

    getter endpoint : URI
    getter retry_delay : Time::Span

    def initialize(base_url : String, @retry_delay : Time::Span = 100.milliseconds)
      uri = URI.parse(base_url)
      unless uri.scheme == "http" && {"127.0.0.1", "localhost", "::1", "[::1]"}.includes?(uri.host) && uri.user.nil? && uri.password.nil? && uri.query.nil? && uri.fragment.nil?
        raise ArgumentError.new("remote-write URL must be loopback HTTP without credentials, query, or fragment")
      end
      uri.path = uri.path.rstrip('/') + ENDPOINT
      @endpoint = uri
      raise ArgumentError.new("retry delay must be nonnegative") if @retry_delay < 0.seconds
    end

    def write(series : Array(Series)) : Int32
      raise ArgumentError.new("remote-write request must contain series") if series.empty?
      body = self.class.encode(series)
      attempts = 0
      loop do
        attempts += 1
        begin
          response = HTTP::Client.new(@endpoint) do |client|
            client.connect_timeout = 3.seconds
            client.read_timeout = 5.seconds
            headers = HTTP::Headers{
              "Content-Encoding" => "snappy",
              "Content-Type" => "application/x-protobuf",
              "User-Agent" => USER_AGENT,
              "X-Prometheus-Remote-Write-Version" => "0.1.0",
            }
            client.post(@endpoint.request_target, headers, body)
          end
          status = response.status_code
          return status if status.in?(200..299)

          retryable = status == 429 || status.in?(500..599)
          if retryable && attempts < MAX_ATTEMPTS
            sleep @retry_delay * attempts
            next
          end
          raise WriteError.new("remote-write HTTP #{status} after #{attempts} attempt(s)")
        rescue ex : IO::Error | Socket::Error
          if attempts < MAX_ATTEMPTS
            sleep @retry_delay * attempts
            next
          end
          raise WriteError.new("remote-write transport failed after #{attempts} attempts: #{ex.message || ex.class.name}")
        end
      end
    end

    def self.encode(series : Array(Series)) : Bytes
      raw_snappy(encode_protobuf(series))
    end

    def self.encode_protobuf(series : Array(Series)) : Bytes
      request = IO::Memory.new
      series.each do |time_series|
        message = IO::Memory.new
        time_series.labels.each do |name, value|
          label = IO::Memory.new
          write_string_field(label, 1, name)
          write_string_field(label, 2, value)
          write_bytes_field(message, 1, label.to_slice)
        end
        time_series.samples.each do |sample|
          encoded_sample = IO::Memory.new
          write_key(encoded_sample, 1, 1)
          encoded_sample.write_bytes(sample.value, IO::ByteFormat::LittleEndian)
          write_key(encoded_sample, 2, 0)
          write_varint(encoded_sample, sample.timestamp_ms.to_u64)
          write_bytes_field(message, 2, encoded_sample.to_slice)
        end
        write_bytes_field(request, 1, message.to_slice)
      end
      request.to_slice
    end

    # Raw Snappy block encoding with literal chunks. It is intentionally
    # dependency-free and valid for all protobuf request sizes used here.
    def self.raw_snappy(input : Bytes) : Bytes
      output = IO::Memory.new
      write_varint(output, input.size.to_u64)
      offset = 0
      while offset < input.size
        remaining = input.size - offset
        literal_length = remaining
        length_minus_one = literal_length - 1
        if length_minus_one < 60
          output.write_byte((length_minus_one << 2).to_u8)
        else
          width = 1
          probe = length_minus_one
          while probe > 0xff
            width += 1
            probe >>= 8
          end
          output.write_byte(((59 + width) << 2).to_u8)
          width.times do |index|
            output.write_byte(((length_minus_one >> (index * 8)) & 0xff).to_u8)
          end
        end
        output.write(input[offset, literal_length])
        offset += literal_length
      end
      output.to_slice
    end

    # Small decoder exposed for deterministic wire-format specs and diagnostics.
    def self.decode_raw_snappy(input : Bytes) : Bytes
      cursor = 0
      expected, cursor = read_varint(input, cursor)
      output = IO::Memory.new
      while cursor < input.size
        tag = input[cursor].to_i
        cursor += 1
        kind = tag & 0x3
        raise ArgumentError.new("wire spec decoder only accepts literal Snappy chunks") unless kind == 0
        length_code = tag >> 2
        if length_code < 60
          length = length_code + 1
        else
          width = length_code - 59
          raise ArgumentError.new("invalid Snappy literal length") if width > 4 || cursor + width > input.size
          length_minus_one = 0_u64
          width.times do |index|
            length_minus_one |= input[cursor + index].to_u64 << (index * 8)
          end
          cursor += width
          length = length_minus_one.to_i + 1
        end
        raise ArgumentError.new("truncated Snappy literal") if cursor + length > input.size
        output.write(input[cursor, length])
        cursor += length
      end
      decoded = output.to_slice
      raise ArgumentError.new("Snappy decoded size mismatch") unless decoded.size == expected
      decoded
    end

    def self.decode_write_request(input : Bytes) : Array(DecodedSeries)
      decoded = [] of DecodedSeries
      cursor = 0
      while cursor < input.size
        key, cursor = read_varint(input, cursor)
        field = (key >> 3).to_i
        wire = (key & 0x7).to_i
        raise ArgumentError.new("unexpected WriteRequest protobuf field") unless field == 1 && wire == 2
        payload, cursor = read_length_delimited(input, cursor)
        decoded << decode_time_series(payload)
      end
      decoded
    end

    def self.valid_metric_name?(name : String) : Bool
      !!(name =~ /\A[a-zA-Z_:][a-zA-Z0-9_:]*\z/)
    end

    def self.valid_label_name?(name : String) : Bool
      !!(name =~ /\A[a-zA-Z_][a-zA-Z0-9_]*\z/)
    end

    def self.escape_label_value(value : String) : String
      value.gsub("\\", "\\\\").gsub("\n", "\\n").gsub("\"", "\\\"")
    end

    private def self.write_key(io : IO, field : Int32, wire_type : Int32) : Nil
      write_varint(io, ((field << 3) | wire_type).to_u64)
    end

    private def self.write_string_field(io : IO, field : Int32, value : String) : Nil
      write_bytes_field(io, field, value.to_slice)
    end

    private def self.write_bytes_field(io : IO, field : Int32, value : Bytes) : Nil
      write_key(io, field, 2)
      write_varint(io, value.size.to_u64)
      io.write(value)
    end

    private def self.write_varint(io : IO, value : UInt64) : Nil
      remaining = value
      while remaining >= 0x80_u64
        io.write_byte(((remaining & 0x7f_u64) | 0x80_u64).to_u8)
        remaining >>= 7
      end
      io.write_byte(remaining.to_u8)
    end

    private def self.read_varint(bytes : Bytes, cursor : Int32) : Tuple(UInt64, Int32)
      value = 0_u64
      shift = 0
      index = cursor
      while index < bytes.size && shift < 64
        byte = bytes[index].to_u64
        value |= (byte & 0x7f_u64) << shift
        index += 1
        return {value, index} if (byte & 0x80_u64) == 0
        shift += 7
      end
      raise ArgumentError.new("truncated or invalid protobuf varint")
    end

    private def self.read_length_delimited(bytes : Bytes, cursor : Int32) : Tuple(Bytes, Int32)
      length, offset = read_varint(bytes, cursor)
      count = length.to_i
      raise ArgumentError.new("truncated protobuf message") if count < 0 || offset + count > bytes.size
      {bytes[offset, count], offset + count}
    end

    private def self.decode_time_series(input : Bytes) : DecodedSeries
      labels = [] of Tuple(String, String)
      samples = [] of Tuple(Int64, Float64)
      cursor = 0
      while cursor < input.size
        key, cursor = read_varint(input, cursor)
        field = (key >> 3).to_i
        wire = (key & 0x7).to_i
        case {field, wire}
        when {1, 2}
          label_bytes, cursor = read_length_delimited(input, cursor)
          labels << decode_label(label_bytes)
        when {2, 2}
          sample_bytes, cursor = read_length_delimited(input, cursor)
          samples << decode_sample(sample_bytes)
        else
          raise ArgumentError.new("unexpected TimeSeries protobuf field")
        end
      end
      DecodedSeries.new(labels, samples)
    end

    private def self.decode_label(input : Bytes) : Tuple(String, String)
      name = ""
      value = ""
      cursor = 0
      while cursor < input.size
        key, cursor = read_varint(input, cursor)
        field = (key >> 3).to_i
        wire = (key & 0x7).to_i
        raise ArgumentError.new("unexpected Label protobuf field") unless {1, 2}.includes?(field) && wire == 2
        bytes, cursor = read_length_delimited(input, cursor)
        if field == 1
          name = String.new(bytes)
        else
          value = String.new(bytes)
        end
      end
      {name, value}
    end

    private def self.decode_sample(input : Bytes) : Tuple(Int64, Float64)
      timestamp = nil.as(Int64?)
      value = nil.as(Float64?)
      cursor = 0
      while cursor < input.size
        key, cursor = read_varint(input, cursor)
        field = (key >> 3).to_i
        wire = (key & 0x7).to_i
        case {field, wire}
        when {1, 1}
          raise ArgumentError.new("truncated Sample value") if cursor + 8 > input.size
          value = IO::ByteFormat::LittleEndian.decode(Float64, input[cursor, 8])
          cursor += 8
        when {2, 0}
          raw, cursor = read_varint(input, cursor)
          timestamp = raw.to_i64
        else
          raise ArgumentError.new("unexpected Sample protobuf field")
        end
      end
      {timestamp || raise(ArgumentError.new("missing Sample timestamp")), value || raise(ArgumentError.new("missing Sample value"))}
    end
  end
end
