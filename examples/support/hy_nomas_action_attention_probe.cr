module HyNomasActionAttentionProbe
  UPSTREAM_REVISION      = "a31c0fe0bcc493be188fd37ddba4ce2f674becb4"
  UPSTREAM_SOURCE_SHA256 = "acf827bb7feb84c9046618737146544843a3f50c3500a3b16925dadeb6c5bdee"
  QUERY_ROWS             = 259_i32
  MASKED_CURRENT_ROWS    = 259_i32
  LEGACY_CURRENT_ROWS    = 257_i32
  HISTORY_ROWS           =   3_i32
  HEADS                  =   2_i32
  HEAD_DIM               =   4_i32
  OUTPUT_ELEMENTS        = QUERY_ROWS * HEADS * HEAD_DIM
  ACTIVE_OUTPUT_ELEMENTS = LEGACY_CURRENT_ROWS * HEADS * HEAD_DIM
  ATOL                   = 2.0e-5

  record Comparison, name : String, max_abs : Float64?, failures : Int32,
    first_bad_index : Int32?, finite : Bool

  class Fixture
    getter manifest_sha256 : String
    getter query : String
    getter current_key : String
    getter current_value : String
    getter legacy_current_key : String
    getter legacy_current_value : String
    getter history_key : String
    getter history_value : String
    getter expected_masked : Array(Float32)
    getter expected_legacy : Array(Float32)
    getter query_labels : Array(Int32)
    getter key_labels : Array(Int32)
    getter range_count : Int32

    def initialize(manifest_path : String, expected_manifest_sha256 : String,
                   oracle_source_path : String, upstream_source_path : String)
      unless expected_manifest_sha256.matches?(/\A[0-9a-fA-F]{64}\z/)
        raise ArgumentError.new("fixture manifest SHA-256 must be 64 hex characters")
      end
      raise ArgumentError.new("fixture manifest cannot be a symlink") if File.symlink?(manifest_path)
      raw_manifest = File.read(manifest_path)
      @manifest_sha256 = Digest::SHA256.hexdigest(raw_manifest)
      unless @manifest_sha256 == expected_manifest_sha256.downcase
        raise ArgumentError.new("fixture manifest digest mismatch")
      end

      document = JSON.parse(raw_manifest)
      raise ArgumentError.new("unexpected action-attention fixture schema") unless document["schema"].as_s == "cogni_ml.hy_nomas_action_attention_reference"
      raise ArgumentError.new("unexpected action-attention fixture version") unless document["schema_version"].as_i == 1

      upstream = document["upstream"]
      raise ArgumentError.new("upstream revision pin mismatch") unless upstream["revision"].as_s == UPSTREAM_REVISION
      raise ArgumentError.new("upstream source path pin mismatch") unless upstream["source_file"].as_s == "hymm/models/dit/nomas_cpt.py"
      raise ArgumentError.new("upstream range builder pin mismatch") unless upstream["range_builder"].as_s == "_build_intra_seg_cache_ranges"
      unless upstream["source_file_sha256"].as_s == UPSTREAM_SOURCE_SHA256
        raise ArgumentError.new("upstream source digest pin mismatch")
      end
      raise ArgumentError.new("upstream source cannot be a symlink") if File.symlink?(upstream_source_path)
      actual_upstream_sha = Digest::SHA256.hexdigest(File.read(upstream_source_path))
      unless actual_upstream_sha == UPSTREAM_SOURCE_SHA256
        raise ArgumentError.new("local upstream source does not match pinned nomas_cpt.py")
      end

      oracle = document["oracle"]
      raise ArgumentError.new("oracle must be CPU-only") unless oracle["device"].as_s == "cpu"
      raise ArgumentError.new("oracle must use F32") unless oracle["dtype"].as_s == "torch.float32"
      raise ArgumentError.new("oracle tolerance mismatch") unless oracle["atol"].as_f == ATOL
      raise ArgumentError.new("oracle source cannot be a symlink") if File.symlink?(oracle_source_path)
      actual_oracle_sha = Digest::SHA256.hexdigest(File.read(oracle_source_path))
      unless actual_oracle_sha == oracle["script_sha256"].as_s
        raise ArgumentError.new("oracle source does not match authenticated fixture metadata")
      end

      geometry = document["geometry"]
      expected_geometry = {
        "batch"            => 1_i64,
        "history_rows"     => HISTORY_ROWS.to_i64,
        "image_rows"       => 250_i64,
        "valid_text_rows"  => 7_i64,
        "padded_text_rows" => 2_i64,
        "query_rows"       => QUERY_ROWS.to_i64,
        "current_rows"     => MASKED_CURRENT_ROWS.to_i64,
        "key_rows"         => (HISTORY_ROWS + MASKED_CURRENT_ROWS).to_i64,
        "heads"            => HEADS.to_i64,
        "head_dim"         => HEAD_DIM.to_i64,
      }
      expected_geometry.each do |key, expected|
        raise ArgumentError.new("fixture geometry mismatch for #{key}") unless geometry[key].as_i == expected
      end

      mask = document["mask"]
      @query_labels = mask["query_labels"].as_a.map(&.as_i.to_i32)
      @key_labels = mask["key_labels"].as_a.map(&.as_i.to_i32)
      unless @query_labels.size == QUERY_ROWS && @key_labels.size == HISTORY_ROWS + MASKED_CURRENT_ROWS
        raise ArgumentError.new("fixture label vector length mismatch")
      end
      unless @query_labels.all? { |label| label >= -1 } && @key_labels.all? { |label| label >= -1 }
        raise ArgumentError.new("fixture contains unsupported negative labels")
      end
      unless @query_labels[-2, 2] == [-1_i32, -1_i32] && @key_labels[-2, 2] == [-1_i32, -1_i32]
        raise ArgumentError.new("fixture padding labels are not hidden")
      end
      @range_count = mask["q_ranges"].as_a.size.to_i32
      raise ArgumentError.new("fixture must retain multiple source ranges") unless @range_count >= 2

      directory = File.dirname(File.expand_path(manifest_path))
      tensors = document["tensors"]
      @query = read_tensor(directory, tensors["query"], OUTPUT_ELEMENTS.to_i64)
      @current_key = read_tensor(directory, tensors["current_key"], OUTPUT_ELEMENTS.to_i64)
      @current_value = read_tensor(directory, tensors["current_value"], OUTPUT_ELEMENTS.to_i64)
      @history_key = read_tensor(directory, tensors["history_key"], HISTORY_ROWS.to_i64 * HEADS * HEAD_DIM)
      @history_value = read_tensor(directory, tensors["history_value"], HISTORY_ROWS.to_i64 * HEADS * HEAD_DIM)
      @expected_masked = decode_f32(read_tensor(directory, tensors["expected_masked"], OUTPUT_ELEMENTS.to_i64))
      @expected_legacy = decode_f32(read_tensor(directory, tensors["expected_legacy"], OUTPUT_ELEMENTS.to_i64))

      # Legacy attention is run on the compact, visible current rows. The first
      # visible current K/V rows are image plus valid text; the last two are padding.
      legacy_bytes = LEGACY_CURRENT_ROWS.to_i64 * HEADS * HEAD_DIM * sizeof(Float32)
      @legacy_current_key = @current_key.byte_slice(0, legacy_bytes.to_i)
      @legacy_current_value = @current_value.byte_slice(0, legacy_bytes.to_i)
    end

    private def read_tensor(directory : String, descriptor : JSON::Any,
                            expected_elements : Int64) : String
      raise ArgumentError.new("fixture tensor dtype must be F32") unless descriptor["dtype"].as_s == "F32"
      raise ArgumentError.new("fixture tensor byte order must be little-endian") unless descriptor["byte_order"].as_s == "little"
      filename = descriptor["file"].as_s
      unless File.basename(filename) == filename && filename != "." && filename != ".."
        raise ArgumentError.new("fixture tensor path must be a basename")
      end
      path = File.join(directory, filename)
      raise ArgumentError.new("fixture tensor cannot be a symlink") if File.symlink?(path)
      raw = File.read(path)
      expected_bytes = expected_elements * sizeof(Float32)
      raise ArgumentError.new("fixture tensor byte count mismatch for #{filename}") unless raw.bytesize.to_i64 == expected_bytes && descriptor["nbytes"].as_i == expected_bytes
      unless Digest::SHA256.hexdigest(raw) == descriptor["sha256"].as_s
        raise ArgumentError.new("fixture tensor digest mismatch for #{filename}")
      end
      shape = descriptor["shape"].as_a.map(&.as_i)
      unless shape.size == 4 && shape[0] == 1_i64 && shape[2] == HEADS.to_i64 && shape[3] == HEAD_DIM.to_i64 &&
             shape[1] * shape[2] * shape[3] == expected_elements
        raise ArgumentError.new("fixture tensor shape mismatch for #{filename}")
      end
      raw
    end

    private def decode_f32(raw : String) : Array(Float32)
      unless raw.bytesize % sizeof(Float32) == 0
        raise ArgumentError.new("fixture F32 sidecar length is not aligned")
      end
      values = raw.to_slice.to_unsafe.as(Pointer(Float32))
      Array(Float32).new(raw.bytesize // sizeof(Float32)) { |index| values[index] }
    end
  end

  def self.upload_f32(raw : String) : ML::MetalBuffer
    buffer = ML::MetalBuffer.new(raw.bytesize.to_i64)
    buffer.write_bytes(raw.to_unsafe.as(Pointer(UInt8)), raw.bytesize)
    buffer
  end

  def self.output_buffer : ML::MetalBuffer
    ML::MetalBuffer.new(OUTPUT_ELEMENTS.to_i64 * sizeof(Float32))
  end

  def self.compare(name : String, actual : Array(Float32), expected : Array(Float32)) : Comparison
    raise ArgumentError.new("comparison vector length mismatch: #{name}") unless actual.size == expected.size
    max_abs = 0.0_f64
    finite = true
    failures = 0_i32
    first_bad = nil.as(Int32?)
    actual.each_with_index do |value, index|
      reference = expected[index]
      unless value.finite? && reference.finite?
        finite = false
        failures += 1
        first_bad ||= index.to_i32
        next
      end
      error = (value.to_f64 - reference.to_f64).abs
      max_abs = Math.max(max_abs, error)
      if error > ATOL
        failures += 1
        first_bad ||= index.to_i32
      end
    end
    Comparison.new(name, finite ? max_abs : nil, failures, first_bad, finite)
  end

  def self.compare_json(json : JSON::Builder, rows : Array(Comparison)) : Nil
    json.array do
      rows.each do |row|
        json.object do
          json.field "name", row.name
          json.field "max_abs", row.max_abs
          json.field "failures", row.failures
          json.field "first_bad_index", row.first_bad_index
          json.field "finite", row.finite
        end
      end
    end
  end
end
