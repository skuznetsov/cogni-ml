require "digest/sha256"
require "json"

module HyNomasReferenceAttentionProbe
  extend self

  SCHEMA            = "cogni_ml.hy_nomas_reference_attention_probe/v1"
  SOURCE_SCHEMA     = "cogni_ml.hy_nomas_reference_attention_source/v1"
  PRECISION         = "reference-bf16-attention"
  SCOPE             = "one native block-0 active-prefix attention operation using the same Q and physical first-T0 BF16 stored K/V; 4849 active rows; no padded Torch, Flash3, cache append, full upstream T0 handoff, rest block, other layers, T1, or generation parity"
  UPSTREAM_REVISION = "a31c0fe0bcc493be188fd37ddba4ce2f674becb4"
  NATIVE_REVISION   = {{ `git rev-parse HEAD`.stringify.strip }}
  TOKENS            = 4_849
  HEADS             =    16
  HEAD_DIM          =   128
  ELEMENTS          = TOKENS * HEADS * HEAD_DIM
  TENSOR_BYTES      = ELEMENTS * 2
  NAMED_BYTES       = ELEMENTS.to_i64 * 4_i64
  LAYOUT            = "[1,tokens,heads,head_dim]; image then valid text"
  TENSOR_NAMES      = ["attention.q_bf16", "attention.k_stored_bf16", "attention.v_stored_bf16", "attention.output_bf16"]
  FIXTURE_KEYS      = ["source_input_receipt_sha256", "source_input_artifact_sha256",
                       "source_history_receipt_sha256", "source_history_artifact_sha256"]

  def validate_options!(precision : String, source_path : String, source_sha256 : String,
                        capture_preqkv : Bool, preqkv_artifact : String, preqkv_receipt : String) : Nil
    if precision == PRECISION
      unless Path[source_path].absolute? && valid_sha256?(source_sha256)
        raise ArgumentError.new("reference-bf16-attention requires an absolute source attention receipt and its trusted SHA-256")
      end
      if capture_preqkv || !preqkv_artifact.empty? || !preqkv_receipt.empty?
        raise ArgumentError.new("reference-bf16-attention does not support pre-QKV companion captures")
      end
    elsif !source_path.empty? || !source_sha256.empty?
      raise ArgumentError.new("attention source receipt options require reference-bf16-attention")
    end
  end

  def source_fixtures!(path : String, expected_sha256 : String) : Hash(String, String)
    unless valid_sha256?(expected_sha256) && Path[path].absolute? && File.file?(path) && !File.symlink?(path)
      raise ArgumentError.new("source attention receipt must be a regular absolute file with a trusted digest")
    end
    raw = File.read(path)
    raise ArgumentError.new("source attention receipt exceeds 64 KiB") if raw.bytesize > 65_536
    unless Digest::SHA256.hexdigest(raw) == expected_sha256
      raise ArgumentError.new("source attention receipt SHA-256 mismatch")
    end
    fixture_fields!(JSON.parse(raw))
  end

  # This binds the requested comparison context. It does not attest source
  # execution: the strict Python comparator checks the source files and Q/K/V.
  def fixture_fields!(source : JSON::Any) : Hash(String, String)
    unless source["schema"].as_s == SOURCE_SCHEMA &&
           source["upstream_revision"].as_s == UPSTREAM_REVISION &&
           source["attention_executed"].as_bool &&
           !source["full_upstream_t0_preprocessing_parity"].as_bool &&
           !source["cache_append_executed"].as_bool &&
           source["device"].as_s == "mps" && source["cpu_fallback_disabled"].as_bool
      raise ArgumentError.new("source attention receipt has the wrong execution scope")
    end
    attention = source["attention"]
    unless attention["query_tokens"].as_i == TOKENS && attention["kv_tokens"].as_i == TOKENS &&
           attention["heads"].as_i == HEADS && attention["head_dim"].as_i == HEAD_DIM &&
           attention["input_dtype"].as_s == "BF16" && attention["output_dtype"].as_s == "BF16" &&
           attention["layout"].as_s == LAYOUT && attention["attention_mask"].as_s == "none" &&
           attention["dropout_p"].as_f == 0.0 && !attention["is_causal"].as_bool
      raise ArgumentError.new("source attention receipt has the wrong geometry or SDPA semantics")
    end
    fixtures = source["input_fixtures"].as_h
    unless fixtures.keys.sort == FIXTURE_KEYS.sort
      raise ArgumentError.new("source attention fixture inventory differs")
    end
    fixtures.transform_values do |value|
      hash = value.as_s
      raise ArgumentError.new("source attention fixture digest is malformed") unless valid_sha256?(hash)
      hash
    end
  end

  def header(fixtures : Hash(String, String)) : String
    validate_fixtures!(fixtures)
    body = JSON.build do |json|
      json.object do
        json.field("__metadata__") do
          json.object do
            json.field("format", SCHEMA)
            json.field("upstream_revision", UPSTREAM_REVISION)
            FIXTURE_KEYS.each { |key| json.field(key, fixtures[key]) }
          end
        end
        TENSOR_NAMES.each_with_index do |name, index|
          json.field(name) do
            json.object do
              json.field("dtype", "BF16")
              json.field("shape", [1, TOKENS, HEADS, HEAD_DIM])
              json.field("data_offsets", [index.to_i64 * TENSOR_BYTES, (index.to_i64 + 1) * TENSOR_BYTES])
            end
          end
        end
      end
    end
    body + (" " * ((8 - body.bytesize % 8) % 8))
  end

  def receipt_json(artifact_name : String, artifact_sha256 : String, artifact_bytes : Int64,
                   tensor_sha256s : Array(String), fixtures : Hash(String, String),
                   source_receipt_sha256 : String, named_bytes : Hash(String, Int64),
                   encode_allocations : Int64, encode_readbacks : Int64,
                   preterminal_waits : Int32, terminal_waits : Int32) : String
    validate_fixtures!(fixtures)
    expected_bytes = sizeof(UInt64).to_i64 + header(fixtures).bytesize + 4_i64 * TENSOR_BYTES
    unless File.basename(artifact_name) == artifact_name && !artifact_name.empty? &&
           valid_sha256?(artifact_sha256) && valid_sha256?(source_receipt_sha256) &&
           tensor_sha256s.size == 4 && tensor_sha256s.all? { |hash| valid_sha256?(hash) } &&
           artifact_bytes == expected_bytes
      raise ArgumentError.new("reference attention artifact identity differs")
    end
    unless encode_allocations == 0 && encode_readbacks == 0 && preterminal_waits == 0 && terminal_waits == 1
      raise ArgumentError.new("reference attention requires allocation/readback-free encode and one terminal drain")
    end
    unless named_bytes["reference_attention_buffers"]? == NAMED_BYTES &&
           named_bytes["total_named_bytes"]? == named_bytes.reject("total_named_bytes").values.sum
      raise ArgumentError.new("reference attention named-buffer accounting differs")
    end
    records = TENSOR_NAMES.map_with_index do |name, index|
      {name: name, dtype: "BF16", shape: [1, TOKENS, HEADS, HEAD_DIM],
       nbytes: TENSOR_BYTES, sha256: tensor_sha256s[index]}
    end
    {
      schema: SCHEMA, schema_version: 1, upstream_revision: UPSTREAM_REVISION,
      native_revision: NATIVE_REVISION, scope: SCOPE,
      full_upstream_t0_preprocessing_parity: false,
      attention_executed: true, cache_append_executed: false,
      device: "mps", attention_backend: "native_mps",
      input_fixtures: fixtures, source_attention_receipt_sha256: source_receipt_sha256,
      attention: {query_tokens: TOKENS, kv_tokens: TOKENS, heads: HEADS, head_dim: HEAD_DIM,
                  input_dtype: "BF16", output_dtype: "BF16", layout: LAYOUT,
                  attention_mask: "none", dropout_p: 0.0, is_causal: false},
      attention_inputs: records.first(3),
      tensor_artifact: {basename: artifact_name, sha256: artifact_sha256, bytes: artifact_bytes, tensors: records},
      named_buffer_accounting: named_bytes, encode_allocations: encode_allocations,
      encode_readbacks: encode_readbacks, preterminal_waits: preterminal_waits,
      terminal_waits: terminal_waits,
    }.to_pretty_json
  end

  private def validate_fixtures!(fixtures : Hash(String, String)) : Nil
    unless fixtures.keys.sort == FIXTURE_KEYS.sort && fixtures.values.all? { |hash| valid_sha256?(hash) }
      raise ArgumentError.new("reference attention fixture identities differ")
    end
  end

  private def valid_sha256?(value : String) : Bool
    /\A[0-9a-f]{64}\z/.matches?(value)
  end
end
