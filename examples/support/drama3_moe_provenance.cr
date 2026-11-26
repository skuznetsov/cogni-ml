require "digest/sha256"
require "json"
require "../../src/ml/audio/drama3/native"

module Drama3MoeProvenance
  TARGET_LAYER  = 39_i32
  TARGET_PREFIX = "backbone.layers.#{TARGET_LAYER}.mlp."

  alias Record = NamedTuple(
    target_layer: Int32,
    target_moe_weights_sha256: String,
    tensor_count: Int32,
    model_path: String,
    config_sha256: String,
    index_sha256: String,
    verified_before_gpu: Bool,
  )

  # Hashes the actual mapped raw tensor payloads retained by WeightStore, in
  # the same shard/name order and byte format as replay_mlx_moe.py.
  def self.verify!(model_path : String, store : ML::Audio::Drama3::WeightStore, expected_source_hashes : Hash(String, JSON::Any)) : Record
    expanded_path = File.expand_path(model_path)
    config_path = File.join(expanded_path, "config.json")
    index_path = File.join(expanded_path, "model.safetensors.index.json")
    config_bytes = File.read(config_path)
    index_bytes = File.read(index_path)
    config_sha256 = Digest::SHA256.hexdigest(config_bytes)
    index_sha256 = Digest::SHA256.hexdigest(index_bytes)

    verify_expected_hash!(expected_source_hashes, "config.json", config_sha256)
    verify_expected_hash!(expected_source_hashes, "model.safetensors.index.json", index_sha256)
    expected_weights_sha256 = required_hash(expected_source_hashes, "target_moe_weights")

    weight_map = JSON.parse(index_bytes)["weight_map"].as_h
    by_shard = Hash(String, Array(String)).new { |hash, shard| hash[shard] = [] of String }
    weight_map.each do |name, shard_value|
      next unless name.starts_with?(TARGET_PREFIX)
      shard = shard_value.as_s
      unless !shard.empty? && File.basename(shard) == shard && shard != "." && shard != ".."
        raise ArgumentError.new("target MoE index contains an invalid shard path for #{name}")
      end
      by_shard[shard] << name
    end
    raise ArgumentError.new("model index has no tensors under #{TARGET_PREFIX}") if by_shard.empty?

    digest = Digest::SHA256.new
    tensor_count = 0_i32
    by_shard.keys.sort.each do |shard|
      by_shard[shard].sort.each do |name|
        tensor = store.tensor(name)
        raw = tensor.raw_bytes || raise ArgumentError.new("target MoE tensor is not backed by raw mapped bytes: #{name}")
        torch_dtype = torch_dtype(tensor.dtype)

        digest.update("#{name}\0")
        digest.update("#{torch_dtype}\0")
        shape_bytes = IO::Memory.new
        tensor.shape.each { |dimension| shape_bytes.write_bytes(dimension.to_i64, IO::ByteFormat::LittleEndian) }
        digest.update(shape_bytes.to_slice)
        digest.update(raw)
        tensor_count += 1
      end
    end

    target_weights_sha256 = digest.hexfinal
    unless target_weights_sha256 == expected_weights_sha256
      raise ArgumentError.new("target MoE weight digest mismatch: expected #{expected_weights_sha256}, found #{target_weights_sha256}")
    end

    {
      target_layer:              TARGET_LAYER,
      target_moe_weights_sha256: target_weights_sha256,
      tensor_count:              tensor_count,
      model_path:                expanded_path,
      config_sha256:             config_sha256,
      index_sha256:              index_sha256,
      verified_before_gpu:       true,
    }
  end

  private def self.required_hash(source_hashes : Hash(String, JSON::Any), key : String) : String
    value = source_hashes[key]?.try(&.as_s) || raise ArgumentError.new("input fixture is missing source hash #{key}")
    raise ArgumentError.new("input fixture source hash #{key} is malformed") unless value.matches?(/\A[0-9a-f]{64}\z/)
    value
  end

  private def self.verify_expected_hash!(source_hashes : Hash(String, JSON::Any), key : String, actual : String) : Nil
    expected = required_hash(source_hashes, key)
    raise ArgumentError.new("model #{key} digest mismatch: expected #{expected}, found #{actual}") unless expected == actual
  end

  private def self.torch_dtype(dtype : String) : String
    case dtype
    when "BOOL" then "bool"
    when "F32"  then "float32"
    when "BF16" then "bfloat16"
    when "F16"  then "float16"
    when "U32"  then "uint32"
    when "I32"  then "int32"
    else             raise ArgumentError.new("unsupported raw target MoE tensor dtype #{dtype}")
    end
  end
end
