require "digest/sha256"
require "json"

# Setup-only admission for the deliberately partial, camera/action-off adapter.
# Cache hashes bind the actual residents; source identity denotes base ancestry.
module HyNomasCinematic
  extend self

  SCHEMA            = "cogni_ml.hy_nomas_cinematic_bf16_cache/v1"
  BASE_INDEX_SHA    = "a235ca592b47161472c7fd94cd3d886fce164480cd0571179c8860002063cbb0"
  BASE_SOURCE_SHA   = "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70"
  UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
  ADAPTER_REVISION  = "bb6a8ca451bfe84947f0ea1c6d51957f983ffffe"
  ADAPTER_SHA       = "765426a6c417eed70a57d72f67cfe002076cfc92b6b35fe185091012ba80c7c2"
  CONFIG_SHA        = "f656c2866953e7a90a785a5c51ce127a79ea61c4217ef30e3cd2d69afcd3d998"

  record IndexIdentity, path : String, sha256 : String,
    lineage_path : String, lineage_sha256 : String

  def load!(path : String, sha : String, *, bf16 : Bool,
            camera_conditioning : Bool, action_conditioning : Bool) : IndexIdentity?
    return nil if path.empty? && sha.empty?
    raise ArgumentError.new("cinematic cache requires both path and SHA") if path.empty? || sha.empty?
    unless bf16 && !camera_conditioning && !action_conditioning
      raise ArgumentError.new("cinematic cache requires BF16 DiT and disabled camera/action conditioning")
    end
    raise ArgumentError.new("cinematic lineage SHA is malformed") unless /\A[0-9a-f]{64}\z/.matches?(sha)
    unless Path[path].absolute? && File.file?(path) && !File.symlink?(path) && File.size(path).in?(1..2_097_152)
      raise ArgumentError.new("cinematic lineage must be a bounded regular absolute file")
    end
    info = File.info(path)
    bytes = File.read(path)
    after = File.info(path)
    unless Digest::SHA256.hexdigest(bytes) == sha && info.same_file?(after) &&
           info.size == after.size && info.modification_time == after.modification_time
      raise ArgumentError.new("cinematic lineage SHA or file changed")
    end
    index_sha = validate_document!(JSON.parse(bytes))
    index_path = File.join(File.dirname(File.realpath(path)), "index.json")
    unless File.file?(index_path) && !File.symlink?(index_path)
      raise ArgumentError.new("cinematic index must be a regular sibling")
    end
    # Existing Admission hashes the index, all Q4 inventory and every BF16
    # control before allocation, then refreshes file snapshots at consumption.
    IndexIdentity.new(index_path, index_sha, File.realpath(path), sha)
  end

  def validate_document!(doc : JSON::Any) : String
    begin
      unless doc.as_h.keys.sort == %w(adapter base_index_sha256 base_source_sha256 index merge producer schema upstream_revision).sort &&
             doc["schema"].as_s == SCHEMA && doc["upstream_revision"].as_s == UPSTREAM_REVISION &&
             doc["base_index_sha256"].as_s == BASE_INDEX_SHA && doc["base_source_sha256"].as_s == BASE_SOURCE_SHA
        raise ArgumentError.new("cinematic cache base/schema provenance mismatch")
      end
      adapter = doc["adapter"]
      unless adapter.as_h.keys.sort == %w(alpha bytes config_sha256 path rank repo revision sha256).sort &&
             adapter["repo"].as_s == "tencent/Hy-Nomas" && adapter["revision"].as_s == ADAPTER_REVISION &&
             adapter["path"].as_s == "DiT/lora_weights/cinematic/adapter.safetensors" &&
             adapter["sha256"].as_s == ADAPTER_SHA && adapter["bytes"].as_i64 == 6_122_591_632_i64 &&
             adapter["rank"].as_i == 256 && adapter["alpha"].as_f == 512.0 &&
             adapter["config_sha256"].as_s == CONFIG_SHA
        raise ArgumentError.new("cinematic adapter identity/rank/scale mismatch")
      end
      merge = doc["merge"]
      unless merge.as_h.keys.sort == %w(action_conditioning active_targets formula skipped_action_targets).sort &&
             merge["formula"].as_s == "bf16(base_f32 + (B_f32 @ A_f32) * 2)" &&
             merge["active_targets"].as_i == 756 && merge["skipped_action_targets"].as_i == 162 &&
             !merge["action_conditioning"].as_bool
        raise ArgumentError.new("cinematic cache merge precision or partial-target scope mismatch")
      end
      index = doc["index"]
      unless index.as_h.keys.sort == %w(basename sha256) && index["basename"].as_s == "index.json" &&
             /\A[0-9a-f]{64}\z/.matches?(index["sha256"].as_s) && doc["producer"].as_h.size > 0
        raise ArgumentError.new("cinematic index identity mismatch")
      end
      index["sha256"].as_s
    rescue ex : TypeCastError | KeyError
      raise ArgumentError.new("cinematic cache lineage has a missing or malformed field")
    end
  end
end
