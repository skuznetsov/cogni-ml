require "json"
require "digest/sha256"
require "./hy_nomas_generation"

module HyNomasGeneration
  # Continuation banks intentionally use a separate record type from the
  # legacy first-segment fixture: decoder.noise is larger than its 1 GiB cap.
  # The record still admits only the producer's exact, bounded inventory.
  struct TimelineRecord
    getter name : String
    getter dtype : String
    getter shape : Array(Int32)
    getter offset : Int64
    getter nbytes : Int64
    getter sha256 : String

    def initialize(value : JSON::Any, payload_bytes : Int64, expected_dtype : String,
                   expected_shape : Array(Int32), expected_offset : Int64,
                   expected_nbytes : Int64, name : String)
      @name = name
      @dtype = value["dtype"].as_s
      @shape = value["shape"].as_a.map(&.as_i)
      @offset = value["offset"].as_i64
      @nbytes = value["nbytes"].as_i64
      @sha256 = value["sha256"].as_s
      unless @dtype == expected_dtype && @shape == expected_shape &&
             @offset == expected_offset && @nbytes == expected_nbytes &&
             @offset >= 0 && @nbytes > 0 && @offset <= payload_bytes &&
             @nbytes <= payload_bytes - @offset && PixelUMMBlockProbe.valid_digest?(@sha256)
        raise ArgumentError.new("continuation tensor record mismatch: #{name}")
      end
    end
  end

  class TimelineInputs
    GIB             = 1024_i64**3
    SCHEMA          = "cogni_ml.hy_nomas_continuation_inputs/v1"
    SCHEMA_VERSION  = 1
    PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
    SOURCE_SHA256   = {
      "configs/experiment/inference.yaml"               => "d4db57b63597f98d4bddddd98355198e4e8c25da366d2e27bc7e74a79d26ec67",
      "hymm/models/dit/modules/posemb_layers.py"        => "27b729d04185dfd7d65ecf9ba0d761d4864153d3e656908997d60a6971afc1aa",
      "hymm/models/vae/pixelumm_decoder/autoencoder.py" => "65e0d77b9f2e9a3ca457d6a51bfdb239f7b9dc3fc8a0ad014025c36d6be9f0cf",
      "hymm/models/vae/pixelumm_decoder/model.py"       => "bea01a8490f8e9a4076da83d4f9fbb3cba7bd6701df5c28fb2e2c04f00131eec",
      "hymm/pipeline/causal_ar.py"                      => "f9aae49b14becb87d958665f763e510c28eeaaad9292f00ac511d1146202949f",
    }
    ROPE_OFFSETS = [0, 30, 60, 90, 120, 150]
    TENSOR_SPECS = {
      "input.latent_bank" => {"F32", [1, 32, 180, 9, 16]},
      "input.qwen_bf16"   => {"BF16", [1, 1280, 3584]},
      "input.rope_cos"    => {"F32", [6, 4320, 128]},
      "input.rope_sin"    => {"F32", [6, 4320, 128]},
      "cache.eps_bf16"    => {"BF16", [6, 1, 32, 30, 9, 16]},
      "decoder.noise"     => {"F32", [1, 3, 720, 288, 512]},
      "decoder.rope_cos"  => {"F32", [1, 5760, 1, 128]},
      "decoder.rope_sin"  => {"F32", [1, 5760, 1, 128]},
    }

    getter base : Inputs
    getter file : TensorFile
    getter records : Hash(String, TimelineRecord)
    @first_rope_verified = false

    def initialize(path : String, sha : String)
      unless PixelUMMBlockProbe.valid_digest?(sha)
        raise ArgumentError.new("continuation manifest identity must be a lowercase SHA-256")
      end
      validate_canonical_file!(path, "continuation manifest")
      @file = TensorFile.new(path, sha, 4_i64 * GIB)
      self.class.validate_document!(@file.document, @file.payload_bytes)
      @records = parse_records(@file.document, @file.payload_bytes)

      bundle = @file.document["qwen_input_bundle"]
      base_path = bundle["manifest_path"].as_s
      base_payload = bundle["payload_path"].as_s
      validate_canonical_file!(base_path, "Qwen input manifest")
      validate_canonical_file!(base_payload, "Qwen input payload")
      @base = Inputs.new(base_path, bundle["manifest_sha256"].as_s, conditioned: true)
      validate_base_binding!(@file.document, bundle)
    end

    # Authenticate the aggregate bundle before writing any of its records into
    # shared Metal buffers. Record digests are also checked while each record
    # is streamed directly into its final shared buffer.
    def verify_payload! : Nil
      @file.verify_payload!
      return if @first_rope_verified
      check = @file.document["producer"]["rope_first_segment_check"]
      {"input.rope_cos" => "cos_output_sha256", "input.rope_sin" => "sin_output_sha256"}.each do |name, key|
        record = @records[name]
        actual = sha_range(@file.payload, record.offset, record.nbytes // 6)
        unless actual == check[key].as_s
          raise ArgumentError.new("first-segment RoPE payload digest mismatch: #{name}")
        end
      end
      @first_rope_verified = true
    end

    def upload!(name : String, arena : ML::Video::Arena) : ML::MetalBuffer
      record = @records[name]? || raise ArgumentError.new("unknown continuation tensor: #{name}")
      verify_payload!
      dtype = record.dtype == "BF16" ? ML::Video::WeightDType::BF16 : ML::Video::WeightDType::F32
      elements = record.nbytes // dtype.bytes_per_element
      unless elements <= Int32::MAX
        raise ArgumentError.new("continuation tensor exceeds arena element-count limit: #{name}")
      end
      buffer = arena.alloc_weight(name, elements.to_i32, dtype)
      unless buffer.valid? && buffer.size == record.nbytes && buffer.storage_mode == ML::StorageMode::Shared
        raise ArgumentError.new("continuation upload extent/storage mismatch: #{name}")
      end
      upload_record!(record, buffer)
      buffer
    end

    def bytes(names : Array(String)) : Int64
      names.sum do |name|
        (@records[name]? || raise ArgumentError.new("unknown continuation tensor: #{name}")).nbytes
      end
    end

    # Pure metadata validation is deliberately public so tests can exercise
    # admission without creating a 1.3 GiB payload or touching Metal.
    def self.validate_document!(document : JSON::Any, payload_bytes : Int64) : Nil
      expected_keys = %w(cache_writer geometry noise_seeds payload payload_bytes payload_sha256 producer prompt qwen_input_bundle qwen_source schema schema_version tensors upstream_revision upstream_sources)
      unless document.as_h.keys.sort == expected_keys.sort
        raise ArgumentError.new("continuation manifest key inventory mismatch")
      end
      unless document["schema"].as_s == SCHEMA && document["schema_version"].as_i == SCHEMA_VERSION &&
             document["upstream_revision"].as_s == PINNED_REVISION &&
             document["payload_bytes"].as_i64 == payload_bytes &&
             document["payload"].as_s.matches?(/\A[A-Za-z0-9_.-]+\.tensors\.bin\z/) &&
             PixelUMMBlockProbe.valid_digest?(document["payload_sha256"].as_s)
        raise ArgumentError.new("continuation schema, payload, or pinned revision mismatch")
      end
      validate_sources!(document["upstream_sources"])
      validate_geometry!(document["geometry"])
      validate_noise_seeds!(document["noise_seeds"])
      validate_cache_writer!(document["cache_writer"])
      validate_prompt_source!(document)
      validate_bundle!(document["qwen_input_bundle"], document["tensors"])
      validate_producer!(document["producer"], document["qwen_input_bundle"])
      validate_tensor_records!(document["tensors"], payload_bytes)
    end

    private def parse_records(document : JSON::Any, payload_bytes : Int64) : Hash(String, TimelineRecord)
      records = {} of String => TimelineRecord
      offset = 0_i64
      TENSOR_SPECS.each do |name, (dtype, shape)|
        width = dtype == "BF16" ? 2_i64 : 4_i64
        nbytes = shape.reduce(1_i64) { |product, extent| product * extent } * width
        records[name] = TimelineRecord.new(document["tensors"][name], payload_bytes,
          dtype, shape, offset, nbytes, name)
        offset += nbytes
      end
      records
    end

    private def validate_base_binding!(document : JSON::Any, bundle : JSON::Any) : Nil
      base_doc = @base.file.document
      unless @base.file.path == bundle["manifest_path"].as_s &&
             @base.file.sha256 == bundle["manifest_sha256"].as_s &&
             @base.file.payload == bundle["payload_path"].as_s &&
             base_doc["payload_sha256"].as_s == bundle["payload_sha256"].as_s &&
             base_doc["upstream_revision"].as_s == PINNED_REVISION &&
             base_doc["prompt"].as_s == document["prompt"].as_s &&
             base_doc["qwen_source"] == document["qwen_source"] &&
             @base.records["input.qwen_bf16"].sha256 == bundle["qwen_tensor_sha256"].as_s &&
             @base.records["input.rope_cos"].sha256 == bundle["first_rope_cos_sha256"].as_s &&
             @base.records["input.rope_sin"].sha256 == bundle["first_rope_sin_sha256"].as_s &&
             @records["input.qwen_bf16"].sha256 == @base.records["input.qwen_bf16"].sha256
        raise ArgumentError.new("continuation Qwen source bundle/base identity mismatch")
      end
    end

    private def upload_record!(record : TimelineRecord, buffer : ML::MetalBuffer) : Nil
      bytes = Bytes.new(buffer.contents.as(UInt8*), record.nbytes.to_i32)
      digest = Digest::SHA256.new
      File.open(@file.payload) do |io|
        io.seek(record.offset)
        position = 0
        while position < bytes.size
          count = Math.min(bytes.size - position, 8 * 1024 * 1024)
          chunk = bytes[position, count]
          io.read_fully(chunk)
          digest.update(chunk)
          position += count
        end
      end
      unless digest.hexfinal == record.sha256
        raise ArgumentError.new("continuation tensor record digest mismatch: #{record.name}")
      end
    end

    private def sha_range(path : String, offset : Int64, nbytes : Int64) : String
      digest = Digest::SHA256.new
      scratch = Bytes.new(1024 * 1024)
      File.open(path) do |io|
        io.seek(offset)
        remaining = nbytes
        while remaining > 0
          chunk = scratch[0, Math.min(remaining, scratch.size.to_i64).to_i32]
          io.read_fully(chunk)
          digest.update(chunk)
          remaining -= chunk.size
        end
      end
      digest.hexfinal
    end

    private def validate_canonical_file!(path : String, label : String) : Nil
      unless Path[path].absolute? && File.file?(path) && !File.symlink?(path) && File.realpath(path) == path
        raise ArgumentError.new("#{label} must be a canonical absolute regular file")
      end
    end

    private def self.validate_sources!(sources : JSON::Any) : Nil
      unless sources.as_h.keys.sort == SOURCE_SHA256.keys.sort &&
             SOURCE_SHA256.all? { |path, digest| sources[path].as_s == digest }
        raise ArgumentError.new("continuation pinned upstream source inventory mismatch")
      end
    end

    private def self.validate_geometry!(geometry : JSON::Any) : Nil
      expected_keys = %w(cache_eps_shape decoder_local_attention_tile_tokens decoder_noise_shape decoder_rope_grid decoder_rope_sections decoder_rope_shape decoder_rope_theta dit_rope_axis_dims dit_rope_grid dit_rope_offsets dit_rope_theta latent_bank_shape qwen_shape segment_latents segment_pixel_frames segments total_latents total_pixel_frames full_latent_draw_shape)
      unless geometry.as_h.keys.sort == expected_keys.sort &&
             geometry["total_latents"].as_i == 180 && geometry["segments"].as_i == 6 &&
             geometry["segment_latents"].as_i == 30 && geometry["total_pixel_frames"].as_i == 720 &&
             geometry["segment_pixel_frames"].as_i == 120 &&
             geometry["latent_bank_shape"].as_a.map(&.as_i) == [1, 32, 180, 9, 16] &&
             geometry["full_latent_draw_shape"].as_a.map(&.as_i) == [1, 32, 720, 9, 16] &&
             geometry["dit_rope_offsets"].as_a.map(&.as_i) == ROPE_OFFSETS &&
             geometry["dit_rope_grid"].as_a.map(&.as_i) == [30, 9, 16] &&
             geometry["dit_rope_axis_dims"].as_a.map(&.as_i) == [16, 56, 56] &&
             geometry["dit_rope_theta"].as_f == 256.0 &&
             geometry["cache_eps_shape"].as_a.map(&.as_i) == [6, 1, 32, 30, 9, 16] &&
             geometry["qwen_shape"].as_a.map(&.as_i) == [1, 1280, 3584] &&
             geometry["decoder_noise_shape"].as_a.map(&.as_i) == [1, 3, 720, 288, 512] &&
             geometry["decoder_rope_grid"].as_a.map(&.as_i) == [10, 18, 32] &&
             geometry["decoder_rope_shape"].as_a.map(&.as_i) == [1, 5760, 1, 128] &&
             geometry["decoder_rope_sections"].as_a.map(&.as_i) == [64, 32, 32] &&
             geometry["decoder_rope_theta"].as_a.map(&.as_f) == [1_000_000.0, 10_000.0, 10_000.0] &&
             geometry["decoder_local_attention_tile_tokens"].as_i == 8
        raise ArgumentError.new("continuation geometry or global RoPE offsets mismatch")
      end
    end

    private def self.validate_noise_seeds!(seeds : JSON::Any) : Nil
      bank = seeds["upstream_interactive_bank"]
      unless seeds.as_h.keys.sort == %w(cache_eps_draw_count cache_eps_draw_order cache_eps_draw_shape cache_eps_generator_device cache_eps_output_dtype cache_eps_sample_dtype cache_eps_seed cache_eps_stock_upstream_cpu_rng_parity latent_draw_shape latent_export_slice latent_generator_device latent_seed pixel_noise_latent_indices pixel_noise_seed pixel_noise_seed_formula upstream_interactive_bank).sort &&
             seeds["latent_seed"].as_i == 42 && seeds["latent_generator_device"].as_s == "mps" &&
             seeds["latent_draw_shape"].as_a.map(&.as_i) == [1, 32, 720, 9, 16] &&
             seeds["latent_export_slice"].as_a.map(&.as_i) == [0, 180] &&
             bank.as_h.keys.sort == %w(generator_device latent_frames note seconds seed video_fps).sort &&
             bank["seconds"].as_i == 120 && bank["video_fps"].as_i == 24 &&
             bank["latent_frames"].as_i == 720 && bank["seed"].as_i == 42 &&
             bank["generator_device"].as_s == "cpu" &&
             bank["note"].as_s == "MPS values are not claimed identical to upstream CPU torch.randn values" &&
             seeds["pixel_noise_seed"].as_i == 42 &&
             seeds["pixel_noise_seed_formula"].as_s == "seed * 1000003 + global_latent_frame_index" &&
             seeds["pixel_noise_latent_indices"].as_a.map(&.as_i) == (0...180).to_a &&
             seeds["cache_eps_seed"].as_i == 42 && seeds["cache_eps_generator_device"].as_s == "mps" &&
             seeds["cache_eps_sample_dtype"].as_s == "F32" && seeds["cache_eps_output_dtype"].as_s == "BF16" &&
             seeds["cache_eps_draw_shape"].as_a.map(&.as_i) == [1, 32, 30, 9, 16] &&
             seeds["cache_eps_draw_count"].as_i == 6 &&
             seeds["cache_eps_draw_order"].as_s == "six sequential segment-shaped draws from a separate MPS generator" &&
             seeds["cache_eps_stock_upstream_cpu_rng_parity"].as_bool == false
        raise ArgumentError.new("continuation noise generation or RNG provenance mismatch")
      end
    end

    private def self.validate_cache_writer!(writer : JSON::Any) : Nil
      fixed = writer["fixed_initial_noise_timeline"]
      unless writer.as_h.keys.sort == %w(cache_at_step cache_eps_dtype cache_eps_draws cache_eps_generator_device cache_eps_sample_dtype cache_eps_seed cache_eps_tensor cache_eps_upstream_cpu_rng_parity cache_latent_source cache_noise_sigma cache_timestep cache_write_formula config_path config_sha256 fixed_initial_noise_timeline preserve_kv_cache resolved_override source_config_default_infer_steps_video).sort &&
             writer["config_path"].as_s == "configs/experiment/inference.yaml" &&
             writer["config_sha256"].as_s == SOURCE_SHA256["configs/experiment/inference.yaml"] &&
             writer["source_config_default_infer_steps_video"].as_i == 8 &&
             writer["resolved_override"].as_h.keys == ["inference.infer_steps_video"] &&
             writer["resolved_override"]["inference.infer_steps_video"].as_i == 4 &&
             writer["preserve_kv_cache"].as_bool &&
             writer["cache_at_step"].as_i == -1 && writer["cache_noise_sigma"].as_f == 0.206897 &&
             writer["cache_timestep"].as_f == 206.897 &&
             writer["cache_latent_source"].as_s == "final denoised segment latent selected by cache_at_step=-1" &&
             writer["cache_write_formula"].as_s == "((1-sigma)*cache_latents.float()+sigma*cache_eps.float()).to(cache_latents.dtype)" &&
             writer["cache_eps_tensor"].as_s == "cache.eps_bf16" && writer["cache_eps_seed"].as_i == 42 &&
             writer["cache_eps_generator_device"].as_s == "mps" &&
             writer["cache_eps_sample_dtype"].as_s == "F32 then cast to BF16" &&
             writer["cache_eps_dtype"].as_s == "BF16" && writer["cache_eps_upstream_cpu_rng_parity"].as_bool == false &&
             fixed.as_h.keys.sort == %w(generator_device latent_frames sample_dtype seconds seed stock_upstream_generator_device stock_upstream_rng_values_identical stock_upstream_sample_dtype video_fps).sort &&
             fixed["seconds"].as_i == 120 && fixed["video_fps"].as_i == 24 && fixed["latent_frames"].as_i == 720 &&
             fixed["seed"].as_i == 42 && fixed["generator_device"].as_s == "mps" &&
             fixed["sample_dtype"].as_s == "F32" && fixed["stock_upstream_generator_device"].as_s == "cpu" &&
             fixed["stock_upstream_sample_dtype"].as_s == "BF16" && fixed["stock_upstream_rng_values_identical"].as_bool == false
        raise ArgumentError.new("cache writer configuration or RNG scope mismatch")
      end
      draws = writer["cache_eps_draws"].as_a
      unless draws.size == 6 && draws.each_with_index.all? do |draw, index|
               draw.as_h.keys.sort == %w(dtype draw_order generator_device global_latent_start global_latent_stop seed segment_index shape).sort &&
               draw["segment_index"].as_i == index && draw["global_latent_start"].as_i == index * 30 &&
               draw["global_latent_stop"].as_i == (index + 1) * 30 &&
               draw["shape"].as_a.map(&.as_i) == [1, 32, 30, 9, 16] &&
               draw["dtype"].as_s == "BF16" && draw["generator_device"].as_s == "mps" &&
               draw["seed"].as_i == 42 && draw["draw_order"].as_i == index
             end
        raise ArgumentError.new("cache epsilon draw inventory or order mismatch")
      end
    end

    private def self.validate_prompt_source!(document : JSON::Any) : Nil
      prompt = document["prompt"].as_s
      source = document["qwen_source"]
      unless !prompt.strip.empty? && source["upstream_revision"].as_s == PINNED_REVISION &&
             source["state_index"].as_i == 26 && source["crop_tokens"].as_i == 1280 &&
             source["hidden_size"].as_i == 3584 && source["prompt_contract"]["prompt"].as_s == prompt &&
             source["prompt_sha256"].as_s == Digest::SHA256.hexdigest(prompt) &&
             source["prompt_contract"]["prompt_sha256"].as_s == source["prompt_sha256"].as_s
        raise ArgumentError.new("continuation Qwen source does not bind prompt and pinned geometry")
      end
      source_document = JSON.parse({
        "schema"            => "cogni_ml.hy_nomas_generation_inputs/v2",
        "schema_version"    => 2,
        "upstream_revision" => PINNED_REVISION,
        "prompt"            => prompt,
        "geometry"          => {"qwen_valid_tokens" => source["valid_tokens"]},
        "qwen_source"       => source,
      }.to_json)
      HyNomasGeneration.conditioned_mask!(source_document)
    end

    private def self.validate_bundle!(bundle : JSON::Any, tensors : JSON::Any) : Nil
      expected_keys = %w(first_rope_atol first_rope_cos_sha256 first_rope_rtol first_rope_sin_sha256 manifest_path manifest_sha256 payload_path payload_sha256 qwen_tensor_sha256)
      unless bundle.as_h.keys.sort == expected_keys.sort &&
             PixelUMMBlockProbe.valid_digest?(bundle["manifest_sha256"].as_s) &&
             PixelUMMBlockProbe.valid_digest?(bundle["payload_sha256"].as_s) &&
             PixelUMMBlockProbe.valid_digest?(bundle["qwen_tensor_sha256"].as_s) &&
             PixelUMMBlockProbe.valid_digest?(bundle["first_rope_cos_sha256"].as_s) &&
             PixelUMMBlockProbe.valid_digest?(bundle["first_rope_sin_sha256"].as_s) &&
             bundle["first_rope_atol"].as_f == PixelUMMBlockProbe::ATOL &&
             bundle["first_rope_rtol"].as_f == PixelUMMBlockProbe::RTOL &&
             Path[bundle["manifest_path"].as_s].absolute? && Path[bundle["payload_path"].as_s].absolute? &&
             bundle["qwen_tensor_sha256"].as_s == tensors["input.qwen_bf16"]["sha256"].as_s
        raise ArgumentError.new("continuation Qwen input bundle identity mismatch")
      end
    end

    private def self.validate_producer!(producer : JSON::Any, bundle : JSON::Any) : Nil
      unless producer.as_h.keys.sort == %w(device host_export mps_fallback model_inference python_version rope_first_segment_check tensor_generation_device torch_version).sort &&
             producer["device"].as_s == "mps" && producer["mps_fallback"].as_s == "disabled" &&
             producer["model_inference"].as_bool == false && producer["tensor_generation_device"].as_s == "mps" &&
             producer["host_export"].as_s == "one packed terminal MPS-to-host readback" &&
             !producer["torch_version"].as_s.empty? && !producer["python_version"].as_s.empty?
        raise ArgumentError.new("continuation producer scope mismatch")
      end
      check = producer["rope_first_segment_check"]
      unless check.as_h.keys.sort == %w(atol cos_max_abs_error cos_output_sha256 cos_source_sha256 rtol sin_max_abs_error sin_output_sha256 sin_source_sha256 source_manifest_sha256).sort &&
             check["cos_source_sha256"].as_s == bundle["first_rope_cos_sha256"].as_s &&
             check["sin_source_sha256"].as_s == bundle["first_rope_sin_sha256"].as_s &&
             check["source_manifest_sha256"].as_s == bundle["manifest_sha256"].as_s &&
             check["atol"].as_f == PixelUMMBlockProbe::ATOL && check["rtol"].as_f == PixelUMMBlockProbe::RTOL &&
             check["cos_max_abs_error"].as_f >= 0.0 && check["cos_max_abs_error"].as_f <= PixelUMMBlockProbe::ATOL + PixelUMMBlockProbe::RTOL &&
             check["sin_max_abs_error"].as_f >= 0.0 && check["sin_max_abs_error"].as_f <= PixelUMMBlockProbe::ATOL + PixelUMMBlockProbe::RTOL &&
             PixelUMMBlockProbe.valid_digest?(check["cos_output_sha256"].as_s) &&
             PixelUMMBlockProbe.valid_digest?(check["sin_output_sha256"].as_s)
        raise ArgumentError.new("first-segment DiT RoPE source comparison metadata mismatch")
      end
    end

    private def self.validate_tensor_records!(tensors : JSON::Any, payload_bytes : Int64) : Nil
      unless tensors.as_h.keys.sort == TENSOR_SPECS.keys.sort
        raise ArgumentError.new("continuation tensor inventory mismatch")
      end
      offset = 0_i64
      TENSOR_SPECS.each do |name, (dtype, shape)|
        nbytes = shape.reduce(1_i64) { |product, extent| product * extent } * (dtype == "BF16" ? 2_i64 : 4_i64)
        record = tensors[name]
        unless record.as_h.keys.sort == %w(dtype nbytes offset sha256 shape).sort &&
               record["dtype"].as_s == dtype && record["shape"].as_a.map(&.as_i) == shape &&
               record["offset"].as_i64 == offset && record["nbytes"].as_i64 == nbytes &&
               PixelUMMBlockProbe.valid_digest?(record["sha256"].as_s) &&
               offset <= payload_bytes && nbytes <= payload_bytes - offset
          raise ArgumentError.new("continuation tensor size, layout, or record identity mismatch: #{name}")
        end
        offset += nbytes
      end
      unless offset == payload_bytes
        raise ArgumentError.new("continuation payload extent differs from the exact tensor inventory")
      end
    end
  end
end
