require "./spec_helper"
require "digest/sha256"
require "json"

private QWEN3VL_SWEEP_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
private QWEN3VL_SWEEP_PROMPT = "a tiny lighthouse above a quiet harbor"

private def qwen3vl_sweep_reference_bundle(directory : String) : String
  raw_tokens = 18
  retained_rows = 4
  drop_idx = 14
  input_ids = Bytes.new(raw_tokens * 8, 0_u8)
  attention_mask = Bytes.new(raw_tokens * 8, 0_u8)
  raw_tokens.times do |index|
    IO::ByteFormat::LittleEndian.encode((index + 100).to_i64, input_ids[index * 8, 8])
    IO::ByteFormat::LittleEndian.encode(1_i64, attention_mask[index * 8, 8])
  end
  embeddings = Bytes.new(retained_rows * 4096 * 2, 0_u8)
  hidden_state = Bytes.new(raw_tokens * 4096 * 2, 0_u8)
  tensors = [] of {String, String, Array(Int32), Bytes}
  tensors << {"input_ids", "int64-le", [1, raw_tokens], input_ids}
  tensors << {"attention_mask", "int64-le", [1, raw_tokens], attention_mask}
  tensors << {"pre_final_rmsnorm_embeddings", "bfloat16-le", [1, retained_rows, 4096], embeddings}
  37.times do |index|
    tensors << {"hidden_state_#{index.to_s.rjust(3, '0')}", "bfloat16-le", [1, raw_tokens, 4096], hidden_state}
  end

  payload_io = IO::Memory.new
  tensors.each { |(_, _, _, bytes)| payload_io.write(bytes) }
  payload = payload_io.to_slice
  manifest = JSON.build do |json|
    json.object do
      json.field "schema", "qwen-image21-text-reference"
      json.field "schema_version", 1
      json.field "model" do
        json.object do
          json.field "repo", "Qwen/Qwen-Image-2.1"
          json.field "revision_sha", QWEN3VL_SWEEP_REVISION
          json.field "revision_source", "argument_and_local_cache_metadata"
          json.field "pipeline_class", "QwenImage21Pipeline"
          json.field "text_encoder_class", "Qwen3VLForConditionalGeneration"
          json.field "processor_class", "Qwen3VLProcessor"
        end
      end
      json.field "prompt", QWEN3VL_SWEEP_PROMPT
      json.field "tokenization" do
        json.object do
          json.field "raw_template_text", "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n#{QWEN3VL_SWEEP_PROMPT}<|im_end|>\n<|im_start|>assistant\n"
          json.field "processor_kwargs" do
            json.object do
              json.field "padding", true
              json.field "padding_side", "left"
              json.field "return_tensors", "pt"
            end
          end
          json.field "tokenizer_truncation", false
        end
      end
      json.field "sequence" do
        json.object do
          json.field "max_sequence_length", 512
          json.field "max_sequence_length_semantics", "post-drop validation guard only; no processor truncation"
          json.field "actual_sequence_length", retained_rows
          json.field "raw_input_shape", [1, raw_tokens]
          json.field "drop_idx", drop_idx
        end
      end
      json.field "embedding" do
        json.object do
          json.field "source", "QwenImage21Pipeline._get_qwen_prompt_embeds"
          json.field "pre_final_rmsnorm", true
          json.field "rmsnorm_hook_observed_and_verified", true
          json.field "shape", [1, retained_rows, 4096]
          json.field "source_dtype", "bfloat16"
          json.field "expected_decoder_layer_count", 36
          json.field "expected_hidden_state_count", 37
          json.field "expected_hidden_state_count_source", "loaded text_encoder.config.text_config.num_hidden_layers + 1"
          json.field "hidden_state_count", 37
        end
      end
      json.field "runtime" do
        json.object do
          json.field "source_dtype", "bfloat16"
          json.field "device", "cpu"
          json.field "diffusers_version", "0.41.0.dev0"
          json.field "diffusers_commit", "8b3c707ebd3ec4881f4190cf42931da07eaf3b65"
          json.field "official_source_file", "diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py"
        end
      end
      json.field "payload_file", "qwen_image21_text_reference.bin"
      json.field "payload_nbytes", payload.size
      json.field "payload_sha256", Digest::SHA256.hexdigest(payload)
      json.field "tensors" do
        json.object do
          offset = 0
          tensors.each do |name, dtype, shape, bytes|
            json.field name do
              json.object do
                json.field "dtype", dtype
                json.field "shape", shape
                json.field "offset_bytes", offset
                json.field "nbytes", bytes.size
                json.field "sha256", Digest::SHA256.hexdigest(bytes)
              end
            end
            offset += bytes.size
          end
        end
      end
    end
  end
  File.write(File.join(directory, "qwen_image21_text_reference.bin"), payload)
  File.write(File.join(directory, "qwen_image21_text_reference.json"), manifest)
  JSON.parse(manifest)["payload_sha256"].as_s
end

describe "qwen3vl text-layer sweep reference binding" do
  it "admits a generic prompt only with its exact payload SHA and derives token counts" do
    root = File.join(Dir.tempdir, "qwen3vl-sweep-spec-#{Random.rand(1_000_000_000)}")
    reference_dir = File.join(root, "reference")
    encoder_dir = File.join(root, "empty-encoder")
    Dir.mkdir(root)
    Dir.mkdir(reference_dir)
    Dir.mkdir(encoder_dir)
    begin
      payload_sha = qwen3vl_sweep_reference_bundle(reference_dir)
      script = File.expand_path("../scripts/qwen3vl_text_layer_sweep.cr", __DIR__)
      args = [
        "run", script,
        "--link-flags", "-fuse-ld=/usr/bin/ld",
        "--",
        "--layers=1", "--full-prompt",
        "--reference-sha256=#{payload_sha}",
        "--text-encoder-dir=#{encoder_dir}",
        "--reference-dir=#{reference_dir}",
      ]
      stdout = IO::Memory.new
      stderr = IO::Memory.new
      status = Process.run("crystal", args, output: stdout, error: stderr)
      status.exit_code.should_not eq(0), "stdout=#{stdout} stderr=#{stderr}"
      stdout.to_s.should contain("reference_validation=passed model_revision=#{QWEN3VL_SWEEP_REVISION}")
      stdout.to_s.should contain("prompt=#{QWEN3VL_SWEEP_PROMPT.inspect}")
      stdout.to_s.should contain("projection_backend=scalar")
      stdout.to_s.should contain("manifest_sha256=#{Digest::SHA256.hexdigest(File.read(File.join(reference_dir, "qwen_image21_text_reference.json")))}")
      stdout.to_s.should contain("raw_tokens=18 retained_tokens=4 drop_idx=14")
      stdout.to_s.should contain("tokens=18")
      stdout.to_s.should contain("retained_rows=[14, 15, 16, 17]")

      bad_args = args.dup
      bad_args[bad_args.index("--reference-sha256=#{payload_sha}").not_nil!] = "--reference-sha256=#{"0" * 64}"
      bad_stdout = IO::Memory.new
      bad_stderr = IO::Memory.new
      bad_status = Process.run("crystal", bad_args, output: bad_stdout, error: bad_stderr)
      bad_status.exit_code.should eq(2)
      bad_stdout.to_s.should_not contain("model_revision=#{QWEN3VL_SWEEP_REVISION}")
      bad_stderr.to_s.should contain("reference SHA-256")

      accelerate_args = args.dup
      accelerate_args << "--projection-backend=accelerate"
      accelerate_stdout = IO::Memory.new
      accelerate_stderr = IO::Memory.new
      accelerate_status = Process.run("crystal", accelerate_args, output: accelerate_stdout, error: accelerate_stderr)
      accelerate_status.exit_code.should_not eq(0), "stdout=#{accelerate_stdout} stderr=#{accelerate_stderr}"
      accelerate_stdout.to_s.should contain("projection_backend=accelerate")

      script_source = File.read(script)
      script_source.should contain("json.field \"fixture_manifest_sha256\"")
      script_source.should contain("projection_backend: projection_backend")
    ensure
      Dir.glob(File.join(reference_dir, "*")).each { |path| File.delete(path) }
      Dir.delete(reference_dir)
      Dir.delete(encoder_dir)
      Dir.delete(root)
    end
  end
end
