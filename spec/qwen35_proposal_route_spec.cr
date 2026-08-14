require "./spec_helper"
require "../src/ml/gguf/qwen35_proposal_route"

describe ML::GGUF::Qwen35ProposalRoute do
  it "uses the same stable identity contract as proposal-route certificates" do
    model_path = File.tempname("qwen35-proposal-route-model", ".gguf")
    File.write(model_path, "mock-model")
    begin
      tok = ML::GGUF::Qwen35Tokenizer.new(
        ["<pad>", "hello", "world"],
        eos_id: 2_i32,
        pad_id: 0_i32,
        add_bos: false,
        model_path: model_path,
      )

      model_id = ML::GGUF::Qwen35ProposalRoute.model_id(model_path)
      tokenizer_id = ML::GGUF::Qwen35ProposalRoute.tokenizer_id(model_id, tok)

      model_info = File.info(model_path)
      expected_model = ML::GGUF::Qwen35PromptCache.short_hash(
        "model\0#{model_path}\0#{model_info.size}\0#{model_info.modification_time.to_unix}")
      expected_tokenizer = ML::GGUF::Qwen35PromptCache.short_hash(
        "tokenizer\0#{expected_model}\0#{tok.vocab.size}\0#{tok.eos_id}\0#{tok.pad_id}\0#{ML::GGUF::Qwen35Tokenizer::ENCODING_REVISION}"
      )
      legacy_tokenizer = ML::GGUF::Qwen35PromptCache.short_hash(
        "tokenizer\0#{expected_model}\0#{tok.vocab.size}\0#{tok.eos_id}\0#{tok.pad_id}"
      )

      model_id.should eq(expected_model)
      tokenizer_id.should eq(expected_tokenizer)
      tokenizer_id.should_not eq(legacy_tokenizer)
    ensure
      File.delete(model_path) if File.exists?(model_path)
    end
  end

  it "resolves exact prompt and route-key proposal certificates" do
    root = File.tempname("qwen35-proposal-route-cache")
    model_path = File.tempname("qwen35-proposal-route-model", ".gguf")
    Dir.mkdir_p(root)
    File.write(model_path, "mock-model")
    begin
      tok = ML::GGUF::Qwen35Tokenizer.new(
        ["<pad>", "hello", "world"],
        eos_id: 2_i32,
        pad_id: 0_i32,
        add_bos: false,
        model_path: model_path,
      )
      prompt = "hello world"
      token_ids = [1_i32, 2_i32]
      model_id = ML::GGUF::Qwen35ProposalRoute.model_id(model_path)
      tokenizer_id = ML::GGUF::Qwen35ProposalRoute.tokenizer_id(model_id, tok)
      store = ML::GGUF::Qwen35PromptCache::Store.new(root)
      store.save_proposal_route(
        model_id: model_id,
        tokenizer_id: tokenizer_id,
        prompt_text: prompt,
        token_ids: token_ids,
        route: ML::GGUF::Qwen35PromptCache::PROPOSAL_ROUTE_PCA_UPDOWN,
        route_rank: 4,
        route_layers: [4, 0, 2],
        route_key: "code_square",
      )

      exact = ML::GGUF::Qwen35ProposalRoute.resolve(root, model_path, tok, prompt, token_ids)
      exact.entry.not_nil!.route.should eq(ML::GGUF::Qwen35PromptCache::PROPOSAL_ROUTE_PCA_UPDOWN)
      exact.entry.not_nil!.route_layers.should eq([0, 2, 4])

      keyed = ML::GGUF::Qwen35ProposalRoute.resolve(root, model_path, tok, "different", [99_i32], "code_square")
      keyed.entry.not_nil!.route_rank.should eq(4)

      miss = ML::GGUF::Qwen35ProposalRoute.resolve(root, model_path, tok, "different", [99_i32])
      miss.entry.should be_nil
    ensure
      FileUtils.rm_rf(root) if Dir.exists?(root)
      File.delete(model_path) if File.exists?(model_path)
    end
  end
end
