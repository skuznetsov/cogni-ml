require "./spec_helper"
require "file_utils"
require "../src/ml/gguf/qwen35_self_spec_plan"

def with_qwen35_self_spec_plan_mock_context(&)
  root = File.tempname("qwen35-self-spec-plan-cache")
  model_path = File.tempname("qwen35-self-spec-plan-model", ".gguf")
  Dir.mkdir_p(root)
  File.write(model_path, "mock-model")
  tok = ML::GGUF::Qwen35Tokenizer.new(
    ["<pad>", "hello", "world"],
    eos_id: 2_i32,
    pad_id: 0_i32,
    add_bos: false,
    model_path: model_path,
  )
  begin
    yield root, model_path, tok
  ensure
    FileUtils.rm_rf(root) if Dir.exists?(root)
    File.delete(model_path) if File.exists?(model_path)
  end
end

def write_qwen35_self_spec_plan_adapter(path : String, hidden : Int32 = 2, rank : Int32 = 1, layers : Array(Int32) = [0])
  adapters = ML::GGUF::Qwen35FFNUpDownAdapterMap.new
  layers.each do |layer|
    adapters[layer] = ML::GGUF::Qwen35FFNUpDownAdapter.new(
      x_mean: Array(Float64).new(hidden, 0.0),
      c_mean: Array(Float64).new(rank, 0.0),
      coeff_weights: Array(Array(Float64)).new(rank) { Array(Float64).new(hidden, 0.0) },
      down_basis: Array(Array(Float32)).new(rank) { Array(Float32).new(hidden, 0.0_f32) },
    )
  end
  ML::GGUF::Qwen35FFNUpDownAdapterArtifact.dump(path, adapters, rank, hidden, "spec")
end

describe ML::GGUF::Qwen35SelfSpecPlan do
  it "returns a route miss without requiring an adapter" do
    with_qwen35_self_spec_plan_mock_context do |root, model_path, tok|
      result = ML::GGUF::Qwen35SelfSpecPlan.resolve(root, model_path, tok, "hello", [1_i32], 2)
      result.status.should eq(ML::GGUF::Qwen35SelfSpecPlan::STATUS_ROUTE_MISS)
      result.route_hit?.should be_false
      result.executable?.should be_false
    end
  end

  it "resolves baseline routes without adapter validation" do
    with_qwen35_self_spec_plan_mock_context do |root, model_path, tok|
      model_id = ML::GGUF::Qwen35ProposalRoute.model_id(model_path)
      tokenizer_id = ML::GGUF::Qwen35ProposalRoute.tokenizer_id(model_id, tok)
      store = ML::GGUF::Qwen35PromptCache::Store.new(root)
      store.save_proposal_route(
        model_id: model_id,
        tokenizer_id: tokenizer_id,
        prompt_text: "hello",
        token_ids: [1_i32],
        route: ML::GGUF::Qwen35PromptCache::PROPOSAL_ROUTE_BASELINE,
        route_key: "base",
      )

      result = ML::GGUF::Qwen35SelfSpecPlan.resolve(root, model_path, tok, "ignored", [99_i32], 2, route_key: "base")
      result.status.should eq(ML::GGUF::Qwen35SelfSpecPlan::STATUS_BASELINE)
      result.adapter_note.should eq("adapter_artifact=not_applicable")
      result.executable?.should be_true
    end
  end

  it "validates a matching pca-updown route and adapter artifact" do
    with_qwen35_self_spec_plan_mock_context do |root, model_path, tok|
      adapter_path = File.join(root, "adapter.json")
      write_qwen35_self_spec_plan_adapter(adapter_path, hidden: 2, rank: 2, layers: [0, 2])
      model_id = ML::GGUF::Qwen35ProposalRoute.model_id(model_path)
      tokenizer_id = ML::GGUF::Qwen35ProposalRoute.tokenizer_id(model_id, tok)
      store = ML::GGUF::Qwen35PromptCache::Store.new(root)
      store.save_proposal_route(
        model_id: model_id,
        tokenizer_id: tokenizer_id,
        prompt_text: "hello",
        token_ids: [1_i32],
        route: ML::GGUF::Qwen35PromptCache::PROPOSAL_ROUTE_PCA_UPDOWN,
        route_rank: 2,
        route_layers: [2, 0],
        route_key: "updown",
      )

      result = ML::GGUF::Qwen35SelfSpecPlan.resolve(root, model_path, tok, "ignored", [99_i32], 2, route_key: "updown", adapter_path: adapter_path)
      result.status.should eq(ML::GGUF::Qwen35SelfSpecPlan::STATUS_PCA_UPDOWN)
      result.requested_layers.should eq([0, 2])
      result.adapter_note.should contain("adapter_artifact=valid")
      result.executable?.should be_true
    end
  end

  it "fails closed when the adapter artifact misses a requested layer" do
    with_qwen35_self_spec_plan_mock_context do |root, model_path, tok|
      adapter_path = File.join(root, "adapter.json")
      write_qwen35_self_spec_plan_adapter(adapter_path, hidden: 2, rank: 2, layers: [0])
      model_id = ML::GGUF::Qwen35ProposalRoute.model_id(model_path)
      tokenizer_id = ML::GGUF::Qwen35ProposalRoute.tokenizer_id(model_id, tok)
      store = ML::GGUF::Qwen35PromptCache::Store.new(root)
      store.save_proposal_route(
        model_id: model_id,
        tokenizer_id: tokenizer_id,
        prompt_text: "hello",
        token_ids: [1_i32],
        route: ML::GGUF::Qwen35PromptCache::PROPOSAL_ROUTE_PCA_UPDOWN,
        route_rank: 2,
        route_layers: [0, 2],
        route_key: "bad",
      )

      result = ML::GGUF::Qwen35SelfSpecPlan.resolve(root, model_path, tok, "ignored", [99_i32], 2, route_key: "bad", adapter_path: adapter_path)
      result.status.should eq(ML::GGUF::Qwen35SelfSpecPlan::STATUS_INVALID_ADAPTER)
      result.adapter_note.should contain("missing_layers")
      result.executable?.should be_false
    end
  end
end
