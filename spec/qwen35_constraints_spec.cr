require "./spec_helper"
require "../src/ml/gguf/qwen35_chat"
require "../src/ml/gguf/qwen35_constraints"

module Qwen35ConstraintsSpecHelper
  def self.fake_tokenizer(vocab : Array(String)) : ML::GGUF::Qwen35Tokenizer
    token_to_id = {} of String => Int32
    vocab.each_with_index { |piece, id| token_to_id[piece] = id.to_i32 }
    ML::GGUF::Qwen35Tokenizer.new(
      vocab,
      eos_id: vocab.size.to_i32 - 1,
      pad_id: vocab.size.to_i32 - 1,
      add_bos: false,
      model_path: "fake.gguf",
      token_to_id: token_to_id,
    )
  end
end

describe ML::GGUF::Qwen35Constraints do
  it "builds token frontiers for finite literal corridors" do
    tok = Qwen35ConstraintsSpecHelper.fake_tokenizer(["<", "<tool", "<tool_call>", "_call", "_call>", "tool", ">"])

    ids = ML::GGUF::Qwen35Constraints.literal_frontier_ids(tok, ["<tool_call>"])
    indexed_ids = ML::GGUF::Qwen35Constraints.literal_frontier_ids(
      ML::GGUF::Qwen35Constraints::TokenTextIndex.new(tok),
      ["<tool_call>"])
    pieces = ids.map { |id| tok.decode_single(id) }

    indexed_ids.should eq(ids)
    pieces.should contain("<")
    pieces.should contain("<tool")
    pieces.should contain("<tool_call>")
    pieces.should_not contain("_call")
    pieces.should_not contain("tool")
  end

  it "advances literal options after an emitted token" do
    next_options = ML::GGUF::Qwen35Constraints.advance_literal_options(
      ["<tool_call>", "<function=read_file>"],
      "<tool",
    )

    next_options.should eq(["_call>"])
  end

  it "returns an empty frontier after the literal corridor is complete or invalid" do
    tok = Qwen35ConstraintsSpecHelper.fake_tokenizer(["<", "x"])

    ML::GGUF::Qwen35Constraints.literal_frontier_ids(tok, [] of String).should be_empty
    ML::GGUF::Qwen35Constraints.advance_literal_options(["<tool_call>"], "nope").should be_empty
  end

  it "extracts Qwen tool-call prefix options from OpenAI-style tools" do
    tools = ML::GGUF::Qwen35Chat.parse_tools_json(%([
      {"type":"function","function":{"name":"read_file","parameters":{"type":"object"}}},
      {"type":"function","function":{"name":"grep","parameters":{"type":"object"}}},
      {"type":"function","function":{"name":"read_file","parameters":{"type":"object"}}}
    ]))

    names = ML::GGUF::Qwen35Constraints.tool_function_names(tools)
    required = ML::GGUF::Qwen35Constraints.tool_required_parameters(tools)
    prefixes = ML::GGUF::Qwen35Constraints.qwen_tool_call_prefix_options(names)

    names.should eq(["read_file", "grep"])
    required["read_file"].should be_empty
    prefixes.should eq([
      "<tool_call>\n<function=read_file>\n",
      "<tool_call>\n<function=grep>\n",
    ])
  end

  it "extracts required parameters and renders parameter-open options" do
    tools = ML::GGUF::Qwen35Chat.parse_tools_json(%([
      {"type":"function","function":{"name":"read_file","parameters":{"type":"object","required":["path","limit"]}}}
    ]))

    required = ML::GGUF::Qwen35Constraints.tool_required_parameters(tools)
    options = ML::GGUF::Qwen35Constraints.qwen_parameter_open_options(required["read_file"])

    required["read_file"].should eq(["path", "limit"])
    options.should eq(["<parameter=path>\n", "<parameter=limit>\n"])
  end

  it "renders single-parameter close options" do
    ML::GGUF::Qwen35Constraints.qwen_single_parameter_close_options.should eq([
      "</parameter>\n</function>\n</tool_call>",
    ])
  end

  it "advances a partially emitted single-parameter close literal" do
    options = ML::GGUF::Qwen35Constraints.qwen_single_parameter_close_options
    ML::GGUF::Qwen35Constraints.advance_literal_options(options, "</par").should eq([
      "ameter>\n</function>\n</tool_call>",
    ])
  end

  it "renders parameter continuation options" do
    ML::GGUF::Qwen35Constraints.qwen_parameter_continue_options(["limit"]).should eq([
      "</parameter>\n<parameter=limit>\n",
    ])
  end

  it "extracts finite enum and boolean parameter value options" do
    tools = ML::GGUF::Qwen35Chat.parse_tools_json(%([
      {"type":"function","function":{"name":"edit_mode","parameters":{"type":"object","properties":{
        "mode":{"type":"string","enum":["fast","safe"]},
        "dry_run":{"type":"boolean"}
      },"required":["mode","dry_run"]}}}
    ]))

    options = ML::GGUF::Qwen35Constraints.tool_finite_parameter_value_options(tools)
    options["edit_mode"]["mode"].should eq(["fast\n", "safe\n"])
    options["edit_mode"]["dry_run"].should eq(["true\n", "false\n"])
  end
end
