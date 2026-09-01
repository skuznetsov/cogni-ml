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
  it "tracks token-option corridors through singleton spans and branch points" do
    corridor = ML::GGUF::Qwen35Constraints::TokenOptionCorridor.from_options([
      [10, 20, 30],
      [10, 20, 40],
      [10, 50],
    ])

    corridor.next_ids.should eq([10])
    corridor = corridor.advance(10)
    corridor.next_ids.should eq([20, 50])
    corridor = corridor.advance(20)
    corridor.next_ids.should eq([30, 40])
    corridor = corridor.advance(30)
    corridor.complete?.should be_true
  end

  it "rejects empty and prefix-ambiguous token-option entries" do
    ML::GGUF::Qwen35Constraints::TokenOptionCorridor.from_options([] of Array(Int32)).empty?.should be_true
    expect_raises(ArgumentError, "token-option corridor options must be non-empty") do
      ML::GGUF::Qwen35Constraints.required_token_option_corridor([[] of Int32])
    end
    expect_raises(ArgumentError, "token-option corridor options must be prefix-free") do
      ML::GGUF::Qwen35Constraints.required_token_option_corridor([
        [1, 2],
        [1, 2, 3],
      ])
    end
  end

  it "maps an emitted token trace back to the selected finite literal option" do
    options = [
      [1, 2, 3],
      [1, 2, 4],
      [1, 2, 4, 5],
    ]

    ML::GGUF::Qwen35Constraints::TokenOptionCorridor.selected_literal_index?(options, [1, 2, 4]).should eq(1)
    ML::GGUF::Qwen35Constraints::TokenOptionCorridor.selected_literal_index?(options, [1, 2, 4, 5]).should eq(2)
    ML::GGUF::Qwen35Constraints::TokenOptionCorridor.selected_literal_index?(options, [1, 9]).should be_nil
  end

  it "builds byte-exact token-option corridors with the native tokenizer" do
    tok = ML::GGUF::Qwen35Tokenizer.new(
      ["a", "b", "c", "ab", "ac", "eos"],
      eos_id: 5,
      pad_id: 5,
      add_bos: false,
      model_path: "fake.gguf",
      token_to_id: {"a" => 0, "b" => 1, "c" => 2, "ab" => 3, "ac" => 4, "eos" => 5},
      bpe_ranks: { {"a", "b"} => 0, {"a", "c"} => 1 },
    )

    corridor = ML::GGUF::Qwen35Constraints.token_option_corridor(tok, ["ab", "ac"])
    corridor.options.should eq([[3], [4]])
    corridor.next_ids.should eq([3, 4])
  end

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

  it "keeps tool labels attached until literal choices diverge" do
    tok = Qwen35ConstraintsSpecHelper.fake_tokenizer([
      "<", "<tool_call>", "s", "shell", "l", "list", "list_directory",
    ])
    index = ML::GGUF::Qwen35Constraints::TokenTextIndex.new(tok)
    remaining = ML::GGUF::Qwen35Constraints.labeled_tool_call_prefixes([
      "shell", "list_directory",
    ])

    common = ML::GGUF::Qwen35Constraints.labeled_literal_frontier(index, remaining)
    common.find { |candidate| candidate.text == "<" }.not_nil!.labels.should eq([
      "shell", "list_directory",
    ])

    remaining = ML::GGUF::Qwen35Constraints.advance_labeled_literal_options(
      remaining,
      "<tool_call>\n<function=",
    )
    choice = ML::GGUF::Qwen35Constraints.labeled_literal_frontier(index, remaining)
    choice.find { |candidate| candidate.text == "shell" }.not_nil!.labels.should eq(["shell"])
    choice.find { |candidate| candidate.text == "list" }.not_nil!.labels.should eq(["list_directory"])
    ML::GGUF::Qwen35Constraints.labeled_frontier_diverged?(choice).should be_true
  end

  it "detects a real choice while one token still prefixes multiple labels" do
    tok = Qwen35ConstraintsSpecHelper.fake_tokenizer(["g", "grep", "glob", "s", "shell"])
    index = ML::GGUF::Qwen35Constraints::TokenTextIndex.new(tok)
    frontier = ML::GGUF::Qwen35Constraints.labeled_literal_frontier(index, {
      "grep"  => "grep>\n",
      "glob"  => "glob>\n",
      "shell" => "shell>\n",
    })

    frontier.find { |candidate| candidate.text == "g" }.not_nil!.labels.should eq(["grep", "glob"])
    ML::GGUF::Qwen35Constraints.labeled_frontier_diverged?(frontier).should be_true
  end

  it "does not confuse a longer tokenizer shortcut with a label choice" do
    tok = Qwen35ConstraintsSpecHelper.fake_tokenizer(["=", "=read", "=list"])
    index = ML::GGUF::Qwen35Constraints::TokenTextIndex.new(tok)
    frontier = ML::GGUF::Qwen35Constraints.labeled_literal_frontier(index, {
      "read_file"      => "=read_file>\n",
      "list_directory" => "=list_directory>\n",
    })

    frontier.find { |candidate| candidate.text == "=" }.not_nil!.labels.should eq([
      "read_file", "list_directory",
    ])
    ML::GGUF::Qwen35Constraints.labeled_frontier_diverged?(frontier).should be_false
  end

  it "returns an empty frontier after the literal corridor is complete or invalid" do
    tok = Qwen35ConstraintsSpecHelper.fake_tokenizer(["<", "x"])

    ML::GGUF::Qwen35Constraints.literal_frontier_ids(tok, [] of String).should be_empty
    ML::GGUF::Qwen35Constraints.advance_literal_options(["<tool_call>"], "nope").should be_empty
  end

  it "rejects an incomplete literal corridor with no tokenizer frontier" do
    tok = Qwen35ConstraintsSpecHelper.fake_tokenizer(["x", "eos"])
    index = ML::GGUF::Qwen35Constraints::TokenTextIndex.new(tok)

    expect_raises(ML::GGUF::Qwen35Constraints::LiteralFrontierError) do
      ML::GGUF::Qwen35Constraints.required_literal_frontier_ids(index, ["<tool_call>"])
    end
  end

  it "keeps representable and completed literal corridors admissible" do
    tok = Qwen35ConstraintsSpecHelper.fake_tokenizer(["<", "<tool", "eos"])
    index = ML::GGUF::Qwen35Constraints::TokenTextIndex.new(tok)

    ML::GGUF::Qwen35Constraints.required_literal_frontier_ids(
      index, ["<tool_call>"]).should eq([0, 1])
    ML::GGUF::Qwen35Constraints.required_literal_frontier_ids(
      index, [""]).should be_empty
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
    prefixes = ML::GGUF::Qwen35Constraints.qwen_tool_required_parameter_prefix_options(tools)

    required["read_file"].should eq(["path", "limit"])
    options.should eq(["<parameter=path>\n", "<parameter=limit>\n"])
    prefixes.should eq([
      "<tool_call>\n<function=read_file>\n<parameter=path>\n",
      "<tool_call>\n<function=read_file>\n<parameter=limit>\n",
    ])
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
    calls = ML::GGUF::Qwen35Constraints.qwen_tool_finite_call_options(tools)

    options["edit_mode"]["mode"].should eq(["fast\n", "safe\n"])
    options["edit_mode"]["dry_run"].should eq(["true\n", "false\n"])
    calls.should contain("<tool_call>\n<function=edit_mode>\n<parameter=mode>\nfast\n</parameter>\n</function>\n</tool_call>")
    calls.should contain("<tool_call>\n<function=edit_mode>\n<parameter=dry_run>\nfalse\n</parameter>\n</function>\n</tool_call>")
  end

  it "extracts bounded integer parameter value options" do
    tools = ML::GGUF::Qwen35Chat.parse_tools_json(%([
      {"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{
        "limit":{"type":"integer","minimum":1,"maximum":3}
      },"required":["limit"]}}}
    ]))

    options = ML::GGUF::Qwen35Constraints.tool_finite_parameter_value_options(tools)
    options["read_file"]["limit"].should eq(["1\n", "2\n", "3\n"])
  end

  it "extracts optional parameters in schema property order" do
    tools = ML::GGUF::Qwen35Chat.parse_tools_json(%([
      {"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{
        "path":{"type":"string"},
        "limit":{"type":"integer"},
        "exact":{"type":"boolean"}
      },"required":["path"]}}}
    ]))

    ML::GGUF::Qwen35Constraints.tool_optional_parameters(tools)["read_file"].should eq(["limit", "exact"])
  end

  it "does not enumerate overly wide integer ranges" do
    tools = ML::GGUF::Qwen35Chat.parse_tools_json(%([
      {"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{
        "limit":{"type":"integer","minimum":1,"maximum":10000}
      },"required":["limit"]}}}
    ]))

    ML::GGUF::Qwen35Constraints.tool_finite_parameter_value_options(tools).should be_empty
    ML::GGUF::Qwen35Constraints.qwen_tool_finite_call_options(tools).should be_empty
  end
end
