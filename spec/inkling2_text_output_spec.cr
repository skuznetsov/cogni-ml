require "spec"
require "../src/ml/serve/text_output"

describe ML::Serve::TextOutput do
  it "emits complete characters across tokens and withholds a split stop sequence" do
    pieces = [] of String
    output = ML::Serve::TextOutput.new(["STOP", "TOP"]) { |text| pieces << text }
    wave = "👋"
    output.append("Hello " + wave.byte_slice(0, 2)).should be_true
    pieces.should eq(["Hello "])
    output.append(wave.byte_slice(2) + "S").should be_true
    pieces.should eq(["Hello ", wave])
    output.append("TO").should be_true
    output.append("P suppressed").should be_false
    output.finish
    pieces.join.should eq("Hello 👋")
  end

  it "flushes a partial stop and replaces an incomplete final codepoint" do
    pieces = [] of String
    output = ML::Serve::TextOutput.new(["END"]) { |text| pieces << text }
    output.append("A E").should be_true
    output.append("N").should be_true
    output.finish
    pieces.join.should eq("A EN")

    output.append("👋".byte_slice(0, 2)).should be_true
    output.finish
    pieces.join.should eq("A EN�")
  end
end
