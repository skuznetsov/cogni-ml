require "spec"
require "../src/ml/audio/drama3/profile"

include ML::Audio::Drama3

describe "DRAMA-3 profiler" do
  it "records nested component paths, categories, and calls" do
    path = File.tempname("drama3-profile", ".json")
    previous_path = ENV["DRAMA3_PROFILE"]?
    ENV["DRAMA3_PROFILE"] = path

    begin
      Profile.configure_from_env!
      Profile.measure("generation", category: "model") do
        2.times do
          Profile.measure("matvec.quant", category: "matvec") { 1 + 1 }
        end
      end
      Profile.write_from_env

      spans = JSON.parse(File.read(path))["spans"].as_a
      generation = spans.find { |span| span["path"].as_s == "generation" }.not_nil!
      matvec = spans.find { |span| span["path"].as_s == "generation/matvec.quant" }.not_nil!

      generation["category"].as_s.should eq("model")
      generation["calls"].as_i.should eq(1)
      generation["inclusive_ns"].as_i
      generation["exclusive_ns"].as_i
      matvec["category"].as_s.should eq("matvec")
      matvec["calls"].as_i.should eq(2)
      # With no other child scopes, the parent exclusive duration plus the
      # two child inclusive durations must account for the parent's inclusive
      # duration. This checks nesting arithmetic without asserting timings.
      parent_inclusive = generation["inclusive_ns"].as_i
      parent_exclusive = generation["exclusive_ns"].as_i
      children_inclusive = matvec["inclusive_ns"].as_i
      (parent_exclusive + children_inclusive).should eq(parent_inclusive)
      matvec["exclusive_ns"].as_i.should be <= matvec["inclusive_ns"].as_i
    ensure
      if previous_path
        ENV["DRAMA3_PROFILE"] = previous_path
      else
        ENV.delete("DRAMA3_PROFILE")
      end
      Profile.configure_from_env!
      File.delete(path) if File.exists?(path)
    end
  end
end
