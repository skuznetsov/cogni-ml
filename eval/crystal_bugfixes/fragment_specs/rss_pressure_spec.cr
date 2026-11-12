require "spec"
require "./support/rss_guard"

describe CogniSpecRSSGuard do
  it "enables the process guard for a system free-memory floor alone" do
    keys = ["COGNI_SPEC_RSS_GUARD", "COGNI_SPEC_MAX_RSS_MB", "COGNI_SPEC_MIN_FREE_PCT"]
    saved = keys.map { |key| ENV[key]? }
    begin
      ENV["COGNI_SPEC_RSS_GUARD"] = "1"
      ENV["COGNI_SPEC_MAX_RSS_MB"] = "0"
      ENV["COGNI_SPEC_MIN_FREE_PCT"] = "12"
      CogniSpecRSSGuard.enabled?.should be_true
      ENV["COGNI_SPEC_RSS_GUARD"] = "0"
      CogniSpecRSSGuard.enabled?.should be_false
    ensure
      keys.zip(saved).each do |key, value|
        if value
          ENV[key] = value
        else
          ENV.delete(key)
        end
      end
    end
  end
end
