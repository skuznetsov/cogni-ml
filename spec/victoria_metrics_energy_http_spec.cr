require "spec"
require "../src/ml/energy/victoria_metrics"

# Use a disposable VictoriaMetrics instance; fixtures are explicitly synthetic.
if endpoint = ENV["COGNI_TEST_VM_URL"]?
  describe "VictoriaMetrics raw HTTP energy integration" do
    it "imports raw samples and measures only the clipped request window" do
      uri = URI.parse(endpoint)
      start = Time.utc.to_unix_ms - 10_000
      fixture = "cogni_energy_fixture_watts"
      labels = {"__name__" => fixture, "test_id" => start.to_s}
      body = {metric: labels, timestamps: [start, start + 1000, start + 2000], values: [1, 3, 1]}.to_json + "\n"
      client = HTTP::Client.new(uri)
      begin
        response = client.post(uri.path.rstrip('/') + "/api/v1/import", body: body)
        response.success?.should be_true
        reader = ML::Energy::VictoriaMetrics.new(endpoint)
        measurement = nil.as(JSON::Any?)
        # Default VM ingestion buffers flush every five seconds.
        100.times do
          begin
            measurement = reader.measure(%(#{fixture}{test_id="#{start}"}), start + 250, start + 1750, kind: "active_power", max_gap_ms: 1100_i64)
            break
          rescue ArgumentError
            sleep 100.milliseconds
          end
        end
        measurement.should_not be_nil
        result = measurement.not_nil!
        result["active_energy_joules"].as_f.should be_close(3.375, 1e-8)
        result["apparent_energy_va_seconds"].raw.should be_nil
        expect_raises(ArgumentError) do
          reader.measure(%(#{fixture}{test_id="#{start}"}), start + 250, start + 1750, kind: "active_power", max_gap_ms: 500_i64)
        end
        expect_raises(ArgumentError) do
          reader.measure(%(#{fixture}{test_id="missing"}), start + 250, start + 1750, kind: "active_power")
        end
      ensure
        client.close
      end
    end
  end
end
