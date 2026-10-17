require "spec"
require "http/server"
require "../src/ml/energy/prometheus_remote_write"

describe ML::Energy::PrometheusRemoteWrite do
  it "encodes a remote-write request as protobuf in a raw Snappy block" do
    series = ML::Energy::PrometheusRemoteWrite::Series.new(
      "cogni_llm_generated_tokens_total",
      {"trial" => "1", "host" => "gputer"},
      [ML::Energy::PrometheusRemoteWrite::Sample.new(1_700_000_000_001_i64, 0.0),
       ML::Energy::PrometheusRemoteWrite::Sample.new(1_700_000_000_002_i64, 4.0)])

    encoded = ML::Energy::PrometheusRemoteWrite.encode([series])
    encoded.should_not be_empty
    decoded = ML::Energy::PrometheusRemoteWrite.decode_raw_snappy(encoded)
    decoded.should eq(ML::Energy::PrometheusRemoteWrite.encode_protobuf([series]))
    decoded_series = ML::Energy::PrometheusRemoteWrite.decode_write_request(decoded)
    decoded_series.size.should eq(1)
    decoded_series[0].labels.should eq([
      {"__name__", "cogni_llm_generated_tokens_total"}, {"host", "gputer"}, {"trial", "1"},
    ])
    decoded_series[0].samples.should eq([{1_700_000_000_001_i64, 0.0}, {1_700_000_000_002_i64, 4.0}])
  end

  it "sets the required headers and retries a transient server failure" do
    attempts = 0
    request_headers = [] of HTTP::Headers
    server = HTTP::Server.new do |context|
      attempts += 1
      request_headers << context.request.headers
      context.response.status_code = attempts == 1 ? 503 : 204
    end
    address = server.bind_unused_port
    spawn { server.listen }
    begin
      client = ML::Energy::PrometheusRemoteWrite.new("http://127.0.0.1:#{address.port}", retry_delay: 1.millisecond)
      series = ML::Energy::PrometheusRemoteWrite::Series.new(
        "cogni_llm_generated_tokens_total", {"host" => "gputer"},
        [ML::Energy::PrometheusRemoteWrite::Sample.new(10_i64, 1.0)])
      client.write([series]).should eq(204)
      attempts.should eq(2)
      headers = request_headers.last
      headers["Content-Encoding"].should eq("snappy")
      headers["Content-Type"].should eq("application/x-protobuf")
      headers["X-Prometheus-Remote-Write-Version"].should eq("0.1.0")
      headers["User-Agent"].should start_with("cogni-ml/")
      request_headers.last["Content-Length"].to_i.should be > 0
    ensure
      server.close
    end
  end

  it "does not retry permanent client errors and bounds retries" do
    attempts = 0
    server = HTTP::Server.new do |context|
      attempts += 1
      context.response.status_code = 400
    end
    address = server.bind_unused_port
    spawn { server.listen }
    begin
      client = ML::Energy::PrometheusRemoteWrite.new("http://127.0.0.1:#{address.port}", retry_delay: 1.millisecond)
      series = ML::Energy::PrometheusRemoteWrite::Series.new(
        "cogni_llm_generated_tokens_total", {"host" => "gputer"},
        [ML::Energy::PrometheusRemoteWrite::Sample.new(10_i64, 1.0)])
      expect_raises(ML::Energy::PrometheusRemoteWrite::WriteError, /HTTP 400/) { client.write([series]) }
      attempts.should eq(1)
    ensure
      server.close
    end
  end

  it "retries 429 and stops after the bounded retry count" do
    attempts = 0
    server = HTTP::Server.new do |context|
      attempts += 1
      context.response.status_code = attempts == 1 ? 429 : 204
    end
    address = server.bind_unused_port
    spawn { server.listen }
    begin
      client = ML::Energy::PrometheusRemoteWrite.new("http://127.0.0.1:#{address.port}", retry_delay: 1.millisecond)
      series = ML::Energy::PrometheusRemoteWrite::Series.new(
        "cogni_llm_generated_tokens_total", {"host" => "gputer"},
        [ML::Energy::PrometheusRemoteWrite::Sample.new(10_i64, 1.0)])
      client.write([series]).should eq(204)
      attempts.should eq(2)
    ensure
      server.close
    end

    attempts = 0
    failing_server = HTTP::Server.new do |context|
      attempts += 1
      context.response.status_code = 503
    end
    address = failing_server.bind_unused_port
    spawn { failing_server.listen }
    begin
      client = ML::Energy::PrometheusRemoteWrite.new("http://127.0.0.1:#{address.port}", retry_delay: 1.millisecond)
      expect_raises(ML::Energy::PrometheusRemoteWrite::WriteError, /after 3 attempt/) { client.write([series]) }
      attempts.should eq(3)
    ensure
      failing_server.close
    end
  end

  it "rejects invalid samples and unordered timestamps" do
    expect_raises(ArgumentError, /finite/) do
      ML::Energy::PrometheusRemoteWrite::Sample.new(1_i64, Float64::NAN)
    end
    expect_raises(ArgumentError, /strictly increasing/) do
      ML::Energy::PrometheusRemoteWrite::Series.new(
        "cogni_llm_generated_tokens_total", {"host" => "gputer"},
        [ML::Energy::PrometheusRemoteWrite::Sample.new(2_i64, 1.0),
         ML::Energy::PrometheusRemoteWrite::Sample.new(2_i64, 2.0)])
    end
  end
end
