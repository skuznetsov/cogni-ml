require "json"
require "http/client"
require "uri"

module ML::Energy
  # Raw observations, rather than query_range's interpolated evaluation grid.
  class PowerSeries
    getter metric : Hash(String, String)
    getter samples : Array(Tuple(Int64, Float64))

    def initialize(@metric, @samples)
    end

    private def covered_segments(start_ms : Int64, end_ms : Int64, max_gap_ms : Int64, &)
      raise ArgumentError.new("measurement window must have positive duration") unless end_ms > start_ms
      raise ArgumentError.new("max sample gap must be positive") unless max_gap_ms > 0
      if @samples.size < 2 || @samples.first[0] > start_ms || @samples.last[0] < end_ms
        raise ArgumentError.new("insufficient raw sample coverage for request window")
      end
      @samples.each_cons(2) do |pair|
        left, right = pair
        next if right[0] <= start_ms || left[0] >= end_ms
        raise ArgumentError.new("raw measurement gap exceeds configured maximum") if right[0] - left[0] > max_gap_ms
        yield left, right, Math.max(start_ms, left[0]), Math.min(end_ms, right[0])
      end
    end

    # Validity is a held status, never a linearly interpolated physical value.
    def require_valid!(start_ms : Int64, end_ms : Int64, max_gap_ms : Int64)
      covered_segments(start_ms, end_ms, max_gap_ms) do |left, right, lo, hi|
        raise ArgumentError.new("collector reported invalid measurement") unless left[1] == 1.0
      end
      at_end = @samples.reverse.find { |point| point[0] <= end_ms }
      raise ArgumentError.new("collector reported invalid measurement") unless at_end && at_end[1] == 1.0
    end

    def integrate(start_ms : Int64, end_ms : Int64, kind : String = "current_ac",
                  voltage : Float64 = 116.0, power_factor : Float64? = nil,
                  max_gap_ms : Int64 = 5000_i64) : JSON::Any
      raise ArgumentError.new("measurement kind must be current_ac or active_power") unless {"current_ac", "active_power"}.includes?(kind)
      if kind == "active_power" && !@metric["__name__"]?.try(&.ends_with?("_watts"))
        raise ArgumentError.new("active_power requires an explicitly selected metric ending in _watts")
      end
      raise ArgumentError.new("voltage must be positive and finite") unless voltage.finite? && voltage > 0
      if pf = power_factor
        raise ArgumentError.new("power factor must be finite and in (0, 1]") unless pf.finite? && pf > 0 && pf <= 1
        raise ArgumentError.new("power factor is only applicable to current_ac") unless kind == "current_ac"
      end
      integral = 0.0
      points = Set(Int64).new
      curve = {} of Int64 => Float64
      covered_segments(start_ms, end_ms, max_gap_ms) do |left, right, lo, hi|
        span = (right[0] - left[0]).to_f64
        low_value = left[1] + (right[1] - left[1]) * (lo - left[0]) / span
        high_value = left[1] + (right[1] - left[1]) * (hi - left[0]) / span
        integral += (low_value + high_value) * 0.5 * (hi - lo) / 1000.0
        points << left[0] << right[0]
        scale = kind == "current_ac" ? voltage : 1.0
        curve[lo - start_ms] = low_value * scale
        curve[hi - start_ms] = high_value * scale
      end
      duration = (end_ms - start_ms) / 1000.0
      apparent = kind == "current_ac" ? integral * voltage : nil
      active = kind == "active_power" ? integral : nil
      estimated = apparent.try { |value| power_factor.try { |pf| value * pf } }
      raise ArgumentError.new("measurement integral overflow") unless integral.finite? && (apparent.nil? || apparent.finite?)
      JSON.parse(JSON.build do |j|
        j.object do
          j.field "status", "available"
          j.field "kind", kind
          j.field "unit_contract", kind == "current_ac" ? "fluke_current_amperes with mode=ac; RMS voltage assumed" : "explicit source metric ending in _watts; source must measure active power"
          j.field "metric", @metric
          j.field "start_ms", start_ms
          j.field "end_ms", end_ms
          j.field "duration_seconds", duration
          j.field "sample_count", points.size
          j.field "max_gap_ms", max_gap_ms
          j.field "integration", "linear interpolation of bracketing raw samples"
          j.field "power_unit", kind == "current_ac" ? "VA" : "W"
          j.field "curve" do
            j.array do
              curve.to_a.sort_by(&.[0]).each do |elapsed, value|
                j.object do
                  j.field "elapsed_ms", elapsed
                  j.field "value", value
                end
              end
            end
          end
          j.field "assumed_voltage_volts", kind == "current_ac" ? voltage : nil
          j.field "assumed_power_factor", power_factor
          j.field "mean_apparent_power_va", apparent.try { |value| value / duration }
          j.field "apparent_energy_va_seconds", apparent
          j.field "mean_active_power_watts", active.try { |value| value / duration }
          j.field "active_energy_joules", active
          j.field "estimated_active_energy_joules", estimated
        end
      end)
    end
  end

  class VictoriaMetrics
    def self.parse_export(raw : String) : PowerSeries
      metric = nil.as(Hash(String, String)?)
      points = {} of Int64 => Float64
      raw.each_line do |line|
        next if line.strip.empty?
        row = JSON.parse(line)
        labels = row["metric"].as_h.transform_values(&.as_s)
        raise ArgumentError.new("raw export must select exactly one series") if metric && metric != labels
        metric = labels
        values = row["values"].as_a
        timestamps = row["timestamps"].as_a
        raise ArgumentError.new("raw export value/timestamp lengths differ") unless values.size == timestamps.size
        values.zip(timestamps).each do |value, timestamp|
          number = value.as_f? || value.as_i64?.try(&.to_f64)
          raise ArgumentError.new("sample must be finite and nonnegative") unless number && number.finite? && number >= 0
          time = timestamp.as_i64
          if old = points[time]?
            raise ArgumentError.new("conflicting duplicate measurement timestamp") unless old == number
          end
          points[time] = number
        end
      end
      raise ArgumentError.new("raw export has no measurement samples") if metric.nil? || points.empty?
      PowerSeries.new(metric.not_nil!, points.to_a.sort_by(&.[0]))
    end

    def initialize(url : String, @bearer_token : String? = nil)
      @base = URI.parse(url)
      unless {"http", "https"}.includes?(@base.scheme) && @base.host && @base.user.nil? && @base.password.nil? && @base.query.nil? && @base.fragment.nil?
        raise ArgumentError.new("VictoriaMetrics URL must be HTTP(S) without credentials, query or fragment")
      end
      if @bearer_token && @base.scheme == "http" && !{"127.0.0.1", "localhost", "::1", "[::1]"}.includes?(@base.host)
        raise ArgumentError.new("bearer authentication requires HTTPS for a non-loopback VictoriaMetrics URL")
      end
    end

    def read_series(selector : String, start_ms : Int64, end_ms : Int64,
                    max_gap_ms : Int64 = 5000_i64) : PowerSeries
      raise ArgumentError.new("selector must not be empty") if selector.strip.empty?
      raise ArgumentError.new("invalid measurement window") unless end_ms > start_ms && max_gap_ms > 0
      uri = @base.dup
      uri.path = uri.path.rstrip('/') + "/api/v1/export"
      uri.query = URI::Params.encode({
        "match[]" => selector,
        "start"   => ((start_ms - max_gap_ms) / 1000.0).to_s,
        "end"     => ((end_ms + max_gap_ms) / 1000.0).to_s,
      })
      headers = HTTP::Headers.new
      if token = @bearer_token
        headers["Authorization"] = "Bearer #{token}"
      end
      response = HTTP::Client.new(uri) do |client|
        client.connect_timeout = 10.seconds
        client.read_timeout = 30.seconds
        client.get(uri.request_target, headers)
      end
      raise ArgumentError.new("VictoriaMetrics export failed with HTTP #{response.status_code}") unless response.status_code == 200
      self.class.parse_export(response.body)
    end

    def measure(selector : String, start_ms : Int64, end_ms : Int64,
                kind : String = "current_ac", voltage : Float64 = 116.0,
                power_factor : Float64? = nil, max_gap_ms : Int64 = 5000_i64) : JSON::Any
      series = read_series(selector, start_ms, end_ms, max_gap_ms)
      if kind == "current_ac"
        unless series.metric["__name__"]? == "fluke_current_amperes" && series.metric["mode"]? == "ac"
          raise ArgumentError.new("current_ac requires fluke_current_amperes with mode=ac")
        end
        labels = series.metric.reject { |name, _| name == "__name__" }
        validity_selector = "fluke_measurement_valid{" + labels.map { |name, value| "#{name}=#{value.to_json}" }.join(',') + "}"
        validity = read_series(validity_selector, start_ms, end_ms, max_gap_ms)
        unless validity.metric.reject { |name, _| name == "__name__" } == labels
          raise ArgumentError.new("collector validity labels do not match physical series")
        end
        validity.require_valid!(start_ms, end_ms, max_gap_ms)
      end
      series.integrate(start_ms, end_ms, kind, voltage, power_factor, max_gap_ms)
    end
  end
end
