require "spec"
require "../src/ml/energy/victoria_metrics"

private def vm_row(values = "[1,3,1]", timestamps = "[0,1000,2000]", device = "meter")
  %({"metric":{"__name__":"fluke_current_amperes","mode":"ac","device":"#{device}"},"values":#{values},"timestamps":#{timestamps}})
end

describe ML::Energy::VictoriaMetrics do
  it "does not transmit bearer tokens over remote plain HTTP" do
    expect_raises(ArgumentError) { ML::Energy::VictoriaMetrics.new("http://example.invalid:8428", "test-token") }
    ML::Energy::VictoriaMetrics.new("http://127.0.0.1:8428", "test-token")
    ML::Energy::VictoriaMetrics.new("https://example.invalid", "test-token")
  end
  it "merges split raw rows, sorts and deduplicates without double counting" do
    series = ML::Energy::VictoriaMetrics.parse_export(vm_row("[3,1]", "[1000,0]") + "\n" + vm_row("[1,3]", "[2000,1000]"))
    result = series.integrate(250_i64, 1750_i64, "current_ac", 116.0, nil, 1100_i64)
    result["apparent_energy_va_seconds"].as_f.should be_close(391.5, 1e-8)
    result["active_energy_joules"].raw.should be_nil
    result["estimated_active_energy_joules"].raw.should be_nil
    result["sample_count"].as_i.should eq(3)
    result["power_unit"].as_s.should eq("VA")
    result["curve"].as_a.map { |point| point["elapsed_ms"].as_i }.should eq([0, 750, 1500])
    result["curve"].as_a.map { |point| point["value"].as_f }.should eq([174.0, 348.0, 174.0])
  end

  it "distinguishes measured joules from an explicit power factor estimate" do
    series = ML::Energy::VictoriaMetrics.parse_export(vm_row)
    watts = ML::Energy::VictoriaMetrics.parse_export(vm_row.gsub("fluke_current_amperes", "bench_power_watts"))
    measured = watts.integrate(0_i64, 2000_i64, "active_power", 116.0, nil, 1100_i64)
    measured["active_energy_joules"].as_f.should eq(4.0)
    expect_raises(ArgumentError, /_watts/) { series.integrate(0_i64, 2000_i64, "active_power") }
    estimated = series.integrate(0_i64, 2000_i64, "current_ac", 116.0, 0.9, 1100_i64)
    estimated["estimated_active_energy_joules"].as_f.should be_close(417.6, 1e-8)
    estimated["active_energy_joules"].raw.should be_nil
  end

  it "rejects missing coverage, long gaps, invalid windows and invalid units" do
    series = ML::Energy::VictoriaMetrics.parse_export(vm_row)
    expect_raises(ArgumentError, /coverage/) { series.integrate(-1_i64, 1000_i64) }
    expect_raises(ArgumentError, /coverage/) { series.integrate(0_i64, 2001_i64) }
    expect_raises(ArgumentError, /gap/) { series.integrate(0_i64, 2000_i64, max_gap_ms: 500_i64) }
    expect_raises(ArgumentError) { series.integrate(1000_i64, 1000_i64) }
    expect_raises(ArgumentError) { series.integrate(0_i64, 1000_i64, voltage: 0.0) }
    expect_raises(ArgumentError) { series.integrate(0_i64, 1000_i64, power_factor: 1.1) }
    expect_raises(ArgumentError) { series.integrate(0_i64, 1000_i64, kind: "watts_maybe") }
  end

  it "rejects ambiguous series, conflicting duplicates, bad values and mismatched arrays" do
    expect_raises(ArgumentError, /one series/) { ML::Energy::VictoriaMetrics.parse_export(vm_row + "\n" + vm_row(device: "other")) }
    expect_raises(ArgumentError, /duplicate/) { ML::Energy::VictoriaMetrics.parse_export(vm_row + "\n" + vm_row("[9]", "[1000]")) }
    expect_raises(ArgumentError) { ML::Energy::VictoriaMetrics.parse_export(vm_row("[-1]", "[0]")) }
    expect_raises(ArgumentError) { ML::Energy::VictoriaMetrics.parse_export(vm_row("[1]", "[0,1000]")) }
    expect_raises(ArgumentError) { ML::Energy::VictoriaMetrics.parse_export("") }
  end

  it "refuses an invalid collector status inside a seemingly covered current window" do
    status = ML::Energy::VictoriaMetrics.parse_export(vm_row("[1,0,1]"))
    expect_raises(ArgumentError, /invalid/) { status.require_valid!(250_i64, 1750_i64, 1100_i64) }
    valid = ML::Energy::VictoriaMetrics.parse_export(vm_row("[1,1,1]"))
    valid.require_valid!(250_i64, 1750_i64, 1100_i64)
  end

  it "refuses credential URLs and unbounded query parameters" do
    expect_raises(ArgumentError) { ML::Energy::VictoriaMetrics.new("http://user:secret@localhost:8428") }
    expect_raises(ArgumentError) { ML::Energy::VictoriaMetrics.new("http://localhost:8428?token=secret") }
  end
end
