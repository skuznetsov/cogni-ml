require "spec"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  lib MPSAPIProfileClockProbe
    fun cpu_probe = gs_mps_api_profile_cpu_probe(
      wall_ns : UInt64*, thread_cpu_ns : UInt64*,
    ) : Int32
    fun clock_probe = gs_mps_api_profile_clock_probe(
      wall_ns : UInt64*, thread_cpu_ns : UInt64*, nested_api_wall_ns : UInt64*,
      nested_mach_wall_ns : UInt64*,
    ) : Int32
  end

  describe "MPS API profile wall clock bridge" do
    it "preserves the original two-output CPU probe ABI" do
      wall_ns = 0_u64
      thread_cpu_ns = 0_u64
      MPSAPIProfileClockProbe.cpu_probe(pointerof(wall_ns),
        pointerof(thread_cpu_ns)).should eq(0)

      wall_ns.should be > 0_u64
      thread_cpu_ns.should be > 0_u64
    end

    it "keeps a CPU-only nested Mach interval within its enclosing API timer" do
      wall_ns = 0_u64
      thread_cpu_ns = 0_u64
      nested_api_wall_ns = 0_u64
      nested_mach_wall_ns = 0_u64
      MPSAPIProfileClockProbe.clock_probe(pointerof(wall_ns),
        pointerof(thread_cpu_ns), pointerof(nested_api_wall_ns),
        pointerof(nested_mach_wall_ns)).should eq(0)

      wall_ns.should be > 0_u64
      thread_cpu_ns.should be > 0_u64
      nested_api_wall_ns.should be > 0_u64
      nested_mach_wall_ns.should be > 0_u64
      wall_ns.should be >= nested_api_wall_ns
      nested_api_wall_ns.should be >= nested_mach_wall_ns
      STDOUT.puts("CPU clock pair: API=#{nested_api_wall_ns}ns Mach=#{nested_mach_wall_ns}ns graph=#{wall_ns}ns thread_cpu=#{thread_cpu_ns}ns")
    end
  end
{% end %}
