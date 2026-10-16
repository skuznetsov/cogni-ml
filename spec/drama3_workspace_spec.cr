require "spec"
require "../src/ml/audio/drama3/resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private def drama3_workspace_backing(buffer : ML::Audio::Drama3::Resident::Buffer) : Tuple(UInt64, Int64)
    {buffer.buffer.handle.address.to_u64, buffer.offset}
  end

  describe "DRAMA-3 resident allocation workspaces" do
    it "reuses fixed slots across graph depths without changing the command ledger" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        outputs_by_depth = [] of Array(Array(UInt32))
        ledgers = [] of Tuple(Int64, Int64, Hash(String, Int64))
        [0, 2].each do |depth|
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth.to_s
          context = ML::Audio::Drama3::Resident::Context.new
          left = context.upload([1.25_f32, -2.5_f32])
          right = context.upload([0.5_f32, 1_f32])
          output_one = context.alloc(2)
          output_two = context.alloc(2)
          workspace = context.new_workspace
          left_backing = drama3_workspace_backing(left)
          output_one_backing = drama3_workspace_backing(output_one)
          output_two_backing = drama3_workspace_backing(output_two)

          first_scratch = nil.as(Tuple(UInt64, Int64)?)
          context.with_workspace(workspace) do
            value = context.add(left, right)
            first_scratch = drama3_workspace_backing(value)
            context.copy(value, output_one)
          end
          context.submit
          allocations_after_warmup = context.allocation_stats[:allocations]
          context.readbacks.should be_empty

          second_scratch = nil.as(Tuple(UInt64, Int64)?)
          context.with_workspace(workspace) do
            value = context.add(output_one, right)
            second_scratch = drama3_workspace_backing(value)
            context.copy(value, output_two)
          end
          context.submit
          context.allocation_stats[:allocations].should eq(allocations_after_warmup)
          first_scratch.should eq(second_scratch)
          drama3_workspace_backing(left).should eq(left_backing)
          drama3_workspace_backing(output_one).should eq(output_one_backing)
          drama3_workspace_backing(output_two).should eq(output_two_backing)
          context.readbacks.should be_empty
          context.graph_stats["pending"].should eq(depth == 2 ? 2_i64 : 0_i64)
          context.graph_stats["completion_waits"].should eq(depth == 2 ? 0_i64 : 2_i64)

          GC.collect
          context.finish
          context.graph_stats["submitted"].should eq(2_i64)
          context.graph_stats["pending"].should eq(0_i64)
          context.graph_stats["completion_waits"].should eq(2_i64)
          first = context.read(output_one, "terminal")
          second = context.read(output_two, "terminal")
          first.map(&.unsafe_as(UInt32)).should eq([1.75_f32, -1.5_f32].map(&.unsafe_as(UInt32)))
          second.map(&.unsafe_as(UInt32)).should eq([2.25_f32, -0.5_f32].map(&.unsafe_as(UInt32)))
          context.readbacks.should eq({"terminal" => 16_i64})

          stats = workspace.stats
          stats["slots"].should eq(1_i64)
          stats["bytes"].should eq(8_i64)
          stats["uses"].should eq(2_i64)
          stats["reuses"].should eq(1_i64)
          outputs_by_depth << [first.map(&.unsafe_as(UInt32)), second.map(&.unsafe_as(UInt32))]
          ledgers << {context.submissions, context.graph_stats["completion_waits"], context.readbacks.dup}
        end
        outputs_by_depth.uniq.size.should eq(1)
        ledgers.uniq.size.should eq(1)
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
        if previous_depth
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous_depth
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
      end
    end

    it "clears a reused zeros slot before its next call" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "0"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        context = ML::Audio::Drama3::Resident::Context.new
        workspace = context.new_workspace
        nonzero = context.upload([3_f32, -7_f32])
        output = context.alloc(2)
        context.with_workspace(workspace) do
          scratch = context.zeros(2)
          context.copy(nonzero, scratch)
        end
        context.submit
        context.with_workspace(workspace) do
          scratch = context.zeros(2)
          context.copy(scratch, output)
        end
        context.submit
        context.read(output, "terminal").map(&.unsafe_as(UInt32)).should eq([0_f32, 0_f32].map(&.unsafe_as(UInt32)))
        workspace.stats["reuses"].should eq(1_i64)
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
        if previous_depth
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous_depth
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
      end
    end

    it "reserves a larger slot once and rejects capacity overflow" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "0"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        context = ML::Audio::Drama3::Resident::Context.new
        workspace = context.new_workspace
        one = nil.as(ML::Audio::Drama3::Resident::Buffer?)
        context.with_workspace(workspace) do
          one = context.alloc(1, reserve_count: 8)
          one.not_nil!.count.should eq(1)
          one.not_nil!.buffer.size.should eq(32_i64)
        end
        context.with_workspace(workspace) do
          eight = context.alloc(8, reserve_count: 8)
          eight.count.should eq(8)
          drama3_workspace_backing(eight).should eq(drama3_workspace_backing(one.not_nil!))
        end
        context.allocation_stats[:allocations].should eq(1_i64)
        workspace.stats["slots"].should eq(1_i64)
        workspace.stats["bytes"].should eq(32_i64)

        context.with_workspace(workspace) do
          expect_raises(ArgumentError) { context.alloc(9, reserve_count: 8) }
          expect_raises(ArgumentError) { context.alloc(1, reserve_count: 0) }
          expect_raises(ArgumentError) { context.alloc(0, reserve_count: 8) }
        end
        context.allocation_stats[:allocations].should eq(1_i64)
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
        if previous_depth
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous_depth
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
      end
    end

    it "keeps uploaded inputs, converted weights, and outputs outside scoped scratch" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "0"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        context = ML::Audio::Drama3::Resident::Context.new
        workspace = context.new_workspace
        stable_input = context.upload([2_f32, -4_f32])
        output = context.alloc(2)
        weight_tensor = ML::Audio::Drama3::Tensor.new([2], [5_f32, 7_f32])
        stable_input_backing = drama3_workspace_backing(stable_input)
        output_backing = drama3_workspace_backing(output)
        weight_one = nil.as(ML::Audio::Drama3::Resident::Buffer?)
        weight_two = nil.as(ML::Audio::Drama3::Resident::Buffer?)
        upload_one = nil.as(ML::Audio::Drama3::Resident::Buffer?)
        upload_two = nil.as(ML::Audio::Drama3::Resident::Buffer?)
        scratch_one = nil.as(Tuple(UInt64, Int64)?)
        scratch_two = nil.as(Tuple(UInt64, Int64)?)

        context.with_workspace(workspace) do
          weight_one = context.tensor(weight_tensor)
          upload_one = context.upload([11_f32, 13_f32])
          scratch = context.alloc(2)
          scratch_one = drama3_workspace_backing(scratch)
          context.copy(upload_one.not_nil!, scratch)
          context.copy(scratch, output)
        end
        context.submit
        context.with_workspace(workspace) do
          weight_two = context.tensor(weight_tensor)
          upload_two = context.upload([17_f32, 19_f32])
          scratch = context.alloc(2)
          scratch_two = drama3_workspace_backing(scratch)
          context.copy(upload_two.not_nil!, scratch)
          context.copy(scratch, output)
        end
        context.submit

        drama3_workspace_backing(stable_input).should eq(stable_input_backing)
        drama3_workspace_backing(output).should eq(output_backing)
        drama3_workspace_backing(weight_one.not_nil!).should eq(drama3_workspace_backing(weight_two.not_nil!))
        drama3_workspace_backing(weight_one.not_nil!).should_not eq(scratch_one)
        drama3_workspace_backing(upload_one.not_nil!).should_not eq(drama3_workspace_backing(upload_two.not_nil!))
        drama3_workspace_backing(upload_one.not_nil!).should_not eq(scratch_one)
        scratch_one.should eq(scratch_two)
        workspace.stats["slots"].should eq(1_i64)

        context.read(output, "terminal").map(&.unsafe_as(UInt32)).should eq([17_f32, 19_f32].map(&.unsafe_as(UInt32)))
        context.read(stable_input, "terminal").map(&.unsafe_as(UInt32)).should eq([2_f32, -4_f32].map(&.unsafe_as(UInt32)))
        context.read(upload_one.not_nil!, "terminal").map(&.unsafe_as(UInt32)).should eq([11_f32, 13_f32].map(&.unsafe_as(UInt32)))
        context.read(upload_two.not_nil!, "terminal").map(&.unsafe_as(UInt32)).should eq([17_f32, 19_f32].map(&.unsafe_as(UInt32)))
        context.read(weight_one.not_nil!, "terminal").map(&.unsafe_as(UInt32)).should eq([5_f32, 7_f32].map(&.unsafe_as(UInt32)))
        context.readbacks.should eq({"terminal" => 40_i64})
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
        if previous_depth
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous_depth
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
      end
    end

    it "keeps the finite status persistent and sticky across workspace reuse" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "0"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        context = ML::Audio::Drama3::Resident::Context.new
        stable_valid = context.upload([1_f32])
        bad = context.upload([Float32::NAN])
        workspace = context.new_workspace
        # The first guard lazily creates finite status while a workspace is
        # active. The following zeros call must use a separate scratch slot.
        context.with_workspace(workspace) do
          context.guard(bad)
          context.zeros(1)
        end
        context.submit
        workspace.stats["slots"].should eq(1_i64)
        workspace.stats["bytes"].should eq(4_i64)
        context.with_workspace(workspace) do
          context.zeros(1)
          context.guard(stable_valid)
        end
        context.submit

        expect_raises(ArgumentError, /non-finite resident intermediate/) do
          context.read(stable_valid, "terminal")
        end
        context.readbacks["control"].should eq(4_i64)
        context.readbacks["terminal"].should eq(0_i64)
        workspace.stats["slots"].should eq(1_i64)
        workspace.stats["reuses"].should eq(1_i64)
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
        if previous_depth
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous_depth
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
      end
    end

    it "retains a grown slot's pending backing through GC and releases both at the terminal boundary" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        context = ML::Audio::Drama3::Resident::Context.new
        workspace = context.new_workspace
        small_left = context.upload([1_f32, -2_f32])
        small_right = context.upload([0.5_f32, 1_f32])
        large_left = context.upload([2_f32, 3_f32, 4_f32, 5_f32])
        large_right = context.upload([10_f32, 20_f32, 30_f32, 40_f32])
        output_one = context.alloc(2)
        output_two = context.alloc(4)
        small_backing = nil.as(Tuple(UInt64, Int64)?)
        large_backing = nil.as(Tuple(UInt64, Int64)?)

        context.with_workspace(workspace) do
          value = context.add(small_left, small_right)
          small_backing = drama3_workspace_backing(value)
          context.copy(value, output_one)
        end
        context.submit
        context.graph_stats["pending"].should eq(1_i64)
        context.graph_stats["completion_waits"].should eq(0_i64)

        context.with_workspace(workspace) do
          value = context.add(large_left, large_right)
          large_backing = drama3_workspace_backing(value)
          context.copy(value, output_two)
        end
        context.submit
        large_backing.should_not eq(small_backing)
        workspace.stats["slots"].should eq(1_i64)
        workspace.stats["retired_slots"].should eq(1_i64)
        workspace.stats["bytes"].should eq(24_i64)
        context.graph_stats["pending"].should eq(2_i64)
        context.graph_stats["completion_waits"].should eq(0_i64)

        GC.collect
        first = context.read(output_one, "terminal")
        second = context.read(output_two, "terminal")
        first.map(&.unsafe_as(UInt32)).should eq([1.5_f32, -1_f32].map(&.unsafe_as(UInt32)))
        second.map(&.unsafe_as(UInt32)).should eq([12_f32, 23_f32, 34_f32, 45_f32].map(&.unsafe_as(UInt32)))
        context.readbacks.should eq({"terminal" => 24_i64})
        context.graph_stats["pending"].should eq(0_i64)
        waits_at_terminal = context.graph_stats["completion_waits"]
        waits_at_terminal.should eq(2_i64)

        context.release_workspace(workspace)
        context.graph_stats["completion_waits"].should eq(waits_at_terminal)
        workspace.stats["released"].should eq(1_i64)
        workspace.stats["slots"].should eq(0_i64)
        workspace.stats["retired_slots"].should eq(0_i64)
        workspace.stats["bytes"].should eq(0_i64)
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
        if previous_depth
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous_depth
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
      end
    end

    it "quarantines retired and current growth backings after queued work aborts" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        context = ML::Audio::Drama3::Resident::Context.new
        workspace = context.new_workspace
        first_inputs = [context.upload([1_f32, 2_f32]), context.upload([3_f32, 4_f32])]
        second_inputs = [context.upload([5_f32, 6_f32, 7_f32, 8_f32]), context.upload([9_f32, 10_f32, 11_f32, 12_f32])]
        output_one = context.alloc(2)
        output_two = context.alloc(4)

        context.with_workspace(workspace) do
          context.copy(context.add(first_inputs[0], first_inputs[1]), output_one)
        end
        context.submit
        context.with_workspace(workspace) do
          context.copy(context.add(second_inputs[0], second_inputs[1]), output_two)
        end
        context.submit
        context.graph_stats["pending"].should eq(2_i64)
        workspace.stats["retired_slots"].should eq(1_i64)

        context.abort
        GC.collect
        workspace.stats["quarantined"].should eq(1_i64)
        workspace.stats["retired_slots"].should eq(1_i64)
        workspace.stats["slots"].should eq(1_i64)
        workspace.stats["bytes"].should eq(24_i64)
        expect_raises(ArgumentError, /workspace|aborted|quarantined/) do
          context.with_workspace(workspace) { }
        end
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
        if previous_depth
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous_depth
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
      end
    end

    it "rejects foreign, reentrant, released, and pending-work release transitions" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        context = ML::Audio::Drama3::Resident::Context.new
        foreign_context = ML::Audio::Drama3::Resident::Context.new
        workspace = context.new_workspace
        expect_raises(ArgumentError, /workspace|context/) do
          foreign_context.with_workspace(workspace) { }
        end

        context.with_workspace(workspace) do
          expect_raises(ArgumentError, /workspace/) do
            context.with_workspace(workspace) { }
          end
          open_scope_waits = context.graph_stats["completion_waits"]
          expect_raises(ArgumentError, /workspace|terminal|queue/) { context.release_workspace(workspace) }
          context.graph_stats["completion_waits"].should eq(open_scope_waits)
          context.alloc(1)
        end
        context.submit
        context.with_workspace(workspace) { context.zeros(1) }
        context.submit
        context.graph_stats["pending"].should eq(1_i64)
        waits_before_release_attempt = context.graph_stats["completion_waits"]
        expect_raises(ArgumentError, /workspace|pending|open/) { context.release_workspace(workspace) }
        context.graph_stats["completion_waits"].should eq(waits_before_release_attempt)
        context.graph_stats["pending"].should eq(1_i64)

        context.finish
        context.release_workspace(workspace)
        workspace.stats["released"].should eq(1_i64)
        workspace.stats["slots"].should eq(0_i64)
        workspace.stats["bytes"].should eq(0_i64)
        expect_raises(ArgumentError, /workspace|released/) { context.with_workspace(workspace) { } }
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
        if previous_depth
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous_depth
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
      end
    end

    it "quarantines a workspace when scoped work aborts" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        context = ML::Audio::Drama3::Resident::Context.new
        workspace = context.new_workspace
        expect_raises(Exception, "workspace test failure") do
          context.with_workspace(workspace) do
            context.zeros(1)
            raise Exception.new("workspace test failure")
          end
        end
        workspace.stats["quarantined"].should eq(1_i64)
        expect_raises(ArgumentError, /workspace|aborted|quarantined/) do
          context.with_workspace(workspace) { }
        end
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
        if previous_depth
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous_depth
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
      end
    end
  end
{% end %}
