require "spec"
require "../src/ml/audio/drama3/tower_resident"

include ML::Audio::Drama3

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private struct Drama3TowerWorkspaceObservation
    getter frame_storage : Array(Float32)
    getter readbacks : Hash(String, Int64)
    getter graph_stats : Hash(String, Int64)
    getter workspace_stats : Hash(String, Int64)
    getter released_stats : Hash(String, Int64)

    def initialize(@frame_storage : Array(Float32), @readbacks : Hash(String, Int64),
                   @graph_stats : Hash(String, Int64), @workspace_stats : Hash(String, Int64),
                   @released_stats : Hash(String, Int64))
    end
  end

  private def drama3_tower_workspace_run(depth : Int32, reuse : Bool) : Drama3TowerWorkspaceObservation
    previous_backend = ENV["DRAMA3_BACKEND"]?
    previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
    begin
      ENV["DRAMA3_BACKEND"] = "metal"
      ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth.to_s
      pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

      path = File.join(__DIR__, "support/drama3/reference_fixture")
      config = TowerConfig.from_json(File.read(File.join(path, "config.json")))
      context = Resident::Context.new
      tower = TowerResident.new(config, WeightStore.new(path), context, 4)
      steps = 3
      frame_width = config.num_codebooks + 1
      draws_count = steps * (config.num_codebooks - 1)

      # Every input and output backing is external to the workspace. Each
      # output row becomes the next call's previous_frame without a host copy.
      main_inputs = context.upload([20_f32, 21_f32, 22_f32])
      hidden_values = Array(Float32).new(steps * config.dim) do |index|
        (((index % 9) - 4).to_f32 * 0.013_f32)
      end
      hidden_inputs = context.upload(hidden_values)
      initial_previous = context.upload([20_f32] + Array(Float32).new(config.num_codebooks, 0_f32))
      draws = context.upload(Array(Float32).new(draws_count) do |index|
        ((index + 1).to_f32 / (draws_count + 1).to_f32)
      end)
      frame_storage = context.upload([111_f32] + Array(Float32).new(steps * frame_width, 0_f32) + [888_f32])
      output_frames = Array(Resident::Buffer).new(steps) do |index|
        frame_storage.slice(1 + index * frame_width, frame_width)
      end
      workspace = reuse ? context.new_workspace : nil
      first_workspace_slots = 0_i64
      first_workspace_bytes = 0_i64
      first_backing_allocations = 0_i64

      steps.times do |step|
        previous_frame = step == 0 ? initial_previous : output_frames[step - 1]
        draw_index = step * (config.num_codebooks - 1)
        sample = ->(logits : Resident::Buffer) do
          draw = draws.slice(draw_index, 1)
          draw_index += 1
          context.sample(logits, draw, 1.0, config.vocab_size, 1.0)
        end

        if active_workspace = workspace
          context.with_workspace(active_workspace) do
            tower.sample_frame(
              main_inputs.slice(step, 1),
              hidden_inputs.slice(step * config.dim, config.dim),
              previous_frame,
              output_frame: output_frames[step],
            ) { |logits| sample.call(logits) }
          end
        else
          tower.sample_frame(
            main_inputs.slice(step, 1),
            hidden_inputs.slice(step * config.dim, config.dim),
            previous_frame,
            output_frame: output_frames[step],
          ) { |logits| sample.call(logits) }
        end

        if active_workspace = workspace
          stats = active_workspace.stats
          if step == 0
            first_workspace_slots = stats["slots"]
            first_workspace_bytes = stats["bytes"]
            first_backing_allocations = context.allocation_stats[:allocations]
          else
            # In particular, attention score requests grow with tower offset.
            # Their reserved max-capacity backing must make later uses allocation-free.
            stats["slots"].should eq(first_workspace_slots)
            stats["bytes"].should eq(first_workspace_bytes)
            context.allocation_stats[:allocations].should eq(first_backing_allocations)
          end
        end

        # One tracked queue submission per offset deliberately leaves depth-two
        # graph work in flight while the next workspace call reuses its slots.
        context.submit
      end

      context.readbacks.should be_empty
      if active_workspace = workspace
        active_workspace.stats["uses"].should eq(steps.to_i64)
        active_workspace.stats["quarantined"].should eq(0_i64)
        active_workspace.stats["slots"].should be > 0_i64
        active_workspace.stats["bytes"].should be > 0_i64
        active_workspace.stats["reuses"].should be > 0_i64
      end

      frame_values = context.read(frame_storage, "terminal").to_a
      frame_values.first.should eq(111_f32)
      frame_values.last.should eq(888_f32)
      context.graph_stats["submitted"].should eq(steps.to_i64)
      context.graph_stats["completed"].should eq(steps.to_i64)
      context.graph_stats["pending"].should eq(0_i64)
      context.graph_stats["retained_bytes"].should eq(0_i64)
      if depth == 2
        context.graph_stats["max_pending"].should eq(2_i64)
      else
        context.graph_stats["max_pending"].should eq(0_i64)
      end

      expected_readbacks = {
        "control"  => 4_i64,
        "terminal" => frame_values.size.to_i64 * 4_i64,
      }
      context.readbacks.should eq(expected_readbacks)

      workspace_stats = if active_workspace = workspace
                          active_workspace.stats.dup
                        else
                          Hash(String, Int64).new
                        end
      graph_stats = context.graph_stats.dup
      readbacks = context.readbacks.dup
      released_stats = if active_workspace = workspace
                         context.release_workspace(active_workspace)
                         active_workspace.stats.dup
                       else
                         Hash(String, Int64).new
                       end
      Drama3TowerWorkspaceObservation.new(frame_values, readbacks, graph_stats, workspace_stats, released_stats)
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

  describe "DRAMA-3 tower scratch workspace" do
    {0, 2}.each do |depth|
      it "reuses reserved score and tower scratch storage at graph depth #{depth}" do
        fresh = drama3_tower_workspace_run(depth, false)
        reused = drama3_tower_workspace_run(depth, true)

        reused.frame_storage.should eq(fresh.frame_storage)
        reused.readbacks.should eq(fresh.readbacks)
        reused.graph_stats["submitted"].should eq(fresh.graph_stats["submitted"])
        reused.graph_stats["completed"].should eq(fresh.graph_stats["completed"])
        reused.graph_stats["max_pending"].should eq(fresh.graph_stats["max_pending"])
        reused.graph_stats["completion_waits"].should eq(fresh.graph_stats["completion_waits"])
        reused.frame_storage[0].should eq(111_f32)
        reused.frame_storage.last.should eq(888_f32)

        stats = reused.workspace_stats
        stats["uses"].should eq(3_i64)
        stats["reuses"].should be > 0_i64
        stats["quarantined"].should eq(0_i64)
        stats["released"].should eq(0_i64)
        stats["slots"].should be > 0_i64
        stats["bytes"].should be > 0_i64

        released = reused.released_stats
        released["released"].should eq(1_i64)
        released["slots"].should eq(0_i64)
        released["bytes"].should eq(0_i64)
        released["uses"].should eq(stats["uses"])
        released["reuses"].should eq(stats["reuses"])
      end
    end
  end
{% end %}
