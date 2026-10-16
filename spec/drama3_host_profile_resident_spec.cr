require "spec"
require "../src/ml/audio/drama3/resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  describe "DRAMA-3 host profile queue transparency" do
    ["0", "2"].each do |depth|
      it "preserves workspace outputs and queue/read ledgers at depth #{depth}" do
        previous = {"DRAMA3_HOST_PROFILE"            => ENV["DRAMA3_HOST_PROFILE"]?,
                    "DRAMA3_COGNIGRAPH_MAX_INFLIGHT" => ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?}
        begin
          ledgers = [] of Hash(String, Int64)
          ["0", "1"].each do |enabled|
            ENV["DRAMA3_HOST_PROFILE"] = enabled
            ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth
            context = ML::Audio::Drama3::Resident::Context.new
            before = context.host_profile.snapshot
            source = context.upload([1_f32, 2_f32, 3_f32])
            output = context.zeros(5)
            workspace = context.new_workspace
            2.times do
              context.with_workspace(workspace) do
                doubled = context.add(source, source)
                context.copy(doubled, output.slice(1, 3))
              end
              context.submit
            end
            context.read(output, "terminal").should eq([0_f32, 2_f32, 4_f32, 6_f32, 0_f32])
            context.release_workspace(workspace)
            context.readbacks.should eq({"terminal" => 20_i64})
            ledger = context.graph_stats
            ledger["submitted"].should eq(2)
            ledger["completed"].should eq(2)
            ledger["completion_waits"].should eq(2)
            ledger["pending"].should eq(0)
            ledgers << ledger
            delta = JSON.parse(ML::Audio::Drama3::HostProfile.delta(before, context.host_profile.snapshot).to_json)
            if enabled == "1"
              delta["counters"]["KernelDispatches"].as_i.should eq(5)
              delta["counters"]["EncoderEnds"].as_i.should eq(2)
              events = delta["events"].as_a.to_h { |event| {event["name"].as_s, event} }
              events["CompletionWait"]["calls"].as_i.should eq(2)
              events["CommandCreate"]["calls"].as_i.should eq(2)
              events["Workspace"]["calls"].as_i.should eq(2)
              events["BufferView"]["calls"].as_i.should be > 0
              events["EncoderAPI"]["calls"].as_i.should be > 0
            else
              delta["events"].as_a.should be_empty
              delta["counters"]["KernelDispatches"].as_i.should eq(0)
            end
          end
          ledgers[0].should eq(ledgers[1])
        ensure
          previous.each do |key, value|
            if value
              ENV[key] = value
            else
              ENV.delete(key)
            end
          end
        end
      end
    end
  end
{% end %}
