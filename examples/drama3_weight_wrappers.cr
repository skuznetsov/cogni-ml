# Single-worker diagnostic only; ordinary Studio inference has no rotation API.
require "../src/ml/audio/drama3/studio/engine"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module ML::Audio::Drama3::Studio
    class Engine
      def enable_weight_wrapper_probe! : Nil
        @context.enable_weight_wrapper_probe!
      end

      def prepare_weight_wrappers(refresh : Bool, clear_views : Bool = false) : Hash(String, Int64)
        @context.prepare_weight_wrappers(refresh, clear_views)
      end

      def begin_weight_wrapper_job! : Nil
        @context.begin_weight_wrapper_job!
      end

      def prepare_weight_extents(narrow : Bool, clear_views : Bool = false) : Hash(String, Int64)
        @context.prepare_weight_extents(narrow, clear_views)
      end

      def weight_wrapper_first_command_bindings
        @context.weight_wrapper_first_command_bindings
      end

      def converted_weight_cache_signature
        @context.converted_weight_cache_signature
      end
    end
  end

  begin
    raise "usage: drama3_weight_wrappers MODEL_DIR" unless ARGV.size == 1
    engine = ML::Audio::Drama3::Studio::Engine.new(File.expand_path(ARGV[0]))
    engine.enable_weight_wrapper_probe!
    STDOUT.puts({event: "ready", model_load_seconds: engine.load_seconds}.to_json)
    STDOUT.flush
    while line = STDIN.gets
      raise "worker request too large" if line.bytesize > 128 * 1024
      input = JSON.parse(line)
      mode = input["wrappers"].as_s
      raise "wrappers must be persistent, views or fresh" unless mode.in?("persistent", "views", "fresh")
      layout = input["layout"]?.try(&.as_s)
      if layout
        raise "layout must be shard or tensor" unless layout.in?("shard", "tensor")
        raise "extent layout supports persistent or views" if mode == "fresh"
      end
      converted_before = engine.converted_weight_cache_signature
      resources_before = ML::Audio::Drama3::HostResourceSnapshot.capture
      started = Time.instant
      preparation = if layout
                      engine.prepare_weight_extents(layout == "tensor", clear_views: mode == "views")
                    else
                      engine.prepare_weight_wrappers(mode == "fresh", clear_views: mode == "views")
                    end
      preparation_seconds = (Time.instant - started).total_seconds
      resources_after = ML::Audio::Drama3::HostResourceSnapshot.capture
      engine.begin_weight_wrapper_job!
      report = engine.generate(input["input"], input["output"].as_s) do |message|
        STDOUT.puts({event: "progress", message: message}.to_json)
        STDOUT.flush
      end
      # This assertion/metadata query never submits or drains GPU work.
      after_job = engine.prepare_weight_wrappers(false)
      bindings = engine.weight_wrapper_first_command_bindings
      converted_after = engine.converted_weight_cache_signature
      converted_unchanged = converted_before == converted_after
      raise "converted GPU weight cache changed after prime" unless converted_before.empty? || converted_unchanged
      gc_before = ML::Audio::Drama3::HostProfile.gc_snapshot
      gc_started = Time.instant
      GC.collect
      gc_seconds = (Time.instant - gc_started).total_seconds
      gc_after = ML::Audio::Drama3::HostProfile.gc_snapshot
      STDOUT.puts({event: "result", report: report,
                   weight_wrappers: {mode: mode, layout: layout || "shard", preparation: preparation, after_job: after_job,
                                     first_command_bindings: bindings,
                                     converted_cache: {before_count: converted_before.size, after_count: converted_after.size,
                                                       unchanged: converted_unchanged},
                                     preparation_seconds: preparation_seconds,
                                     resources_before: resources_before, resources_after: resources_after},
                   worker_gc: {wall_seconds: gc_seconds,
                               stats: ML::Audio::Drama3::HostProfile.gc_delta(gc_before, gc_after)}}.to_json)
      STDOUT.flush
    end
  rescue ex
    STDOUT.puts({event: "error", message: ex.message || "wrapper probe failed"}.to_json)
    STDOUT.flush
    exit 1
  end
{% else %}
  abort "wrapper probe requires a Darwin build with -Ddrama3_metal"
{% end %}
