# GPU-only known-input recurrence experiment. No production file requires it.
require "digest/sha256"
require "../src/ml/audio/drama3/generator_resident"
require "./support/drama3_gdn_scan"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module Drama3GDNScan
    TEXT    = "Hello, this is a voice test."
    LAYERS  = [0, 38]
    WARMUPS = 2
    PAIRS   = 8

    # Bind the diagnostic and its capture/comparison seams to compiled bytes.
    # This is a selected-source receipt, not an attestation of the whole build.
    COMPILED_SOURCE_SHA256 = {
      "examples/drama3_gdn_scan_probe.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/drama3_gdn_scan_probe.cr") }}),
      "examples/support/drama3_gdn_scan.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/support/drama3_gdn_scan.cr") }}),
      "examples/support/drama3_gdn_scan.metal" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/support/drama3_gdn_scan.metal") }}),
      "examples/support/drama3_prefill_compare.metal" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/support/drama3_prefill_compare.metal") }}),
      "src/ml/audio/drama3/backbone_resident.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/backbone_resident.cr") }}),
      "src/ml/audio/drama3/backbone_prefill.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/backbone_prefill.cr") }}),
      "src/ml/audio/drama3/resident.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/resident.cr") }}),
      "src/ml/audio/drama3/backbone.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/backbone.cr") }}),
      "src/ml/audio/drama3/generator_resident.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/generator_resident.cr") }}),
      "src/ml/audio/drama3/prompt.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/prompt.cr") }}),
      "src/ml/audio/drama3/prompt_resident.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/prompt_resident.cr") }}),
      "src/ml/audio/drama3/tower.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/tower.cr") }}),
      "src/ml/audio/drama3/tower_resident.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/tower_resident.cr") }}),
      "src/ml/audio/drama3/native.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/native.cr") }}),
      "src/ml/audio/drama3/resident_batch.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/resident_batch.cr") }}),
      "src/ml/audio/drama3/kernels/backbone_resident.metal" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/kernels/backbone_resident.metal") }}),
      "src/ml/audio/drama3/kernels/resident.metal" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/kernels/resident.metal") }}),
      "src/ml/audio/drama3/kernels/resident_batch.metal" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/kernels/resident_batch.metal") }}),
      "src/ml/audio/drama3/kernels/tower_resident.metal" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/kernels/tower_resident.metal") }}),
    }

    class Capture
      getter rows : Int32, counts = {} of Int32 => Int32
      @tables = {} of Tuple(Int32, String) => Buffer
      @armed : Tuple(Int32, Int32)?

      def initialize(@context : Context, @rows : Int32, @config : ML::Audio::Drama3::BackboneConfig)
        LAYERS.each do |layer|
          @counts[layer] = 0
          {"q" => @config.key_heads * @config.key_dim, "k" => @config.key_heads * @config.key_dim,
           "v" => @config.value_heads * @config.value_dim, "a" => @config.value_heads,
           "b" => @config.value_heads, "output" => @config.value_heads * @config.value_dim}.each do |name, width|
            @tables[{layer, name}] = @context.alloc(@rows * width)
          end
          ["alog", "dt"].each { |name| @tables[{layer, name}] = @context.alloc(@config.value_heads) }
          @tables[{layer, "states"}] = @context.alloc((@rows + 1) * state_count)
        end
      end

      def state_count : Int32
        @config.value_heads * @config.key_dim * @config.value_dim
      end

      def arm(layer : Int32, position : Int32) : Nil
        raise "nested GDN capture" if @armed
        @armed = {layer, position} if LAYERS.includes?(layer)
      end

      def disarm : Nil
        @armed = nil
      end

      def active? : Bool
        !!@armed
      end

      def before(bindings : Array(Buffer), params : Array(UInt64)) : Nil
        layer, row = @armed.not_nil!
        raise "GDN capture order" unless row == @counts[layer] && row < @rows
        c = @config
        raise "GDN capture geometry" unless params == [c.value_heads.to_u64, c.value_dim.to_u64, c.key_heads.to_u64, c.key_dim.to_u64]
        ["q", "k", "v", "a", "b"].each_with_index do |name, index|
          value = bindings[index]
          @context.copy(value, @tables[{layer, name}].slice(row * value.count, value.count))
        end
        if row == 0
          @context.copy(bindings[5], @tables[{layer, "alog"}])
          @context.copy(bindings[6], @tables[{layer, "dt"}])
          @context.copy(bindings[7], states(layer, 0))
        end
      end

      def after(bindings : Array(Buffer)) : Nil
        layer, row = @armed.not_nil!
        @context.copy(bindings[8], @tables[{layer, "output"}].slice(row * bindings[8].count, bindings[8].count))
        @context.copy(bindings[7], states(layer, row + 1))
        @counts[layer] += 1
      end

      def states(layer : Int32, row : Int32) : Buffer
        @tables[{layer, "states"}].slice(row * state_count, state_count)
      end

      def expected(layer : Int32, base : Int32, count : Int32) : Buffer
        width = @config.value_heads * @config.value_dim
        @tables[{layer, "output"}].slice(base * width, count * width)
      end

      def inputs(layer : Int32, base : Int32, count : Int32) : Inputs
        c = @config
        kw, vw = c.key_heads * c.key_dim, c.value_heads * c.value_dim
        Inputs.new(@tables[{layer, "q"}].slice(base * kw, count * kw),
          @tables[{layer, "k"}].slice(base * kw, count * kw),
          @tables[{layer, "v"}].slice(base * vw, count * vw),
          @tables[{layer, "a"}].slice(base * c.value_heads, count * c.value_heads),
          @tables[{layer, "b"}].slice(base * c.value_heads, count * c.value_heads),
          @tables[{layer, "alog"}], @tables[{layer, "dt"}], states(layer, base),
          count, c.value_heads, c.value_dim, c.key_heads, c.key_dim)
      end

      def assert_complete! : Nil
        raise "incomplete GDN capture" unless @counts == LAYERS.to_h { |layer| {layer, @rows} }
      end
    end

    def self.sha(path : String) : String
      digest = Digest::SHA256.new
      File.open(path, "rb") do |io|
        bytes = Bytes.new(1024 * 1024)
        while (count = io.read(bytes)) > 0
          digest.update(bytes[0, count])
        end
      end
      digest.hexfinal
    end

    def self.json(value) : JSON::Any
      JSON.parse(value.to_json)
    end

    def self.verify_compiled_sources!(root : String) : Hash(String, String)
      COMPILED_SOURCE_SHA256.each do |relative, compiled_sha256|
        path = File.join(root, relative)
        raise "compiled source missing: #{relative}" unless File.file?(path)
        raise "runtime source differs from compiled bytes: #{relative}" unless sha(path) == compiled_sha256
      end
      raise "embedded scan shader differs" unless sha_bytes(SOURCE) == COMPILED_SOURCE_SHA256["examples/support/drama3_gdn_scan.metal"]
      raise "embedded comparison shader differs" unless sha_bytes(COMPARE_SOURCE) == COMPILED_SOURCE_SHA256["examples/support/drama3_prefill_compare.metal"]
      COMPILED_SOURCE_SHA256
    end

    def self.configure(depth : String) : Nil
      raise "graph depth must be 0 or 2" unless depth.in?("0", "2")
      ENV.keys.each { |key| ENV.delete(key) if key.starts_with?("DRAMA3_") }
      ENV.delete("QWEN35_NATIVE_TOKENIZER_OFF")
      {"DRAMA3_BACKEND" => "metal", "DRAMA3_DELTA_TRANSPOSE" => "1",
       "DRAMA3_MOE_BATCH" => "1", "DRAMA3_EXPERT_THREADS" => "64",
       "DRAMA3_DENSE_REDUCTION" => "tree", "DRAMA3_BACKBONE_DENSE_REDUCTION" => "tree",
       "DRAMA3_EXPERT_DOWN_REDUCTION" => "native", "DRAMA3_COGNIGRAPH_MAX_INFLIGHT" => depth,
       "DRAMA3_MMAP_SHARED" => "0", "DRAMA3_GDN_POST_FUSION" => "0"}.each { |key, value| ENV[key] = value }
      ML::Audio::Drama3::Profile.configure_from_env!
      raise "component profiling changes synchronization" if ML::Audio::Drama3::Profile.enabled?
    end

    # All activations stay resident. Read just the packed comparison receipts.
    def self.receipts(context : Context, checks : Array(Tuple(String, Buffer))) : Hash(String, Array(Float32))
      packed = context.alloc(checks.size * 5)
      checks.each_with_index { |(_, result), index| context.copy(result, packed.slice(index * 5, 5)) }
      values = context.read(packed, "diagnostic")
      checks.each_with_index.to_h { |(name, _), index| {name, values[index * 5, 5]} }
    end

    def self.exact?(value : Array(Float32)) : Bool
      value[0] == 0 && value[4] == 0
    end

    def self.qualify(context : Context) : JSON::Any
      reference = context.upload([1_f32, 0_f32, 0_f32])
      defect = context.upload([(1_f32.unsafe_as(UInt32) + 1).unsafe_as(Float32), 1e-3_f32, Float32::NAN])
      result = receipts(context, [{"clean", compare(context, reference, reference)}, {"seeded", compare(context, defect, reference)}])
      raise "GDN comparator clean control" unless exact?(result["clean"])
      raise "GDN comparator missed defect" unless result["seeded"][0] == 3 && result["seeded"][1] == 1 && result["seeded"][4] == 1
      json(result)
    end

    def self.check_plan(context : Context, plan : Plan, capture : Capture, layer : Int32, base : Int32) : JSON::Any
      before, reads = context.graph_stats.dup, context.readbacks.dup
      MODES.each { |mode| plan.encode(mode) }
      raise "GDN plan introduced a completion boundary" unless context.graph_stats == before && context.readbacks == reads
      count = plan.inputs.rows
      checks = [] of Tuple(String, Buffer)
      checks << {"replay_vs_capture.output", compare(context, plan.output("sequential"), capture.expected(layer, base, count))}
      checks << {"replay_vs_capture.state", compare(context, plan.state("sequential"), capture.states(layer, base + count))}
      ["serial_tile", "affine_scan"].each do |mode|
        count.times do |row|
          width = plan.inputs.value_width
          checks << {"#{mode}.output.#{row}", compare(context, plan.output(mode).slice(row * width, width), plan.output("sequential").slice(row * width, width))}
        end
        checks << {"#{mode}.state", compare(context, plan.state(mode), plan.state("sequential"))}
      end
      plan.blocks.times do |block|
        checks << {"affine_scan.prefix.#{block}", compare(context,
          plan.prefix_states.slice(block * plan.inputs.state_count, plan.inputs.state_count),
          capture.states(layer, base + block * plan.block_size))}
      end
      # Verify that actual captured x/k and initial states are not vacuous.
      checks << {"q_varies", compare(context, plan.inputs.q.slice(0, plan.inputs.key_width),
        plan.inputs.q.slice(plan.inputs.key_width, plan.inputs.key_width))} if count > 1
      checks << {"initial_nonzero", compare(context, plan.inputs.initial, context.zeros(plan.inputs.state_count))}
      result = receipts(context, checks)
      raise "sequential replay did not reproduce native capture" unless exact?(result["replay_vs_capture.output"]) && exact?(result["replay_vs_capture.state"])
      raise "captured q did not vary" unless count == 1 || result["q_varies"][0] > 0
      raise "incorrect initial state provenance" unless (result["initial_nonzero"][0] > 0) == (base > 0)
      json({checks: result, comparison_columns: ["bit_mismatches", "tolerance_failures", "max_abs", "max_tolerance_ratio", "nonfinite"],
            serial_tile_native_gate_pass: result.select { |name, _| name.starts_with?("serial_tile.") }.values.all? { |value| exact?(value) },
            affine_scan_native_gate_pass: result.select { |name, _| name.starts_with?("affine_scan.") }.values.all? { |value| exact?(value) },
            affine_scan_tolerance_gate_pass: result.select { |name, _| name.starts_with?("affine_scan.") }.values.all? { |value| value[1] == 0 && value[4] == 0 }})
    end

    def self.pairs(context : Context, plan : Plan, mode : String) : Array(JSON::Any)
      result = [] of JSON::Any
      (WARMUPS + PAIRS).times do |index|
        raise "pair not drained" unless context.graph_stats["pending"] == 0 && context.graph_stats["retained_bytes"] == 0
        order = index % 4 == 0 || index % 4 == 3 ? ["sequential", mode] : [mode, "sequential"]
        commands = {} of String => Int64
        encodes = {} of String => Float64
        reads_before = context.readbacks.dup
        order.each do |arm|
          before = context.graph_stats.dup
          commands[arm] = before["submitted"]
          started = Time.monotonic
          plan.encode(arm)
          encodes[arm] = (Time.monotonic - started).total_seconds
          raise "encoding synchronized" unless context.graph_stats == before && context.readbacks == reads_before
          context.submit
        end
        # Comparison work is a separate command, outside the timed arm indices.
        proof = receipts(context, [
          {"output", compare(context, plan.output(mode), plan.output("sequential"))},
          {"state", compare(context, plan.state(mode), plan.state("sequential"))},
        ])
        entry = json({pair: index, warmup: index < WARMUPS, mode: mode, order: order,
                      command_indices: commands, host_encode_seconds: encodes, checks: proof,
                      native_gate_pass: proof.values.all? { |value| exact?(value) },
                      graph: context.graph_stats, readbacks: context.readbacks,
                      timing_source: "drama3_gpu_command events joined by command_indices; reset+full corridor, excludes comparison/allocation"})
        result << entry
        STDERR.puts({event: "gdn_scan_pair", mode: mode, pair: index, blocks: plan.blocks, layer_rows: plan.inputs.rows}.to_json)
      end
      result
    end

    def self.run : Nil
      raise "usage: drama3_gdn_scan_probe MODEL OUTPUT_JSON GRAPH_DEPTH" unless ARGV.size == 3
      model_path, output_path, depth = ARGV
      raise "refusing existing report" if File.exists?(output_path)
      root = File.expand_path("..", __DIR__)
      verify_compiled_sources!(root)
      configure(depth)
      config = JSON.parse(File.read(File.join(model_path, "config.json")))
      bc = ML::Audio::Drama3::BackboneConfig.new(config)
      raise "checkpoint geometry drift" unless [bc.layers, bc.key_dim, bc.value_dim, bc.key_heads, bc.value_heads] == [40, 128, 128, 16, 32]
      tokenizer = ML::Audio::Drama3::Tokenizer.new(File.join(model_path, "tokenizer.json"), config["eos_token_id"].as_i, config["pad_token_id"].as_i)
      prompt = ML::Audio::Drama3::Prompt.build(TEXT, tokenizer, config["audio_decoder_config"]["num_codebooks"].as_i)
      raise "prompt drift" unless prompt.size == 29
      context = Context.new
      qualification = qualify(context)
      store = ML::Audio::Drama3::WeightStore.new(model_path)
      backbone = ML::Audio::Drama3::BackboneResident.new(bc, store, context, prompt.size + 1)
      tower = ML::Audio::Drama3::TowerResident.new(ML::Audio::Drama3::TowerConfig.from_json(config.to_json), store, context, prompt.size + 1)
      inputs = context.alloc(prompt.size * bc.hidden)
      prompt.each_with_index do |frame, row|
        device = tower.upload_frame(frame)
        value = tower.embed_backbone(device, backbone.embedding(device.slice(0, 1)))
        context.copy(value, inputs.slice(row * bc.hidden, bc.hidden))
      end
      context.finish
      capture = Capture.new(context, prompt.size, bc)
      backbone.gdn_scan_capture = capture
      context.gdn_scan_capture = capture
      before, reads = context.graph_stats.dup, context.readbacks.dup
      started = Time.monotonic
      backbone.prefill(inputs, prompt.size)
      raise "capture introduced a boundary" unless context.graph_stats == before && context.readbacks == reads
      context.finish
      capture_seconds = (Time.monotonic - started).total_seconds
      capture.assert_complete!
      backbone.gdn_scan_capture = nil
      context.gdn_scan_capture = nil
      STDERR.puts({event: "gdn_scan_capture", counts: capture.counts, seconds: capture_seconds}.to_json)
      context.gdn_scan_timeline!
      groups = [] of JSON::Any
      LAYERS.each do |layer|
        [{0, 29}, {7, 19}].each do |(base, count)|
          [8, 16, count].uniq.each do |block_size|
            plan = Plan.new(context, capture.inputs(layer, base, count), block_size)
            proof = check_plan(context, plan, capture, layer, base)
            modes = block_size == 8 ? ["serial_tile", "affine_scan"] : ["affine_scan"]
            measurements = modes.to_h { |mode| {mode, pairs(context, plan, mode)} }
            groups << json({layer: layer, base: base, rows: count, block_size: block_size, blocks: plan.blocks,
                            ragged_tail: count % block_size, scratch_bytes: plan.scratch_bytes, validation: proof, measurements: measurements})
          end
        end
      end
      files = ["config.json", "tokenizer.json", "model.safetensors.index.json"]
      index = JSON.parse(File.read(File.join(model_path, "model.safetensors.index.json")))
      files.concat(index["weight_map"].as_h.values.map(&.as_s)).uniq!
      model_hashes = files.sort.to_h { |relative| {relative, sha(File.join(model_path, relative))} }
      source_hashes = verify_compiled_sources!(root)
      source_head = IO::Memory.new
      status = Process.run("git", ["rev-parse", "HEAD"], chdir: root, output: source_head)
      raise "cannot record source HEAD" unless status.success?
      report = {schema: "drama3-gdn-scan-probe-v1", source_head: source_head.to_s.strip,
                executable_sha256: sha(Process.executable_path.not_nil!), compiled_source_sha256: source_hashes,
                source_tree_checked_before_and_after: true,
                embedded_shader_sha256: sha_bytes(SOURCE), model_files_sha256: model_hashes, text: TEXT,
                prompt_ids: prompt.map(&.[0]), rows: prompt.size, layers: LAYERS, geometry: [bc.value_heads, bc.value_dim, bc.key_heads, bc.key_dim],
                controls: ENV.to_h.select { |key, _| key.starts_with?("DRAMA3_") }, graph_depth: depth.to_i,
                comparator_qualification: qualification, capture_counts: capture.counts, capture_seconds: capture_seconds,
                capture_encoding_added_reads_submits_waits: false, groups: groups, graph: context.graph_stats, readbacks: context.readbacks,
                activation_readbacks: 0, production_routing_changed: false,
                scope: "actual-input recurrence only; preallocated scratch, all outputs/state; no full-prefill/WAV or refreshed MLX claim",
                native_gate: "zero output/state/prefix finite bit differences; nonfinite rejects",
                tolerance_metric_only: "6e-5 + 3e-4 * abs(reference), unchanged; does not override native bit gate"}
      File.write(output_path, report.to_pretty_json + "\n")
      puts({event: "gdn_scan_complete", output: output_path, groups: groups.size, graph: context.graph_stats}.to_json)
    end

    def self.sha_bytes(source : String) : String
      Digest::SHA256.hexdigest(source)
    end
  end

  module ML::Audio::Drama3::Resident
    class Context
      property gdn_scan_capture : Drama3GDNScan::Capture?

      def dispatch(name : String, source : String, bindings : Array(Buffer), params : Array(UInt64), count : Int32, group : Int32 = 64) : Nil
        capture = @gdn_scan_capture
        selected = name == "dr3_delta_step_transposed" && capture && capture.active?
        capture.not_nil!.before(bindings, params) if selected
        previous_def(name, source, bindings, params, count, group)
        capture.not_nil!.after(bindings) if selected
      end
    end
  end

  module ML::Audio::Drama3
    class BackboneResident
      property gdn_scan_capture : Drama3GDNScan::Capture?

      private def linear_attention_ready(qkv : Resident::Buffer, z : Resident::Buffer, b : Resident::Buffer,
                                         a : Resident::Buffer, prefix : String, layer : Int32, position : Int32) : Resident::Buffer
        capture = @gdn_scan_capture
        capture.try(&.arm(layer, position))
        previous_def(qkv, z, b, a, prefix, layer, position)
      ensure
        capture.try(&.disarm)
      end
    end
  end

  Drama3GDNScan.run
{% else %}
  abort "requires a Darwin -Ddrama3_metal build"
{% end %}
