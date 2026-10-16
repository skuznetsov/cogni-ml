# Diagnostic sequence producer/parity and completed-GPU benchmark. This tool
# may export fixtures at terminal boundaries; production inference never does.
require "digest/sha256"
require "../src/ml/audio/drama3/generator_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
module Drama3PrefillBenchmark
  include ML::Audio::Drama3
  alias Buffer = Resident::Buffer
  alias Context = Resident::Context
  SOURCE = {{ read_file("#{__DIR__}/support/drama3_prefill_compare.metal") }}
  record Check, name : String, base : Int32, count : Int32, result : Buffer

  def self.sha(path : String) : String
    digest = Digest::SHA256.new
    File.open(path) do |io|
      bytes = Bytes.new(1024 * 1024)
      while (n = io.read(bytes)) > 0
        digest.update(bytes[0, n])
      end
    end
    digest.hexfinal
  end

  def self.load_raw(ctx : Context, path : String) : Buffer
    raise "raw F32 byte count" unless File.size(path) > 0 && File.size(path) % 4 == 0 && File.size(path) // 4 <= Int32::MAX
    buffer = ctx.alloc((File.size(path) // 4).to_i32)
    File.open(path) { |io| io.read_fully(Bytes.new(buffer.buffer.contents.as(UInt8*), File.size(path).to_i32)) }
    buffer
  end

  def self.save_raw(ctx : Context, buffer : Buffer, path : String) : Nil
    raise "refusing existing fixture #{path}" if File.exists?(path)
    values = ctx.read(buffer, "diagnostic")
    File.open(path, "wb") { |io| values.each { |value| io.write_bytes(value, IO::ByteFormat::LittleEndian) } }
  end

  def self.compare(ctx : Context, actual : Buffer, reference : Buffer, atol = 6e-5_f32) : Buffer
    raise "comparison shape" unless actual.count == reference.count
    parts_count = (actual.count + 1023) // 1024
    parts = ctx.alloc(parts_count * 5)
    result = ctx.alloc(5)
    ctx.dispatch("dr3_prefill_compare", SOURCE, [actual, reference, parts],
      [actual.count.to_u64, atol.unsafe_as(UInt32).to_u64, 3e-4_f32.unsafe_as(UInt32).to_u64], parts_count)
    ctx.dispatch("dr3_prefill_compare_reduce", SOURCE, [parts, result], [parts_count.to_u64], 1, 1)
    result
  end

  def self.canonical(ctx : Context, ids : Buffer, values : Buffer, rows : Int32, ranks : Int32) : Buffer
    raise "canonical route shape" unless ids.count == rows * ranks && values.count % (rows * ranks) == 0
    output = ctx.alloc(values.count)
    ctx.dispatch("dr3_prefill_canonical", SOURCE, [ids, values, output],
      [rows.to_u64, ranks.to_u64, (values.count // (rows * ranks)).to_u64], values.count)
    output
  end

  def self.capture(ctx : Context, tables : Hash(String, Buffer), name : String, value : Buffer,
                   row : Int32, rows : Int32, rank : Int32? = nil, top_k : Int32 = 1) : Nil
    width = value.count * (rank ? top_k : 1)
    table = tables[name] ||= ctx.alloc(rows * width)
    offset = row * width + (rank || 0) * value.count
    ctx.copy(value, table.slice(offset, value.count))
  end

  def self.capture_moe(ctx : Context, tables : Hash(String, Buffer), name : String, value : Buffer,
                       row : Int32, rows : Int32, layer : Int32, top_k : Int32) : Nil
    renamed = {"expert_ids_argpartition_order" => "expert_ids", "routed_sum_official" => "routed_sum", "moe_output_official" => "moe_output"}
    rank = nil.as(Int32?)
    base = name
    if name.includes?(".rank_")
      base, suffix = name.split(".rank_", 2)
      rank = suffix.to_i
    end
    base = renamed[base]? || base
    return if base.starts_with?("diagnostic_") || base.starts_with?("routed_prefix_")
    capture(ctx, tables, "backbone.layers.#{layer}.mlp.#{base}", value, row, rows, rank, top_k)
  end

  def self.execute(ctx : Context, model : BackboneResident, input : Buffer, rows : Int32, chunk : Int32) : Buffer
    width = model.config.hidden
    output = ctx.alloc(rows * width)
    base = 0
    while base < rows
      n = Math.min(chunk, rows - base)
      value = if chunk == 1
                model.step(input.slice(base * width, width))
              else
                model.prefill(input.slice(base * width, n * width), n)
              end
      ctx.copy(value, output.slice(base * width, n * width))
      ctx.submit
      base += n
    end
    output
  end

  def self.run : Nil
    raise "usage: drama3_prefill_benchmark MODEL FIXTURE OUTPUT_DIR export|parity|bench CHUNK [UPSTREAM_MANIFEST]" unless ARGV.size.in?(5..6)
    model_path, fixture_path, output_dir, mode = ARGV[0, 4]
    chunk = ARGV[4].to_i
    raise "chunk must be 2..32" unless chunk.in?(2..32)
    raise "component profiling introduces waits" if Profile.enabled?
    Dir.mkdir_p(output_dir)
    config = JSON.parse(File.read(File.join(model_path, "config.json")))
    bc = BackboneConfig.new(config)
    store = WeightStore.new(model_path)
    fixture = JSON.parse(File.read(fixture_path))
    tokenizer = Tokenizer.new(File.join(model_path, "tokenizer.json"), config["eos_token_id"].as_i, config["pad_token_id"].as_i)
    prompt = Prompt.build(fixture["text"].as_s, tokenizer, config["audio_decoder_config"]["num_codebooks"].as_i)
    raise "prompt IDs differ" unless prompt.map(&.[0]) == fixture["prompt_ids"].as_a.map(&.as_i)
    rows = prompt.size
    input_path = File.join(output_dir, "input.f32")
    if mode == "export"
      ctx = Context.new
      backbone = BackboneResident.new(bc, store, ctx, rows + 1)
      tower = TowerResident.new(TowerConfig.from_json(config.to_json), store, ctx, rows + 1)
      input = ctx.alloc(rows * bc.hidden)
      prompt.each_with_index do |frame, row|
        device = tower.upload_frame(frame)
        mixed = tower.embed_backbone(device, backbone.embedding(device.slice(0, 1)))
        ctx.copy(mixed, input.slice(row * bc.hidden, bc.hidden))
      end
      save_raw(ctx, input, input_path)
      File.write(File.join(output_dir, "input.json"), {shape: [rows, bc.hidden], dtype: "float32-le", sha256: sha(input_path),
        prompt_fixture: fixture_path, prompt_sha256: sha(fixture_path), purpose: "immutable terminal fixture from existing GPU embedding kernels",
        model_config_sha256: sha(File.join(model_path, "config.json")), model_index_sha256: sha(File.join(model_path, "model.safetensors.index.json")),
        readbacks: ctx.readbacks}.to_pretty_json + "\n")
      return
    end
    input_metadata = JSON.parse(File.read(File.join(output_dir, "input.json")))
    raise "input identity" unless sha(input_path) == input_metadata["sha256"].as_s && input_metadata["shape"].as_a.map(&.as_i) == [rows, bc.hidden]
    if mode == "parity"
      ctx = Context.new
      input = load_raw(ctx, input_path)
      tables = {} of String => Buffer
      scalar = BackboneResident.new(bc, store, ctx, rows + 1)
      scalar.diagnostic_layer = 39
      row = 0
      scalar.buffer_trace = ->(name : String, value : Buffer) { capture(ctx, tables, name, value, row, rows); nil }
      scalar.moe_trace = ->(name : String, value : Buffer) { capture_moe(ctx, tables, name, value, row, rows, 39, bc.top_k); nil }
      scalar_out = ctx.alloc(rows * bc.hidden)
      rows.times do |i|
        row = i
        value = scalar.step(input.slice(i * bc.hidden, bc.hidden))
        ctx.copy(value, scalar_out.slice(i * bc.hidden, bc.hidden))
        ctx.submit
      end
      checks = [] of Check
      # Poison probes distinguish a real numerical oracle from a vacuous one.
      checks << Check.new("oracle:poison", 0, 3,
        compare(ctx, ctx.upload([0_f32, 1e-3_f32, Float32::NAN]), ctx.upload([0_f32, 0_f32, 0_f32])))
      batch_tables = {} of String => Buffer
      batch = BackboneResident.new(bc, store, ctx, rows + 1)
      batch.diagnostic_layer = 39
      base = 0
      batch.prefill_trace = ->(name : String, value : Buffer) {
        reference = tables[name]? || raise "missing scalar table #{name}"
        width = reference.count // rows
        checks << Check.new("native_bitwise:" + name, base, value.count, compare(ctx, value, reference.slice(base * width, value.count)))
        table = batch_tables[name] ||= ctx.alloc(reference.count)
        ctx.copy(value, table.slice(base * width, value.count))
        nil
      }
      batch_out = ctx.alloc(rows * bc.hidden)
      while base < rows
        n = Math.min(chunk, rows - base)
        value = batch.prefill(input.slice(base * bc.hidden, n * bc.hidden), n)
        ctx.copy(value, batch_out.slice(base * bc.hidden, n * bc.hidden))
        ctx.submit
        base += n
      end
      scalar.buffer_trace = nil
      scalar.moe_trace = nil
      batch.prefill_trace = nil
      # Following decode catches persistent-state differences beyond outputs.
      scalar_next = scalar.step(input.slice(0, bc.hidden))
      batch_next = batch.step(input.slice(0, bc.hidden))
      checks << Check.new("native_bitwise:decode_continuation", rows, bc.hidden, compare(ctx, batch_next, scalar_next))
      head = store.matrix("lm_head.weight")
      scalar_logits = ctx.matvec(head, scalar_out.slice((rows-1)*bc.hidden, bc.hidden))
      batch_logits = ctx.matvec(head, batch_out.slice((rows-1)*bc.hidden, bc.hidden))
      checks << Check.new("native_bitwise:terminal_logits", rows-1, scalar_logits.count, compare(ctx, batch_logits, scalar_logits, 3e-5_f32))
      # Saved MLX terminal endpoint remains a separate, stricter guard.
      fixture["stages"].as_a.each do |stage|
        name = stage["name"].as_s
        expected = ctx.upload(stage["values"].as_a.map { |v| v.as_f.to_f32 })
        actual = name == "prompt.backbone.hidden" ? batch_out.slice((rows-1)*bc.hidden, bc.hidden) : batch_logits
        checks << Check.new("saved_mlx_terminal:" + name, rows-1, actual.count, compare(ctx, actual, expected, 3e-5_f32))
      end
      # Upstream intermediate fixture comparisons use their original tolerance.
      if manifest_path = ARGV[5]?
        manifest = JSON.parse(File.read(manifest_path))
        raise "upstream format" unless manifest["format"].as_s == "drama3-backbone-prefill-intermediates-v1"
        raise "upstream revision" unless manifest["upstream"]["revision"].as_s == "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
        raise "model revision" unless manifest["model"]["revision"].as_s == "5331ab509359c0e4a7e353a19e255ddf3e8152a8"
        raise "upstream input identity" unless manifest["input"]["used_f32_sha256"].as_s == sha(input_path) && manifest["input"]["used_rows"].as_i == rows
        raise "upstream checkpoint identity" unless manifest["checkpoint"]["config_sha256"].as_s == sha(File.join(model_path, "config.json")) && manifest["checkpoint"]["index_sha256"].as_s == sha(File.join(model_path, "model.safetensors.index.json"))
        required = ["backbone.hidden"]
        bc.layers.times do |layer|
          ["attention_residual", "hidden"].each { |stage| required << "backbone.layers.#{layer}.#{stage}" }
        end
        moe_prefix = "backbone.layers.39.mlp."
        %w(normalized_input router_logits routed_sum shared_expert_output shared_gate_logits moe_output canonical_expert_ids canonical_selected_weights canonical_expert_gate_outputs canonical_expert_up_outputs canonical_expert_activation_outputs canonical_expert_down_outputs).each { |stage| required << moe_prefix + stage }
        ["scalar", "sequence"].each do |arm|
          stages = manifest["arms"][arm]["tensors"].as_a
          required.each do |name|
            matches = stages.select { |stage| stage["name"].as_s == name }
            raise "missing/duplicate upstream stage #{arm}:#{name}" unless matches.size == 1
            stage = matches.first
            path = File.expand_path(stage["file"].as_s, File.dirname(manifest_path))
            raise "upstream stage path" unless path.starts_with?(File.expand_path(File.dirname(manifest_path)) + "/")
            raise "upstream stage identity" unless sha(path) == stage["sha256"].as_s && File.size(path) == stage["bytes"].as_i64
            expected = load_raw(ctx, path)
            shape = stage["shape"].as_a.map(&.as_i64)
            raise "upstream stage shape" unless shape.first == rows && shape.reduce(1_i64) { |a, b| a * b } == expected.count
            if stage["dtype"].as_s == "int32"
              converted = ctx.alloc(expected.count)
              ctx.dispatch("dr3_prefill_i32_f32", SOURCE, [expected, converted], [expected.count.to_u64], expected.count)
              expected = converted
            else
              raise "upstream stage dtype" unless stage["dtype"].as_s == "float32"
            end
            [tables, batch_tables].each_with_index do |native, index|
              actual = if name.starts_with?(moe_prefix + "canonical_")
                         raw_name = name.sub(".canonical_", ".")
                         canonical(ctx, native[moe_prefix + "expert_ids"], native[raw_name], rows, bc.top_k)
                       else
                         native[name]
                       end
              label = index == 0 ? "native_scalar" : "native_batch"
              checks << Check.new("upstream_#{arm}:#{label}:" + name, 0, actual.count,
                compare(ctx, actual, expected, name == "backbone.hidden" ? 3e-5_f32 : 6e-5_f32))
            end
          end
        end
      end
      aggregate = ctx.alloc(checks.size * 5)
      checks.each_with_index { |check, i| ctx.copy(check.result, aggregate.slice(i * 5, 5)) }
      before_reads = ctx.readbacks.dup
      values = ctx.read(aggregate, "diagnostic")
      results = checks.map_with_index do |check, i|
        metrics = values[i*5,5]
        {name: check.name, base: check.base, count: check.count, bitwise_mismatches: metrics[0].to_i64,
         pointwise_failures: metrics[1].to_i64, max_abs: metrics[2], max_ratio: metrics[3], nonfinite: metrics[4].to_i64}
      end
      poison = results.find { |r| r[:name] == "oracle:poison" }.not_nil!
      raise "GPU comparator poison escaped" unless poison[:bitwise_mismatches] == 2 && poison[:pointwise_failures] == 1 && poison[:nonfinite] == 1
      native_pass = results.select { |r| r[:name].starts_with?("native_bitwise:") }.all? { |r| r[:bitwise_mismatches] == 0 && r[:nonfinite] == 0 }
      terminal_pass = results.select { |r| r[:name].starts_with?("saved_mlx_terminal:") }.all? { |r| r[:pointwise_failures] == 0 && r[:nonfinite] == 0 }
      upstream_pass = ARGV[5]? ? results.select { |r| r[:name].starts_with?("upstream_") }.all? { |r| r[:pointwise_failures] == 0 && r[:nonfinite] == 0 } : nil
      suffix = ARGV[5]? ? "-upstream" : ""
      path = File.join(output_dir, "parity-chunk#{chunk}-depth#{ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]? || "0"}#{suffix}.json")
      raise "refusing existing result" if File.exists?(path)
      File.write(path, {rows: rows, chunk: chunk, input_sha256: sha(input_path), results: results,
        graph: ctx.graph_stats, readbacks_before_terminal_metrics: before_reads, readbacks: ctx.readbacks,
        native_bitwise_pass: native_pass, saved_mlx_terminal_pass: terminal_pass,
        upstream_pointwise_pass: upstream_pass,
        upstream_manifest_sha256: ARGV[5]?.try { |p| sha(p) },
        source_sha256: sha(__FILE__)}.to_pretty_json + "\n")
      norm_path = File.join(output_dir, "moe-normalized.f32")
      save_raw(ctx, tables["backbone.layers.39.mlp.normalized_input"], norm_path) unless File.exists?(norm_path)
      puts({result: path, checks: results.size, native_bitwise_mismatches: results.select { |r| r[:name].starts_with?("native_bitwise:") }.sum { |r| r[:bitwise_mismatches] }, pointwise_failures: results.reject { |r| r[:name].starts_with?("oracle:") }.sum { |r| r[:pointwise_failures] }}.to_json)
      raise "native batch arithmetic drift" unless native_pass
      raise "saved MLX terminal pointwise gate failed" unless terminal_pass
      # A completed diagnostic capture is not a passing upstream numerical gate.
      # Preserve its measurements, but never let a caller promote on exit zero.
      raise "upstream intermediate pointwise gate remains open" if upstream_pass == false
    elsif mode == "bench"
      observations = [] of NamedTuple(component: String, iteration: Int32, chunk: Int32, seconds: Float64, graph: Hash(String, Int64), readbacks: Hash(String, Int64))
      ctx = Context.new
      ["moe", "backbone"].each do |component|
        input = load_raw(ctx, component == "moe" ? File.join(output_dir, "moe-normalized.f32") : input_path)
        # Excluded prime followed by counterbalanced A/B/B/A pairs.
        [1, chunk, 1, chunk, chunk, 1, chunk, 1, 1, chunk].each_with_index do |size, iteration|
          model = BackboneResident.new(bc, store, ctx, rows + 1)
          model.diagnostic_layer = 39
          ctx.finish # Diagnostic preparation only, outside the measured run.
          before = ctx.graph_stats
          STDERR.puts({event: "prefill_benchmark_begin", component: component, iteration: iteration, chunk: size}.to_json)
          started = Time.instant
          if component == "backbone"
            execute(ctx, model, input, rows, size)
          else
            base = 0
            while base < rows
              n = Math.min(size, rows - base)
              if size == 1
                model.diagnose_moe(input.slice(base * bc.hidden, bc.hidden), 39)
              else
                model.diagnose_moe_rows(input.slice(base * bc.hidden, n * bc.hidden), 39, n)
              end
              ctx.submit
              base += n
            end
          end
          ctx.finish
          seconds = (Time.instant - started).total_seconds
          delta = ctx.graph_stats.dup
          before.each { |key, value| delta[key] -= value unless key == "depth" }
          STDERR.puts({event: "prefill_benchmark_end", component: component, iteration: iteration, chunk: size, seconds: seconds}.to_json)
          observations << {component: component, iteration: iteration, chunk: size, seconds: seconds, graph: delta, readbacks: ctx.readbacks.dup}
          GC.collect
        end
      end
      path = File.join(output_dir, "benchmark-chunk#{chunk}-depth#{ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]? || "0"}.json")
      raise "refusing existing result" if File.exists?(path)
      File.write(path, {rows: rows, chunk: chunk, persistent_context: true, input_sha256: sha(input_path), observations: observations, source_sha256: sha(__FILE__)}.to_pretty_json + "\n")
      puts({result: path}.to_json)
    else
      raise "unknown probe mode"
    end
  end
end
Drama3PrefillBenchmark.run
{% else %}
  abort "build with -Ddrama3_metal on macOS"
{% end %}
