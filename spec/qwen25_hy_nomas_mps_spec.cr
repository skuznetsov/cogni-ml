require "spec"
require "json"
require "digest/sha256"
require "./support/video_allocation_counter"
require "../src/ml/gguf/qwen25_hy_nomas"

# Fixture I/O only: no neural CPU reference implementation. JSON contains
# exact F32 decodes of BF16 values produced by the official MPS decoder.
module Qwen25FixtureSpec
  extend self
  alias Plan = ML::GGUF::Qwen25VLBlockPlan

  # A fixture may keep exact BF16 source bytes in one bounded sibling file.
  # This is source I/O during preparation, before any command is encoded.
  class Source
    @path : String? = nil
    @size = 0_i64

    def initialize(fixture : JSON::Any, manifest_path : String)
      records = [] of JSON::Any
      gather(fixture, records)
      if sidecars = fixture["sidecars"]?
        raise "unexpected fixture sidecars" unless sidecars.as_h.keys == ["bf16"]
        info = sidecars["bf16"]
        name = info["path"].as_s
        raise "sidecar must be a sibling basename" unless name.matches?(/\A[A-Za-z0-9_.-]+\z/) && name != "." && name != ".."
        raise "unexpected sidecar encoding" unless info["encoding"].as_s == "bfloat16-le"
        @size = info["nbytes"].as_i64
        raise "sidecar exceeds fixture budget" unless 0 < @size <= 512_i64 * 1024 * 1024
        @path = File.join(File.dirname(File.realpath(manifest_path)), name)
        path = @path.not_nil!
        raise "sidecar must be a regular sibling file" unless File.file?(path) && !File.symlink?(path) && File.realpath(path) == path
        raise "sidecar size mismatch" unless File.size(path) == @size
        expected_hash = info["sha256"].as_s
        raise "invalid sidecar hash" unless expected_hash.matches?(/\A[0-9a-f]{64}\z/)
        raise "sidecar byte size must be even" unless @size.even?
        digest = Digest::SHA256.new
        File.open(path) do |io|
          bytes = Bytes.new(1024 * 1024)
          remaining = @size
          while remaining > 0
            chunk = bytes[0, Math.min(remaining, bytes.size).to_i32]
            io.read_fully(chunk)
            digest.update(chunk)
            (chunk.size // 2).times do |i|
              word = chunk[i * 2].to_u16 | (chunk[i * 2 + 1].to_u16 << 8)
              raise "nonfinite BF16 sidecar value" if word & 0x7f80 == 0x7f80
            end
            remaining -= chunk.size
          end
        end
        raise "sidecar hash mismatch" unless digest.hexfinal == expected_hash
        spans = records.map { |record| span(record) }.sort_by(&.[0])
        cursor = 0_i64
        spans.each do |offset, size|
          raise "sidecar spans must be contiguous and nonoverlapping" unless offset == cursor
          cursor += size
        end
        raise "sidecar has unreferenced bytes" unless cursor == @size
      else
        raise "external tensor without a sidecar" unless records.empty?
      end
    end

    def span(record : JSON::Any) : Tuple(Int64, Int64)
      raise "sidecar tensor must be BF16" unless record["dtype"].as_s == "bfloat16"
      raise "tensor has both inline and external values" if record["values"]?
      storage = record["storage"]
      raise "unexpected tensor storage" unless storage["sidecar"].as_s == "bf16"
      offset, size = storage["offset_bytes"].as_i64, storage["nbytes"].as_i64
      elements = record["shape"].as_a.reduce(1_i64) do |count, dimension|
        dim = dimension.as_i64
        raise "invalid sidecar shape" unless dim > 0 && count <= @size // 2 // dim
        count * dim
      end
      raise "sidecar tensor byte count mismatch" unless size == elements * 2
      raise "sidecar tensor out of bounds" unless offset >= 0 && offset.even? && size > 0 && size <= @size && offset <= @size - size
      {offset, size}
    end

    def read_into(record : JSON::Any, buffer : ML::MetalBuffer) : Nil
      offset, size = span(record)
      raise "sidecar destination size mismatch" unless buffer.size == size
      File.open(@path.not_nil!) do |io|
        io.seek(offset)
        io.read_fully(Slice.new(buffer.contents.as(UInt8*), size.to_i32))
      end
    end

    def values(record : JSON::Any) : Array(Float32)
      offset, size = span(record)
      bytes = Bytes.new(size.to_i32)
      File.open(@path.not_nil!) do |io|
        io.seek(offset)
        io.read_fully(bytes)
      end
      Array(Float32).new(bytes.size // 2) do |i|
        word = bytes[i * 2].to_u32 | (bytes[i * 2 + 1].to_u32 << 8)
        (word << 16).unsafe_as(Float32)
      end
    end

    private def gather(node : JSON::Any, records : Array(JSON::Any)) : Nil
      if hash = node.as_h?
        if hash.has_key?("storage") && hash.has_key?("dtype") && hash.has_key?("shape")
          records << node
        else
          hash.each_value { |value| gather(value, records) }
        end
      elsif array = node.as_a?
        array.each { |value| gather(value, records) }
      end
    end
  end

  def values(record : JSON::Any, shape : Array(Int32), dtype : String, source : Source? = nil) : Array(Float32)
    record["dtype"].as_s.should eq(dtype)
    record["shape"].as_a.map(&.as_i).should eq(shape)
    result = if record["storage"]?
               source.not_nil!.values(record)
             else
               record["values"].as_a.map { |v| v.as_f.to_f32 }
             end
    result.size.to_i64.should eq(Plan.elements(shape))
    result.each { |v| v.finite?.should be_true }
    result
  end

  def upload(record : JSON::Any, shape : Array(Int32), dtype : String,
             owned : Array(ML::MetalBuffer), source : Source? = nil) : ML::MetalBuffer
    if record["storage"]?
      record["dtype"].as_s.should eq(dtype)
      record["shape"].as_a.map(&.as_i).should eq(shape)
      raise "only BF16 sidecar uploads admitted" unless dtype == "bfloat16"
      buffer = ML::MetalBuffer.new(Plan.elements(shape) * 2)
      owned << buffer
      source.not_nil!.read_into(record, buffer)
      return buffer
    end
    data = values(record, shape, dtype)
    bytes = dtype == "bfloat16" ? 2_i64 : 4_i64
    buffer = ML::MetalBuffer.new(data.size.to_i64 * bytes)
    owned << buffer
    if dtype == "bfloat16"
      words = data.map do |v|
        bits = v.unsafe_as(UInt32)
        raise "fixture BF16 value is not an exact decode" unless bits & 0xffff == 0
        (bits >> 16).to_u16
      end
      buffer.write_bytes(words.to_unsafe.as(UInt8*), words.size * 2)
    else
      buffer.write(data)
    end
    buffer
  end

  def read_bf16(buffer : ML::MetalBuffer) : Array(Float32)
    words = Slice(UInt16).new((buffer.size // 2).to_i32)
    buffer.read_bytes(words.to_unsafe.as(UInt8*), words.size * 2)
    words.map { |v| (v.to_u32 << 16).unsafe_as(Float32) }.to_a
  end

  def failure_count(actual : Array(Float32), expected : Array(Float32)) : Int32
    raise "comparison size mismatch" unless actual.size == expected.size
    actual.each_with_index.count do |value, i|
      ref = expected[i]
      !value.finite? || !ref.finite? || (value.to_f64 - ref).abs > 2e-5 + 2e-5 * ref.abs
    end
  end

  def weight_name(name : String, index : Int32) : String
    suffix = {"input_norm" => "input_layernorm.weight", "q_w" => "self_attn.q_proj.weight",
              "q_bias" => "self_attn.q_proj.bias", "k_w" => "self_attn.k_proj.weight",
              "k_bias" => "self_attn.k_proj.bias", "v_w" => "self_attn.v_proj.weight",
              "v_bias" => "self_attn.v_proj.bias", "o_w" => "self_attn.o_proj.weight",
              "post_norm" => "post_attention_layernorm.weight", "gate_w" => "mlp.gate_proj.weight",
              "up_w" => "mlp.up_proj.weight", "down_w" => "mlp.down_proj.weight"}[name]
    "layers.#{index}.#{suffix}"
  end

  def stage_name(name : String, index : Int32) : String
    case name
    when "q_rope"       then "post_rope_q"
    when "k_rope"       then "post_rope_k"
    when "attended"     then "attended"
    when "residual"     then "residual_after_attention"
    when "silu_product" then "silu_product"
    when "hidden_out"   then "hidden_states_out"
    else
      suffix = {"input_norm" => "input_layernorm", "post_norm" => "post_attention_layernorm",
                "q_proj" => "self_attn.q_proj", "k_proj" => "self_attn.k_proj",
                "v_proj" => "self_attn.v_proj", "o_proj" => "self_attn.o_proj",
                "gate_proj" => "mlp.gate_proj", "up_proj" => "mlp.up_proj",
                "down_proj" => "mlp.down_proj"}[name]
      "layers.#{index}.#{suffix}"
    end
  end

  def qualify(path : String) : Nil
    fixture = JSON.parse(File.read(path))
    fixture["schema"].as_s.should eq("cogni_ml.qwen25_vl_text_decoder_hidden_state_reference")
    fixture["schema_version"].as_i.should eq(1)
    conf = fixture["config"]
    layers = fixture["layers"].as_a
    layers.size.should eq(conf["num_hidden_layers"].as_i)
    (1..2).includes?(layers.size).should be_true
    conf["attention_implementation"].as_s.should eq("sdpa")
    c = ML::GGUF::Qwen25VLConfig.new(dim: conf["hidden_size"].as_i,
      ffn_dim: conf["intermediate_size"].as_i, heads: conf["num_attention_heads"].as_i,
      kv_heads: conf["num_key_value_heads"].as_i, head_dim: conf["head_dim"].as_i,
      layers: layers.size, rms_norm_eps: conf["rms_norm_eps"].as_f.to_f32,
      rope_theta: conf["rope_theta"].as_f.to_f32,
      # Earlier schema-v1 tiny captures used a fixed context of 32 and did
      # not serialize that admission-only field.
      max_positions: conf["max_position_embeddings"]?.try(&.as_i) || 32)
    tokens = conf["sequence_length"].as_i
    inputs = fixture["inputs"]
    fixture_source = Source.new(fixture, path)
    owned = [] of ML::MetalBuffer
    plans = [] of Plan
    chains = [] of ML::Metal::MPSCommandChain
    retained_before = Plan.retained_count
    begin
      input = upload(inputs["hidden_states"], [1, tokens, c.dim], "bfloat16", owned, fixture_source)
      # HF SDPA consumes a boolean mask. The separately identified native
      # bias is its GPU-derived finite BF16 0/min encoding, never relabeled as
      # the upstream mask. Fixtures admit no fully masked attention rows.
      mask = upload(inputs["additive_attention_bias"], [1, 1, tokens, tokens], "bfloat16", owned, fixture_source)
      pos = upload(inputs["position_ids"], [tokens], "float32", owned, fixture_source)
      inv = upload(inputs["inv_freq"], [c.head_dim // 2], "float32", owned, fixture_source)
      weight_sets = layers.each_with_index.map do |layer, index|
        layer["index"].as_i.should eq(index)
        Plan.weight_shapes(c).to_h do |name, shape|
          {name, upload(layer["weights"][weight_name(name, index)], shape, "bfloat16", owned, fixture_source)}
        end
      end.to_a
      # Admission failures must not leak native plans or named stage buffers.
      live = ML::MetalBuffer.stats[:live_bytes]
      expect_raises(ArgumentError, /bindings/) do
        Plan.new(c, tokens, input, mask, pos, inv, {} of String => ML::MetalBuffer)
      end
      expect_raises(ArgumentError, /budget/) do
        Plan.new(c, tokens, input, mask, pos, inv, weight_sets[0], max_workspace_bytes: 1)
      end
      aliased = weight_sets[0].dup
      aliased["o_w"] = aliased["q_w"]
      expect_raises(Exception, /overlap|alias|disjoint/) do
        Plan.new(c, tokens, input, mask, pos, inv, aliased)
      end
      ML::MetalBuffer.stats[:live_bytes].should eq(live)
      Plan.retained_count.should eq(retained_before)
      weight_sets.each_with_index do |weights, index|
        source = index == 0 ? input : plans.last.output
        plans << Plan.new(c, tokens, source, mask, pos, inv, weights, layer_index: index)
      end
      if plans.size == 2
        plans[1].input.same?(plans[0].output).should be_true
        weight_sets[1]["q_w"].same?(weight_sets[0]["q_w"]).should be_false
      end
      chain = ML::Metal::MPSCommandChain.new
      chains << chain
      allocations = ML::MetalBuffer.video_spec_allocations
      plans.each(&.encode(chain))
      ML::MetalBuffer.video_spec_allocations.should eq(allocations)
      chain.wait_count.should eq(0)
      expect_raises(ArgumentError, /terminal wait/) { plans[0].release }
      other = ML::Metal::MPSCommandChain.new
      chains << other
      expect_raises(ArgumentError, /in-flight chain/) { plans[0].encode(other) }
      chain.commit_and_wait
      chain.wait_count.should eq(1)
      # All comparisons and diagnostic reads happen after the sole request wait.
      report = [] of NamedTuple(layer: Int32, stage: String, count: Int32, failures: Int32, max_abs: Float64)
      plans.each_with_index do |plan, index|
        Plan.stage_shapes(c, tokens).each do |name, shape|
          expected = if name == "rotary_cos" || name == "rotary_sin"
                       values(inputs[name], [3, 1, tokens, c.head_dim], "bfloat16", fixture_source)[0, tokens * c.head_dim]
                     else
                       values(layers[index]["stages"][stage_name(name, index)], shape, "bfloat16", fixture_source)
                     end
          actual = read_bf16(plan.stage(name))
          failures = failure_count(actual, expected)
          max_abs = actual.each_with_index.max_of { |v, i| (v.to_f64 - expected[i]).abs }
          report << {layer: index, stage: name, count: actual.size, failures: failures, max_abs: max_abs}
        end
      end
      if root = ENV["COGNI_QWEN25_REPORT_DIR"]?
        File.write(File.join(root, "#{File.basename(path)}.native.json"), report.to_json)
      end
      failures = report.select { |r| r[:failures] > 0 }
      if failures.empty?
        puts "Qwen2.5-VL #{File.basename(path)}: #{report.size} BF16 stages pass, one terminal wait, zero named-buffer allocations during encode"
      else
        puts "Qwen2.5-VL first mismatch: #{failures.first.to_json} (#{failures.size}/#{report.size} stages fail)"
      end
      failures.should be_empty
      expect_raises(ArgumentError, /completed chain/) { plans[0].encode(chain) }
    ensure
      chains.each do |chain|
        begin
          chain.commit_and_wait
        rescue
          # A drained command error still permits resource cleanup. An
          # undrained chain correctly makes plan.release fail closed below.
        end
      end
      plans.reverse_each(&.release)
      chains.each(&.release)
      owned.reverse_each(&.release)
    end
    Plan.retained_count.should eq(retained_before)
  end
end

describe "Qwen2.5-VL BF16 MPS qualification" do
  it "reads exact BF16 sidecar bytes and rejects corrupt or overlapping spans" do
    manifest = File.join(Dir.tempdir, "qwen25-source-#{Process.pid}.json")
    sidecar = "qwen25-source-#{Process.pid}.bf16"
    path = File.join(Dir.tempdir, sidecar)
    bytes = Bytes[0x80, 0x3f, 0x80, 0xbf]
    record = {dtype: "bfloat16", shape: [2], storage: {sidecar: "bf16", offset_bytes: 0, nbytes: 4}}
    fixture = JSON.parse({sidecars: {bf16: {path: sidecar, encoding: "bfloat16-le", nbytes: 4,
                                            sha256: Digest::SHA256.hexdigest(bytes)}}, record:   record}.to_json)
    begin
      File.write(manifest, fixture.to_json)
      File.write(path, bytes)
      Qwen25FixtureSpec::Source.new(fixture, manifest).values(fixture["record"]).should eq([1_f32, -1_f32])
      overlapping = JSON.parse({sidecars: fixture["sidecars"], records: [record, record]}.to_json)
      expect_raises(Exception, /nonoverlapping/) { Qwen25FixtureSpec::Source.new(overlapping, manifest) }
      File.write(path, Bytes[0, 0, 0, 0])
      expect_raises(Exception, /hash mismatch/) { Qwen25FixtureSpec::Source.new(fixture, manifest) }
    ensure
      File.delete?(manifest)
      File.delete?(path)
    end
  end

  it "rejects a known BF16 error, nonfinite data and size mismatch" do
    Qwen25FixtureSpec.failure_count([1_f32], [1_f32]).should eq(0)
    Qwen25FixtureSpec.failure_count([1.0078125_f32], [1_f32]).should eq(1)
    Qwen25FixtureSpec.failure_count([Float32::NAN], [1_f32]).should eq(1)
    Qwen25FixtureSpec.failure_count([1_f32], [Float32::INFINITY]).should eq(1)
    expect_raises(Exception, /size mismatch/) { Qwen25FixtureSpec.failure_count([1_f32], [] of Float32) }
  end

  {% unless flag?(:cpu_only) %}
    if paths = ENV["COGNI_QWEN25_REFERENCES"]?
      paths.split(':').each do |path|
        it "matches every upstream boundary in #{File.basename(path)}" do
          Qwen25FixtureSpec.qualify(path)
        end
      end
    else
      pending "requires guarded official fixtures via COGNI_QWEN25_REFERENCES"
    end
  {% end %}
end
