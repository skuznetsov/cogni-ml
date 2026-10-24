# Test-only same-input SiLU arithmetic probe. The production kernel remains
# unchanged; this spec compares its output and explicit arithmetic candidates
# against a pinned PyTorch 2.8 MPS fixture.
require "spec"
require "json"
require "digest/sha256"
require "./support/video_allocation_counter"
require "../src/ml/video/metal_mps_command_chain"
require "../src/ml/video/hy_nomas_mps_activation"
require "../src/ml/video/hy_nomas_block"

module HyNomasPointwiseProbe
  REVISION       = "911cea6458a89c2663513ff7099561ec190dd491"
  CONTROL_SOURCE = {{ read_file("#{__DIR__}/support/hy_nomas_pointwise_controls.metal") }}

  # Tuple entries are {report label, Metal function name}. The first entry is
  # the unchanged production function; the rest are diagnostic-only controls.
  VARIANTS = {
    {"production_exp_div", "hy_nomas_silu_f32"},
    {"default_exp_div", "hy_probe_silu_exp_div"},
    {"precise_exp_div", "hy_probe_silu_precise_exp_div"},
    {"precise_exp_precise_div", "hy_probe_silu_precise_exp_precise_div"},
    {"default_exp_recip_mul", "hy_probe_silu_exp_recip_mul"},
    {"precise_exp_recip_mul", "hy_probe_silu_precise_exp_recip_mul"},
    {"default_exp2_div", "hy_probe_silu_exp2_div"},
    {"precise_exp2_div", "hy_probe_silu_precise_exp2_div"},
  }
  GRAPH_VARIANT = "pytorch28_mpsgraph_silu"

  struct Operation
    getter name : String
    getter case_name : String
    getter stream : String
    getter case_batches : Int32
    getter shape : Array(Int32)
    getter input : Array(Float32)
    getter expected : Array(Float32)
    getter rounded_expected : Array(Float32)

    def initialize(@name : String, @case_name : String, @stream : String, @case_batches : Int32,
                   @shape : Array(Int32), @input : Array(Float32),
                   @expected : Array(Float32), @rounded_expected : Array(Float32))
    end
  end

  def self.trusted_digest!(value : String) : Nil
    raise ArgumentError.new("Expected a trusted SHA-256 digest") unless value.matches?(/\A[0-9a-f]{64}\z/)
  end

  def self.digest(path : String) : String
    digest = Digest::SHA256.new
    bytes = Bytes.new(1024 * 1024)
    File.open(path, "rb") do |file|
      while (count = file.read(bytes)) > 0
        digest.update(bytes[0, count])
      end
    end
    digest.final.hexstring
  end

  def self.values(record : JSON::Any, context : String) : Array(Float32)
    shape = record["shape"].as_a.map do |item|
      value = item.as_i64
      raise ArgumentError.new("Invalid #{context} shape") unless value > 0 && value <= Int32::MAX
      value.to_i32
    end
    raise ArgumentError.new("Empty #{context} shape") if shape.empty?
    count = 1_i64
    shape.each do |dimension|
      raise ArgumentError.new("#{context} tensor is too large") if count > Int32::MAX // dimension
      count *= dimension
    end
    result = record["values"].as_a.map do |item|
      value = item.as_f
      unless value.finite? && value.abs <= Float32::MAX.to_f64
        raise ArgumentError.new("#{context} contains a non-finite or out-of-F32 value")
      end
      value.to_f32
    end
    raise ArgumentError.new("#{context} value count does not match its shape") unless result.size == count
    raise ArgumentError.new("#{context} contains a non-finite F32 value") unless result.all?(&.finite?)
    result
  end

  def self.shape(record : JSON::Any, context : String) : Array(Int32)
    raw = record["shape"].as_a
    raise ArgumentError.new("Empty #{context} shape") if raw.empty?
    raw.map do |item|
      value = item.as_i64
      raise ArgumentError.new("Invalid #{context} shape") unless value > 0 && value <= Int32::MAX
      value.to_i32
    end
  end

  def self.record_sha256_f32_le(values : Array(Float32)) : String
    bytes = IO::Memory.new(values.size * sizeof(Float32))
    values.each { |value| bytes.write_bytes(value, IO::ByteFormat::LittleEndian) }
    Digest::SHA256.hexdigest(bytes.to_slice)
  end

  def self.case_batches!(op : JSON::Any, case_name : String, stream : String,
                         input_shape : Array(Int32), config : JSON::Any) : Int32
    dim = config["dim"].as_i
    video_tokens = config["video_tokens"].as_i
    text_tokens = config["text_tokens"].as_i
    base_batches = config["batches"].as_i

    # New fixtures carry the source case's resolved shape config. For already
    # generated fixtures, the input record is itself the hash-pinned original
    # case shape; only broadcast has a documented per-case batch override.
    batch_count = if case_entry = op.as_h["case_config"]?
                    case_config = case_entry
                    unless case_config.as_h.keys.sort == ["batches", "dim", "text_tokens", "video_tokens"]
                      raise ArgumentError.new("Invalid per-case SiLU shape config")
                    end
                    unless case_config["dim"].as_i == dim &&
                           case_config["video_tokens"].as_i == video_tokens &&
                           case_config["text_tokens"].as_i == text_tokens
                      raise ArgumentError.new("Per-case SiLU shape config disagrees with the frozen fixture config")
                    end
                    case_config["batches"].as_i
                  elsif case_name == "broadcast"
                    input_shape[0]?
                  else
                    base_batches
                  end

    unless batch_count && batch_count > 0 && batch_count <= Int32::MAX
      raise ArgumentError.new("Invalid per-case SiLU batch count")
    end
    expected_shape = if case_name == "broadcast"
                       [batch_count, dim]
                     else
                       tokens = stream == "img" ? video_tokens : text_tokens
                       [batch_count, tokens, dim]
                     end
    unless input_shape == expected_shape
      raise ArgumentError.new("SiLU input layout disagrees with its immutable source case: #{case_name}/#{stream}")
    end
    batch_count.to_i32
  end

  def self.operations(fixture : JSON::Any) : Array(Operation)
    config = fixture["config"]
    dim = config["dim"].as_i
    batches = config["batches"].as_i
    video_tokens = config["video_tokens"].as_i
    text_tokens = config["text_tokens"].as_i
    raise ArgumentError.new("Invalid pointwise fixture config") unless {dim, batches, video_tokens, text_tokens}.all?(&.positive?)

    raw_operations = fixture["operations"].as_a
    expected_names = [
      "broadcast.img.vec_silu", "broadcast.txt.vec_silu",
      "per_token.img.vec_silu", "per_token.txt.vec_silu",
    ]
    names = raw_operations.map { |op| op["name"].as_s }
    unless names.size == expected_names.size && names.sort == expected_names.sort
      raise ArgumentError.new("Pointwise fixture must contain exactly the four admitted SiLU operations")
    end

    provenance_hashes = fixture["provenance"]["input_hashes_f32_le"]
    parsed = raw_operations.map do |op|
      name = op["name"].as_s
      kind = op["kind"].as_s
      case_name = op["case"].as_s
      stream = op["stream"].as_s
      expected_source = stream == "img" ? "#{case_name}.inputs.vec" : "#{case_name}.inputs.vec_txt"
      unless kind == "silu" && {"broadcast", "per_token"}.includes?(case_name) &&
             {"img", "txt"}.includes?(stream) && name == "#{case_name}.#{stream}.vec_silu" &&
             op["source_input"].as_s == expected_source
        raise ArgumentError.new("Invalid SiLU operation identity or source link: #{name}")
      end

      x_record = op["inputs"]["x"]
      x_shape = shape(x_record, "#{name}/x")
      case_batch_count = case_batches!(op, case_name, stream, x_shape, config)
      x = values(x_record, "#{name}/x")
      expected = values(op["expected"], "#{name}/expected")
      rounded_expected = values(op["rounded_expected"], "#{name}/rounded_expected")
      expected_shape = shape(op["expected"], "#{name}/expected")
      rounded_shape = shape(op["rounded_expected"], "#{name}/rounded_expected")
      unless x_shape == expected_shape && x_shape == rounded_shape && x.size == expected.size &&
             expected.size == rounded_expected.size
        raise ArgumentError.new("SiLU fixture tensor layouts disagree: #{name}")
      end
      input_hash = record_sha256_f32_le(x)
      unless op["input_sha256_f32_le"].as_s == input_hash && provenance_hashes[name].as_s == input_hash
        raise ArgumentError.new("SiLU input provenance hash mismatch: #{name}")
      end
      unless rounded_expected.map(&.unsafe_as(UInt32)) == expected.map { |value| bf16_bits(value) }
        raise ArgumentError.new("SiLU reference BF16 round-trip mismatch: #{name}")
      end

      Operation.new(name, case_name, stream, case_batch_count, x_shape, x, expected, rounded_expected)
    end
    parsed.sort_by(&.name)
  end

  def self.load(path : String, expected_sha256 : String) : JSON::Any
    trusted_digest!(expected_sha256)
    bytes = File.read(path)
    raise ArgumentError.new("Pointwise fixture SHA-256 mismatch") unless Digest::SHA256.hexdigest(bytes) == expected_sha256
    fixture = JSON.parse(bytes)
    unless fixture["schema_version"].as_i == 1 && fixture["kind"].as_s == "hy_nomas_same_input_pointwise_diagnostic" &&
           fixture["upstream_revision"].as_s == REVISION
      raise ArgumentError.new("Unsupported pointwise diagnostic fixture")
    end
    source_sha = fixture["source_fixture_sha256"].as_s
    trusted_digest!(source_sha)
    source_checkpoint = fixture["source_checkpoint_sha256"]
    trusted_digest!(source_checkpoint.as_s) unless source_checkpoint.raw.nil?
    provenance = fixture["provenance"]
    unless provenance["upstream_revision"].as_s == REVISION && provenance["reference_device"].as_s == "mps" &&
           provenance["compute_dtype"].as_s == "F32" && provenance["operation"].as_s == "pinned get_activation_layer('silu') -> torch.nn.SiLU" &&
           provenance["checkpoint_loaded"].as_bool == false && provenance["block_instantiated"].as_bool == false &&
           provenance["terminal_mps_synchronize_count"].as_i == 1 && provenance["torch_version"].as_s.starts_with?("2.8.")
      raise ArgumentError.new("Unsupported pointwise reference provenance")
    end
    unless fixture["weights_are_synthetic"].as_bool == source_checkpoint.raw.nil?
      raise ArgumentError.new("Pointwise source/checkpoint provenance mismatch")
    end
    operations(fixture)
    fixture
  end

  def self.bf16_bits(value : Float32) : UInt32
    bits = value.unsafe_as(UInt32)
    return ((bits >> 16) | 0x40_u32) << 16 if (bits & 0x7fffffff_u32) > 0x7f800000_u32
    (bits &+ 0x7fff_u32 &+ ((bits >> 16) & 1_u32)) & 0xffff0000_u32
  end

  def self.compare(actual : Array(Float32), reference : Array(Float32))
    violations = 0
    max_abs = 0.0_f32
    max_gate_ratio = 0.0_f32
    first = nil.as(Int32?)
    bitwise_mismatches = 0
    bf16_bin_mismatches = 0
    Math.min(actual.size, reference.size).times do |index|
      value = actual[index]
      target = reference[index]
      bitwise_mismatches += 1 unless value.unsafe_as(UInt32) == target.unsafe_as(UInt32)
      bf16_bin_mismatches += 1 unless bf16_bits(value) == bf16_bits(target)
      if value.finite? && target.finite?
        delta = (value - target).abs
        gate = 2e-5_f32 + 2e-5_f32 * target.abs
        max_abs = Math.max(max_abs, delta)
        max_gate_ratio = Math.max(max_gate_ratio, delta / gate)
        next if delta <= gate
      end
      violations += 1
      first ||= index
    end
    violations += (actual.size - reference.size).abs
    {
      count:               actual.size,
      reference_count:     reference.size,
      strict_violations:   violations,
      max_abs:             max_abs,
      max_gate_ratio:      max_gate_ratio,
      bitwise_mismatches:  bitwise_mismatches,
      bf16_bin_mismatches: bf16_bin_mismatches,
      first_index:         first,
      actual:              first.try { |index| actual[index] },
      reference:           first.try { |index| reference[index] },
      actual_bits:         first.try { |index| actual[index].unsafe_as(UInt32).to_s(16) },
      reference_bits:      first.try { |index| reference[index].unsafe_as(UInt32).to_s(16) },
    }
  end

  def self.write_fresh_report(path : String, contents : String) : Nil
    fd = LibC.open(path, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
    raise IO::Error.new("Pointwise report path must be fresh and writable") if fd < 0
    report = IO::FileDescriptor.new(fd)
    begin
      report.write(contents.to_slice)
    ensure
      report.close
    end
  end

  def self.pipeline(name : String) : ML::Metal::ComputePipeline
    ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, CONTROL_SOURCE) }
  end

  def self.dispatch(e : ML::Metal::ComputeEncoder, pipeline : ML::Metal::ComputePipeline,
                    input : ML::MetalBuffer, output : ML::MetalBuffer, count : Int32) : Nil
    e.set_pipeline(pipeline)
    e.set_buffer(input, 0).set_buffer(output, 1, ML::Metal::BufferAccess::Write)
    e.set_value(count.to_u32, 2).dispatch_1d(count)
  end

  def self.run(path : String, fixture_sha256 : String, report_path : String) : Nil
    fixture = load(path, fixture_sha256)
    ops = operations(fixture)
    arena = ML::Video::Arena.new
    chain = nil.as(ML::Metal::MPSCommandChain?)
    graph_plans = [] of ML::Video::HyNomasMPSActivation::SiLUPlan
    begin
      prod_pipeline = ML::Video::HyNomasDualStreamBlock.pipeline("hy_nomas_silu_f32")
      candidate_pipelines = {} of String => ML::Metal::ComputePipeline
      VARIANTS.each do |_, function|
        next if function == "hy_nomas_silu_f32"
        candidate_pipelines[function] = pipeline(function)
      end
      round_pipeline = pipeline("hy_probe_silu_roundtrip")

      ops.each_with_index do |op, op_index|
        arena.upload("#{op_index}.input", op.input)
        VARIANTS.each do |variant, _function|
          arena.alloc("#{op_index}.#{variant}.raw", op.input.size)
          arena.alloc("#{op_index}.#{variant}.rounded", op.input.size)
        end
        graph_raw = arena.alloc("#{op_index}.#{GRAPH_VARIANT}.raw", op.input.size)
        arena.alloc("#{op_index}.#{GRAPH_VARIANT}.rounded", op.input.size)
        graph_plans << ML::Video::HyNomasMPSActivation::SiLUPlan.new(
          op.shape, arena["#{op_index}.input"], graph_raw,
        )
      end

      allocations = ML::MetalBuffer.video_spec_allocations
      chain = ML::Metal::MPSCommandChain.new
      ops.each_with_index do |op, op_index|
        input = arena["#{op_index}.input"]
        encoder = chain.not_nil!.encoder
        VARIANTS.each do |variant, function|
          output = arena["#{op_index}.#{variant}.raw"]
          rounded = arena["#{op_index}.#{variant}.rounded"]
          active_pipeline = function == "hy_nomas_silu_f32" ? prod_pipeline : candidate_pipelines[function]
          dispatch(encoder, active_pipeline, input, output, op.input.size)
          dispatch(encoder, round_pipeline, output, rounded, op.input.size)
        end

        graph_plans[op_index].encode(chain.not_nil!)
        # Reuse the same fixed bindings on the same serial chain; completion
        # is still the single terminal wait below.
        graph_plans[op_index].encode(chain.not_nil!)
        expect_raises(ArgumentError, /Drain/) { graph_plans[op_index].release(chain) }
        graph_output = arena["#{op_index}.#{GRAPH_VARIANT}.raw"]
        graph_rounded = arena["#{op_index}.#{GRAPH_VARIANT}.rounded"]
        dispatch(chain.not_nil!.encoder, round_pipeline, graph_output,
          graph_rounded, op.input.size)
      end
      ML::MetalBuffer.video_spec_allocations.should eq(allocations), "pointwise encode must not allocate MetalBuffers"

      chain.not_nil!.commit_and_wait
      chain.not_nil!.wait_count.should eq(1), "pointwise probe must use one terminal wait"
      report_lines = [] of String
      report_lines << {
        phase:                    "manifest",
        upstream_revision:        REVISION,
        pointwise_fixture_sha256: fixture_sha256,
        source_fixture_sha256:    fixture["source_fixture_sha256"].as_s,
        source_checkpoint_sha256: fixture["source_checkpoint_sha256"].raw,
        weights_are_synthetic:    fixture["weights_are_synthetic"].as_bool,
        config:                   fixture["config"],
        reference:                fixture["provenance"],
        gate_atol:                2e-5,
        gate_rtol:                2e-5,
        rounded_gate:             "finite 2e-5 + 2e-5*abs(reference); BF16 bin count reported separately",
        production_changes:       false,
        candidate_variants:       VARIANTS.map(&.[0]).to_a + [GRAPH_VARIANT],
        case_batches:             ops.map { |op| {operation: op.name, batches: op.case_batches} },
        terminal_waits:           chain.not_nil!.wait_count,
      }.to_json

      strict_failures = [] of String
      ops.each_with_index do |op, op_index|
        arena["#{op_index}.input"].read(op.input.size).map(&.unsafe_as(UInt32)).should eq(op.input.map(&.unsafe_as(UInt32))),
          "frozen pointwise input changed: #{op.name}"
        VARIANTS.each do |variant, _function|
          raw = arena["#{op_index}.#{variant}.raw"].read(op.input.size)
          rounded = arena["#{op_index}.#{variant}.rounded"].read(op.input.size)
          rounded.map(&.unsafe_as(UInt32)).should eq(raw.map { |value| bf16_bits(value) }),
            "native BF16 round-trip must match its raw result: #{op.name}/#{variant}"
          {"raw" => {raw, op.expected}, "bf16_roundtrip" => {rounded, op.rounded_expected}}.each do |phase, pair|
            stats = compare(pair[0], pair[1])
            report_lines << {operation: op.name, case: op.case_name, stream: op.stream,
                             case_batches: op.case_batches,
                             variant: variant, phase: phase, stats: stats}.to_json
            strict_failures << "#{op.name}/#{variant}/#{phase}: #{stats[:strict_violations]}" if stats[:strict_violations] > 0
          end
          first_bin = raw.each_index.find { |index| bf16_bits(raw[index]) != bf16_bits(op.expected[index]) }
          if index = first_bin
            expected_rounded = op.rounded_expected[index]
            actual_rounded = rounded[index]
            report_lines << {operation: op.name, variant: variant, phase: "first_bf16_bin_difference",
                             index: index, actual: raw[index], reference: op.expected[index],
                             actual_bits: raw[index].unsafe_as(UInt32).to_s(16),
                             reference_bits: op.expected[index].unsafe_as(UInt32).to_s(16),
                             actual_rounded: actual_rounded, reference_rounded: expected_rounded,
                             rounded_bits_differ: actual_rounded.unsafe_as(UInt32) != expected_rounded.unsafe_as(UInt32)}.to_json
          end
        end

        graph_raw = arena["#{op_index}.#{GRAPH_VARIANT}.raw"].read(op.input.size)
        graph_rounded = arena["#{op_index}.#{GRAPH_VARIANT}.rounded"].read(op.input.size)
        graph_rounded.map(&.unsafe_as(UInt32)).should eq(graph_raw.map { |value| bf16_bits(value) }),
          "native BF16 round-trip must match its MPSGraph SiLU result: #{op.name}"
        {"raw"            => {graph_raw, op.expected},
         "bf16_roundtrip" => {graph_rounded, op.rounded_expected}}.each do |phase, pair|
          stats = compare(pair[0], pair[1])
          report_lines << {operation: op.name, case: op.case_name, stream: op.stream,
                           case_batches: op.case_batches,
                           variant: GRAPH_VARIANT, phase: phase, stats: stats}.to_json
          strict_failures << "#{op.name}/#{GRAPH_VARIANT}/#{phase}: #{stats[:strict_violations]}" if stats[:strict_violations] > 0
          strict_failures << "#{op.name}/#{GRAPH_VARIANT}/#{phase}: bitwise=#{stats[:bitwise_mismatches]}" if stats[:bitwise_mismatches] > 0
        end
        if index = graph_raw.each_index.find { |value_index| bf16_bits(graph_raw[value_index]) != bf16_bits(op.expected[value_index]) }
          expected_rounded = op.rounded_expected[index]
          actual_rounded = graph_rounded[index]
          report_lines << {operation: op.name, variant: GRAPH_VARIANT, phase: "first_bf16_bin_difference",
                           index: index, actual: graph_raw[index], reference: op.expected[index],
                           actual_bits: graph_raw[index].unsafe_as(UInt32).to_s(16),
                           reference_bits: op.expected[index].unsafe_as(UInt32).to_s(16),
                           actual_rounded: actual_rounded, reference_rounded: expected_rounded,
                           rounded_bits_differ: actual_rounded.unsafe_as(UInt32) != expected_rounded.unsafe_as(UInt32)}.to_json
        end
      end
      report_lines << {phase: "complete", operation_count: ops.size, variant_count: VARIANTS.size + 1,
                       strict_numerical_failure_records: strict_failures.size,
                       status: strict_failures.empty? ? "strict_gate_pass" : "strict_gate_failed"}.to_json
      write_fresh_report(report_path, report_lines.join('\n') + "\n")
      strict_failures.should be_empty, strict_failures.first(12).join('\n')
    ensure
      begin
        chain.try { |pending| pending.commit_and_wait unless pending.completed? }
      ensure
        if chain.nil? || chain.not_nil!.completed?
          begin
            graph_plans.each { |plan| plan.release(chain) }
            chain.try(&.release)
          ensure
            arena.release
          end
        end
      end
    end
  end
end

describe HyNomasPointwiseProbe do
  it "validates broadcast batch geometry from the pinned per-case shape" do
    config = JSON.parse(%({"batches":1,"dim":2,"video_tokens":2,"text_tokens":1}))
    legacy_op = JSON.parse(%({"case":"broadcast","stream":"img"}))
    HyNomasPointwiseProbe.case_batches!(legacy_op, "broadcast", "img", [2, 2], config).should eq(2)

    explicit_op = JSON.parse(%({"case_config":{"batches":2,"dim":2,"video_tokens":2,"text_tokens":1}}))
    HyNomasPointwiseProbe.case_batches!(explicit_op, "broadcast", "img", [2, 2], config).should eq(2)

    wrong_batch = JSON.parse(%({"case_config":{"batches":1,"dim":2,"video_tokens":2,"text_tokens":1}}))
    expect_raises(ArgumentError, /layout disagrees/) do
      HyNomasPointwiseProbe.case_batches!(wrong_batch, "broadcast", "img", [2, 2], config)
    end
    expect_raises(ArgumentError, /layout disagrees/) do
      HyNomasPointwiseProbe.case_batches!(legacy_op, "broadcast", "img", [2, 3], config)
    end
  end

  it "retains the finite gate and reports F32 and BF16 disagreement separately" do
    HyNomasPointwiseProbe.compare([0.25_f32], [0.25_f32])[:strict_violations].should eq(0)
    close_values = HyNomasPointwiseProbe.compare([0.7675781_f32], [0.7675783_f32])
    close_values[:strict_violations].should eq(0)
    close_values[:bf16_bin_mismatches].should eq(1)
    HyNomasPointwiseProbe.compare([0.26_f32], [0.25_f32])[:strict_violations].should eq(1)
    HyNomasPointwiseProbe.compare([Float32::NAN], [0.25_f32])[:strict_violations].should eq(1)
    HyNomasPointwiseProbe.compare([Float32::INFINITY], [Float32::INFINITY])[:strict_violations].should eq(1)
  end

  it "rejects changed fixture bytes and malformed tensors before Metal allocation" do
    before = ML::MetalBuffer.video_spec_allocations
    file = File.tempfile("hy-nomas-pointwise", ".json")
    begin
      file << "{}"
      file.flush
      expect_raises(ArgumentError, /SHA-256 mismatch/) { HyNomasPointwiseProbe.load(file.path, "0" * 64) }
      expect_raises(ArgumentError, /trusted SHA/) { HyNomasPointwiseProbe.load(file.path, "self-declared") }
      expect_raises(ArgumentError, /value count/) do
        HyNomasPointwiseProbe.values(JSON.parse(%({"shape":[2],"values":[0.25]})), "seed")
      end
      expect_raises(ArgumentError, /non-finite/) do
        HyNomasPointwiseProbe.values(JSON.parse(%({"shape":[1],"values":[1e100]})), "seed")
      end
      ML::MetalBuffer.video_spec_allocations.should eq(before)
    ensure
      path = file.path
      file.close
      File.delete(path) if File.exists?(path)
    end
  end

  it "compares same-input PyTorch MPS SiLU against native arithmetic candidates" do
    unless fixture = ENV["HY_NOMAS_POINTWISE_FIXTURE"]?
      pending!("Set HY_NOMAS_POINTWISE_FIXTURE, HY_NOMAS_POINTWISE_FIXTURE_SHA256, and HY_NOMAS_POINTWISE_REPORT")
    end
    fixture_sha256 = ENV["HY_NOMAS_POINTWISE_FIXTURE_SHA256"]
    report = ENV["HY_NOMAS_POINTWISE_REPORT"]
    HyNomasPointwiseProbe.run(fixture, fixture_sha256, report)
  end
end
