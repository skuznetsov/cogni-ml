# Diagnostic only: frozen upstream operands isolate local arithmetic from
# propagation through earlier block operations. Production kernels are unchanged.
require "spec"
require "json"
require "digest/sha256"
require "./support/video_allocation_counter"
require "../src/ml/video/hy_nomas_block"
require "./support/hy_nomas_mps_projection_controls"
require "./support/hy_nomas_mps_reductions"
require "./support/hy_nomas_rms_controls"

module HyAttentionProbe
  SOURCE = ML::Video::MetalOps::SOURCE + ML::Video::HyNomasDualStreamBlock::SOURCE +
           {{ read_file("#{__DIR__}/support/hy_nomas_attention_probe.metal") }} +
           RMS_ROPE_CONTROL_SOURCE
  REVISION = "911cea6458a89c2663513ff7099561ec190dd491"

  def self.fresh_report(path : String, & : IO ->) : Nil
    fd = LibC.open(path, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
    raise IO::Error.new("Diagnostic report path must be fresh and writable") if fd < 0
    report = IO::FileDescriptor.new(fd)
    begin
      yield report
    ensure
      report.close
    end
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

  def self.trusted_digest!(value : String) : Nil
    raise ArgumentError.new("Expected a trusted SHA-256 digest") unless value.matches?(/\A[0-9a-f]{64}\z/)
  end

  def self.load(path : String, expected : String, source_mode : String = "any") : JSON::Any
    trusted_digest!(expected)
    bytes = File.read(path)
    raise ArgumentError.new("Probe fixture SHA-256 mismatch") unless Digest::SHA256.hexdigest(bytes) == expected
    fixture = JSON.parse(bytes)
    unless fixture["schema_version"].as_i == 1 && fixture["kind"].as_s == "hy_nomas_same_input_attention_diagnostic" &&
           fixture["upstream_revision"].as_s == REVISION
      raise ArgumentError.new("Unsupported attention diagnostic fixture")
    end
    expected_labels = ["broadcast/f32", "broadcast/bf16_weights_roundtrip_f32_compute",
                       "per_token/f32", "per_token/bf16_weights_roundtrip_f32_compute"]
    unless fixture["cases"].as_a.map { |data| data["label"].as_s }.sort == expected_labels.sort
      raise ArgumentError.new("Probe requires exactly the four admitted case/weight variants")
    end
    unless {"any", "actual", "synthetic"}.includes?(source_mode) &&
           (source_mode == "any" || fixture["weights_are_synthetic"].as_bool == (source_mode == "synthetic"))
      raise ArgumentError.new("Diagnostic source mode mismatch")
    end
    provenance = fixture["provenance"]
    unless provenance["activation_storage"].as_s == "F32" &&
           provenance["canonical_qkv_boundary"].as_s == "BF16_RNE_roundtrip_F32" &&
           provenance["effective_weight_policy"].as_s == "source_to_F32_or_source_to_BF16_RNE_to_F32_on_MPS"
      raise ArgumentError.new("Unsupported probe activation or effective-weight policy")
    end
    fixture
  end

  def self.values(record : JSON::Any) : Array(Float32)
    shape = record["shape"].as_a.map(&.as_i64)
    unless !shape.empty? && shape.all? { |size| size > 0 && size <= Int32::MAX }
      raise ArgumentError.new("Invalid probe tensor shape")
    end
    count = 1_i64
    shape.each do |size|
      raise ArgumentError.new("Probe tensor is too large") if count > Int32::MAX // size
      count *= size
    end
    result = record["values"].as_a.map do |item|
      value = item.as_f
      unless value.finite? && value.abs <= Float32::MAX.to_f64
        raise ArgumentError.new("Probe tensor contains a nonfinite or out-of-range value")
      end
      value.to_f32
    end
    unless result.size == count && result.all?(&.finite?)
      raise ArgumentError.new("Probe tensor length or finite-value mismatch")
    end
    result
  end

  def self.config(data : JSON::Any) : ML::Video::HyNomasConfig
    c = data["config"]
    result = ML::Video::HyNomasConfig.new(
      batches: c["batches"].as_i, video_tokens: c["video_tokens"].as_i,
      text_tokens: c["text_tokens"].as_i, dim: c["dim"].as_i,
      heads: c["heads"].as_i, head_dim: c["head_dim"].as_i,
      ffn_dim: c["ffn_dim"].as_i,
    )
    result.validate!
    result
  end

  # Validate the exact admitted layouts before any diagnostic GPU allocation.
  # Score/probability layout is [B,Q,H,K], not [B,H,Q,K].
  def self.validate_operation!(op : JSON::Any, c : ML::Video::HyNomasConfig) : Nil
    inputs = op["inputs"].as_h
    inputs.each_value { |record| values(record) }
    expected = values(op["expected"])
    values(op["rounded_expected"]).size.should eq(expected.size)
    if origin = op["origin_expected"]?
      values(origin).size.should eq(expected.size)
    end
    joint_shape = [c.batches, c.joint_tokens, c.heads, c.head_dim]
    score_shape = [c.batches, c.joint_tokens, c.heads, c.joint_tokens]
    require_shape = ->(name : String, shape : Array(Int32)) do
      unless inputs[name]["shape"].as_a.map(&.as_i) == shape
        raise ArgumentError.new("Probe input layout mismatch: #{op["name"]}/#{name}")
      end
    end
    count = c.joint_rows * c.dim
    output_shape = joint_shape
    case op["kind"].as_s
    when "linear", "rms", "rope"
      prefix = op["weight_prefix"]?.try(&.as_s) || ""
      stream = op["name"].as_s.split('.').first
      raise ArgumentError.new("Invalid probe stream") unless {"img", "txt"}.includes?(stream)
      tokens = stream == "img" ? c.video_tokens : c.text_tokens
      count = c.batches * tokens * c.dim
      unless values(inputs["x"]).size == count
        raise ArgumentError.new("Probe input length mismatch")
      end
      if op["kind"].as_s == "linear"
        unless {"q", "k", "v", "proj"}.any? { |name| prefix == "#{stream}_attn_#{name}" }
          raise ArgumentError.new("Unsupported probe projection")
        end
        require_shape.call("x", [c.batches, tokens, c.dim])
        output_shape = prefix.ends_with?("_proj") ? [c.batches, tokens, c.dim] : [c.batches, tokens, c.heads, c.head_dim]
        unless {"normal", "none"}.includes?(op["bias_mode"]?.try(&.as_s) || "normal")
          raise ArgumentError.new("Unsupported diagnostic bias mode")
        end
      else
        output_shape = [c.batches, tokens, c.heads, c.head_dim]
        require_shape.call("x", [c.batches, tokens, c.heads, c.head_dim])
        if op["kind"].as_s == "rms"
          unless {"q", "k"}.any? { |name| prefix == "#{stream}_attn_#{name}_norm" }
            raise ArgumentError.new("Unsupported probe RMSNorm")
          end
        else
          raise ArgumentError.new("Only video RoPE is admitted") unless stream == "img"
          require_shape.call("cos", [c.video_tokens, c.head_dim])
          require_shape.call("sin", [c.video_tokens, c.head_dim])
        end
      end
    when "qk"
      require_shape.call("q", joint_shape)
      require_shape.call("k", joint_shape)
      count = ML::Video::HyNomasDualStreamBlock.checked_count(c.batches, c.joint_tokens, c.heads, c.joint_tokens)
      output_shape = score_shape
    when "softmax"
      require_shape.call("scores", score_shape)
      count = ML::Video::HyNomasDualStreamBlock.checked_count(c.batches, c.joint_tokens, c.heads, c.joint_tokens)
      output_shape = score_shape
    when "pv"
      require_shape.call("probabilities", score_shape)
      require_shape.call("v", joint_shape)
    when "online_scores"
      require_shape.call("scores", score_shape)
      require_shape.call("v", joint_shape)
    when "attention"
      {"q", "k", "v"}.each { |name| require_shape.call(name, joint_shape) }
    else
      raise ArgumentError.new("Unknown attention probe operation")
    end
    scale(op, c) if {"qk", "attention"}.includes?(op["kind"].as_s)
    raise ArgumentError.new("Probe output length mismatch") unless expected.size == count
    unless op["expected"]["shape"].as_a.map(&.as_i) == output_shape &&
           op["rounded_expected"]["shape"].as_a.map(&.as_i) == output_shape
      raise ArgumentError.new("Probe output layout mismatch")
    end
    if origin = op["origin_expected"]?
      allowed_shapes = [output_shape]
      allowed_shapes << [c.batches, c.joint_tokens, c.dim] if op["kind"].as_s == "attention"
      unless allowed_shapes.includes?(origin["shape"].as_a.map(&.as_i))
        raise ArgumentError.new("Original upstream output layout mismatch")
      end
    end
  end

  def self.same_values!(left : JSON::Any, right : JSON::Any, label : String) : Nil
    unless values(left).map(&.unsafe_as(UInt32)) == values(right).map(&.unsafe_as(UInt32))
      raise ArgumentError.new("Frozen operand identity mismatch: #{label}")
    end
  end

  # Verification of frozen fixture lineage only, never an inference path.
  def self.scaled_values!(scaled : JSON::Any, original : JSON::Any, factor : Float32, label : String) : Nil
    unless values(scaled).map(&.unsafe_as(UInt32)) == values(original).map { |value| (value * factor).unsafe_as(UInt32) }
      raise ArgumentError.new("Frozen scaled operand mismatch: #{label}")
    end
  end

  def self.validate_links!(data : JSON::Any) : Nil
    c = config(data)
    variant = data["source_variant"].as_s
    expected_resident = case variant
                        when "f32"                                then "F32"
                        when "bf16_weights_roundtrip_f32_compute" then "BF16"
                        else                                           raise ArgumentError.new("Unsupported diagnostic weight variant")
                        end
    unless data["resident_dtype"].as_s == expected_resident &&
           data["label"].as_s == "#{data["source_case"].as_s}/#{variant}"
      raise ArgumentError.new("Diagnostic label/resident dtype mismatch")
    end
    ops = data["operations"].as_a.to_h { |op| {op["name"].as_s, op} }
    required = {} of String => String
    {"img", "txt"}.each do |stream|
      {"q", "k", "v", "proj"}.each do |part|
        base = "#{stream}.attn.#{part}.linear"
        {"", "_nobias", "_explicit_bias"}.each do |suffix|
          required[base + suffix] = "linear"
          op = ops[base + suffix]
          unless op["weight_prefix"].as_s == "#{stream}_attn_#{part}"
            raise ArgumentError.new("Probe projection identity mismatch")
          end
          unless op["bias_mode"].as_s == (suffix == "_nobias" ? "none" : "normal")
            raise ArgumentError.new("Probe projection bias mode mismatch")
          end
          same_values!(ops[base]["inputs"]["x"], op["inputs"]["x"], base + suffix)
        end
        if part != "proj"
          same_values!(ops["#{stream}.attn.q.linear"]["inputs"]["x"], ops[base]["inputs"]["x"], "shared QKV projection input")
        end
      end
      {"q", "k"}.each do |part|
        name = "#{stream}.attn.#{part}.rms"
        required[name] = "rms"
        unless ops[name]["weight_prefix"].as_s == "#{stream}_attn_#{part}_norm"
          raise ArgumentError.new("Probe RMSNorm identity mismatch")
        end
        same_values!(ops[name]["inputs"]["x"], ops["#{stream}.attn.#{part}.linear"]["origin_expected"], name)
        if stream == "img"
          rope = "#{stream}.attn.#{part}.rope"
          required[rope] = "rope"
          same_values!(ops[rope]["inputs"]["x"], ops[name]["origin_expected"], rope)
        end
      end
    end
    canonical = ops["attention.canonical_sdpa"]
    required["attention.canonical_sdpa"] = "attention"
    {"q", "k", "v"}.each do |part|
      frozen = data["canonical_qkv"][part]
      unless frozen["shape"].as_a.map(&.as_i) == [c.batches, c.joint_tokens, c.heads, c.head_dim]
        raise ArgumentError.new("Canonical attention record layout mismatch")
      end
      same_values!(frozen, canonical["inputs"][part], "canonical QKV/#{part}")
      unless values(canonical["inputs"][part]).all? { |value| value.unsafe_as(UInt32) == bf16_bits(value) }
        raise ArgumentError.new("Canonical attention input must hold BF16-expanded F32 values")
      end
      img_name = part == "v" ? "img.attn.v.linear" : "img.attn.#{part}.rope"
      txt_name = part == "v" ? "txt.attn.v.linear" : "txt.attn.#{part}.rms"
      img = values(ops[img_name]["origin_expected"])
      txt = values(ops[txt_name]["origin_expected"])
      boundary_bits = [] of UInt32
      c.batches.times do |batch|
        img[batch * c.video_tokens * c.dim, c.video_tokens * c.dim].each { |value| boundary_bits << bf16_bits(value) }
        txt[batch * c.text_tokens * c.dim, c.text_tokens * c.dim].each { |value| boundary_bits << bf16_bits(value) }
      end
      unless boundary_bits == values(frozen).map(&.unsafe_as(UInt32))
        raise ArgumentError.new("Frozen attention boundary lineage mismatch: #{part}")
      end
    end
    unless scale(canonical, c) == 1.0_f32 / Math.sqrt(c.head_dim.to_f32)
      raise ArgumentError.new("Canonical attention scale mismatch")
    end
    {"postscale", "query_prescale", "split_prescale"}.each do |mode|
      qk = ops["qk.#{mode}"]
      softmax = ops["softmax.#{mode}"]
      pv = ops["pv.#{mode}"]
      explicit = ops["attention.explicit.#{mode}"]
      required["qk.#{mode}"] = "qk"
      required["softmax.#{mode}"] = "softmax"
      required["pv.#{mode}"] = "pv"
      required["attention.explicit.#{mode}"] = "attention"
      expected_scale = mode == "postscale" ? scale(canonical, c) : 1.0_f32
      unless scale(qk, c) == expected_scale && scale(explicit, c) == expected_scale
        raise ArgumentError.new("Diagnostic scaling mode mismatch")
      end
      {"q", "k"}.each do |part|
        same_values!(qk["inputs"][part], explicit["inputs"][part], "#{mode}/#{part}")
        same_values!(qk["inputs"][part], canonical["inputs"][part], "postscale canonical/#{part}") if mode == "postscale"
      end
      if mode == "query_prescale"
        scaled_values!(qk["inputs"]["q"], canonical["inputs"]["q"], (1.0 / Math.sqrt(c.head_dim.to_f64)).to_f32, "query prescale Q")
        same_values!(qk["inputs"]["k"], canonical["inputs"]["k"], "query prescale K")
      elsif mode == "split_prescale"
        factor = Math.sqrt(1.0 / Math.sqrt(c.head_dim.to_f64)).to_f32
        {"q", "k"}.each { |part| scaled_values!(qk["inputs"][part], canonical["inputs"][part], factor, "split prescale/#{part}") }
      end
      same_values!(qk["expected"], softmax["inputs"]["scores"], "#{mode}/scores")
      same_values!(softmax["expected"], pv["inputs"]["probabilities"], "#{mode}/probabilities")
      same_values!(pv["inputs"]["v"], canonical["inputs"]["v"], "#{mode}/V")
      same_values!(explicit["inputs"]["v"], canonical["inputs"]["v"], "#{mode}/explicit V")
      same_values!(explicit["expected"], pv["expected"], "#{mode}/explicit output")
    end
    required["online_scores.postscale"] = "online_scores"
    online = ops["online_scores.postscale"]
    same_values!(online["inputs"]["scores"], ops["qk.postscale"]["expected"], "online frozen scores")
    same_values!(online["inputs"]["v"], canonical["inputs"]["v"], "online V")
    required["attention.mps_math_sdpa"] = "attention"
    required["softmax.mps_math_probabilities"] = "softmax"
    required["pv.mps_math_probabilities"] = "pv"
    direct = ops["attention.mps_math_sdpa"]
    probability = ops["softmax.mps_math_probabilities"]
    pv = ops["pv.mps_math_probabilities"]
    {"q", "k", "v"}.each { |part| same_values!(direct["inputs"][part], canonical["inputs"][part], "direct MPS/#{part}") }
    unless scale(direct, c) == scale(canonical, c)
      raise ArgumentError.new("Direct MPS attention scale mismatch")
    end
    same_values!(direct["expected"], canonical["expected"], "direct MPS output/canonical SDPA")
    same_values!(probability["inputs"]["scores"], ops["qk.postscale"]["expected"], "direct MPS softmax score control")
    same_values!(pv["inputs"]["probabilities"], probability["expected"], "direct MPS frozen probabilities")
    same_values!(pv["inputs"]["v"], canonical["inputs"]["v"], "direct MPS PV input")
    same_values!(pv["expected"], direct["expected"], "direct MPS PV output")
    unless ops.keys.sort == required.keys.sort
      raise ArgumentError.new("Unexpected diagnostic operation inventory")
    end
    required.each do |name, kind|
      raise ArgumentError.new("Diagnostic operation kind mismatch: #{name}") unless ops[name]["kind"].as_s == kind
    end
  end

  def self.scale(op : JSON::Any, c : ML::Video::HyNomasConfig) : Float32
    canonical = 1.0_f32 / Math.sqrt(c.head_dim.to_f32)
    value = op["scale"]?.try(&.as_f) || canonical.to_f64
    unless value.finite? && value > 0.0 && value <= 1.0 && (value.to_f32 == canonical || value == 1.0)
      raise ArgumentError.new("Unsupported diagnostic attention scale")
    end
    value.to_f32
  end

  def self.online(e : ML::Metal::ComputeEncoder, scores : ML::MetalBuffer, value : ML::MetalBuffer,
                  output : ML::MetalBuffer, c : ML::Video::HyNomasConfig) : Nil
    e.set_pipeline(pipeline("hy_probe_online_scores"))
    e.set_buffer(scores, 0).set_buffer(value, 1).set_buffer(output, 2, ML::Metal::BufferAccess::Write)
    e.set_value(c.joint_tokens.to_u32, 3).set_value(c.joint_tokens.to_u32, 4)
    e.set_value(c.heads.to_u32, 5).set_value(c.head_dim.to_u32, 6)
    e.dispatch_threadgroups({c.batches * c.joint_tokens * c.heads, 1, 1}, {((c.head_dim + 31) // 32) * 32, 1, 1})
  end

  def self.pipeline(name : String) : ML::Metal::ComputePipeline
    ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
  end

  def self.round(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer, y : ML::MetalBuffer, count : Int32) : Nil
    e.set_pipeline(pipeline("hy_probe_round"))
    e.set_buffer(x, 0).set_buffer(y, 1, ML::Metal::BufferAccess::Write)
    e.set_value(count.to_u32, 2).dispatch_1d(count)
  end

  def self.encode(e : ML::Metal::ComputeEncoder, op : JSON::Any, index : Int32,
                  c : ML::Video::HyNomasConfig, arena : ML::Video::Arena, weights : ML::Video::Arena) : Nil
    input = ->(name : String) { arena["#{index}.input.#{name}"] }
    output = arena["#{index}.output"]
    prefix = op["weight_prefix"]?.try(&.as_s) || ""
    rows = values(op["expected"]).size // c.dim
    threads = ((c.head_dim + 31) // 32) * 32
    groups = c.batches * c.joint_tokens * c.heads
    case op["kind"].as_s
    when "linear"
      no_bias = op["bias_mode"]?.try(&.as_s) == "none"
      bias = no_bias ? arena["zero"] : weights["#{prefix}_bias"]
      bias_dtype = no_bias ? ML::Video::WeightDType::F32 : weights.dtype("#{prefix}_bias")
      ML::Video::MetalOps.linear(e, input.call("x"), weights["#{prefix}_weight"], bias, output,
        rows, c.dim, c.dim, weights.dtype("#{prefix}_weight"), bias_dtype)
    when "rms"
      ML::Video::MetalOps.norm(e, input.call("x"), weights["#{prefix}_weight"], arena["zero"], output,
        rows * c.heads, c.head_dim, c.norm_eps, rms: true, affine: true,
        weight_dtype: weights.dtype("#{prefix}_weight"))
    when "rope"
      e.set_pipeline(ML::Video::HyNomasDualStreamBlock.pipeline("hy_nomas_rope_f32"))
      e.set_buffer(input.call("x"), 0).set_buffer(input.call("cos"), 1).set_buffer(input.call("sin"), 2)
      e.set_buffer(output, 3, ML::Metal::BufferAccess::Write)
      e.set_value(c.video_tokens.to_u32, 4).set_value(c.heads.to_u32, 5)
      e.set_value(c.head_dim.to_u32, 6).set_value(c.batches.to_u32, 7)
      e.dispatch_1d(rows * c.dim)
    when "qk"
      e.set_pipeline(pipeline("hy_probe_qk"))
      e.set_buffer(input.call("q"), 0).set_buffer(input.call("k"), 1)
      e.set_buffer(output, 2, ML::Metal::BufferAccess::Write)
      e.set_value(c.joint_tokens.to_u32, 3).set_value(c.joint_tokens.to_u32, 4)
      e.set_value(c.heads.to_u32, 5).set_value(c.head_dim.to_u32, 6)
      e.set_value(scale(op, c), 7)
      e.dispatch_threadgroups({groups, 1, 1}, {threads, 1, 1})
    when "softmax"
      e.set_pipeline(pipeline("hy_probe_softmax"))
      e.set_buffer(input.call("scores"), 0).set_buffer(output, 1, ML::Metal::BufferAccess::Write)
      e.set_value(c.joint_tokens.to_u32, 2)
      # This serial diagnostic has no padded rows: exactly one thread per row.
      e.dispatch_threadgroups({groups, 1, 1}, {1, 1, 1})
    when "pv"
      e.set_pipeline(pipeline("hy_probe_pv"))
      e.set_buffer(input.call("probabilities"), 0).set_buffer(input.call("v"), 1)
      e.set_buffer(output, 2, ML::Metal::BufferAccess::Write)
      e.set_value(c.joint_tokens.to_u32, 3).set_value(c.joint_tokens.to_u32, 4)
      e.set_value(c.heads.to_u32, 5).set_value(c.head_dim.to_u32, 6)
      e.set_value((c.joint_rows * c.dim).to_u32, 7).dispatch_1d(c.joint_rows * c.dim)
    when "online_scores"
      online(e, input.call("scores"), input.call("v"), output, c)
    when "attention"
      # The unchanged production kernel, with explicit diagnostic scale for
      # the pre-scaled-input counterfactuals. Canonical scale is unchanged.
      e.set_pipeline(ML::Video::MetalOps.pipeline("video_attention_f32"))
      e.set_buffer(input.call("q"), 0).set_buffer(input.call("k"), 1).set_buffer(input.call("v"), 2)
      e.set_buffer(output, 3, ML::Metal::BufferAccess::Write)
      e.set_value(c.joint_tokens.to_u32, 4).set_value(c.joint_tokens.to_u32, 5)
      e.set_value(c.heads.to_u32, 6).set_value(c.head_dim.to_u32, 7).set_value(scale(op, c), 8)
      e.dispatch_threadgroups({groups, 1, 1}, {threads, 1, 1})
    end
    round(e, output, arena["#{index}.rounded"], values(op["expected"]).size)
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
    bins = 0
    bits = 0
    Math.min(actual.size, reference.size).times do |index|
      value = actual[index]
      target = reference[index]
      bins += 1 unless bf16_bits(value) == bf16_bits(target)
      bits += 1 unless value.unsafe_as(UInt32) == target.unsafe_as(UInt32)
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
    {count: actual.size, reference_count: reference.size, violations: violations,
     max_abs: max_abs, max_gate_ratio: max_gate_ratio, bf16_bin_mismatches: bins,
     bitwise_mismatches: bits,
     first_index: first, actual: first.try { |i| actual[i] }, reference: first.try { |i| reference[i] },
     actual_bits: first.try { |i| actual[i].unsafe_as(UInt32).to_s(16) },
     reference_bits: first.try { |i| reference[i].unsafe_as(UInt32).to_s(16) }}
  end

  def self.run_case(data : JSON::Any, source : ML::Video::CheckpointFile,
                    source_dtype : ML::GGUF::SafeTensorDType, report : IO,
                    mps_controls : Bool = false) : Array(String)
    c = config(data)
    ops = data["operations"].as_a
    raise ArgumentError.new("Empty or duplicate probe operations") unless !ops.empty? && ops.map { |op| op["name"].as_s }.uniq.size == ops.size
    ops.each { |op| validate_operation!(op, c) }
    validate_links!(data)
    resident = ML::Video::WeightDType.parse(data["resident_dtype"].as_s)
    arena = ML::Video::Arena.new
    weights = ML::Video::Arena.new
    mps_arena = ML::Video::Arena.new
    prepared = nil.as(ML::Video::PreparedWeights?)
    projections = nil.as(MPSProjectionControls?)
    reductions = nil.as(MPSReductionControls?)
    rms_controls = nil.as(MPSRMSControls?)
    rope_controls = nil.as(MPSRopeControls?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    failures = [] of String
    begin
      arena.upload("zero", Array(Float32).new(c.dim, 0.0_f32))
      ops.each_with_index do |op, index|
        op["inputs"].as_h.each { |name, record| arena.upload("#{index}.input.#{name}", values(record)) }
        count = values(op["expected"]).size
        arena.alloc("#{index}.output", count)
        arena.alloc("#{index}.rounded", count)
      end
      arena.alloc("native_factored", c.joint_rows * c.dim)
      arena.alloc("native_factored.rounded", c.joint_rows * c.dim)
      budget = 2_i64 * File.size(source.path) + arena.resident_bytes + 1024_i64 * 1024
      plan = ML::Video::WeightPlan.new([source], ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(c, source_dtype),
        budget, resident_dtype: resident, require_all: true, reserve_bytes: arena.resident_bytes,
        allow_f32_to_bf16: source_dtype.f32? && resident.bf16?)
      prepared = ML::Video::PreparedWeights.new(plan, weights)
      if mps_controls
        mps_arena.upload("zero", Array(Float32).new(c.dim, 0.0_f32))
        ops.each_with_index do |op, index|
          next unless {"linear", "qk", "softmax", "pv", "attention", "rms", "rope"}.includes?(op["kind"].as_s)
          op["inputs"].as_h.each { |name, record| mps_arena.upload("#{index}.input.#{name}", values(record)) }
          count = values(op["expected"]).size
          mps_arena.alloc("#{index}.output", count)
          mps_arena.alloc("#{index}.rounded", count)
        end
        projections = MPSProjectionControls.new(c, ops, mps_arena, weights)
        reductions = MPSReductionControls.new(c, ops, mps_arena)
        rms_controls = MPSRMSControls.new(c, ops, mps_arena, weights)
        rope_controls = MPSRopeControls.new(c, ops, mps_arena)
      end
      allocations = ML::MetalBuffer.video_spec_allocations
      command = nil.as(ML::Metal::CommandBuffer?)
      encoder = if mps_controls
                  chain = ML::Metal::MPSCommandChain.new
                  chain.not_nil!.encoder
                else
                  command = ML::Metal::CommandBuffer.new
                  ML::Metal::ComputeEncoder.new(command.not_nil!)
                end
      prepared.not_nil!.encode(encoder)
      ops.each_with_index { |op, index| encode(encoder, op, index, c, arena, weights) }
      qk_index = ops.index! { |op| op["name"].as_s == "qk.postscale" }
      attention_index = ops.index! { |op| op["name"].as_s == "attention.canonical_sdpa" }
      online(encoder, arena["#{qk_index}.output"], arena["#{attention_index}.input.v"], arena["native_factored"], c)
      round(encoder, arena["native_factored"], arena["native_factored.rounded"], c.joint_rows * c.dim)
      if mps_controls
        ops.each_with_index do |op, index|
          case op["kind"].as_s
          when "linear"
            projections.not_nil!.encode(chain.not_nil!.encoder, op, index)
          when "qk", "softmax", "pv", "attention"
            reductions.not_nil!.encode(chain.not_nil!, op, index)
          when "rms"
            rms_controls.not_nil!.encode(chain.not_nil!, op, index)
          when "rope"
            rope_controls.not_nil!.encode(chain.not_nil!.encoder, op, index)
          else
            next
          end
          round(chain.not_nil!.encoder, mps_arena["#{index}.output"], mps_arena["#{index}.rounded"], values(op["expected"]).size)
        end
      else
        encoder.end_encoding
      end
      ML::MetalBuffer.video_spec_allocations.should eq(allocations), "diagnostic encode must not allocate MetalBuffers"
      # Only terminal diagnostic completion/readback; no production waits added.
      seconds = if mps_controls
                  chain.not_nil!.commit_and_wait
                  chain.not_nil!.wait_count.should eq(1)
                  chain.not_nil!.gpu_elapsed_seconds
                else
                  command.not_nil!.commit_and_wait_gpu_elapsed_seconds
                end
      label = data["label"].as_s
      snapshots = {} of String => Array(Float32)
      rounded_snapshots = {} of String => Array(Float32)
      ops.each_with_index do |op, index|
        op["inputs"].as_h.each do |name, record|
          expected = values(record)
          arena["#{index}.input.#{name}"].read(expected.size).map(&.unsafe_as(UInt32)).should eq(expected.map(&.unsafe_as(UInt32))),
            "frozen upstream input must remain bitwise identical: #{label}/#{op["name"]}/#{name}"
        end
        expected = values(op["expected"])
        actual = arena["#{index}.output"].read(expected.size)
        rounded = arena["#{index}.rounded"].read(expected.size)
        snapshots[op["name"].as_s] = actual
        rounded_snapshots[op["name"].as_s] = rounded
        rounded.map(&.unsafe_as(UInt32)).should eq(actual.map { |value| bf16_bits(value) }),
          "native boundary must be the correct rounding of its own raw result"
        expected_rounded = values(op["rounded_expected"])
        expected_rounded.map(&.unsafe_as(UInt32)).should eq(expected.map { |value| bf16_bits(value) }),
          "reference boundary must be the correct rounding of its own raw result"
        {"raw" => {actual, expected}, "rounded" => {rounded, expected_rounded}}.each do |phase, pair|
          stats = compare(pair[0], pair[1])
          report.puts({label: label, operation: op["name"].as_s, kind: op["kind"].as_s,
                       reference_mode: op["reference_mode"]?.try(&.as_s), phase: phase, stats: stats}.to_json)
          if stats[:violations] > 0 || (phase == "rounded" && stats[:bitwise_mismatches] > 0)
            failures << "#{label}/#{op["name"]}/#{phase}: strict=#{stats[:violations]} bits=#{stats[:bitwise_mismatches]}"
          end
        end
        if origin = op["origin_expected"]?
          stats = compare(expected, values(origin))
          report.puts({label: label, operation: op["name"].as_s, phase: "reference_reproduction", stats: stats}.to_json)
          failures << "#{label}/#{op["name"]}/reference_reproduction" if stats[:violations] > 0
        end
        first_bin = actual.each_index.find { |i| bf16_bits(actual[i]) != bf16_bits(expected[i]) }
        if i = first_bin
          midpoint = (rounded[i].to_f64 + expected_rounded[i].to_f64) / 2.0
          report.puts({label: label, operation: op["name"].as_s, phase: "first_bf16_bin_difference", index: i,
                       actual: actual[i], reference: expected[i], actual_bits: actual[i].unsafe_as(UInt32).to_s(16),
                       reference_bits: expected[i].unsafe_as(UInt32).to_s(16), rounded: rounded[i], reference_rounded: expected_rounded[i],
                       midpoint: midpoint, actual_minus_midpoint: actual[i].to_f64 - midpoint,
                       reference_minus_midpoint: expected[i].to_f64 - midpoint}.to_json)
        end
      end
      factored = arena["native_factored"].read(c.joint_rows * c.dim)
      production = arena["#{attention_index}.output"].read(c.joint_rows * c.dim)
      stats = compare(factored, production)
      report.puts({label: label, operation: "native_factored_vs_production", phase: "bitwise_control", stats: stats}.to_json)
      failures << "#{label}/native_factored_vs_production" if stats[:bitwise_mismatches] > 0
      {"raw"     => {factored, values(ops[attention_index]["expected"])},
       "rounded" => {arena["native_factored.rounded"].read(c.joint_rows * c.dim), values(ops[attention_index]["rounded_expected"])}}.each do |phase, pair|
        report.puts({label: label, operation: "native_factored_vs_upstream", phase: phase, stats: compare(pair[0], pair[1])}.to_json)
      end
      indexed = ops.to_h { |op| {op["name"].as_s, op} }
      canonical_reference = values(indexed["attention.canonical_sdpa"]["expected"])
      {"raw"     => {snapshots["online_scores.postscale"], canonical_reference},
       "rounded" => {rounded_snapshots["online_scores.postscale"], values(indexed["attention.canonical_sdpa"]["rounded_expected"])}}.each do |phase, pair|
        report.puts({label: label, operation: "native_online_frozen_qk_vs_canonical_sdpa", phase: phase, stats: compare(pair[0], pair[1])}.to_json)
      end
      canonical_rounded = values(indexed["attention.canonical_sdpa"]["rounded_expected"])
      production.each_index do |index|
        next if rounded_snapshots["attention.canonical_sdpa"][index].unsafe_as(UInt32) == canonical_rounded[index].unsafe_as(UInt32)
        points = {} of String => {native: Float32, reference: Float32, native_rounded: Float32, reference_rounded: Float32}
        {"attention.canonical_sdpa", "online_scores.postscale", "pv.mps_math_probabilities"}.each do |name|
          points[name] = {native: snapshots[name][index], reference: values(indexed[name]["expected"])[index],
                          native_rounded: rounded_snapshots[name][index], reference_rounded: values(indexed[name]["rounded_expected"])[index]}
        end
        row = index // c.head_dim
        report.puts({label: label, operation: "canonical_boundary_intervention", phase: "causal_control", index: index, points: points,
                     native_qk_row: snapshots["qk.postscale"][row * c.joint_tokens, c.joint_tokens],
                     reference_qk_row: values(indexed["qk.postscale"]["expected"])[row * c.joint_tokens, c.joint_tokens],
                     qk_reference_mode: "explicit_mps_postscale"}.to_json)
      end
      mps_failures = 0
      mps_operations = 0
      mps_rms_failures = 0
      mps_rms_operations = 0
      source_rope_failures = 0
      source_rope_operations = 0
      if mps_controls
        ops.each_with_index do |op, index|
          next unless {"linear", "qk", "softmax", "pv", "attention"}.includes?(op["kind"].as_s)
          mps_operations += 1
          op["inputs"].as_h.each do |name, record|
            mps_arena["#{index}.input.#{name}"].read(values(record).size).map(&.unsafe_as(UInt32)).should eq(values(record).map(&.unsafe_as(UInt32))),
              "MPS control must preserve frozen upstream operands"
          end
          actual = mps_arena["#{index}.output"].read(values(op["expected"]).size)
          rounded = mps_arena["#{index}.rounded"].read(actual.size)
          rounded.map(&.unsafe_as(UInt32)).should eq(actual.map { |value| bf16_bits(value) })
          {"mps_raw"     => {actual, values(op["expected"])},
           "mps_rounded" => {rounded, values(op["rounded_expected"])}}.each do |phase, pair|
            stats = compare(pair[0], pair[1])
            report.puts({label: label, operation: op["name"].as_s, kind: op["kind"].as_s,
                         reference_mode: op["reference_mode"]?.try(&.as_s), phase: phase, stats: stats}.to_json)
            if stats[:violations] > 0 || (phase == "mps_rounded" && stats[:bitwise_mismatches] > 0)
              mps_failures += 1
              failures << "#{label}/#{op["name"]}/#{phase}: strict=#{stats[:violations]} bits=#{stats[:bitwise_mismatches]}"
            end
          end
        end

        ops.each_with_index do |op, index|
          next unless rms_controls.not_nil!.has_plan?(index.to_i32)
          mps_rms_operations += 1
          op["inputs"].as_h.each do |name, record|
            mps_arena["#{index}.input.#{name}"].read(values(record).size).map(&.unsafe_as(UInt32)).should eq(values(record).map(&.unsafe_as(UInt32))),
              "MPS RMS control must preserve frozen upstream operands"
          end
          actual = mps_arena["#{index}.output"].read(values(op["expected"]).size)
          rounded = mps_arena["#{index}.rounded"].read(actual.size)
          rounded.map(&.unsafe_as(UInt32)).should eq(actual.map { |value| bf16_bits(value) })
          {"mps_rms_raw"     => {actual, values(op["expected"])},
           "mps_rms_rounded" => {rounded, values(op["rounded_expected"])}}.each do |phase, pair|
            stats = compare(pair[0], pair[1])
            stream_tokens = op["name"].as_s.starts_with?("img.") ? c.video_tokens : c.text_tokens
            report.puts({label: label, operation: op["name"].as_s, kind: op["kind"].as_s,
                         phase: phase, stats: stats, input_shape: [c.batches, stream_tokens, c.heads, c.head_dim],
                         mps_graph_shape: [c.batches, stream_tokens * c.heads, c.head_dim], flattened_rows: true}.to_json)
            if stats[:violations] > 0 || (phase == "mps_rms_rounded" && stats[:bitwise_mismatches] > 0)
              mps_rms_failures += 1
              failures << "#{label}/#{op["name"]}/#{phase}: strict=#{stats[:violations]} bits=#{stats[:bitwise_mismatches]}"
            end
          end
        end

        ops.each_with_index do |op, index|
          next unless rope_controls.not_nil!.has_plan?(index.to_i32)
          source_rope_operations += 1
          op["inputs"].as_h.each do |name, record|
            mps_arena["#{index}.input.#{name}"].read(values(record).size).map(&.unsafe_as(UInt32)).should eq(values(record).map(&.unsafe_as(UInt32))),
              "Source RoPE control must preserve frozen upstream operands"
          end
          actual = mps_arena["#{index}.output"].read(values(op["expected"]).size)
          rounded = mps_arena["#{index}.rounded"].read(actual.size)
          rounded.map(&.unsafe_as(UInt32)).should eq(actual.map { |value| bf16_bits(value) })
          {"source_rope_raw"     => {actual, values(op["expected"])},
           "source_rope_rounded" => {rounded, values(op["rounded_expected"])}}.each do |phase, pair|
            stats = compare(pair[0], pair[1])
            report.puts({label: label, operation: op["name"].as_s, kind: op["kind"].as_s,
                         phase: phase, stats: stats,
                         control_source: "separate_product_add_uncontracted"}.to_json)
            if stats[:violations] > 0 || (phase == "source_rope_rounded" && stats[:bitwise_mismatches] > 0)
              source_rope_failures += 1
              failures << "#{label}/#{op["name"]}/#{phase}: strict=#{stats[:violations]} bits=#{stats[:bitwise_mismatches]}"
            end
          end
        end
      end
      report.puts({label: label, phase: "summary", operations: ops.size, numerical_failures: failures.size,
                   mps_control_operations: mps_operations, mps_control_failures: mps_failures,
                   mps_rms_control_operations: mps_rms_operations, mps_rms_control_failures: mps_rms_failures,
                   source_rope_control_operations: source_rope_operations, source_rope_control_failures: source_rope_failures,
                   resident_weight_bytes: plan.destination_bytes, staging_weight_bytes: plan.staging_bytes,
                   scratch_and_input_bytes: arena.resident_bytes + mps_arena.resident_bytes, encode_metal_buffer_allocations: ML::MetalBuffer.video_spec_allocations - allocations,
                   gpu_seconds_diagnostic_command: seconds, terminal_waits: 1}.to_json)
      report.flush
      failures
    ensure
      # MPSGraph may already have continued a root before an encode error.
      # Drain that exceptional submission before releasing any plan/buffer.
      begin
        chain.try { |pending| pending.commit_and_wait unless pending.completed? }
      ensure
        # Preserve resources if completion is unknown. A terminal error with
        # completed? true still permits cleanup without leaking the plans.
        if chain.nil? || chain.not_nil!.completed?
          begin
            chain.try(&.release)
          ensure
            reductions.try(&.release)
            projections.try(&.release)
            rms_controls.try(&.release)
            rope_controls.try(&.release)
            prepared.try(&.release)
            mps_arena.release
            arena.release
            weights.release
          end
        end
      end
    end
  end
end

describe HyAttentionProbe do
  it "retains the fixed finite gate and detects seeded corruption" do
    HyAttentionProbe.compare([0.25_f32], [0.25_f32])[:violations].should eq(0)
    HyAttentionProbe.compare([0.26_f32], [0.25_f32])[:violations].should eq(1)
    HyAttentionProbe.compare([Float32::NAN], [0.25_f32])[:violations].should eq(1)
    HyAttentionProbe.compare([Float32::INFINITY], [Float32::INFINITY])[:violations].should eq(1)
    HyAttentionProbe.compare([0.0_f32], [] of Float32)[:violations].should eq(1)
    HyAttentionProbe.compare([0.7675781_f32], [0.7675783_f32])[:bf16_bin_mismatches].should eq(1)
  end

  it "rejects malformed tensor lengths and nonfinite values before allocation" do
    before = ML::MetalBuffer.video_spec_allocations
    expect_raises(ArgumentError) { HyAttentionProbe.values(JSON.parse(%({"shape":[2],"values":[0.25]}))) }
    expect_raises(ArgumentError) { HyAttentionProbe.values(JSON.parse(%({"shape":[0],"values":[]}))) }
    expect_raises(ArgumentError) { HyAttentionProbe.values(JSON.parse(%({"shape":[1],"values":[1e100]}))) }
    ML::MetalBuffer.video_spec_allocations.should eq(before)
  end

  it "rejects changed fixture bytes against an independent digest" do
    before = ML::MetalBuffer.video_spec_allocations
    file = File.tempfile("hy-attention-digest", ".json")
    begin
      file << "{}"
      file.flush
      expect_raises(ArgumentError, /SHA-256 mismatch/) { HyAttentionProbe.load(file.path, "0" * 64) }
      expect_raises(ArgumentError, /trusted SHA/) { HyAttentionProbe.load(file.path, "self-declared") }
      expect_raises(IO::Error, /fresh/) { HyAttentionProbe.fresh_report(file.path) { |report| report << "overwrite" } }
      File.read(file.path).should eq("{}")
      ML::MetalBuffer.video_spec_allocations.should eq(before)
    ensure
      path = file.path
      file.close
      File.delete(path) if File.exists?(path)
    end
  end

  it "rejects transposed attention layouts and invalid scales before allocation" do
    before = ML::MetalBuffer.video_spec_allocations
    c = ML::Video::HyNomasConfig.new(batches: 1, video_tokens: 3, text_tokens: 2,
      dim: 8, heads: 2, head_dim: 4, ffn_dim: 32)
    tensor = ->(shape : Array(Int32)) { {shape: shape, values: Array(Float64).new(shape.product, 0.25)} }
    op = JSON.parse({name: "qk.postscale", kind: "qk", scale: 0.5,
                     inputs: {q: tensor.call([1, 5, 2, 4]), k: tensor.call([1, 5, 2, 4])},
                     expected: tensor.call([1, 2, 5, 5]),
                     rounded_expected: tensor.call([1, 2, 5, 5])}.to_json)
    expect_raises(ArgumentError, /output layout/) { HyAttentionProbe.validate_operation!(op, c) }
    {0.0, -1.0, 1e100}.each do |bad|
      expect_raises(ArgumentError, /attention scale/) { HyAttentionProbe.scale(JSON.parse({scale: bad}.to_json), c) }
    end
    ML::MetalBuffer.video_spec_allocations.should eq(before)
  end

  it "rejects missing cases and a synthetic fixture presented as actual weights" do
    before = ML::MetalBuffer.video_spec_allocations
    labels = ["broadcast/f32", "broadcast/bf16_weights_roundtrip_f32_compute",
              "per_token/f32", "per_token/bf16_weights_roundtrip_f32_compute"]
    file = File.tempfile("hy-attention-source", ".json")
    begin
      [labels, [] of String].each do |admitted|
        content = {schema_version: 1, kind: "hy_nomas_same_input_attention_diagnostic",
                   upstream_revision: HyAttentionProbe::REVISION,
                   cases: admitted.map { |label| {label: label} }, weights_are_synthetic: true}.to_json
        File.write(file.path, content)
        expect_raises(ArgumentError) do
          HyAttentionProbe.load(file.path, Digest::SHA256.hexdigest(content), "actual")
        end
      end
      ML::MetalBuffer.video_spec_allocations.should eq(before)
    ensure
      path = file.path
      file.close
      File.delete(path) if File.exists?(path)
    end
  end

  it "compares isolated operations on identical upstream inputs without widening the gate" do
    unless path = ENV["HY_NOMAS_ATTENTION_FIXTURE"]?
      pending!("Set HY_NOMAS_ATTENTION_FIXTURE and trusted fixture/weight digests")
    end
    source_mode = ENV["HY_NOMAS_ATTENTION_SOURCE"]
    {"actual", "synthetic"}.includes?(source_mode).should be_true
    fixture_digest = ENV["HY_NOMAS_ATTENTION_FIXTURE_SHA256"]
    fixture = HyAttentionProbe.load(path, fixture_digest, source_mode)
    # The normal fixture digest protects the expected data; this mutation
    # control separately demonstrates that its lineage validator is active.
    broken = JSON.parse(fixture["cases"][0].to_json)
    origin = broken["operations"].as_a.find { |op| op["name"].as_s == "img.attn.v.linear" }.not_nil!["origin_expected"]["values"].as_a
    origin[0] = JSON::Any.new(origin[0].as_f + 1.0)
    before = ML::MetalBuffer.video_spec_allocations
    expect_raises(ArgumentError, /boundary lineage/) { HyAttentionProbe.validate_links!(broken) }
    ML::MetalBuffer.video_spec_allocations.should eq(before)
    weights_path = ENV["HY_NOMAS_ATTENTION_WEIGHTS"]
    weights_digest = ENV["HY_NOMAS_ATTENTION_WEIGHTS_SHA256"]
    HyAttentionProbe.trusted_digest!(weights_digest)
    HyAttentionProbe.digest(weights_path).should eq(weights_digest)
    fixture["source_checkpoint_sha256"].as_s.should eq(weights_digest)
    source_dtype = ML::GGUF::SafeTensorDType.parse(fixture["source_dtype"].as_s)
    {ML::GGUF::SafeTensorDType::F32, ML::GGUF::SafeTensorDType::BF16}.includes?(source_dtype).should be_true
    checkpoint = ML::Video::CheckpointFile.new(weights_path)
    mps_mode = ENV["HY_NOMAS_ATTENTION_MPS_CONTROLS"]? || "0"
    {"0", "1"}.includes?(mps_mode).should be_true
    begin
      failures = [] of String
      HyAttentionProbe.fresh_report(ENV["HY_NOMAS_ATTENTION_REPORT"]) do |report|
        report.puts({phase: "manifest", upstream_revision: HyAttentionProbe::REVISION,
                     fixture_sha256: fixture_digest, checkpoint_sha256: weights_digest,
                     source_fixture_sha256: fixture["source_fixture_sha256"].as_s,
                     source_mode: source_mode, gate_atol: 2e-5, gate_rtol: 2e-5,
                     rounded_gate: "bitwise", production_changes: false, mps_controls: mps_mode == "1"}.to_json)
        fixture["cases"].as_a.each { |data| failures.concat(HyAttentionProbe.run_case(data, checkpoint, source_dtype, report, mps_mode == "1")) }
        report.puts({phase: "complete", cases: fixture["cases"].as_a.size, failed_comparison_records: failures.size,
                     status: failures.empty? ? "parity_pass" : "numerical_gate_failed"}.to_json)
      end
      STDERR.puts "Hy-Nomas same-input attention diagnostic: failed strict comparison records=#{failures.size}"
      failures.should be_empty, failures.first(12).join('\n')
    ensure
      checkpoint.close
    end
  end
end
