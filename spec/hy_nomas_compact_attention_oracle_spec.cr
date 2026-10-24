require "spec"
require "json"
require "digest/sha256"
require "../src/ml/video/hy_nomas_attention_compact"
require "../src/ml/video/metal_mps_command_chain"
require "./support/video_allocation_counter"

private COMPACT_ORACLE_SCHEMA              = "hy_nomas_compact_attention_reference_v1"
private COMPACT_ORACLE_TORCH_VERSION       = "2.8.0"
private COMPACT_ORACLE_TORCH_GIT           = "a1cb3cc05d46d198467bebbb6e8fba50a325d4e7"
private COMPACT_ORACLE_TORCH_SOURCE_SHA256 = "167ee89bddb489c154b5d5ca72cf042745443f557d624fc48175f2d29f45a351"
private COMPACT_ORACLE_FORMULA             = "avalanche32(native_flat_index + salt*0x9e3779b9); ((hash & 65535) - 32768) / 65536"
private COMPACT_ORACLE_HEADS               =   16
private COMPACT_ORACLE_HEAD_DIM            =  128
private COMPACT_ORACLE_J_TOKENS            = 6560
private COMPACT_ORACLE_J_CHUNK             =  128
private COMPACT_ORACLE_J_REFERENCE_HEADS   =   16

private COMPACT_ORACLE_INPUT_SOURCE = <<-METAL
  #include <metal_stdlib>
  using namespace metal;
  inline float hy_compact_oracle_value(uint index, uint salt) {
    uint mixed = index + salt * 0x9e3779b9u;
    mixed ^= mixed >> 16;
    mixed *= 0x7feb352du;
    mixed ^= mixed >> 15;
    mixed *= 0x846ca68bu;
    mixed ^= mixed >> 16;
    const int quantized = int(mixed & 65535u) - 32768;
    return float(quantized) / 65536.0f;
  }
  inline float hy_compact_oracle_bf16_rne(float value) {
    uint bits = as_type<uint>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return as_type<float>(bits & 0xffff0000u);
  }
  kernel void hy_compact_oracle_fill(
      device float *output [[buffer(0)]], constant uint &count [[buffer(1)]],
      constant uint &salt [[buffer(2)]], constant uint &bf16_boundary [[buffer(3)]],
      uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float value = hy_compact_oracle_value(i, salt);
    output[i] = bf16_boundary == 0u ? value : hy_compact_oracle_bf16_rne(value);
  }
METAL

private def compact_oracle_sha256(data : Bytes) : String
  Digest::SHA256.hexdigest(data)
end

private def compact_oracle_file_sha256(path : String) : String
  digest = Digest::SHA256.new
  buffer = Bytes.new(1 << 20)
  File.open(path, "rb") do |file|
    while (count = file.read(buffer)) > 0
      digest.update(buffer[0, count])
    end
  end
  digest.final.hexstring
end

private def compact_oracle_required_env(name : String) : String
  value = ENV[name]?
  raise ArgumentError.new("Set #{name}") unless value && !value.empty?
  value
end

private def compact_oracle_digest_env(name : String) : String
  value = compact_oracle_required_env(name)
  raise ArgumentError.new("#{name} must be a 64-character lowercase SHA-256") unless value.matches?(/\A[0-9a-f]{64}\z/)
  value
end

private def compact_oracle_int_env(name : String, minimum : Int32, maximum : Int32) : Int32
  raw = compact_oracle_required_env(name)
  value = raw.to_i?
  raise ArgumentError.new("#{name} must be an integer") unless value
  unless value >= minimum && value <= maximum
    raise ArgumentError.new("#{name} must be within #{minimum}..#{maximum}")
  end
  value
end

private def compact_oracle_load_fixture(case_name : String) : {JSON::Any, Bytes, String, String}
  fixture_path = compact_oracle_required_env("HY_NOMAS_COMPACT_ATTENTION_FIXTURE")
  fixture_pin = compact_oracle_digest_env("HY_NOMAS_COMPACT_ATTENTION_FIXTURE_SHA256")
  fixture_text = File.read(fixture_path)
  fixture_bytes = fixture_text.to_slice
  fixture_digest = compact_oracle_sha256(fixture_bytes)
  raise ArgumentError.new("Compact attention fixture SHA-256 mismatch") unless fixture_digest == fixture_pin
  fixture = JSON.parse(String.new(fixture_bytes))
  unless fixture["schema"].as_s == COMPACT_ORACLE_SCHEMA && fixture["case"].as_s == case_name
    raise ArgumentError.new("Unsupported compact attention fixture schema or case")
  end

  producer_pin = compact_oracle_digest_env("HY_NOMAS_COMPACT_ATTENTION_PRODUCER_SHA256")
  producer_path = File.join(__DIR__, "support/hy_nomas_compact_attention_reference.py")
  producer_digest = compact_oracle_file_sha256(producer_path)
  provenance = fixture["provenance"]
  unless provenance["producer_sha256"].as_s == producer_pin && producer_pin == producer_digest
    raise ArgumentError.new("Compact attention producer source hash mismatch")
  end
  unless provenance["torch_version"].as_s == COMPACT_ORACLE_TORCH_VERSION &&
         provenance["torch_git_commit"].as_s == COMPACT_ORACLE_TORCH_GIT &&
         provenance["torch_tag"].as_s == "v2.8.0" &&
         provenance["producer"].as_s == "hy_nomas_compact_attention_reference.py" &&
         provenance["torch_attention_source"].as_s == "aten/src/ATen/native/mps/operations/Attention.mm" &&
         provenance["torch_attention_source_sha256"].as_s == COMPACT_ORACLE_TORCH_SOURCE_SHA256 &&
         provenance["reference_device"].as_s == "mps" &&
         provenance["mps_fallback_disabled"].as_bool
    raise ArgumentError.new("Compact attention PyTorch MPS provenance mismatch")
  end
  macos_major = provenance["macos_version"].as_s.split('.').first.to_i?
  unless macos_major && macos_major >= 15
    raise ArgumentError.new("Compact attention fixture was not produced on macOS 15 or newer")
  end

  input = fixture["input"]
  input_bf16_boundary = input["bf16_boundary_roundtrip"].as_bool
  unless input["dtype"].as_s == "float32" && input["layout"].as_s == "B,T,H,D" &&
         input["formula"].as_s == COMPACT_ORACLE_FORMULA &&
         input["checkpoint"].as_s == "none; direct deterministic synthetic Q/K/V" &&
         input["salts"]["q"].as_i == 1 && input["salts"]["k"].as_i == 2 &&
         input["salts"]["v"].as_i == 3
    raise ArgumentError.new("Compact attention synthetic input contract mismatch")
  end
  reference = fixture["reference"]
  unless reference["operator"].as_s == "torch.nn.functional.scaled_dot_product_attention" &&
         reference["device"].as_s == "mps" && reference["dtype"].as_s == "float32" &&
         reference["attention_mask"].as_s == "none" && reference["dropout_p"].as_f == 0.0 &&
         !reference["is_causal"].as_bool && reference["mps_fallback_disabled"].as_bool &&
         reference["scale"].as_s == "default 1/sqrt(head_dim)" &&
         reference["comparison_gate"].as_s == "2e-5 + 2e-5*abs(reference); BF16 RNE bitwise"
    raise ArgumentError.new("Compact attention reference operator contract mismatch")
  end

  geometry = fixture["geometry"]
  expected_tokens = case case_name
                    when "t257-full"     then 257
                    when "j6560-sampled" then COMPACT_ORACLE_J_TOKENS
                    else                      raise ArgumentError.new("Unknown compact attention oracle case")
                    end
  expected_scope = case case_name
                   when "t257-full" then "full T=257 query shape"
                   else                  "selected mathematical query rows only; no full-T6560 parity claim"
                   end
  unless reference["query_slice_scope"].as_s == expected_scope
    raise ArgumentError.new("Compact attention reference query scope mismatch")
  end
  unless geometry["batches"].as_i == 1 && geometry["tokens"].as_i == expected_tokens &&
         geometry["heads"].as_i == COMPACT_ORACLE_HEADS &&
         geometry["head_dim"].as_i == COMPACT_ORACLE_HEAD_DIM &&
         geometry["native_input_layout"].as_s == "B,T,H,D" &&
         geometry["pytorch_qkv_layout"].as_s == "B,H,T,D" &&
         geometry["reference_output_layout"].as_s == "B,T,H,D"
    raise ArgumentError.new("Compact attention reference geometry mismatch")
  end

  sidecar_path = compact_oracle_required_env("HY_NOMAS_COMPACT_ATTENTION_OUTPUTS")
  sidecar_pin = compact_oracle_digest_env("HY_NOMAS_COMPACT_ATTENTION_OUTPUTS_SHA256")
  outputs = fixture["outputs"]
  unless outputs["filename"].as_s == File.basename(sidecar_path) &&
         outputs["byte_order"].as_s == "little" && outputs["dtype"].as_s == "float32" &&
         outputs["sha256"].as_s == sidecar_pin
    raise ArgumentError.new("Compact attention output sidecar provenance mismatch")
  end
  expected_sidecar_bytes = case case_name
                           when "t257-full" then 257_i64 * COMPACT_ORACLE_HEADS * COMPACT_ORACLE_HEAD_DIM * 4
                           else                  3_i64 * COMPACT_ORACLE_J_CHUNK * COMPACT_ORACLE_HEADS * COMPACT_ORACLE_HEAD_DIM * 4
                           end
  sidecar_size = File.size(sidecar_path)
  unless sidecar_size == expected_sidecar_bytes && sidecar_size == outputs["bytes"].as_i64
    raise ArgumentError.new("Compact attention output sidecar size mismatch")
  end
  sidecar = Bytes.new(sidecar_size.to_i)
  File.open(sidecar_path, "rb") { |file| file.read_fully(sidecar) }
  raise ArgumentError.new("Compact attention output sidecar SHA-256 mismatch") unless compact_oracle_sha256(sidecar) == sidecar_pin

  records = fixture["records"].as_a
  expected_records = [] of Tuple(String, Int32, Int32, Int32, Int32, Int32?)
  if case_name == "t257-full"
    reference_shape = fixture["reference_shape"]
    unless reference_shape["query_rows"].as_i == 257 &&
           reference_shape["heads_per_group"].as_i == COMPACT_ORACLE_HEADS
      raise ArgumentError.new("T257 full reference shape mismatch")
    end
    expected_records << {"full", 0, 257, 0, COMPACT_ORACLE_HEADS, nil}
  else
    reference_shape = fixture["reference_shape"]
    unless reference_shape["query_rows"].as_i == COMPACT_ORACLE_J_CHUNK &&
           reference_shape["heads_per_group"].as_i == COMPACT_ORACLE_J_REFERENCE_HEADS
      raise ArgumentError.new("J6560 PyTorch reference shape mismatch")
    end
    # Match the native overlapping tail schedule: first, middle, and final
    # full-width chunks, including the final chunk beginning at T-C.
    [{0, 0, 128}, {25, 3200, 128}, {51, 6432, 128}].each do |chunk_index, query_start, rows|
      label = "h0_q#{query_start}_r#{rows}"
      expected_records << {label, query_start, rows, 0, COMPACT_ORACLE_J_REFERENCE_HEADS, chunk_index}
    end
  end
  unless records.size == expected_records.size
    raise ArgumentError.new("Compact attention fixture record count mismatch")
  end
  expected_offset = 0_i64
  records.each_with_index do |record, index|
    label, query_start, rows, head_start, group_heads, chunk_index = expected_records[index]
    shape = record["shape"].as_a.map(&.as_i)
    byte_count = rows.to_i64 * group_heads * COMPACT_ORACLE_HEAD_DIM * 4
    offset = record["offset_bytes"].as_i64
    unless record["label"].as_s == label && record["query_start"].as_i == query_start &&
           record["query_rows"].as_i == rows && record["head_start"].as_i == head_start &&
           record["group_heads"].as_i == group_heads && shape == [1, rows, group_heads, COMPACT_ORACLE_HEAD_DIM] &&
           record["layout"].as_s == "B,T,H,D" && offset == expected_offset &&
           record["bytes"].as_i64 == byte_count
      raise ArgumentError.new("Compact attention fixture record mismatch: #{label}")
    end
    if chunk_index
      raise ArgumentError.new("Compact attention chunk index mismatch: #{label}") unless record["chunk_index"].as_i == chunk_index
    end
    stop = offset + byte_count
    if offset < 0 || stop > sidecar.size
      raise ArgumentError.new("Compact attention fixture record is outside its sidecar")
    end
    slice = sidecar[offset.to_i, byte_count.to_i]
    unless compact_oracle_sha256(slice) == record["sha256"].as_s
      raise ArgumentError.new("Compact attention record SHA-256 mismatch: #{label}")
    end
    expected_offset = stop
  end
  unless expected_offset == sidecar.size
    raise ArgumentError.new("Compact attention sidecar has unreferenced bytes")
  end
  {fixture, sidecar, fixture_digest, sidecar_pin}
end

private def compact_oracle_int32(bytes : Bytes, offset : Int32) : Float32
  bits = bytes[offset].to_u32 |
         (bytes[offset + 1].to_u32 << 8) |
         (bytes[offset + 2].to_u32 << 16) |
         (bytes[offset + 3].to_u32 << 24)
  bits.unsafe_as(Float32)
end

private def compact_oracle_bf16_bits(value : Float32) : UInt32
  bits = value.unsafe_as(UInt32)
  ((bits.to_u64 + 0x7fff_u64 + ((bits >> 16) & 1)) & 0xffff0000_u64).to_u32
end

private def compact_oracle_fresh_report(path : String) : IO::FileDescriptor
  fd = LibC.open(path, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
  raise IO::Error.new("Compact attention report path must be fresh and writable") if fd < 0
  IO::FileDescriptor.new(fd)
end

private def compact_oracle_emit(report : IO::FileDescriptor?, value : T) : Nil forall T
  line = value.to_json
  STDERR.puts(line)
  if report
    report.puts(line)
    report.flush
  end
end

describe "Hy-Nomas compact attention independent PyTorch-MPS oracle" do
  it "compares the selected compact plan against pinned synthetic references" do
    fixture_env = ENV["HY_NOMAS_COMPACT_ATTENTION_FIXTURE"]?
    pending!("Set the exact synthetic fixture and its trusted SHA-256 pins") unless fixture_env

    case_name = compact_oracle_required_env("HY_NOMAS_COMPACT_ATTENTION_CASE")
    unless {"t257-full", "j6560-sampled"}.includes?(case_name)
      raise ArgumentError.new("HY_NOMAS_COMPACT_ATTENTION_CASE must be t257-full or j6560-sampled")
    end
    tokens = case case_name
             when "t257-full" then 257
             else                  COMPACT_ORACLE_J_TOKENS
             end
    default_chunk = case case_name
                    when "t257-full" then 257
                    else                  COMPACT_ORACLE_J_CHUNK
                    end
    query_chunk = if ENV["HY_NOMAS_COMPACT_ATTENTION_QUERY_CHUNK"]?
                    compact_oracle_int_env("HY_NOMAS_COMPACT_ATTENTION_QUERY_CHUNK", 9, tokens)
                  else
                    default_chunk
                  end
    heads_per_group = if ENV["HY_NOMAS_COMPACT_ATTENTION_HEADS_PER_GROUP"]?
                        compact_oracle_int_env("HY_NOMAS_COMPACT_ATTENTION_HEADS_PER_GROUP", 1, COMPACT_ORACLE_HEADS)
                      else
                        COMPACT_ORACLE_HEADS
                      end
    if case_name == "j6560-sampled" && query_chunk != COMPACT_ORACLE_J_CHUNK
      raise ArgumentError.new("J6560 sampled references require C=128")
    end
    if case_name == "j6560-sampled" && !{4, COMPACT_ORACLE_HEADS}.includes?(heads_per_group)
      raise ArgumentError.new("J6560 sampled reference diagnostics permit G=16 or explicit G=4")
    end

    # All manifest, producer, PyTorch source, sidecar, and C/G checks complete
    # before Arena.new reaches Metal device initialization or allocates buffers.
    fixture, expected_bytes, fixture_digest, outputs_digest = compact_oracle_load_fixture(case_name)
    input_bf16_boundary = fixture["input"]["bf16_boundary_roundtrip"].as_bool
    batches = 1
    count = ML::Video::HyNomasDualStreamBlock.checked_count(
      batches, tokens, COMPACT_ORACLE_HEADS, COMPACT_ORACLE_HEAD_DIM
    )
    compact_source = File.join(__DIR__, "../src/ml/video/hy_nomas_attention_compact_bridge.mm")
    compact_source_sha = compact_oracle_file_sha256(compact_source)
    report = nil.as(IO::FileDescriptor?)
    arena = nil.as(ML::Video::Arena?)
    plan = nil.as(ML::Video::HyNomasCompactAttentionMPSPlan?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    terminal_wait_attempted = false
    begin
      if report_path = ENV["HY_NOMAS_COMPACT_ATTENTION_REPORT"]?
        report = compact_oracle_fresh_report(report_path)
      end
      arena = ML::Video::Arena.new
      {"q", "k", "v"}.each { |name| arena.not_nil!.alloc(name, count) }
      arena.not_nil!.alloc("output", count)
      plan = ML::Video::HyNomasCompactAttentionMPSPlan.new(
        batches, tokens, COMPACT_ORACLE_HEADS, COMPACT_ORACLE_HEAD_DIM,
        arena.not_nil!["q"], arena.not_nil!["k"], arena.not_nil!["v"],
        arena.not_nil!["output"], query_chunk, heads_per_group
      )
      fill = ML::Metal::PipelineCache.get("hy_compact_oracle_fill") do
        ML::Metal::ComputePipeline.new("hy_compact_oracle_fill", COMPACT_ORACLE_INPUT_SOURCE)
      end
      chain = ML::Metal::MPSCommandChain.new
      bf16_boundary_flag = input_bf16_boundary ? 1_u32 : 0_u32
      { {"q", 1_u32}, {"k", 2_u32}, {"v", 3_u32} }.each do |name, salt|
        chain.not_nil!.encoder.set_pipeline(fill)
          .set_buffer(arena.not_nil![name], 0, ML::Metal::BufferAccess::Write)
          .set_value(count.to_u32, 1).set_value(salt, 2)
          .set_value(bf16_boundary_flag, 3).dispatch_1d(count)
      end
      allocations_before = ML::MetalBuffer.video_spec_allocations
      plan.not_nil!.encode(chain.not_nil!)
      encode_allocations = ML::MetalBuffer.video_spec_allocations - allocations_before
      encode_allocations.should eq(0)
      terminal_wait_attempted = true
      chain.not_nil!.commit_and_wait
      raise "Compact attention oracle consumer used more than one terminal wait" unless chain.not_nil!.wait_count == 1

      actual = arena.not_nil!["output"].contents.as(Float32*)
      finite_violations = 0_i64
      bf16_mismatches = 0_i64
      raw_mismatches = 0_i64
      compared = 0_i64
      max_abs = 0.0_f64
      fixture["records"].as_a.each do |record|
        query_start = record["query_start"].as_i
        rows = record["query_rows"].as_i
        head_start = record["head_start"].as_i
        group_heads = record["group_heads"].as_i
        offset = record["offset_bytes"].as_i
        label = record["label"].as_s
        record_finite = 0_i64
        record_bf16 = 0_i64
        record_raw = 0_i64
        record_max_abs = 0.0_f64
        first_finite_index = nil.as(Int64?)
        first_finite_actual = nil.as(Float32?)
        first_finite_reference = nil.as(Float32?)
        first_bf16_index = nil.as(Int64?)
        first_bf16_actual = nil.as(Float32?)
        first_bf16_reference = nil.as(Float32?)
        rows.times do |row|
          group_heads.times do |group_head|
            COMPACT_ORACLE_HEAD_DIM.times do |dimension|
              expected_index = ((row * group_heads + group_head) * COMPACT_ORACLE_HEAD_DIM + dimension) * 4
              expected = compact_oracle_int32(expected_bytes, offset + expected_index)
              actual_index = ((query_start + row) * COMPACT_ORACLE_HEADS + head_start + group_head) * COMPACT_ORACLE_HEAD_DIM + dimension
              value = actual[actual_index]
              delta = (value - expected).abs
              unless value.finite? && expected.finite? &&
                     delta.to_f64 <= 2e-5_f64 + 2e-5_f64 * expected.abs.to_f64
                finite_violations += 1
                record_finite += 1
                if first_finite_index.nil?
                  first_finite_index = actual_index.to_i64
                  first_finite_actual = value
                  first_finite_reference = expected
                end
              end
              unless compact_oracle_bf16_bits(value) == compact_oracle_bf16_bits(expected)
                bf16_mismatches += 1
                record_bf16 += 1
                if first_bf16_index.nil?
                  first_bf16_index = actual_index.to_i64
                  first_bf16_actual = value
                  first_bf16_reference = expected
                end
              end
              if value.unsafe_as(UInt32) != expected.unsafe_as(UInt32)
                raw_mismatches += 1
                record_raw += 1
              end
              if delta.finite?
                record_max_abs = Math.max(record_max_abs, delta.to_f64)
              end
              compared += 1
            end
          end
        end
        max_abs = Math.max(max_abs, record_max_abs)
        compact_oracle_emit(report, {
          phase: "record", label: label, query_start: query_start,
          query_rows: rows, head_start: head_start, group_heads: group_heads,
          compared: rows.to_i64 * group_heads * COMPACT_ORACLE_HEAD_DIM,
          finite_violations: record_finite, bf16_mismatches: record_bf16,
          raw_mismatches: record_raw, max_abs: record_max_abs,
          first_finite_index: first_finite_index, first_finite_actual: first_finite_actual,
          first_finite_reference: first_finite_reference,
          first_bf16_index: first_bf16_index, first_bf16_actual: first_bf16_actual,
          first_bf16_reference: first_bf16_reference,
        })
      end
      status = finite_violations == 0 && bf16_mismatches == 0 ? "parity_pass" : "numerical_gate_failed"
      compact_oracle_emit(report, {
        phase: "complete", status: status, case: case_name,
        batches: batches, tokens: tokens, heads: COMPACT_ORACLE_HEADS,
        head_dim: COMPACT_ORACLE_HEAD_DIM, query_chunk: query_chunk,
        heads_per_group: heads_per_group, compared: compared,
        finite_violations: finite_violations, bf16_mismatches: bf16_mismatches,
        raw_mismatches: raw_mismatches, max_abs: max_abs,
        input_bf16_boundary_roundtrip: input_bf16_boundary,
        fixture_sha256: fixture_digest, outputs_sha256: outputs_digest,
        producer_sha256: fixture["provenance"]["producer_sha256"].as_s,
        torch_version: fixture["provenance"]["torch_version"].as_s,
        torch_attention_source_sha256: COMPACT_ORACLE_TORCH_SOURCE_SHA256,
        compact_bridge_sha256: compact_source_sha, terminal_waits: chain.not_nil!.wait_count,
        tracked_metal_buffer_allocations_during_encode: encode_allocations,
        reference_query_rows: fixture["reference_shape"]["query_rows"].as_i,
        reference_heads_per_group: fixture["reference_shape"]["heads_per_group"].as_i,
        scope: case_name == "j6560-sampled" ? "selected query rows only; no full-T6560 bitwise claim" : "full T257 output",
      })
      finite_violations.should eq(0)
      bf16_mismatches.should eq(0)
    ensure
      begin
        if active = chain
          active.commit_and_wait unless active.completed? || terminal_wait_attempted
        end
      ensure
        if chain.nil? || chain.not_nil!.completed?
          begin
            plan.try(&.release)
          ensure
            begin
              chain.try(&.release)
            ensure
              begin
                arena.try(&.release)
              ensure
                report.try(&.close)
              end
            end
          end
        else
          # Keep GPU-owned buffers and plan retained when completion is unknown;
          # close only the diagnostics file, which has no GPU lifetime.
          report.try(&.close)
        end
      end
    end
  end
end
