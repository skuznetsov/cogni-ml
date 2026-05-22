require "./spec_helper"
require "../src/ml/gguf/qwen35_metal"
require "../src/ml/gguf/quant_matmul"
require "../src/ml/gguf/reader"

QWEN36_IQ4_NL_METAL = "#{ENV["HOME"]}/.cache/lm-studio/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-IQ4_NL.gguf"

private def cosine_iq4(a : Array(Float32), b : Array(Float32)) : Float64
  dot = 0.0
  na = 0.0
  nb = 0.0
  a.each_with_index do |av, i|
    bv = b[i]
    dot += av.to_f64 * bv.to_f64
    na += av.to_f64 * av.to_f64
    nb += bv.to_f64 * bv.to_f64
  end
  dot / (Math.sqrt(na) * Math.sqrt(nb))
end

private def max_abs_diff_iq4(a : Array(Float32), b : Array(Float32)) : Float32
  m = 0.0_f32
  a.each_with_index do |av, i|
    d = (av - b[i]).abs
    m = d if d > m
  end
  m
end

describe ML::GGUF::Qwen35Metal do
  pending!("Qwen3.6 IQ4_NL model not present") unless File.exists?(QWEN36_IQ4_NL_METAL)
  pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

  it "matmul_iq4_nl GEMV matches CPU reference on a real Qwen3.6 tensor subset" do
    g = ML::GGUF::GGUFFile.new(QWEN36_IQ4_NL_METAL)
    info = g.tensor("blk.0.attn_gate.weight").not_nil!
    info.type.iq4_nl?.should be_true
    in_dim = info.dims[0].to_i32
    out_dim = 8
    row_bytes = ((in_dim + 31) // 32) * 18
    w_raw = g.read_tensor_raw(info)[0, row_bytes * out_dim].dup
    g.close

    batch = 2
    x = Array(Float32).new(batch * in_dim) do |i|
      (((i * 17) % 31) - 15).to_f32 / 13.0_f32
    end
    zero_bias = Array(Float32).new(out_dim, 0.0_f32)

    gpu = ML::GGUF::Qwen35Metal.matmul_iq4_nl(x, w_raw, in_dim, out_dim, batch)
    cpu = ML::GGUF::QuantMatmul.matmul_add(
      x, batch, in_dim, w_raw, ML::GGUF::TensorType::IQ4_NL, out_dim, zero_bias
    )

    cos = cosine_iq4(gpu, cpu)
    diff = max_abs_diff_iq4(gpu, cpu)
    mag = cpu.map(&.abs).max
    puts "  [metal_iq4_nl_gemv] cos=#{cos.round(6)}, max_abs_diff=#{diff}, rel=#{diff / mag}"

    cos.should be >= 0.99999
    (diff / mag).should be <= 1.0e-5_f32
  end

  it "routes IQ4_NL through generic Metal matmul and matmul_many for GEMV only" do
    g = ML::GGUF::GGUFFile.new(QWEN36_IQ4_NL_METAL)
    if region = g.mmap_region
      base, size = region
      ML::GGUF::Qwen35Metal.register_mmap(base, size)
    end
    info = g.tensor("blk.0.attn_gate.weight").not_nil!
    in_dim = info.dims[0].to_i32
    out_dim = 4
    row_bytes = ((in_dim + 31) // 32) * 18
    raw = g.read_tensor_raw(info)[0, row_bytes * out_dim].dup
    g.close

    qw = ML::GGUF::QuantWeight.new(raw, ML::GGUF::TensorType::IQ4_NL, out_dim, in_dim)
    x = Array(Float32).new(in_dim) { |i| (((i * 19) % 37) - 18).to_f32 / 17.0_f32 }

    generic = ML::GGUF::Qwen35Metal.matmul(qw, x, 1).not_nil!
    direct = ML::GGUF::Qwen35Metal.matmul_iq4_nl(x, raw, in_dim, out_dim, 1)
    generic.should eq(direct)

    many = ML::GGUF::Qwen35Metal.matmul_many([qw, qw], x).not_nil!
    many.size.should eq(2)
    many[0].should eq(direct)
    many[1].should eq(direct)

    batch = ML::GGUF::Qwen35Metal::GEMM_BATCH_THRESHOLD + 1
    x_many = Array(Float32).new(batch * in_dim, 0.125_f32)
    ML::GGUF::Qwen35Metal.matmul(qw, x_many, batch).should be_nil
  end

  it "routes F32 weights through generic Metal GEMV" do
    in_dim = 64
    out_dim = 6
    batch = 2
    raw = Bytes.new(in_dim * out_dim * sizeof(Float32))
    w = raw.to_unsafe.as(Pointer(Float32))
    (in_dim * out_dim).times do |i|
      w[i] = (((i * 7) % 23) - 11).to_f32 / 19.0_f32
    end
    qw = ML::GGUF::QuantWeight.new(raw, ML::GGUF::TensorType::F32, out_dim, in_dim)
    x = Array(Float32).new(batch * in_dim) { |i| (((i * 13) % 29) - 14).to_f32 / 11.0_f32 }
    zero_bias = Array(Float32).new(out_dim, 0.0_f32)

    gpu = ML::GGUF::Qwen35Metal.matmul(qw, x, batch).not_nil!
    cpu = ML::GGUF::QuantMatmul.matmul_add(
      x, batch, in_dim, raw, ML::GGUF::TensorType::F32, out_dim, zero_bias
    )

    max_abs_diff_iq4(gpu, cpu).should be <= 2.0e-5_f32
  end
end
