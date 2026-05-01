require "./spec_helper"
require "../src/ml/gguf/qwen35_mtp"

private def bf16_bytes(values : Array(Float32)) : Bytes
  out = Bytes.new(values.size * 2)
  values.each_with_index do |v, i|
    h = (v.unsafe_as(UInt32) >> 16).to_u16
    out[i * 2] = (h & 0xff).to_u8
    out[i * 2 + 1] = (h >> 8).to_u8
  end
  out
end

private def write_mtp_fixture(path : String) : Nil
  specs = {
    "mtp.fc.weight"                                  => {2, 4},
    "mtp.layers.0.input_layernorm.weight"            => {2},
    "mtp.layers.0.self_attn.q_proj.weight"           => {4, 2},
    "mtp.layers.0.self_attn.q_norm.weight"           => {1},
    "mtp.layers.0.self_attn.k_proj.weight"           => {1, 2},
    "mtp.layers.0.self_attn.k_norm.weight"           => {1},
    "mtp.layers.0.self_attn.v_proj.weight"           => {1, 2},
    "mtp.layers.0.self_attn.o_proj.weight"           => {2, 3},
    "mtp.layers.0.post_attention_layernorm.weight"   => {2},
    "mtp.layers.0.mlp.gate_proj.weight"              => {5, 2},
    "mtp.layers.0.mlp.up_proj.weight"                => {5, 2},
    "mtp.layers.0.mlp.down_proj.weight"              => {2, 5},
    "mtp.norm.weight"                                => {2},
    "mtp.pre_fc_norm_embedding.weight"               => {2},
    "mtp.pre_fc_norm_hidden.weight"                  => {2},
  }

  payloads = {} of String => Bytes
  specs.each_with_index do |(name, shape), idx|
    n = shape.product
    vals = Array(Float32).new(n) { |i| (idx + 1).to_f32 + i.to_f32 / 16.0_f32 }
    payloads[name] = bf16_bytes(vals)
  end

  offset = 0
  header = JSON.build do |json|
    json.object do
      json.field "__metadata__" do
        json.object do
          json.field "format", "pt"
          json.field "source_repo", "fixture"
        end
      end
      specs.each do |name, shape|
        bytes = payloads[name]
        json.field name do
          json.object do
            json.field "dtype", "BF16"
            json.field "shape" do
              json.array { shape.each { |dim| json.number dim } }
            end
            json.field "data_offsets" do
              json.array do
                json.number offset
                json.number offset + bytes.size
              end
            end
          end
        end
        offset += bytes.size
      end
    end
  end

  File.open(path, "wb") do |io|
    io.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
    io.write(header.to_slice)
    specs.each_key { |name| io.write(payloads[name]) }
  end
end

describe ML::GGUF::SafetensorsFile do
  it "loads BF16 tensors from a safetensors sidecar" do
    path = File.join(Dir.tempdir, "qwen35_mtp_#{Random.rand(1_000_000)}.safetensors")
    write_mtp_fixture(path)
    begin
      st = ML::GGUF::SafetensorsFile.new(path)
      info = st.tensor("mtp.fc.weight").not_nil!
      info.dtype.should eq(ML::GGUF::SafeTensorDType::BF16)
      info.shape.should eq([2_i64, 4_i64])
      st.tensor_bytes(info).size.should eq(16)
      vals = st.read_tensor_f32(info)
      vals[0].should eq(1.0_f32)
      vals[1].should be_close(1.0625_f32, 1e-6_f32)
    ensure
      File.delete?(path)
    end
  end
end

describe ML::GGUF::Qwen35MTPWeights do
  it "loads the required Qwen MTP tensor set" do
    path = File.join(Dir.tempdir, "qwen35_mtp_#{Random.rand(1_000_000)}.safetensors")
    write_mtp_fixture(path)
    begin
      mtp = ML::GGUF::Qwen35MTPWeights.from_safetensors(path)
      mtp.fc.out_dim.should eq(2)
      mtp.fc.in_dim.should eq(4)
      mtp.q_proj.out_dim.should eq(4)
      mtp.o_proj.in_dim.should eq(3)
      mtp.ffn_gate.out_dim.should eq(5)
      mtp.input_layernorm.size.should eq(2)
      mtp.pre_fc_norm_hidden.size.should eq(2)
      mtp.total_raw_bytes.should be > 0
    ensure
      File.delete?(path)
    end
  end
end

describe ML::GGUF::Qwen35MTP do
  it "multiplies BF16 dense row-major matrices" do
    raw = bf16_bytes([
      1.0_f32, 2.0_f32, 3.0_f32,
      -1.0_f32, 0.5_f32, 4.0_f32,
    ])
    w = ML::GGUF::DenseBF16Weight.new("fixture", raw, 2, 3)
    out = ML::GGUF::Qwen35MTP.matvec_bf16(w, [2.0_f32, -1.0_f32, 0.25_f32])
    out[0].should be_close(0.75_f32, 1e-6_f32)
    out[1].should be_close(-1.5_f32, 1e-6_f32)
  end

  it "applies Qwen3.5 sidecar RMSNorm with one-centered weights" do
    out = ML::GGUF::Qwen35MTP.rms_norm_sidecar([3.0_f32, 4.0_f32], [0.0_f32, 1.0_f32], 0.0_f32)
    out[0].should be_close(0.84852815_f32, 1e-6_f32)
    out[1].should be_close(2.2627418_f32, 1e-6_f32)
  end

  it "extracts stable top-k logits" do
    top = ML::GGUF::Qwen35MTP.top_k([0.5_f32, 3.0_f32, 3.0_f32, -1.0_f32], 3)
    top.should eq([{1, 3.0_f32}, {2, 3.0_f32}, {0, 0.5_f32}])
  end
end
