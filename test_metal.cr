require "./src/ml"
require "./src/ml/gguf/nomic_bert"
require "./src/ml/gguf/metal_backend"
require "http/client"
require "json"
MODEL = "/Users/sergey/.cache/lm-studio/models/nomic-ai/nomic-embed-text-v2-moe-GGUF/nomic-embed-text-v2-moe.Q5_K_M.gguf"
ML::Metal::Device.init!
cpu = ML::GGUF::NomicBertMoE(ML::GGUF::F32Backend).from_gguf(MODEL)
gpu = ML::GGUF::NomicBertMoE.from_gguf(MODEL, ML::GGUF::MetalBackend.new)
["Hello world", "Crystal programming language", "def method_name(arg)", "PostgreSQL extension for vector search"].each do |text|
  gpu.embed(text)
  t0 = Time.instant; ec = cpu.embed(text); cm = (Time.instant - t0).total_milliseconds
  best = Float64::MAX; eg = [] of Float32
  10.times { t0 = Time.instant; eg = gpu.embed(text); ms = (Time.instant - t0).total_milliseconds; best = ms if ms < best }
  dot = 0.0_f64; ec.size.times { |i| dot += ec[i].to_f64 * eg[i].to_f64 }
  # LM Studio ref
  client = HTTP::Client.new("127.0.0.1", 1234)
  resp = client.post("/v1/embeddings", headers: HTTP::Headers{"Content-Type" => "application/json"},
    body: {"model" => "text-embedding-nomic-embed-text-v2-moe", "input" => text}.to_json)
  ref = JSON.parse(resp.body)["data"][0]["embedding"].as_a.map(&.as_f.to_f32)
  client.close; norm = Math.sqrt(ref.sum { |v| v * v }); ref.map! { |v| v / norm }
  cos_ref = 0.0_f64; eg.size.times { |i| cos_ref += eg[i].to_f64 * ref[i].to_f64 }
  STDERR.puts "#{text[0, 30].ljust(32)} GPU=#{best.round(1)}ms cos(G,C)=#{dot.round(3)} cos(G,Ref)=#{cos_ref.round(3)}"
end
