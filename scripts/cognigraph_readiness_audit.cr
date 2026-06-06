#!/usr/bin/env crystal
# Static readiness audit for introducing ML::Metal::ComputeGraph/GraphEncoder
# into Qwen/Gemma runtime paths. This is intentionally conservative: it finds
# API and access-annotation blockers before graph scheduling is promoted.

ROOT = File.expand_path("..", __DIR__)
TARGETS = {
  "nomic_backend" => "src/ml/gguf/metal_backend.cr",
  "qwen35"        => "src/ml/gguf/qwen35_metal.cr",
  "gemma4"        => "src/ml/gguf/gemma4_metal.cr",
  "compute_graph" => "src/ml/metal/compute_graph.cr",
  "dispatch"      => "src/ml/metal/dispatch.cr",
}

WRITEISH = /(out|dst|cache|partial|tile|ids?|values?|logits|qkv|q_buf|k_buf|v_buf|gate_buf|up_buf|down_buf|normed_buf|ffn|ctx|projected|combined|state|z_buf|residual_buf)/

record Hit, path : String, line : Int32, text : String

def read_lines(rel : String) : Array(String)
  File.read_lines(File.join(ROOT, rel))
end

def hits(rel : String, pattern : Regex) : Array(Hit)
  read_lines(rel).each_with_index.compact_map do |line, idx|
    line =~ pattern ? Hit.new(rel, idx + 1, line.strip) : nil
  end.to_a
end

def count(rel : String, pattern : Regex) : Int32
  hits(rel, pattern).size
end

puts "CogniGraph readiness audit"
puts "root=#{ROOT}"
puts

puts "== Current ComputeGraph users =="
TARGETS.each do |name, rel|
  next unless File.exists?(File.join(ROOT, rel))
  cg = count(rel, /ComputeGraph/)
  ge = count(rel, /GraphEncoder/)
  next if cg == 0 && ge == 0
  puts "#{name}: ComputeGraph=#{cg} GraphEncoder=#{ge} file=#{rel}"
end
puts

puts "== Qwen/Gemma encoder shape =="
{"qwen35" => TARGETS["qwen35"], "gemma4" => TARGETS["gemma4"]}.each do |name, rel|
  typed = count(rel, /enc\s*:\s*ML::Metal::ComputeEncoder/)
  enc_new = count(rel, /ComputeEncoder\.new/)
  graph_ref = count(rel, /GraphEncoder|ComputeGraph/)
  puts "#{name}: typed_compute_encoder_params=#{typed} compute_encoder_new=#{enc_new} graph_refs=#{graph_ref}"
end
puts

puts "== GraphEncoder API delta vs ComputeEncoder =="
compute_api = hits(TARGETS["dispatch"], /^\s*def\s+([a-zA-Z0-9_!?]+)/).map { |h| h.text.match(/^def\s+([a-zA-Z0-9_!?]+)/).try(&.[1]) }.compact.to_set
# dispatch.cr has class indentation; use a broader scan too.
compute_api = hits(TARGETS["dispatch"], /def\s+([a-zA-Z0-9_!?]+)/).map { |h| h.text.match(/def\s+([a-zA-Z0-9_!?]+)/).try(&.[1]) }.compact.to_set
graph_api = hits(TARGETS["compute_graph"], /def\s+([a-zA-Z0-9_!?]+)/).map { |h| h.text.match(/def\s+([a-zA-Z0-9_!?]+)/).try(&.[1]) }.compact.to_set
needed = %w[set_pipeline set_buffer set_value set_bytes set_threadgroup_memory dispatch dispatch_1d dispatch_threadgroups dispatch_threadgroups_indirect memory_barrier end_encoding]
needed.each do |m|
  puts "#{m}: compute=#{compute_api.includes?(m)} graph=#{graph_api.includes?(m)}"
end
puts

puts "== Potential graph-unsafe set_buffer calls =="
puts "Heuristic: write-like buffer name, but no explicit BufferAccess/ReadWrite/Write token on the call line. Review before GraphEncoder promotion."
{"qwen35" => TARGETS["qwen35"], "gemma4" => TARGETS["gemma4"]}.each do |name, rel|
  candidates = [] of Hit
  read_lines(rel).each_with_index do |line, idx|
    next unless line.includes?(".set_buffer(")
    next if line.includes?("BufferAccess") || line.includes?("::Write") || line.includes?("::ReadWrite")
    next unless line =~ WRITEISH
    candidates << Hit.new(rel, idx + 1, line.strip)
  end
  puts "#{name}: candidates=#{candidates.size}"
  candidates.first(40).each do |h|
    puts "  #{h.path}:#{h.line}: #{h.text}"
  end
  puts "  ... truncated #{candidates.size - 40}" if candidates.size > 40
end
puts

puts "== Decision hints =="
puts "- Nomic backend is the current explicit ComputeGraph user."
puts "- Qwen/Gemma are manual resident command-buffer wave users until typed encoder APIs and access annotations are graph-ready."
puts "- Safe first graft should target a tiny stateless helper or a debug-only graph path with parity/spec gates, not mutable KV/SSM state corridors."
