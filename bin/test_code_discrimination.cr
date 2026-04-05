#!/usr/bin/env crystal
# Phase 0.5: Can nomic-embed distinguish semantically different code?
#
# Test groups:
#   A. Similar syntax, different semantics (MUST discriminate → low cosine)
#   B. Different syntax, same semantics (SHOULD group → high cosine)
#   C. Completely different code (baseline → low cosine)
#   D. Tool call discrimination (grep vs read_file vs edit_file)
#
# Gate:
#   PASS: A pairs cosine < 0.85, B pairs cosine > 0.85, D pairs cosine < 0.80
#   SOFT: A < 0.90, D < 0.85
#   FAIL: A > 0.90 (can't distinguish read vs write)

require "../src/ml/gguf/metal_backend"
require "../src/ml/gguf/nomic_bert"

MODEL = ENV["EMBED_MODEL"]? || (Path.home / ".cache/lm-studio/models/nomic-ai/nomic-embed-text-v2-moe-GGUF/nomic-embed-text-v2-moe.Q5_K_M.gguf").to_s

# ── Test pairs ──

# Group A: Similar syntax, DIFFERENT semantics
GROUP_A = {
  {"File.read(path)",
   "File.write(path, content)",
   "read vs write file"},

  {"array.push(item)",
   "array.delete(item)",
   "push vs delete"},

  {"user = User.find(id)",
   "user = User.create(params)",
   "find vs create (DB)"},

  {"socket.send(data)",
   "socket.receive(buffer)",
   "send vs receive"},

  {"mutex.lock\nshared_var += 1\nmutex.unlock",
   "mutex.lock\nshared_var -= 1\nmutex.unlock",
   "increment vs decrement (subtle)"},

  {"if file.exists?\n  process(file)\nend",
   "if file.exists?\n  File.delete(file)\nend",
   "process vs delete file"},

  {"def initialize(@name : String)\nend",
   "def finalize\n  @name = nil\nend",
   "constructor vs destructor"},

  {"response = HTTP::Client.get(url)",
   "response = HTTP::Client.post(url, body: data)",
   "GET vs POST request"},
}

# Group B: Different syntax, SAME semantics
GROUP_B = {
  {"items.select { |i| i > 0 }",
   "items.reject { |i| i <= 0 }",
   "select-positive vs reject-non-positive"},

  {"result = arr.map { |x| x * 2 }",
   "result = [] of Int32\narr.each { |x| result << x * 2 }",
   "map vs manual loop"},

  {"return nil if input.nil?",
   "input.try { |v| return v } ; return nil",
   "nil guard styles"},

  {"File.open(path) { |f| f.read_string }",
   "File.read(path)",
   "file read two styles"},
}

# Group C: Completely different (baseline)
GROUP_C = {
  {"SELECT * FROM users WHERE id = ?",
   "def fibonacci(n)\n  return n if n <= 1\n  fibonacci(n-1) + fibonacci(n-2)\nend",
   "SQL vs recursion"},

  {"docker compose up -d",
   "class Matrix\n  def transpose\n    # ...\n  end\nend",
   "docker command vs matrix class"},
}

# Group D: Tool call discrimination (critical for agent)
GROUP_D = {
  {"grep: search for pattern 'render_content' in Crystal files",
   "read_file: open src/tui/app.cr and show contents",
   "grep vs read_file"},

  {"edit_file: replace 'old_method' with 'new_method' in src/app.cr",
   "read_file: open src/app.cr to inspect the code",
   "edit vs read"},

  {"shell: run 'crystal build src/main.cr'",
   "grep: search for 'main' in source files",
   "shell vs grep"},

  {"list_directory: show files in src/tui/",
   "read_file: open src/tui/app.cr",
   "list_dir vs read_file"},

  {"We need to search for the function definition",
   "The function is defined on line 42, it takes two arguments",
   "intent-to-search vs fact-about-function"},
}

def cosine(a : Array(Float32), b : Array(Float32)) : Float64
  dot = 0.0_f64; na = 0.0_f64; nb = 0.0_f64
  a.size.times do |i|
    av = a[i].to_f64; bv = b[i].to_f64
    dot += av * bv; na += av * av; nb += bv * bv
  end
  return 0.0 if na <= 1e-12 || nb <= 1e-12
  dot / (Math.sqrt(na) * Math.sqrt(nb))
end

def l2_normalize(vec : Array(Float32)) : Array(Float32)
  norm = Math.sqrt(vec.sum { |x| x.to_f64 * x.to_f64 })
  return vec if norm <= 1e-12
  inv = (1.0 / norm).to_f32
  vec.map { |x| x * inv }
end

# ── Run ──

puts "Loading nomic-embed-text-v2-moe..."
ML::Metal::Device.init!
model = ML::GGUF::NomicBertMoE.from_gguf(MODEL, ML::GGUF::MetalBackend.new)
puts "Loaded. dim=#{model.dim}, vocab=#{model.vocab_size}"

# Warmup
model.embed("warmup")

def run_group(model, name : String, pairs, expect_low : Bool)
  puts "\n═══ Group #{name} ═══"
  cosines = [] of Float64
  pairs.each do |pair|
    a_text, b_text, label = pair
    a_vec = l2_normalize(model.embed(a_text))
    b_vec = l2_normalize(model.embed(b_text))
    cos = cosine(a_vec, b_vec)
    cosines << cos
    marker = if expect_low
               cos < 0.85 ? "✓" : (cos < 0.90 ? "~" : "✗")
             else
               cos > 0.85 ? "✓" : (cos > 0.75 ? "~" : "✗")
             end
    puts "  #{marker} cos=#{cos.round(4)}  #{label}"
  end
  avg = cosines.sum / cosines.size
  min = cosines.min
  max = cosines.max
  puts "  ── avg=#{avg.round(4)} min=#{min.round(4)} max=#{max.round(4)}"
  {avg, min, max}
end

a_avg, _, a_max = run_group(model, "A: Same syntax, DIFFERENT semantics (want LOW)", GROUP_A, expect_low: true)
b_avg, b_min, _ = run_group(model, "B: Different syntax, SAME semantics (want HIGH)", GROUP_B, expect_low: false)
c_avg, _, _ = run_group(model, "C: Completely different (baseline)", GROUP_C, expect_low: true)
d_avg, _, d_max = run_group(model, "D: Tool call discrimination (want LOW)", GROUP_D, expect_low: true)

puts "\n═══ VERDICT ═══"
puts "Group A (must discriminate): avg=#{a_avg.round(4)}, max=#{a_max.round(4)}"
puts "Group B (must group):        avg=#{b_avg.round(4)}, min=#{b_min.round(4)}"
puts "Group C (baseline):          avg=#{c_avg.round(4)}"
puts "Group D (tool discrimination): avg=#{d_avg.round(4)}, max=#{d_max.round(4)}"

puts ""
if a_max < 0.85 && b_min > 0.85 && d_max < 0.80
  puts "🟢 PASS — nomic-embed discriminates code semantics well"
  puts "   Encoder suitable for Cogniformerus-native model"
elsif a_max < 0.90 && d_max < 0.85
  puts "🟡 SOFT PASS — discriminates but with marginal separation"
  puts "   May work with learned K/V projections in cross-attention"
elsif a_avg < 0.90
  puts "🟠 MARGINAL — some discrimination but unreliable"
  puts "   Consider code-specific encoder (CodeBERT, StarEncoder)"
else
  puts "🔴 FAIL — nomic-embed cannot distinguish code semantics"
  puts "   Need different encoder for this architecture"
end
