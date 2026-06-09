#!/usr/bin/env crystal

require "../src/ml/gguf/reader"

def usage
  STDERR.puts "usage: crystal scripts/export_gguf_tensor_probe.cr --model PATH --tensor NAME --out PATH"
  exit 2
end

model_path = nil
tensor_name = nil
out_path = nil
row_limit = nil
i = 0
while i < ARGV.size
  case ARGV[i]
  when "--model"
    i += 1; model_path = ARGV[i]?
  when "--tensor"
    i += 1; tensor_name = ARGV[i]?
  when "--out"
    i += 1; out_path = ARGV[i]?
  when "--row-limit"
    i += 1; row_limit = ARGV[i]?.try(&.to_u32)
  when "--list-q4"
    i += 1; model_path = ARGV[i]?
    usage unless model_path
    g = ML::GGUF::GGUFFile.new(model_path.not_nil!)
    g.tensors.select { |t| t.type.q4_k? }.first(80).each do |t|
      puts "#{t.name}\t#{t.dims.join("x")}\t#{t.type.name}\t#{t.data_bytes}"
    end
    exit
  when "--list-q8"
    i += 1; model_path = ARGV[i]?
    usage unless model_path
    g = ML::GGUF::GGUFFile.new(model_path.not_nil!)
    g.tensors.select { |t| t.type.q8_0? }.first(80).each do |t|
      puts "#{t.name}\t#{t.dims.join("x")}\t#{t.type.name}\t#{t.data_bytes}"
    end
    exit
  when "--list-q6"
    i += 1; model_path = ARGV[i]?
    usage unless model_path
    g = ML::GGUF::GGUFFile.new(model_path.not_nil!)
    g.tensors.select { |t| t.type.q6_k? }.first(80).each do |t|
      puts "#{t.name}\t#{t.dims.join("x")}\t#{t.type.name}\t#{t.data_bytes}"
    end
    exit
  else
    usage
  end
  i += 1
end

usage unless model_path && tensor_name && out_path

g = ML::GGUF::GGUFFile.new(model_path.not_nil!)
info = g.tensor(tensor_name.not_nil!) || raise "tensor not found: #{tensor_name}"
raise "expected 2D tensor, got dims=#{info.dims}" unless info.dims.size == 2
raise "expected Q4_K, Q6_K, or Q8_0, got #{info.type.name}" unless info.type.q4_k? || info.type.q6_k? || info.type.q8_0?

in_dim = info.dims[0].to_u32
out_dim = info.dims[1].to_u32
if limit = row_limit
  raise "--row-limit must be > 0" if limit == 0
  raise "--row-limit #{limit} exceeds out_dim #{out_dim}" if limit > out_dim
  out_dim = limit
end
type_id = if info.type.q4_k?
            12_u32
          elsif info.type.q6_k?
            14_u32
          else
            8_u32
          end
raw = g.read_tensor_raw(info)
if limit = row_limit
  full_out_dim = info.dims[1].to_u64
  raise "raw tensor bytes not divisible by out_dim" unless raw.size.to_u64 % full_out_dim == 0
  row_bytes = raw.size.to_u64 // full_out_dim
  raw = raw[0, (row_bytes * limit).to_i]
end

File.open(out_path.not_nil!, "wb") do |io|
  io.write_bytes 0x43564750_u32, IO::ByteFormat::LittleEndian # CVGP
  io.write_bytes type_id, IO::ByteFormat::LittleEndian
  io.write_bytes out_dim, IO::ByteFormat::LittleEndian
  io.write_bytes in_dim, IO::ByteFormat::LittleEndian
  io.write_bytes raw.size.to_u64, IO::ByteFormat::LittleEndian
  io.write raw
end

puts "exported tensor=#{info.name} type=#{info.type.name} out=#{out_dim} in=#{in_dim} bytes=#{raw.size} path=#{out_path}"
