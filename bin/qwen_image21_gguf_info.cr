require "option_parser"
require "../src/ml/gguf/qwen_image21_gguf"

model_path : String? = nil
list_tensors = false

OptionParser.parse do |parser|
  parser.banner = "Usage: qwen_image21_gguf_info --model PATH [--list-tensors]"
  parser.on("--model PATH", "Qwen-Image 2.1 DiT GGUF path") { |value| model_path = File.expand_path(value) }
  parser.on("--list-tensors", "Print storage and source shapes for every tensor") { list_tensors = true }
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
end

abort("--model is required") unless model_path
path = model_path.not_nil!
file : ML::GGUF::GGUFFile? = nil

begin
  file = ML::GGUF::GGUFFile.new(path, mmap_tensors: false)
  inventory = ML::GGUF::QwenImage21GGUFInventory.from_file(file)
  required_payload_bytes = file.tensors.max_of? { |tensor| tensor.offset.to_i64 + tensor.data_bytes } || 0_i64
  required_file_bytes = file.data_offset + required_payload_bytes
  actual_file_bytes = File.size(path)

  puts "architecture=#{inventory.architecture}"
  puts "gguf_version=#{file.version}"
  puts "tensors=#{file.tensors.size}"
  puts "transformer_blocks=#{inventory.block_count}"
  puts "orig_shapes=#{inventory.orig_shape_count}"
  puts "tensor_data_bytes=#{inventory.total_tensor_bytes}"
  puts "required_file_bytes=#{required_file_bytes}"
  puts "actual_file_bytes=#{actual_file_bytes}"
  puts "tensor_data_complete=#{actual_file_bytes >= required_file_bytes}"
  puts "reader_compatible=#{inventory.reader_compatible?}"
  puts "unsupported_types=#{inventory.unsupported_type_labels.join(",")}" unless inventory.reader_compatible?
  puts "tensor_types=#{inventory.type_counts.map { |name, count| "#{name}:#{count}" }.join(",")}"

  if list_tensors
    file.tensors.each do |tensor|
      storage = tensor.dims.join("x")
      logical = inventory.logical_shape(tensor.name).join("x")
      puts "tensor=#{tensor.name} type=#{tensor.type.name} storage=#{storage} source=#{logical} bytes=#{tensor.data_bytes}"
    end
  end
rescue ex : ArgumentError
  STDERR.puts "invalid Qwen-Image 2.1 GGUF: #{ex.message}"
  exit 2
ensure
  file.try(&.close)
end
