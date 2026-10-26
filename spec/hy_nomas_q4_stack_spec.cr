require "spec"
require "digest/sha256"
require "file_utils"
require "json"
require "./support/video_allocation_counter"
require "../src/ml/video/hy_nomas_q4_stack"

private class HyNomasQ4StackSpecFixture
  getter directory : String
  getter index_path : String
  getter index_sha256 : String
  getter source_sha256 : String
  getter source_bytes : Int64
  getter payload_bytes : Int64
  getter stage_bytes : Int64
  getter config : ML::Video::HyNomasConfig
  @index_sha256 : String
  @globals_sha256 : String

  def initialize
    @directory = File.tempname("hy-nomas-q4-stack")
    FileUtils.mkdir_p(@directory)
    @index_path = File.join(@directory, "index.json")
    @index_sha256 = ""
    @source_sha256 = Digest::SHA256.hexdigest("synthetic-full-checkpoint")
    @source_bytes = 34_213_991_280_i64
    @config = ML::Video::HyNomasConfig.new(batches: 1, video_tokens: 1,
      text_tokens: 1, dim: 64, heads: 1, head_dim: 64, ffn_dim: 256)
    @config.validate!
    @stage_bytes = ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(
      @config, block_index: 0).sum do |binding|
      binding.shape.reduce(1_i64) { |count, dimension| count * dimension } * 2_i64
    end
    @payload_bytes = 0_i64
    @block_entries = [] of NamedTuple(block_index: Int32, manifest: String,
      manifest_sha256: String, payload_sha256: String, bf16: String,
      bf16_sha256: String)
    54.times do |block_index|
      @block_entries << write_block(block_index.to_i32)
    end
    globals_name = "globals-bf16.safetensors"
    globals_path = File.join(@directory, globals_name)
    write_file(globals_path, "global BF16 control")
    @globals_sha256 = sha256(globals_path)
    @index_sha256 = write_index
  end

  def cleanup : Nil
    FileUtils.rm_rf(@directory)
  end

  def rewrite_index(&block : JSON::Any -> JSON::Any) : Nil
    document = JSON.parse(File.read(@index_path))
    yield document
    body = document.to_json
    File.write(@index_path, body)
    @index_sha256 = Digest::SHA256.hexdigest(body)
  end

  def stack(*, budget_bytes : Int64 = @payload_bytes + @stage_bytes + 32_i64,
            source_sha256 : String = @source_sha256) : ML::Video::HyNomasQ4Stack
    ML::Video::HyNomasQ4Stack.new(@index_path, @index_sha256,
      source_sha256, @config, budget_bytes,
      stage_bytes: @stage_bytes, reserve_bytes: 32_i64)
  end

  private def write_block(block_index : Int32)
    bindings = ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(
      @config, block_index: block_index).sort_by(&.source)
    payload = Bytes.new(0)
    tensor_layout = [] of NamedTuple(binding: ML::Video::WeightBinding,
      elements: Int64, encoding: String, offset: Int64, nbytes: Int64)
    cursor = 0_i64
    bindings.each do |binding|
      elements = binding.shape.reduce(1_i64) { |count, dimension| count * dimension }
      encoding = binding.shape.size == 2 ? "q4_k_8range" : "bf16-le"
      nbytes = if binding.shape.size == 2
                 ML::Video::Q4KRangeWeights.payload_bytes(elements.to_i32)
               else
                 elements * 2_i64
               end
      tensor_layout << {binding: binding, elements: elements,
                        encoding: encoding, offset: cursor, nbytes: nbytes}
      cursor += nbytes
    end
    payload = Bytes.new(cursor.to_i32) do |index|
      ((index + block_index) & 0xff).to_u8
    end
    payload_name = "block-#{block_index}.bin"
    payload_path = File.join(@directory, payload_name)
    write_file(payload_path, payload)
    payload_sha256 = sha256(payload_path)

    manifest_body = JSON.build do |json|
      json.object do
        json.field "schema", ML::Video::HyNomasQ4Cache::SCHEMA
        json.field "schema_version", ML::Video::HyNomasQ4Cache::SCHEMA_VERSION
        json.field "source" do
          json.object do
            json.field "sha256", @source_sha256
            json.field "bytes", @source_bytes
          end
        end
        json.field "block_index", block_index
        json.field "geometry" do
          json.object do
            ML::Video::HyNomasQ4Cache.geometry(@config).each do |name, value|
              json.field name, value
            end
          end
        end
        json.field "codec" do
          json.object do
            json.field "matrix", "q4_k"
            json.field "ranges", 8
            json.field "block_elements", ML::Video::Q4KRangeWeights::BLOCK_ELEMENTS
            json.field "block_bytes", ML::Video::Q4KRangeWeights::Q4_K_BLOCK_BYTES
            json.field "vector", "bf16-le"
          end
        end
        json.field "payload" do
          json.object do
            json.field "basename", payload_name
            json.field "nbytes", cursor
            json.field "sha256", payload_sha256
          end
        end
        json.field "tensors" do
          json.object do
            tensor_layout.each do |tensor|
              json.field tensor[:binding].source do
                json.object do
                  json.field "shape", tensor[:binding].shape
                  json.field "source_dtype", "F32"
                  json.field "resident_dtype", "BF16"
                  json.field "encoding", tensor[:encoding]
                  json.field "offset_bytes", tensor[:offset]
                  json.field "nbytes", tensor[:nbytes]
                end
              end
            end
          end
        end
      end
    end
    manifest_name = "block-#{block_index}.json"
    manifest_path = File.join(@directory, manifest_name)
    File.write(manifest_path, manifest_body)

    bf16_name = "block-#{block_index}-bf16.safetensors"
    bf16_path = File.join(@directory, bf16_name)
    write_file(bf16_path, "BF16 control #{block_index}")
    @payload_bytes += cursor
    {block_index: block_index, manifest: manifest_name,
     manifest_sha256: sha256(manifest_path), payload_sha256: payload_sha256,
     bf16: bf16_name, bf16_sha256: sha256(bf16_path)}
  end

  private def write_index : String
    body = JSON.build do |json|
      json.object do
        json.field "schema", ML::Video::HyNomasQ4StackIndex::SCHEMA
        json.field "schema_version", ML::Video::HyNomasQ4StackIndex::SCHEMA_VERSION
        json.field "source" do
          json.object do
            json.field "sha256", @source_sha256
            json.field "bytes", @source_bytes
          end
        end
        json.field "blocks" do
          json.array do
            @block_entries.each do |block|
              json.object do
                json.field "block_index", block[:block_index]
                json.field "manifest", block[:manifest]
                json.field "manifest_sha256", block[:manifest_sha256]
                json.field "payload_sha256", block[:payload_sha256]
                json.field "bf16", block[:bf16]
                json.field "bf16_sha256", block[:bf16_sha256]
              end
            end
          end
        end
        json.field "globals" do
          json.object do
            json.field "bf16", "globals-bf16.safetensors"
            json.field "sha256", @globals_sha256
          end
        end
      end
    end
    File.write(@index_path, body)
    Digest::SHA256.hexdigest(body)
  end

  private def write_file(path : String, bytes : Bytes) : Nil
    File.open(path, "wb") { |io| io.write(bytes) }
  end

  private def write_file(path : String, text : String) : Nil
    File.write(path, text)
  end

  private def sha256(path : String) : String
    Digest::SHA256.hexdigest(File.read(path))
  end
end

describe ML::Video::HyNomasQ4Stack do
  it "authenticates the ordered 54-block cache and admits bytes before Metal" do
    fixture = HyNomasQ4StackSpecFixture.new
    begin
      allocations = ML::MetalBuffer.video_spec_allocations
      stack = fixture.stack
      stack.block_count.should eq(54)
      stack.blocks.map(&.block_index).should eq((0...54).to_a)
      stack.cache_payload_bytes.should eq(fixture.payload_bytes)
      stack.admitted_bytes.should eq(fixture.payload_bytes + fixture.stage_bytes + 32_i64)
      ML::MetalBuffer.video_spec_allocations.should eq(allocations)
    ensure
      fixture.cleanup
    end
  end

  it "rejects incorrect source pins, reordered or substituted blocks, and short aggregate budgets before Metal" do
    fixture = HyNomasQ4StackSpecFixture.new
    begin
      allocations = ML::MetalBuffer.video_spec_allocations
      expect_raises(ArgumentError, /source checkpoint mismatch/) do
        fixture.stack(source_sha256: "a" * 64)
      end
      expect_raises(ArgumentError, /memory budget/) do
        fixture.stack(budget_bytes: fixture.payload_bytes + fixture.stage_bytes + 31_i64)
      end

      fixture.rewrite_index do |document|
        document.as_h["blocks"].as_a[1].as_h["block_index"] = JSON.parse("0")
      end
      expect_raises(ArgumentError, /block order/) { fixture.stack }
      ML::MetalBuffer.video_spec_allocations.should eq(allocations)
    ensure
      fixture.cleanup
    end
  end

  it "rejects duplicate cache payload identities and corrupt payload files" do
    fixture = HyNomasQ4StackSpecFixture.new
    begin
      allocations = ML::MetalBuffer.video_spec_allocations
      fixture.rewrite_index do |document|
        first_hash = document.as_h["blocks"].as_a[0].as_h["payload_sha256"]
        document.as_h["blocks"].as_a[1].as_h["payload_sha256"] = first_hash
      end
      expect_raises(ArgumentError, /duplicate block cache identities/) { fixture.stack }

      # Restore a fresh index, then change a payload without changing the
      # authenticated index or manifest. Admission must catch this before any
      # Metal cache buffer is allocated.
      fixture.cleanup
      fixture = HyNomasQ4StackSpecFixture.new
      payload_path = File.join(fixture.directory, "block-7.bin")
      File.write(payload_path, "corrupt")
      expect_raises(ArgumentError, /payload file size mismatch/) { fixture.stack }
      ML::MetalBuffer.video_spec_allocations.should eq(allocations)
    ensure
      fixture.cleanup
    end
  end
end
