require "spec"
require "../examples/support/hy_nomas_bf16_serial_probe"

private def with_bf16_serial_admission_checkpoint(dtype : String, block_index : Int32,
                                                  &block : ML::Video::CheckpointFile ->)
  bytes_per_element = dtype == "BF16" ? 2 : 4
  header = JSON.build do |json|
    json.object do
      32.times do |index|
        json.field "double_blocks.#{block_index}.weight#{index}" do
          json.object do
            json.field "dtype", dtype
            json.field "shape", [1]
            json.field "data_offsets", [index * bytes_per_element, (index + 1) * bytes_per_element]
          end
        end
      end
    end
  end

  temporary = File.tempfile("hy-bf16-serial-admission", ".safetensors")
  checkpoint = nil.as(ML::Video::CheckpointFile?)
  begin
    temporary.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
    temporary.write(header.to_slice)
    temporary.write(Bytes.new(32 * bytes_per_element))
    temporary.flush
    checkpoint = ML::Video::CheckpointFile.new(temporary.path)
    yield checkpoint
  ensure
    checkpoint.try(&.close)
    path = temporary.path
    temporary.close
    File.delete(path) if File.exists?(path)
  end
end

private def bf16_serial_admission_plan(file : ML::Video::CheckpointFile,
                                       resident_dtype : ML::Video::WeightDType = ML::Video::WeightDType::BF16,
                                       allow_f32_to_bf16 : Bool = false) : ML::Video::WeightPlan
  bindings = file.tensors.values.map do |info|
    ML::Video::WeightBinding.new(info.name, info.name, info.shape)
  end
  ML::Video::WeightPlan.new([file], bindings, 4096_i64,
    resident_dtype: resident_dtype, require_all: true,
    allow_f32_to_bf16: allow_f32_to_bf16)
end

describe HyNomasBF16CacheProbe do
  it "admits exactly 32 BF16 tensors for either selected block and rejects a swapped block identity" do
    allocations = ML::MetalBuffer.hy_text_probe_allocations
    readbacks = ML::MetalBuffer.hy_text_probe_readbacks

    {0, 1}.each do |block_index|
      with_bf16_serial_admission_checkpoint("BF16", block_index) do |file|
        plan = bf16_serial_admission_plan(file)
        plan.entries.size.should eq(32)
        plan.staging_bytes.should eq(0)
        HyNomasPretrainedSerialProbe.validate_bf16_plan!(plan, block_index)
      end
    end

    with_bf16_serial_admission_checkpoint("BF16", 0) do |block0|
      expect_raises(ArgumentError) do
        HyNomasPretrainedSerialProbe.validate_bf16_plan!(bf16_serial_admission_plan(block0), 1)
      end
    end
    with_bf16_serial_admission_checkpoint("BF16", 1) do |block1|
      expect_raises(ArgumentError) do
        HyNomasPretrainedSerialProbe.validate_bf16_plan!(bf16_serial_admission_plan(block1), 0)
      end
      expect_raises(ArgumentError) do
        HyNomasPretrainedSerialProbe.validate_bf16_plan!(bf16_serial_admission_plan(block1), 2)
      end
    end

    ML::MetalBuffer.hy_text_probe_allocations.should eq(allocations)
    ML::MetalBuffer.hy_text_probe_readbacks.should eq(readbacks)
  end

  it "rejects F32 source or resident storage and conversion-enabled plans" do
    allocations = ML::MetalBuffer.hy_text_probe_allocations
    readbacks = ML::MetalBuffer.hy_text_probe_readbacks

    with_bf16_serial_admission_checkpoint("F32", 0) do |f32_source|
      converted = bf16_serial_admission_plan(f32_source,
        resident_dtype: ML::Video::WeightDType::BF16, allow_f32_to_bf16: true)
      converted.staging_bytes.should be > 0
      expect_raises(ArgumentError) do
        HyNomasPretrainedSerialProbe.validate_bf16_plan!(converted, 0)
      end

      f32_resident = bf16_serial_admission_plan(f32_source,
        resident_dtype: ML::Video::WeightDType::F32)
      expect_raises(ArgumentError) do
        HyNomasPretrainedSerialProbe.validate_bf16_plan!(f32_resident, 0)
      end
    end

    with_bf16_serial_admission_checkpoint("BF16", 0) do |bf16_source|
      conversion_enabled = bf16_serial_admission_plan(bf16_source,
        resident_dtype: ML::Video::WeightDType::BF16, allow_f32_to_bf16: true)
      conversion_enabled.staging_bytes.should eq(0)
      conversion_enabled.allow_f32_to_bf16.should be_true
      expect_raises(ArgumentError) do
        HyNomasPretrainedSerialProbe.validate_bf16_plan!(conversion_enabled, 0)
      end
    end

    ML::MetalBuffer.hy_text_probe_allocations.should eq(allocations)
    ML::MetalBuffer.hy_text_probe_readbacks.should eq(readbacks)
  end

  it "validates exact block-1 oracle names and rejects a block-0 oracle" do
    allocations = ML::MetalBuffer.hy_text_probe_allocations
    readbacks = ML::MetalBuffer.hy_text_probe_readbacks

    with_bf16_serial_admission_checkpoint("BF16", 1) do |source1|
      plan1 = bf16_serial_admission_plan(source1)
      HyNomasPretrainedSerialProbe.validate_bf16_plan!(plan1, 1)

      with_bf16_serial_admission_checkpoint("BF16", 1) do |oracle1|
        HyNomasBF16CacheProbe.validate_oracle!(plan1, oracle1, block_index: 1)
      end
      with_bf16_serial_admission_checkpoint("BF16", 0) do |oracle0|
        expect_raises(ArgumentError) do
          HyNomasBF16CacheProbe.validate_oracle!(plan1, oracle0, block_index: 1)
        end
      end
    end

    with_bf16_serial_admission_checkpoint("BF16", 0) do |source0|
      plan0 = bf16_serial_admission_plan(source0)
      with_bf16_serial_admission_checkpoint("BF16", 1) do |oracle1|
        expect_raises(ArgumentError) do
          HyNomasBF16CacheProbe.validate_oracle!(plan0, oracle1, block_index: 1)
        end
      end
    end

    ML::MetalBuffer.hy_text_probe_allocations.should eq(allocations)
    ML::MetalBuffer.hy_text_probe_readbacks.should eq(readbacks)
  end

  it "admits a distinct two-source identity pair and rejects missing or aliased paths before Metal" do
    with_bf16_serial_admission_checkpoint("BF16", 0) do |block0|
      with_bf16_serial_admission_checkpoint("BF16", 1) do |block1|
        sources = [
          {path: block0.path, sha256: "a" * 64},
          {path: block1.path, sha256: "b" * 64},
        ]
        allocations = ML::MetalBuffer.hy_text_probe_allocations
        readbacks = ML::MetalBuffer.hy_text_probe_readbacks

        HyNomasPretrainedSerialProbe.validate_bf16_sources!(sources)
        expect_raises(ArgumentError) do
          HyNomasPretrainedSerialProbe.validate_bf16_sources!([sources[0]])
        end
        expect_raises(ArgumentError) do
          HyNomasPretrainedSerialProbe.validate_bf16_sources!([sources[0], sources[0]])
        end

        expect_raises(ArgumentError, /SHA256/) do
          malformed = [sources[0], {path: block1.path, sha256: "bad-digest"}]
          HyNomasPretrainedSerialProbe.validate_bf16_sources!(malformed)
        end

        hard_link = File.tempfile("hy-bf16-hard-link-alias", ".safetensors")
        hard_link_path = hard_link.path
        hard_link.close
        File.delete(hard_link_path)
        begin
          File.link(block0.path, hard_link_path)
          alias_sources = [sources[0], {path: hard_link_path, sha256: "c" * 64}]
          expect_raises(ArgumentError, /alias/) do
            HyNomasPretrainedSerialProbe.validate_bf16_sources!(alias_sources)
          end
        ensure
          File.delete(hard_link_path) if File.exists?(hard_link_path)
        end

        ML::MetalBuffer.hy_text_probe_allocations.should eq(allocations)
        ML::MetalBuffer.hy_text_probe_readbacks.should eq(readbacks)
      end
    end
  end

  it "admits distinct source/oracle pairs and rejects cross-role identity or digest reuse before Metal" do
    with_bf16_serial_admission_checkpoint("BF16", 0) do |source0|
      with_bf16_serial_admission_checkpoint("BF16", 1) do |source1|
        with_bf16_serial_admission_checkpoint("BF16", 0) do |oracle0|
          with_bf16_serial_admission_checkpoint("BF16", 1) do |oracle1|
            sources = [
              {path: source0.path, sha256: "a" * 64},
              {path: source1.path, sha256: "b" * 64},
            ]
            oracles = [
              {path: oracle0.path, sha256: "c" * 64},
              {path: oracle1.path, sha256: "d" * 64},
            ]
            allocations = ML::MetalBuffer.hy_text_probe_allocations
            readbacks = ML::MetalBuffer.hy_text_probe_readbacks

            HyNomasBF16SerialProbe.validate_source_oracle_pairs!(sources, oracles)

            # The same cache path cannot be presented as its oracle, even with
            # a different digest label.
            same_cache_oracle = [
              {path: source0.path, sha256: "e" * 64},
              oracles[1],
            ]
            expect_raises(ArgumentError) do
              HyNomasBF16SerialProbe.validate_source_oracle_pairs!(sources, same_cache_oracle)
            end

            # Cross-index aliasing is also invalid when each pair separately
            # contains two distinct regular paths.
            cross_index_oracle = [
              oracles[0],
              {path: source0.path, sha256: "f" * 64},
            ]
            expect_raises(ArgumentError) do
              HyNomasBF16SerialProbe.validate_source_oracle_pairs!(sources, cross_index_oracle)
            end

            # A distinct inode cannot claim the same content identity as any
            # cache source.
            same_digest_oracle = [
              oracles[0],
              {path: oracle1.path, sha256: sources[0][:sha256]},
            ]
            expect_raises(ArgumentError) do
              HyNomasBF16SerialProbe.validate_source_oracle_pairs!(sources, same_digest_oracle)
            end

            hard_link = File.tempfile("hy-bf16-cross-role-hard-link", ".safetensors")
            hard_link_path = hard_link.path
            hard_link.close
            File.delete(hard_link_path)
            begin
              File.link(source0.path, hard_link_path)
              hard_link_oracle = [
                oracles[0],
                {path: hard_link_path, sha256: "9" * 64},
              ]
              expect_raises(ArgumentError, /alias|same file|identity/i) do
                HyNomasBF16SerialProbe.validate_source_oracle_pairs!(sources, hard_link_oracle)
              end
            ensure
              File.delete(hard_link_path) if File.exists?(hard_link_path)
            end

            ML::MetalBuffer.hy_text_probe_allocations.should eq(allocations)
            ML::MetalBuffer.hy_text_probe_readbacks.should eq(readbacks)
          end
        end
      end
    end
  end
end
