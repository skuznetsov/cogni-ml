require "spec"
require "../src/ml/core/buffer"
require "../src/ml/video/hy_nomas_reference_kv_cache"

describe ML::Video::HyNomasReferenceKVCache do
  it "counts separate active BF16 cache planes and F32 read stages" do
    geometry = ML::Video::HyNomasReferenceKVCache.geometry(8, 5, 2, 4)

    geometry.source_elements.should eq(64)
    geometry.cache_elements.should eq(40)
    geometry.shape.should eq({5, 2, 4})
    geometry.named_bytes.should eq(480_i64)
    ML::Video::HyNomasReferenceKVCache.named_bytes(8, 5, 2, 4).should eq(480_i64)
  end

  it "uses storage rows only for the source extent and active rows for cache capacity" do
    geometry = ML::Video::HyNomasReferenceKVCache.geometry(5_600, 4_849, 16, 128)

    geometry.source_elements.should eq(11_468_800)
    geometry.cache_elements.should eq(9_930_752)
    geometry.named_bytes.should eq(119_169_024_i64)
  end

  it "rejects invalid geometry and dispatch overflow with checked CPU arithmetic" do
    expect_raises(ArgumentError, /dimensions must be positive/) do
      ML::Video::HyNomasReferenceKVCache.geometry(0, 1, 1, 1)
    end
    expect_raises(ArgumentError, /active rows exceed/) do
      ML::Video::HyNomasReferenceKVCache.geometry(3, 4, 1, 1)
    end
    expect_raises(ArgumentError, /element count exceeds/) do
      ML::Video::HyNomasReferenceKVCache.geometry(Int32::MAX, 1, 2, 2)
    end
  end

  it "rejects invalid constructor geometry before reading buffers or allocating Metal state" do
    key = uninitialized ML::MetalBuffer
    value = uninitialized ML::MetalBuffer

    expect_raises(ArgumentError, /dimensions must be positive/) do
      ML::Video::HyNomasReferenceKVCache::Plan.new(key, value,
        storage_rows: 0, active_rows: 1, heads: 1, head_dim: 1)
    end
  end

  it "pins the store to the shared source-BF16 rounding helper and loads physical BF16 values" do
    source = ML::Video::HyNomasReferenceKVCache::SOURCE

    source.should contain("hy_nomas_bf16_roundtrip_f32(source[id])")
    source.should contain("cache[id] = ushort(as_type<uint>(rounded) >> 16u)")
    source.should contain("as_type<float>(uint(cache[id]) << 16u)")
    source.should contain("if (id >= count) return;")
  end
end
