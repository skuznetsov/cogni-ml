require "spec"
require "../src/ml/audio/drama3/backbone_resident"
require "../src/ml/audio/drama3/tower_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  include ML::Audio::Drama3

  class ML::Audio::Drama3::BackboneResident
    def wide_attention_for_growth_spec? : Bool
      @wide_attention
    end

    def has_score_scratch_for_growth_spec? : Bool
      !@full_attention_scores.nil?
    end

    def full_attention_ready_for_growth_spec(q : ML::Audio::Drama3::Resident::Buffer,
                                             k : ML::Audio::Drama3::Resident::Buffer,
                                             v : ML::Audio::Drama3::Resident::Buffer,
                                             gate : ML::Audio::Drama3::Resident::Buffer,
                                             layer : Int32, position : Int32) : ML::Audio::Drama3::Resident::Buffer
      full_attention_ready(q, k, v, gate, layer, position)
    end

    def seed_full_attention_prefix_for_growth_spec(layer : Int32, position : Int32,
                                                   key_values : Array(Float32),
                                                   value_values : Array(Float32)) : Nil
      count = position * @config.kv_heads * @config.head_dim
      raise ArgumentError.new("growth spec cache prefix length") unless position > 0 && key_values.size == count && value_values.size == count
      @ctx.copy(@ctx.upload(key_values), @keys[layer].not_nil!.slice(0, count))
      @ctx.copy(@ctx.upload(value_values), @values[layer].not_nil!.slice(0, count))
      @position = position
    end

    def read_full_attention_prefix_for_growth_spec(layer : Int32, positions : Int32) : Tuple(ML::Audio::Drama3::Vector, ML::Audio::Drama3::Vector)
      count = positions * @config.kv_heads * @config.head_dim
      keys = @ctx.read(@keys[layer].not_nil!.slice(0, count), "terminal")
      values = @ctx.read(@values[layer].not_nil!.slice(0, count), "terminal")
      {keys, values}
    end
  end

  describe "DRAMA-3 resident cache growth primitives" do
    it "relocates both head-major KV caches across distinct old and new strides" do
      context = Resident::Context.new
      heads = 2
      head_dim = 3
      old_capacity = 3
      new_capacity = 5
      valid_length = 2
      values_per_cache = heads * old_capacity * head_dim
      copy_count = heads * valid_length * head_dim

      key_values = Array(Float32).new(values_per_cache) { |index| (index + 1).to_f32 }
      value_values = Array(Float32).new(values_per_cache) { |index| (1000 + index).to_f32 }
      old_keys = context.upload(key_values)
      old_values = context.upload(value_values)
      new_keys = context.alloc_persistent(heads * new_capacity * head_dim)
      new_values = context.alloc_persistent(heads * new_capacity * head_dim)

      context.dispatch(
        "drama3_tower_relocate_kv", RESIDENT_TOWER_SOURCE,
        [old_keys, old_values, new_keys, new_values],
        [heads.to_u64, head_dim.to_u64, old_capacity.to_u64, new_capacity.to_u64, valid_length.to_u64],
        copy_count,
      )

      compact_keys = context.alloc(copy_count)
      compact_values = context.alloc(copy_count)
      [
        {new_keys, compact_keys},
        {new_values, compact_values},
      ].each do |source, compact|
        context.dispatch(
          "drama3_tower_compact_cache", RESIDENT_TOWER_SOURCE, [source, compact],
          [heads.to_u64, head_dim.to_u64, new_capacity.to_u64, valid_length.to_u64], compact.count,
        )
      end

      expected_keys = [] of Float32
      expected_values = [] of Float32
      heads.times do |head|
        valid_length.times do |time|
          head_dim.times do |dimension|
            index = (head * old_capacity + time) * head_dim + dimension
            expected_keys << key_values[index]
            expected_values << value_values[index]
          end
        end
      end

      context.read(compact_keys, "terminal").map(&.unsafe_as(UInt32)).should eq(expected_keys.map(&.unsafe_as(UInt32)))
      context.read(compact_values, "terminal").map(&.unsafe_as(UInt32)).should eq(expected_values.map(&.unsafe_as(UInt32)))
    end

    it "pins backbone attention width and scratch from maximum capacity, including 1024 to 1025" do
      fixture = File.join(__DIR__, "support/drama3/reference_fixture")
      config = BackboneConfig.new(JSON.parse(File.read(File.join(fixture, "config.json"))))
      store = WeightStore.new(fixture)
      narrow_context = Resident::Context.new
      wide_context = Resident::Context.new
      narrow = BackboneResident.new(config, store, narrow_context, 1, maximum_capacity: 1024)
      wide = BackboneResident.new(config, store, wide_context, 1, maximum_capacity: 1025)

      narrow.wide_attention_for_growth_spec?.should be_false
      narrow.has_score_scratch_for_growth_spec?.should be_false
      wide.wide_attention_for_growth_spec?.should be_true
      wide.has_score_scratch_for_growth_spec?.should be_true
      (wide_context.allocation_stats[:allocations] - narrow_context.allocation_stats[:allocations]).should eq(1_i64)

      wide.grow_capacity!(1024)
      wide.capacity.should eq(1024)
      wide.wide_attention_for_growth_spec?.should be_true
      wide.has_score_scratch_for_growth_spec?.should be_true
      wide.grow_capacity!(1025)
      wide.capacity.should eq(1025)
      wide.wide_attention_for_growth_spec?.should be_true
      wide.has_score_scratch_for_growth_spec?.should be_true
    end

    it "keeps wide attention bit-exact at row zero and after growing through position 1024" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        fixture = File.join(__DIR__, "support/drama3/reference_fixture")
        config = BackboneConfig.new(JSON.parse(File.read(File.join(fixture, "config.json"))))
        store = WeightStore.new(fixture)
        layer = config.layers - 1
        kv_width = config.kv_heads * config.head_dim

        [0, 2].each do |depth|
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth.to_s
          dynamic_context = Resident::Context.new
          fixed_context = Resident::Context.new
          dynamic = BackboneResident.new(config, store, dynamic_context, 1, maximum_capacity: 1025)
          fixed = BackboneResident.new(config, store, fixed_context, 1025, maximum_capacity: 1025)

          # Compare the same wide kernel at the first row with different score
          # strides: dynamic capacity one versus fixed capacity 1025.
          first_q = Array(Float32).new(config.heads * config.head_dim) { |i| ((i * 7 % 23) - 11).to_f32 / 16_f32 }
          first_k = Array(Float32).new(kv_width) { |i| ((i * 5 % 19) - 9).to_f32 / 16_f32 }
          first_v = Array(Float32).new(kv_width) { |i| ((i * 11 % 29) - 14).to_f32 / 16_f32 }
          first_gate = Array(Float32).new(config.heads * config.head_dim) { |i| ((i * 3 % 17) - 8).to_f32 / 16_f32 }
          dynamic_first = dynamic.full_attention_ready_for_growth_spec(
            dynamic_context.upload(first_q), dynamic_context.upload(first_k), dynamic_context.upload(first_v),
            dynamic_context.upload(first_gate), layer, 0,
          )
          fixed_first = fixed.full_attention_ready_for_growth_spec(
            fixed_context.upload(first_q), fixed_context.upload(first_k), fixed_context.upload(first_v),
            fixed_context.upload(first_gate), layer, 0,
          )
          dynamic_context.read(dynamic_first, "terminal").map(&.unsafe_as(UInt32)).should eq(
            fixed_context.read(fixed_first, "terminal").map(&.unsafe_as(UInt32)),
          )

          # Seed a deterministic synthetic KV state, then test the actual
          # position-1024 attention dispatch after the 1024 -> 1025 relocation.
          dynamic.grow_capacity!(1024)
          prefix_count = 1024 * kv_width
          key_prefix = Array(Float32).new(prefix_count) { |i| ((i * 19 % 61) - 30).to_f32 / 64_f32 }
          value_prefix = Array(Float32).new(prefix_count) { |i| ((i * 23 % 67) - 33).to_f32 / 64_f32 }
          dynamic.seed_full_attention_prefix_for_growth_spec(layer, 1024, key_prefix, value_prefix)
          fixed.seed_full_attention_prefix_for_growth_spec(layer, 1024, key_prefix, value_prefix)
          dynamic.grow_capacity!(1025)

          grown_prefix = dynamic.read_full_attention_prefix_for_growth_spec(layer, 1024)
          fixed_prefix = fixed.read_full_attention_prefix_for_growth_spec(layer, 1024)
          grown_prefix[0].map(&.unsafe_as(UInt32)).should eq(fixed_prefix[0].map(&.unsafe_as(UInt32)))
          grown_prefix[1].map(&.unsafe_as(UInt32)).should eq(fixed_prefix[1].map(&.unsafe_as(UInt32)))

          boundary_q = Array(Float32).new(config.heads * config.head_dim) { |i| ((i * 13 % 31) - 15).to_f32 / 32_f32 }
          boundary_k = Array(Float32).new(kv_width) { |i| ((i * 17 % 37) - 18).to_f32 / 32_f32 }
          boundary_v = Array(Float32).new(kv_width) { |i| ((i * 29 % 41) - 20).to_f32 / 32_f32 }
          boundary_gate = Array(Float32).new(config.heads * config.head_dim) { |i| ((i * 7 % 29) - 14).to_f32 / 32_f32 }
          dynamic_boundary = dynamic.full_attention_ready_for_growth_spec(
            dynamic_context.upload(boundary_q), dynamic_context.upload(boundary_k), dynamic_context.upload(boundary_v),
            dynamic_context.upload(boundary_gate), layer, 1024,
          )
          fixed_boundary = fixed.full_attention_ready_for_growth_spec(
            fixed_context.upload(boundary_q), fixed_context.upload(boundary_k), fixed_context.upload(boundary_v),
            fixed_context.upload(boundary_gate), layer, 1024,
          )
          dynamic_context.read(dynamic_boundary, "terminal").map(&.unsafe_as(UInt32)).should eq(
            fixed_context.read(fixed_boundary, "terminal").map(&.unsafe_as(UInt32)),
          )
        end
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
        if previous_depth
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous_depth
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
      end
    end

    it "allocates cache replacements outside reusable workspace slots" do
      fixture = File.join(__DIR__, "support/drama3/reference_fixture")
      backbone_config = BackboneConfig.new(JSON.parse(File.read(File.join(fixture, "config.json"))))
      tower_config = TowerConfig.from_json(File.read(File.join(fixture, "config.json")))
      store = WeightStore.new(fixture)

      backbone_context = Resident::Context.new
      backbone = BackboneResident.new(backbone_config, store, backbone_context, 1, maximum_capacity: 2)
      backbone_workspace = backbone_context.new_workspace
      before_backbone = backbone_context.allocation_stats[:allocations]
      backbone_context.with_workspace(backbone_workspace) { backbone.grow_capacity!(2) }
      backbone_workspace.stats["slots"].should eq(0_i64)
      full_attention_layers = backbone_config.layers // backbone_config.full_interval
      expected_backbone_replacements = 2_i64 * full_attention_layers.to_i64
      (backbone_context.allocation_stats[:allocations] - before_backbone).should eq(expected_backbone_replacements)

      tower_context = Resident::Context.new
      tower = TowerResident.new(tower_config, store, tower_context, 1, maximum_capacity: 2)
      tower_workspace = tower_context.new_workspace
      before_tower = tower_context.allocation_stats[:allocations]
      tower_context.with_workspace(tower_workspace) { tower.grow_capacity!(2) }
      tower_workspace.stats["slots"].should eq(0_i64)
      expected_tower_replacements = 2_i64 * tower_config.layer_count.to_i64
      (tower_context.allocation_stats[:allocations] - before_tower).should eq(expected_tower_replacements)
    end

    it "rejects invalid maximums and growth before allocating replacement buffers" do
      fixture = File.join(__DIR__, "support/drama3/reference_fixture")
      backbone_config = BackboneConfig.new(JSON.parse(File.read(File.join(fixture, "config.json"))))
      tower_config = TowerConfig.from_json(File.read(File.join(fixture, "config.json")))
      store = WeightStore.new(fixture)
      backbone_context = Resident::Context.new
      tower_context = Resident::Context.new

      expect_raises(ArgumentError) { BackboneResident.new(backbone_config, store, backbone_context, 1, maximum_capacity: 4097) }
      expect_raises(ArgumentError) { TowerResident.new(tower_config, store, tower_context, 1, maximum_capacity: 4097) }
      backbone_context.allocation_stats[:allocations].should eq(0_i64)
      tower_context.allocation_stats[:allocations].should eq(0_i64)

      backbone = BackboneResident.new(backbone_config, store, backbone_context, 1, maximum_capacity: 2)
      tower = TowerResident.new(tower_config, store, tower_context, 1, maximum_capacity: 2)
      before_backbone = backbone_context.allocation_stats[:allocations]
      before_tower = tower_context.allocation_stats[:allocations]

      expect_raises(ArgumentError) { backbone.grow_capacity!(1) }
      expect_raises(ArgumentError) { backbone.grow_capacity!(3) }
      expect_raises(ArgumentError) { tower.grow_capacity!(1) }
      expect_raises(ArgumentError) { tower.grow_capacity!(3) }
      backbone_context.allocation_stats[:allocations].should eq(before_backbone)
      tower_context.allocation_stats[:allocations].should eq(before_tower)
    end
  end
{% end %}
