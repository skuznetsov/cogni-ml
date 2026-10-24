require "json"
require "../../src/ml/video/hy_nomas_block"
require "../../src/ml/video/hy_nomas_mps_pointwise"

# Frozen-input controls isolate contraction from earlier modulation/SiLU drift.
# The caller owns the arena and performs one terminal wait before reading.
class HyNomasModulationControls
  SOURCE = ML::Video::HyNomasMPSPointwise::SOURCE
  FIELDS = {"attn_shift", "attn_scale", "attn_gate", "ff_shift", "ff_scale", "ff_gate"}

  @pipelines : Hash(String, ML::Metal::ComputePipeline)

  def initialize(@config : ML::Video::HyNomasConfig, @arena : ML::Video::Arena,
                 variant : JSON::Any, data : JSON::Any)
    @pipelines = {"modulate", "residual"}.to_h do |kind|
      name = "hy_nomas_#{kind}_uncontracted_f32"
      {kind, ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }}
    end
    {"img", "txt"}.each do |stream|
      rows = stream == "img" ? @config.image_rows : @config.text_rows
      prefix = "modulation_control.#{stream}"
      checkpoints = variant["checkpoints"]
      {"norm"       => checkpoints["#{stream}.attn.norm"],
       "modulation" => checkpoints["#{stream}.modulation"],
       "projected"  => checkpoints["#{stream}.attn.projected"],
       "input"      => data["inputs"][stream]}.each do |name, record|
        @arena.upload("#{prefix}.#{name}", record["values"].as_a.map(&.as_f.to_f32))
      end
      (FIELDS.to_a + ["modulate.native", "modulate.uncontracted", "residual.native", "residual.uncontracted"]).each do |name|
        @arena.alloc("#{prefix}.#{name}", rows * @config.dim)
      end
    end
  end

  def encode(e : ML::Metal::ComputeEncoder) : Nil
    {"img", "txt"}.each do |stream|
      rows = stream == "img" ? @config.image_rows : @config.text_rows
      tokens = stream == "img" ? @config.video_tokens : @config.text_tokens
      mod_rows = stream == "img" ? @config.effective_img_modulation_rows : @config.effective_txt_modulation_rows
      prefix = "modulation_control.#{stream}"
      count = rows * @config.dim
      e.set_pipeline(ML::Video::HyNomasDualStreamBlock.pipeline("hy_nomas_unpack_modulation_f32"))
      e.set_buffer(@arena["#{prefix}.modulation"], 0)
      FIELDS.each_with_index { |name, index| e.set_buffer(@arena["#{prefix}.#{name}"], index + 1, ML::Metal::BufferAccess::Write) }
      e.set_value(rows.to_u32, 7).set_value(tokens.to_u32, 8).set_value(@config.dim.to_u32, 9)
      e.set_value(mod_rows.to_u32, 10).set_value(@config.batches.to_u32, 11).dispatch_1d(count)
      ML::Video::MetalOps.modulate(e, @arena["#{prefix}.norm"], @arena["#{prefix}.attn_shift"],
        @arena["#{prefix}.attn_scale"], @arena["#{prefix}.modulate.native"], rows, @config.dim, rows)
      ML::Video::MetalOps.add(e, @arena["#{prefix}.input"], @arena["#{prefix}.projected"],
        @arena["#{prefix}.residual.native"], count, @arena["#{prefix}.attn_gate"], @config.dim, rows)
      {"modulate" => {"norm", "attn_shift", "attn_scale"},
       "residual" => {"input", "projected", "attn_gate"}}.each do |kind, names|
        e.set_pipeline(@pipelines[kind])
        names.each_with_index { |name, index| e.set_buffer(@arena["#{prefix}.#{name}"], index) }
        e.set_buffer(@arena["#{prefix}.#{kind}.uncontracted"], 3, ML::Metal::BufferAccess::Write)
        e.set_value(count.to_u32, 4).dispatch_1d(count)
      end
    end
  end

  def snapshots : Hash(String, Array(Float32))
    result = {} of String => Array(Float32)
    {"img", "txt"}.each do |stream|
      rows = stream == "img" ? @config.image_rows : @config.text_rows
      {"modulate", "residual"}.each do |kind|
        {"native", "uncontracted"}.each do |mode|
          result["#{stream}.#{kind}.#{mode}"] = @arena["modulation_control.#{stream}.#{kind}.#{mode}"].read(rows * @config.dim)
        end
      end
    end
    result
  end
end
