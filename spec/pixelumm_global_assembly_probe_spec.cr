require "spec"
require "../examples/support/pixelumm_global_assembly_probe"

describe PixelUMMGlobalAssemblyProbe do
  it "uses the admitted default T30 geometry and one global T60 window schedule" do
    default = ML::Video::PixelUMMGeneration::Config.new
    global = ML::Video::PixelUMMGeneration::Config.new(total_latent_frames: 60)

    default.total_latent_frames.should eq(30)
    default.window_starts.should eq([0, 5, 10, 15, 20])
    default.pixel_frames.should eq(120)
    global.window_starts.should eq((0..50).step(5).to_a)
    global.window_count.should eq(11)
    global.window_starts.includes?(25).should be_true
    global.pixel_frame_coverage_counts.all? { |count| count == 1 || count == 2 }.should be_true
  end

  it "keeps every synthetic BF16 prediction exactly representable" do
    (0..50).step(5) do |start|
      40.times do |local_frame|
        PixelUMMGlobalAssemblyProbe::CHANNELS.times do |channel|
          PixelUMMGlobalAssemblyProbe::SPATIAL_POSITIONS.each do |spatial|
            value = PixelUMMGlobalAssemblyProbe.fixture_value(start, local_frame, channel, spatial)
            PixelUMMGlobalAssemblyProbe.round_bf16(value).should eq(value)
          end
        end
      end
    end
  end

  it "distinguishes global seam blending from concatenating independent T30 outputs" do
    global_values = [] of Float32
    independent_values = [] of Float32
    (120...140).each do |frame|
      PixelUMMGlobalAssemblyProbe::CHANNELS.times do |channel|
        PixelUMMGlobalAssemblyProbe::SPATIAL_POSITIONS.each do |spatial|
          global_values << PixelUMMGlobalAssemblyProbe.global_display_value(frame, channel, spatial)
          independent_values << PixelUMMGlobalAssemblyProbe.independent_concat_display_value(frame, channel, spatial)
        end
      end
    end

    max_delta = global_values.zip(independent_values).max_of { |global, independent| (global - independent).abs }
    max_delta.should be > 2e-5_f32
  end

  it "returns interleaved global and independent samples for the Metal oracle" do
    values = PixelUMMGlobalAssemblyProbe.expected_samples
    values.size.should eq(240 * 3 * 3 * 2)

    first_seam_global = values[(120 * 3 * 3) * 2]
    first_seam_independent = values[(120 * 3 * 3) * 2 + 1]
    (first_seam_global - first_seam_independent).abs.should be > 2e-5_f32
  end
end
