require "spec"
require "../examples/support/pixelumm_block_probe"

describe PixelUMMBlockProbe::Fixture do
  it "rejects a mixed-runtime or CPU preview oracle before reading its payload" do
    { {"2.8.0", true}, {"2.6.0", false} }.each do |version, mps_preflight|
      file = File.tempfile("pixelumm-preview-provenance", ".json")
      path = File.realpath(file.path)
      begin
        raw = {
          schema:                   "cogni_ml.pixelumm_block_reference/v1",
          upstream_revision:        PixelUMMBlockProbe::UPSTREAM_REVISION,
          checkpoint_subset_sha256: PixelUMMBlockProbe::SUBSET_SHA256,
          tile_tokens:              8,
          token_order:              "tile_major",
          grid:                     [10, 18, 32],
          torch_version:            version,
          memory_preflight:         {applies: mps_preflight},
          payload:                  "nonexistent-preview-payload.bin",
        }.to_json
        file.print(raw)
        file.close
        expect_raises(ArgumentError, /preview requires the Torch 2.6.0 MPS oracle/) do
          PixelUMMBlockProbe::Fixture.new(path, Digest::SHA256.hexdigest(raw))
        end
      ensure
        file.close unless file.closed?
        File.delete?(path)
      end
    end
  end
end
