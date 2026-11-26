require "./hy_nomas_decoder_replay"

# Authenticated fixtures shared only by opt-in saved-latent diagnostics.
module PixelUMMAfterDiTProbe
  extend self
  INPUTS            = "/opt/mldata_ssd/cogni-ml/hy-nomas-geometry-20261030/input-prep-v2/inputs.json"
  INPUT_SHA         = "dcbe51749742701a2ee1a6d09a10e2f20e663f3ff54f646d801c743e17c7a104"
  LATENT            = "/opt/mldata_ssd/cogni-ml/hy-nomas-quality-next-20261031/cinematic-v1/native-v1/latent.f32le"
  LATENT_SHA        = "ea79dc6896f3e956035acd0caa69b1730554d5459d80893989023493de4b5bba"
  NOISE_SHA         = "166ea19d2812c0b4dedfedefd4f101b2abe8b4ec48364fb63705f781e1828547"
  CINEMATIC         = "/opt/mldata_ssd/cogni-ml/hy-nomas-quality-next-20261031/cinematic-v1/cinematic-cache-v1/cinematic-lineage.json"
  CINEMATIC_SHA     = "fb16e1322bc1ae294a381c4973af39ebb02fdd3405305c80b31c934f776625a8"
  ORACLE_REPORT     = "/opt/mldata_ssd/cogni-ml/hy-nomas-quality-20261030/results/cinematic-20261031-v1/decoder-source.json"
  ORACLE_REPORT_SHA = "a91ad030b8cd682f4ca6c00ebfad41c4a3ea56a9bcff41155318f6fb12d2ea7e"
  ORACLE_PIXELS     = "/opt/mldata_ssd/cogni-ml/hy-nomas-quality-20261030/results/cinematic-20261031-v1/decoder-source-display.f32le"
  ORACLE_SHA        = "5ed0974dfd3b28d940c370fa8faef6a23b288d86adb03c6810758ae16c55cc9f"
  BUDGET            = 24_i64 * HyNomasGeneration::GIB

  def authenticate!(path : String, sha : String) : String
    canonical = File.realpath(path)
    raise ArgumentError.new("probe fixture must be a regular file") unless File.file?(canonical) && !File.symlink?(canonical)
    raise ArgumentError.new("probe fixture SHA-256 mismatch: #{canonical}") unless PixelUMMBlockProbe.sha_file(canonical) == sha
    canonical
  end

  def export!(path : String, buffer : ML::MetalBuffer)
    HyNomasGeneration.write_fresh_output(path) do |f|
      f.write(Bytes.new(buffer.contents.as(UInt8*), buffer.size.to_i32))
    end
    sha = PixelUMMBlockProbe.sha_file(path)
    raise "decoder display output differs from the pinned upstream oracle" unless sha == ORACLE_SHA
    {path: path, sha256: sha, upstream_bitwise_equal: true}
  end
end
