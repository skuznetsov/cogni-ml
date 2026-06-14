require "./diffusion_gemma_cpu"

module ML::GGUF
  alias DiffusionGemmaPromptRouteMap = Array(Array(Array(DiffusionGemmaCPU::ExpertRoute)))

  class DiffusionGemmaPromptRouteArtifact
    FORMAT_V2 = "diffusion_gemma_prompt_route_artifact_v2"
    FORMAT_V1 = "diffusion_gemma_prompt_route_artifact_v1"

    getter routes : DiffusionGemmaPromptRouteMap
    getter elapsed_ms : Float64
    getter route_rows : Int32
    getter route_slots : Int32
    getter checksum : Float64
    getter source : String
    getter artifact_path : String

    def initialize(@routes : DiffusionGemmaPromptRouteMap,
                   @elapsed_ms : Float64,
                   @route_rows : Int32,
                   @route_slots : Int32,
                   @checksum : Float64,
                   @artifact_path : String,
                   @source : String = "artifact")
    end

    def self.load(path : String,
                  arm : String,
                  expected_prompt_len : Int32,
                  expected_max_layers : Int32,
                  expected_prompt_tokens_sha256 : String,
                  expected_arm_env_sha256 : String,
                  expected_model_sha256 : String,
                  print_diagnostic : Bool = true) : DiffusionGemmaPromptRouteArtifact
      t0 = Time.instant
      metadata = {} of String => String
      routes = DiffusionGemmaPromptRouteMap.new(expected_max_layers) do
        Array(Array(DiffusionGemmaCPU::ExpertRoute)).new(expected_prompt_len) { [] of DiffusionGemmaCPU::ExpertRoute }
      end

      File.each_line(path) do |line|
        next if line.empty?
        if line.starts_with?("#")
          comment = line[1, line.bytesize - 1].strip
          if comment.includes?("=")
            key, value = comment.split("=", 2)
            metadata[key] = value if value
          end
          next
        end

        parts = line.split('\t')
        next if parts[0]? == "kind"
        raise "route artifact row must have 6 fields, got #{parts.size}" unless parts.size == 6
        raise "route artifact row kind must be route, got #{parts[0].inspect}" unless parts[0] == "route"
        layer = parts[1].to_i
        row = parts[2].to_i
        route_index = parts[3].to_i
        expert = parts[4].to_i
        weight = parts[5].to_f32
        raise "route artifact layer out of range: #{layer}" if layer < 0 || layer >= expected_max_layers
        raise "route artifact row out of range: #{row}" if row < 0 || row >= expected_prompt_len
        row_routes = routes[layer][row]
        if route_index != row_routes.size
          raise "route artifact route_index mismatch at layer=#{layer} row=#{row}: #{route_index} != #{row_routes.size}"
        end
        row_routes << DiffusionGemmaCPU::ExpertRoute.new(expert.to_i32, weight)
      end

      validate_metadata!(
        metadata,
        arm,
        expected_prompt_len,
        expected_max_layers,
        expected_prompt_tokens_sha256,
        expected_arm_env_sha256,
        expected_model_sha256
      )
      routes.each_with_index do |layer_routes, layer|
        layer_routes.each_with_index do |row_routes, row|
          raise "route artifact missing routes at layer=#{layer} row=#{row}" if row_routes.empty?
        end
      end

      route_rows = routes.sum(&.size)
      route_slots = routes.sum { |layer| layer.sum(&.size) }
      validate_count_metadata!(metadata, expected_max_layers, route_rows, route_slots)
      checksum = metadata["checksum"]?.try(&.to_f64) || 0.0
      elapsed_ms = (Time.instant - t0).total_milliseconds
      artifact = DiffusionGemmaPromptRouteArtifact.new(
        routes,
        elapsed_ms,
        route_rows,
        route_slots,
        checksum,
        path
      )
      artifact.print_load_diagnostic(arm) if print_diagnostic
      artifact
    end

    def print_load_diagnostic(arm : String) : Nil
      puts [
        "# route_artifact_load",
        arm,
        "path=#{@artifact_path}",
        "layers=#{@routes.size}",
        "rows=#{@route_rows}",
        "slots=#{@route_slots}",
        "load_ms=#{self.class.format_f64(@elapsed_ms)}",
        "artifact_checksum=#{self.class.format_f64(@checksum)}",
      ].join('\t')
    end

    private def self.validate_metadata!(metadata : Hash(String, String),
                                        arm : String,
                                        expected_prompt_len : Int32,
                                        expected_max_layers : Int32,
                                        expected_prompt_tokens_sha256 : String,
                                        expected_arm_env_sha256 : String,
                                        expected_model_sha256 : String) : Nil
      case metadata["format"]?
      when FORMAT_V2
      when FORMAT_V1
        raise "route artifact v1 is missing prompt/env/model boundary metadata; regenerate artifact"
      else
        raise "route artifact format mismatch"
      end
      raise "route artifact arm mismatch: #{metadata["arm"]?} != #{arm}" unless metadata["arm"]? == arm
      raise "route artifact prompt_len mismatch" unless metadata["prompt_len"]?.try(&.to_i) == expected_prompt_len
      raise "route artifact max_layers mismatch" unless metadata["max_layers"]?.try(&.to_i) == expected_max_layers
      unless metadata["prompt_tokens_sha256"]? == expected_prompt_tokens_sha256
        raise "route artifact prompt_tokens_sha256 mismatch"
      end
      unless metadata["arm_env_sha256"]? == expected_arm_env_sha256
        raise "route artifact arm_env_sha256 mismatch"
      end
      unless metadata["model_fingerprint"]? == expected_model_sha256
        raise "route artifact model_fingerprint mismatch"
      end
    end

    private def self.validate_count_metadata!(metadata : Hash(String, String),
                                              expected_max_layers : Int32,
                                              route_rows : Int32,
                                              route_slots : Int32) : Nil
      if layers = metadata["layers"]?
        raise "route artifact layers mismatch" unless layers.to_i == expected_max_layers
      end
      if rows = metadata["rows"]?
        raise "route artifact rows mismatch" unless rows.to_i == route_rows
      end
      if slots = metadata["slots"]?
        raise "route artifact slots mismatch" unless slots.to_i == route_slots
      end
    end

    def self.format_f64(value : Float64) : String
      "%.6f" % value
    end
  end
end
