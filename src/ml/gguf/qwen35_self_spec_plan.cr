require "./qwen35_ffn_updown_adapter"
require "./qwen35_prompt_cache"
require "./qwen35_proposal_route"
require "./qwen35_tokenizer"

module ML::GGUF
  # Product-facing planning boundary for same-weight self-spec routes.
  #
  # This type validates route certificates and optional PCA-updown adapter
  # artifacts, but it intentionally does not execute approximate proposal bodies.
  # Exact verifier integration remains a separate runner boundary.
  module Qwen35SelfSpecPlan
    extend self

    STATUS_DISABLED        = "disabled"
    STATUS_ROUTE_MISS      = "route_miss"
    STATUS_BASELINE        = "baseline"
    STATUS_PCA_UPDOWN      = "pca_updown"
    STATUS_INVALID_ADAPTER = "invalid_adapter"
    STATUS_UNSUPPORTED     = "unsupported"

    record Result,
      status : String,
      route_resolution : Qwen35ProposalRoute::Resolution?,
      route_entry : Qwen35PromptCache::ProposalRouteEntry?,
      adapter_artifact : NamedTuple(source: String, hidden_dim: Int32, rank: Int32, adapters: Qwen35FFNUpDownAdapterMap)?,
      adapter_note : String,
      requested_rank : Int32?,
      requested_layers : Array(Int32),
      reason : String? do
      def route_hit? : Bool
        !@route_entry.nil?
      end

      def executable? : Bool
        @status == STATUS_BASELINE || @status == STATUS_PCA_UPDOWN
      end
    end

    def resolve(route_root : String?,
                model_path : String,
                tok : Qwen35Tokenizer,
                prompt_text : String,
                token_ids : Array(Int32),
                model_hidden_dim : Int32,
                route_key : String? = nil,
                adapter_path : String? = nil) : Result
      unless root = route_root
        return disabled("route memory not configured")
      end

      resolution = Qwen35ProposalRoute.resolve(root, model_path, tok, prompt_text, token_ids, route_key)
      unless entry = resolution.entry
        return Result.new(STATUS_ROUTE_MISS, resolution, nil, nil, "adapter_artifact=not_applicable", nil, [] of Int32, "route miss")
      end

      case entry.route
      when Qwen35PromptCache::PROPOSAL_ROUTE_BASELINE
        return Result.new(STATUS_BASELINE, resolution, entry, nil, "adapter_artifact=not_applicable", nil, entry.route_layers, nil)
      when Qwen35PromptCache::PROPOSAL_ROUTE_PCA_UPDOWN
        requested_rank = entry.route_rank || 0
        return invalid_adapter(resolution, entry, nil, requested_rank, entry.route_layers, "missing adapter path") unless adapter_path

        begin
          artifact = Qwen35FFNUpDownAdapterArtifact.load(adapter_path.not_nil!)
          requested_layers = entry.route_layers.empty? ? artifact[:adapters].keys.sort : entry.route_layers
          adapter_note = validate_adapter_artifact(artifact, model_hidden_dim, requested_rank, requested_layers)
          if adapter_note.starts_with?("adapter_artifact=valid")
            Result.new(STATUS_PCA_UPDOWN, resolution, entry, artifact, adapter_note, requested_rank, requested_layers, nil)
          else
            Result.new(STATUS_INVALID_ADAPTER, resolution, entry, artifact, adapter_note, requested_rank, requested_layers, adapter_note)
          end
        rescue ex
          invalid_adapter(resolution, entry, nil, requested_rank, entry.route_layers, ex.message || ex.class.name)
        end
      else
        Result.new(STATUS_UNSUPPORTED, resolution, entry, nil, "adapter_artifact=not_applicable", entry.route_rank, entry.route_layers, "unsupported route #{entry.route}")
      end
    end

    private def disabled(reason : String) : Result
      Result.new(STATUS_DISABLED, nil, nil, nil, "adapter_artifact=not_applicable", nil, [] of Int32, reason)
    end

    private def invalid_adapter(resolution,
                                entry,
                                artifact,
                                requested_rank : Int32?,
                                requested_layers : Array(Int32),
                                reason : String) : Result
      safe_reason = reason.gsub(/\s+/, "_")
      Result.new(STATUS_INVALID_ADAPTER, resolution, entry, artifact, "adapter_artifact=invalid reason=#{safe_reason}", requested_rank, requested_layers, reason)
    end

    private def validate_adapter_artifact(artifact,
                                          model_hidden_dim : Int32,
                                          requested_rank : Int32,
                                          requested_layers : Array(Int32)) : String
      if artifact[:hidden_dim] != model_hidden_dim
        return "adapter_artifact=invalid reason=hidden_dim expected=#{model_hidden_dim} actual=#{artifact[:hidden_dim]}"
      end
      if requested_rank <= 0
        return "adapter_artifact=invalid reason=rank requested_rank=#{requested_rank}"
      end
      missing = requested_layers.reject { |layer_id| artifact[:adapters].has_key?(layer_id) }
      return "adapter_artifact=invalid reason=missing_layers layers=#{missing.join(',')}" unless missing.empty?

      short_rank = requested_layers.select do |layer_id|
        adapter = artifact[:adapters][layer_id]?
        adapter && adapter.rank < requested_rank
      end
      unless short_rank.empty?
        return "adapter_artifact=invalid reason=rank_too_small layers=#{short_rank.join(',')} requested_rank=#{requested_rank}"
      end

      "adapter_artifact=valid source=#{artifact[:source]} hidden=#{artifact[:hidden_dim]} rank=#{artifact[:rank]} checked_layers=#{requested_layers.join(',')}"
    end
  end
end
