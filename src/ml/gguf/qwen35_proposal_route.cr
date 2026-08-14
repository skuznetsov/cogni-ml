require "./qwen35_prompt_cache"
require "./qwen35_tokenizer"

module ML::GGUF
  # Shared identity and lookup helpers for proposal-route certificates.
  #
  # These certificates select a proposal body only. Exact verification remains
  # the correctness boundary, so callers must still fail closed when a route is
  # missing, stale, or incompatible with the active decode corridor.
  module Qwen35ProposalRoute
    extend self

    record Resolution,
      store : Qwen35PromptCache::Store,
      model_id : String,
      tokenizer_id : String,
      entry : Qwen35PromptCache::ProposalRouteEntry?

    def model_id(path : String) : String
      info = File.info(path)
      Qwen35PromptCache.short_hash("model\0#{path}\0#{info.size}\0#{info.modification_time.to_unix}")
    end

    def tokenizer_id(model_id : String, tok : Qwen35Tokenizer) : String
      Qwen35PromptCache.short_hash(
        "tokenizer\0#{model_id}\0#{tok.vocab.size}\0#{tok.eos_id}\0#{tok.pad_id}\0#{Qwen35Tokenizer::ENCODING_REVISION}"
      )
    end

    def resolve(root : String,
                model_path : String,
                tok : Qwen35Tokenizer,
                prompt_text : String,
                token_ids : Array(Int32),
                route_key : String? = nil) : Resolution
      model = model_id(model_path)
      tokenizer = tokenizer_id(model, tok)
      store = Qwen35PromptCache::Store.new(root)
      entry = if key = route_key
                store.lookup_proposal_route_key(model, tokenizer, key)
              else
                store.lookup_proposal_route(model, tokenizer, prompt_text, token_ids)
              end
      Resolution.new(store, model, tokenizer, entry)
    end
  end
end
