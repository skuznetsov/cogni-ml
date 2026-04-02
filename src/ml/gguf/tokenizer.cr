# SentencePiece Unigram tokenizer — reads vocabulary from GGUF metadata.
#
# Implements Viterbi decoding: finds the token segmentation that
# maximizes the sum of log-probability scores.
#
# Compatible with T5/nomic-embed tokenizer format.

require "./reader"

module ML::GGUF
  class UnigramTokenizer
    getter vocab_size : Int32
    getter bos_id : Int32
    getter eos_id : Int32
    getter unk_id : Int32
    getter pad_id : Int32
    getter add_bos : Bool
    getter add_eos : Bool
    getter add_space_prefix : Bool

    # Token data
    @tokens : Array(String)
    @scores : Array(Float32)
    @types : Array(Int32)        # 1=normal, 2=unknown, 3=control
    @token_to_id : Hash(String, Int32)
    @trie : Array(TrieNode)

    # For Viterbi: max token length to bound the inner loop
    @max_token_len : Int32

    private class TrieNode
      property token_id : Int32
      getter children : Hash(UInt8, Int32)

      def initialize(@token_id : Int32 = -1)
        @children = {} of UInt8 => Int32
      end
    end

    def initialize(gguf : GGUFFile)
      @bos_id = (gguf.get_int("tokenizer.ggml.bos_token_id") || 0).to_i32
      @eos_id = (gguf.get_int("tokenizer.ggml.eos_token_id") || 2).to_i32
      @unk_id = (gguf.get_int("tokenizer.ggml.unknown_token_id") || 3).to_i32
      @pad_id = (gguf.get_int("tokenizer.ggml.padding_token_id") || 1).to_i32
      @add_bos = gguf.metadata["tokenizer.ggml.add_bos_token"]?.as?(Bool) || true
      @add_eos = gguf.metadata["tokenizer.ggml.add_eos_token"]?.as?(Bool) || true
      @add_space_prefix = gguf.metadata["tokenizer.ggml.add_space_prefix"]?.as?(Bool) || true

      # Load vocab
      tokens_arr = gguf.metadata["tokenizer.ggml.tokens"]?.as?(Array(Value)) || [] of Value
      scores_arr = gguf.metadata["tokenizer.ggml.scores"]?.as?(Array(Value)) || [] of Value
      types_arr = gguf.metadata["tokenizer.ggml.token_type"]?.as?(Array(Value)) || [] of Value

      @tokens = tokens_arr.map { |v| v.as?(String) || "" }
      @scores = scores_arr.map { |v| v.as?(Float32) || 0.0_f32 }
      @types = types_arr.map { |v| (v.as?(Int32) || v.as?(UInt32).try(&.to_i32) || 0) }
      @vocab_size = @tokens.size

      # Build lookup (only normal tokens, not control/unknown)
      @token_to_id = {} of String => Int32
      @trie = [TrieNode.new]
      @max_token_len = 0
      @tokens.each_with_index do |tok, i|
        next if tok.empty?
        next if @types[i]? == 3  # Skip control tokens
        @token_to_id[tok] = i.to_i32
        @max_token_len = Math.max(@max_token_len, tok.bytesize)
        trie_insert(tok.to_slice, i.to_i32)
      end
    end

    # Tokenize text → array of token IDs
    def encode(text : String) : Array(Int32)
      # Normalize: replace whitespace runs with single space, prepend ▁
      normalized = normalize(text)
      return wrap_special([] of Int32) if normalized.empty?

      # Viterbi segmentation
      ids = viterbi(normalized)
      wrap_special(ids)
    end

    private def normalize(text : String) : String
      # SentencePiece: replace spaces with ▁, add prefix ▁
      result = String.build do |s|
        s << '▁' if @add_space_prefix
        prev_space = false
        text.each_char do |ch|
          if ch.whitespace?
            unless prev_space
              s << '▁'
              prev_space = true
            end
          else
            s << ch
            prev_space = false
          end
        end
      end
      result
    end

    private def wrap_special(ids : Array(Int32)) : Array(Int32)
      result = [] of Int32
      result << @bos_id if @add_bos
      result.concat(ids)
      result << @eos_id if @add_eos
      result
    end

    # Viterbi algorithm: find optimal segmentation maximizing Σ scores
    #
    # best_score[i] = max score to reach position i in the string
    # best_len[i]   = length of the token that ends at position i
    #
    # We work on UTF-8 bytes for exact GGUF vocab matching.
    private def viterbi(text : String) : Array(Int32)
      bytes = text.to_slice
      n = bytes.size

      # DP arrays
      best_score = Array(Float64).new(n + 1, -Float64::MAX)
      best_len = Array(Int32).new(n + 1, 0)
      best_id = Array(Int32).new(n + 1, -1)
      best_score[0] = 0.0

      n.times do |i|
        next if best_score[i] == -Float64::MAX

        node_idx = 0
        max_len = Math.min(@max_token_len, n - i)
        len = 0
        while len < max_len
          next_idx = @trie[node_idx].children[bytes[i + len]]?
          break unless next_idx

          node_idx = next_idx
          len += 1
          token_id = @trie[node_idx].token_id
          if token_id >= 0
            score = best_score[i] + @scores[token_id].to_f64
            if score > best_score[i + len]
              best_score[i + len] = score
              best_len[i + len] = len
              best_id[i + len] = token_id
            end
          end
        end

        # Always allow single-byte fallback (UNK) to guarantee coverage
        if best_score[i] + (-100.0) > best_score[i + 1]
          # Check if a single UTF-8 char starts here
          char_len = utf8_char_len(bytes[i])
          if char_len > 0 && i + char_len <= n
            fallback_score = best_score[i] + (-100.0)
            if fallback_score > best_score[i + char_len]
              best_score[i + char_len] = fallback_score
              best_len[i + char_len] = char_len
              best_id[i + char_len] = @unk_id
            end
          end
        end
      end

      # Trace back
      ids = [] of Int32
      pos = n
      while pos > 0
        len = best_len[pos]
        id = best_id[pos]
        if len <= 0 || id < 0
          # Should not happen, but safety fallback
          pos -= 1
          ids.unshift(@unk_id)
          next
        end

        ids.unshift(id)
        pos -= len
      end

      ids
    end

    private def trie_insert(bytes : Bytes, token_id : Int32) : Nil
      node_idx = 0
      bytes.each do |byte|
        next_idx = @trie[node_idx].children[byte]?
        unless next_idx
          next_idx = @trie.size
          @trie[node_idx].children[byte] = next_idx
          @trie << TrieNode.new
        end
        node_idx = next_idx
      end
      @trie[node_idx].token_id = token_id
    end

    # Get the byte length of a UTF-8 character from its first byte
    private def utf8_char_len(byte : UInt8) : Int32
      return 1 if byte & 0x80 == 0
      return 2 if byte & 0xE0 == 0xC0
      return 3 if byte & 0xF0 == 0xE0
      return 4 if byte & 0xF8 == 0xF0
      1
    end
  end
end
