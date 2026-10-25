require "digest/sha256"
require "json"
require "./qwen35_tokenizer"

module ML::GGUF
  # CPU-only input preparation for the pinned Hy-Nomas Qwen2.5-VL video path.
  # This intentionally supports one tokenizer schema and one chat template;
  # it is not a general Hugging Face tokenizer or Jinja implementation.
  class Qwen25HyNomasPrompt
    MODEL_ID                    = "Qwen/Qwen2.5-VL-7B-Instruct"
    MODEL_REVISION              = "eb6ba55b40ada64977147b52e50a4145ee89ac4f"
    TOKENIZER_SHA256            = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
    TOKENIZER_CONFIG_SHA256     = "4abd3520120e266da84c0864fee064d1fb10806f02225911a47253dd38dc5f56"
    TEMPLATE_NAME               = "li-dit-encode-video-json"
    TEMPLATE_SHA256             = "3e7b69bdf2a057bf2d6a8322dfb8ad619e686e4e8240b65f29e94492bee0c4a6"
    UPSTREAM_REVISION           = "911cea6458a89c2663513ff7099561ec190dd491"
    UPSTREAM_ENCODER_SHA256     = "aa84b366d6e0a5cf9c73e591e6827746e8cab97852a112d1b4783dc7c2d507f1"
    UPSTREAM_CONSTANTS_SHA256   = "b4ab0055d0a2fade687f012f0e305915f5305df58dc82e2ad8bde25ef4236af9"
    BASE_SEQUENCE_LENGTH        = 1280
    MAX_CROP_START              =  256
    MAX_RAW_SEQUENCE_LENGTH     = BASE_SEQUENCE_LENGTH + MAX_CROP_START
    MAX_PROMPT_BYTES            = 64 * 1024
    EXPECTED_CROP_START         = 108
    PAD_TOKEN                   = "<|endoftext|>"
    EOS_TOKEN                   = "<|im_end|>"
    USER_MARKER                 = "<|im_start|>user\n"
    USER_MARKER_IDS             = [151644_i32, 872_i32, 198_i32]
    FORBIDDEN_MULTIMODAL_TOKENS = [
      "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    ]
    FORBIDDEN_MULTIMODAL_IDS = {
      "<|vision_start|>" => 151652_i32,
      "<|vision_end|>"   => 151653_i32,
      "<|vision_pad|>"   => 151654_i32,
      "<|image_pad|>"    => 151655_i32,
      "<|video_pad|>"    => 151656_i32,
    }

    SYSTEM_PROMPT = "You are a helpful assistant. Describe the video by detailing the following aspects:         1. The main content and theme of the video.         2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.         3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.         4. background environment, light, style and atmosphere.         5. camera angles, movements, and transitions used in the video."

    # Exact Split regex from the pinned tokenizer.json. In particular, this
    # Qwen2.5 pattern does not absorb combining marks into letter runs.
    QWEN25_PRETOKENIZER = /(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/

    class Prepared
      getter prompt : String
      getter prompt_sha256 : String
      getter rendered_prompt : String
      getter rendered_prompt_sha256 : String
      getter crop_start : Int32
      getter raw_length : Int32
      getter text_length : Int32
      getter template_name : String
      getter tokenizer_json_sha256 : String
      getter tokenizer_config_sha256 : String

      @tokens : Array(Int32)
      @attention_mask : Array(Int32)
      @positions : Array(Float32)
      @cropped_tokens : Array(Int32)
      @cropped_attention_mask : Array(Int32)
      @marker_token_ids : Array(Int32)
      @messages : Array(NamedTuple(role: String, content: String))

      def initialize(@prompt : String, @prompt_sha256 : String,
                     @rendered_prompt : String, @rendered_prompt_sha256 : String,
                     @crop_start : Int32, @raw_length : Int32,
                     @tokens : Array(Int32), @attention_mask : Array(Int32),
                     @positions : Array(Float32), @cropped_tokens : Array(Int32),
                     @cropped_attention_mask : Array(Int32), @marker_token_ids : Array(Int32),
                     @tokenizer_json_sha256 : String, @tokenizer_config_sha256 : String,
                     @messages : Array(NamedTuple(role: String, content: String)))
        @text_length = BASE_SEQUENCE_LENGTH.to_i32
        @template_name = TEMPLATE_NAME
      end

      # Return copies so callers cannot mutate a plan after admission.
      def tokens : Array(Int32)
        @tokens.dup
      end

      def attention_mask : Array(Int32)
        @attention_mask.dup
      end

      # The text model uses identical cache positions on all three RoPE axes.
      # Keep one axis here; callers may upload this single compact buffer.
      def positions : Array(Float32)
        @positions.dup
      end

      def cropped_tokens : Array(Int32)
        @cropped_tokens.dup
      end

      def cropped_attention_mask : Array(Int32)
        @cropped_attention_mask.dup
      end

      def marker_token_ids : Array(Int32)
        @marker_token_ids.dup
      end

      def messages : Array(NamedTuple(role: String, content: String))
        @messages.dup
      end

      def forbidden_multimodal_token_ids : Hash(String, Int32)
        FORBIDDEN_MULTIMODAL_IDS.dup
      end
    end

    private class Tokenizer
      getter bpe : Qwen35Tokenizer
      getter pad_id : Int32
      getter eos_id : Int32
      getter tokenizer_sha256 : String
      getter config_sha256 : String

      def initialize(root : String)
        tokenizer_path, tokenizer_json, @tokenizer_sha256 = self.class.read_pinned_asset(
          root, "tokenizer.json", TOKENIZER_SHA256,
        )
        config_path, config_json, @config_sha256 = self.class.read_pinned_asset(
          root, "tokenizer_config.json", TOKENIZER_CONFIG_SHA256,
        )
        @bpe, @pad_id, @eos_id = self.class.parse_assets(tokenizer_path, tokenizer_json, config_path, config_json)
      end

      def encode(text : String) : Array(Int32)
        if ENV["QWEN35_NATIVE_TOKENIZER_OFF"]? == "1"
          raise ArgumentError.new("Qwen2.5 prompt preparation requires the native BPE encoder")
        end
        @bpe.encode(text.unicode_normalize(:nfc), add_bos_override: false)
      end

      def self.read_pinned_asset(root : String, name : String, expected_sha256 : String) : Tuple(String, String, String)
        path = File.join(root, name)
        real_path = begin
          File.realpath(path)
        rescue ex
          raise ArgumentError.new("Qwen2.5 tokenizer asset #{name} is unavailable: #{ex.message}")
        end
        unless real_path == path && File.file?(real_path)
          raise ArgumentError.new("Qwen2.5 tokenizer asset #{name} must be a regular file directly inside the checkpoint root")
        end

        bytes = File.read(real_path)
        actual_sha256 = Digest::SHA256.hexdigest(bytes)
        unless actual_sha256 == expected_sha256
          raise ArgumentError.new("Qwen2.5 tokenizer asset #{name} digest does not match the pinned revision")
        end
        {real_path, bytes, actual_sha256}
      end

      def self.parse_assets(tokenizer_path : String, tokenizer_json : String,
                            config_path : String, config_json : String) : Tuple(Qwen35Tokenizer, Int32, Int32)
        root = JSON.parse(tokenizer_json)
        config = JSON.parse(config_json)
        validate_tokenizer_config!(config, config_path)
        validate_tokenizer_schema!(root, tokenizer_path)

        model = field(root, "model", tokenizer_path)
        vocab_json = field(model, "vocab", tokenizer_path).as_h
        merges_json = field(model, "merges", tokenizer_path).as_a
        added_json = field(root, "added_tokens", tokenizer_path).as_a

        raise ArgumentError.new("Qwen2.5 tokenizer vocabulary size changed") unless vocab_json.size == 151_643
        raise ArgumentError.new("Qwen2.5 tokenizer merge count changed") unless merges_json.size == 151_387
        raise ArgumentError.new("Qwen2.5 added-token inventory changed") unless added_json.size == 22

        entries = Array(String?).new(151_665, nil)
        vocab_json.each do |piece, raw_id|
          assign_token!(entries, int_value(raw_id, "model.vocab.#{piece}").to_i, piece, "model.vocab")
        end

        token_types = Array(Int32).new(entries.size, 0_i32)
        added_json.each_with_index do |raw_entry, index|
          entry = raw_entry
          id = int_value(field(entry, "id", "added_tokens[#{index}]"), "added_tokens[#{index}].id").to_i
          content = string_value(field(entry, "content", "added_tokens[#{index}]"), "added_tokens[#{index}].content")
          validate_added_token!(entry, index)
          assign_token!(entries, id, content, "added_tokens[#{index}]")
          token_types[id] = 3_i32
        end

        vocab = entries.map_with_index do |piece, id|
          piece || raise ArgumentError.new("Qwen2.5 tokenizer vocabulary has a gap at id #{id}")
        end
        token_to_id = {} of String => Int32
        vocab.each_with_index do |piece, id|
          raise ArgumentError.new("Qwen2.5 tokenizer has duplicate vocabulary piece #{piece.inspect}") if token_to_id.has_key?(piece)
          token_to_id[piece] = id.to_i32
        end

        bpe_ranks = {} of Tuple(String, String) => Int32
        merges_json.each_with_index do |raw_merge, rank|
          merge = string_value(raw_merge, "model.merges[#{rank}]")
          parts = merge.split(' ', limit: 2)
          unless parts.size == 2 && !parts[0].empty? && !parts[1].empty?
            raise ArgumentError.new("Qwen2.5 tokenizer has malformed BPE merge at rank #{rank}")
          end
          pair = {parts[0], parts[1]}
          raise ArgumentError.new("Qwen2.5 tokenizer has duplicate BPE merge at rank #{rank}") if bpe_ranks.has_key?(pair)
          bpe_ranks[pair] = rank.to_i32
        end

        pad_id = token_to_id[PAD_TOKEN]?
        eos_id = token_to_id[EOS_TOKEN]?
        raise ArgumentError.new("Qwen2.5 tokenizer is missing its padding token") unless pad_id == 151_643
        raise ArgumentError.new("Qwen2.5 tokenizer is missing its end-of-message token") unless eos_id == 151_645
        pad_id = pad_id.not_nil!
        eos_id = eos_id.not_nil!

        tokenizer = Qwen35Tokenizer.new(
          vocab,
          eos_id,
          pad_id,
          false,
          tokenizer_path,
          token_to_id: token_to_id,
          bpe_ranks: bpe_ranks,
          token_types: token_types,
          pretokenizer: QWEN25_PRETOKENIZER,
        )
        {tokenizer, pad_id, eos_id}
      rescue ex : JSON::ParseException
        raise ArgumentError.new("Qwen2.5 tokenizer JSON is malformed: #{ex.message}")
      rescue ex : KeyError | TypeCastError
        raise ArgumentError.new("Qwen2.5 tokenizer schema is malformed: #{ex.message}")
      end

      private def self.validate_tokenizer_config!(config : JSON::Any, path : String) : Nil
        unless bool_value(field(config, "add_bos_token", path), "add_bos_token", path) == false &&
               bool_value(field(config, "add_prefix_space", path), "add_prefix_space", path) == false &&
               bool_value(field(config, "split_special_tokens", path), "split_special_tokens", path) == false
          raise ArgumentError.new("Qwen2.5 tokenizer configuration enables an unsupported tokenization option")
        end

        pad = string_value(field(config, "pad_token", path), "pad_token")
        eos = string_value(field(config, "eos_token", path), "eos_token")
        raise ArgumentError.new("Qwen2.5 tokenizer padding token changed") unless pad == PAD_TOKEN
        raise ArgumentError.new("Qwen2.5 tokenizer EOS token changed") unless eos == EOS_TOKEN

        if side = config.as_h["padding_side"]?
          raise ArgumentError.new("Qwen2.5 tokenizer must use right padding") unless string_value(side, "padding_side") == "right"
        end
      end

      private def self.validate_tokenizer_schema!(root : JSON::Any, path : String) : Nil
        normalizer = field(root, "normalizer", path)
        raise ArgumentError.new("Qwen2.5 tokenizer normalizer must be NFC") unless string_value(field(normalizer, "type", path), "normalizer.type") == "NFC"

        model = field(root, "model", path)
        raise ArgumentError.new("Qwen2.5 tokenizer model must be BPE") unless string_value(field(model, "type", path), "model.type") == "BPE"
        unless field(model, "dropout", path).raw.nil? && field(model, "unk_token", path).raw.nil? &&
               string_value(field(model, "continuing_subword_prefix", path), "model.continuing_subword_prefix").empty? &&
               string_value(field(model, "end_of_word_suffix", path), "model.end_of_word_suffix").empty? &&
               bool_value(field(model, "fuse_unk", path), "model.fuse_unk", path) == false &&
               bool_value(field(model, "byte_fallback", path), "model.byte_fallback", path) == false
          raise ArgumentError.new("Qwen2.5 tokenizer BPE options changed")
        end

        pre_tokenizer = field(root, "pre_tokenizer", path)
        raise ArgumentError.new("Qwen2.5 tokenizer pretokenizer must be a two-stage sequence") unless string_value(field(pre_tokenizer, "type", path), "pre_tokenizer.type") == "Sequence"
        stages = field(pre_tokenizer, "pretokenizers", path).as_a
        raise ArgumentError.new("Qwen2.5 tokenizer pretokenizer stage count changed") unless stages.size == 2

        split = stages[0]
        raise ArgumentError.new("Qwen2.5 tokenizer first pretokenizer must be isolated regex Split") unless string_value(field(split, "type", path), "pre_tokenizer[0].type") == "Split" &&
                                                                                                            string_value(field(split, "behavior", path), "pre_tokenizer[0].behavior") == "Isolated" &&
                                                                                                            bool_value(field(split, "invert", path), "pre_tokenizer[0].invert", path) == false
        pattern = field(split, "pattern", path)
        pattern_text = string_value(field(pattern, "Regex", path), "pre_tokenizer[0].pattern.Regex")
        expected_pattern = QWEN25_PRETOKENIZER.source
        raise ArgumentError.new("Qwen2.5 tokenizer regex differs from the native implementation") unless pattern_text == expected_pattern

        byte_level = stages[1]
        unless string_value(field(byte_level, "type", path), "pre_tokenizer[1].type") == "ByteLevel" &&
               bool_value(field(byte_level, "add_prefix_space", path), "pre_tokenizer[1].add_prefix_space", path) == false &&
               bool_value(field(byte_level, "trim_offsets", path), "pre_tokenizer[1].trim_offsets", path) == false &&
               bool_value(field(byte_level, "use_regex", path), "pre_tokenizer[1].use_regex", path) == false
          raise ArgumentError.new("Qwen2.5 tokenizer ByteLevel options changed")
        end
      end

      private def self.validate_added_token!(entry : JSON::Any, index : Int32) : Nil
        %w[single_word lstrip rstrip normalized].each do |key|
          unless bool_value(field(entry, key, "added_tokens[#{index}]"), "added_tokens[#{index}].#{key}", "added_tokens[#{index}]") == false
            raise ArgumentError.new("Qwen2.5 tokenizer added token #{index} has unsupported #{key} behavior")
          end
        end
        bool_value(field(entry, "special", "added_tokens[#{index}]"), "added_tokens[#{index}].special", "added_tokens[#{index}]")
      end

      private def self.assign_token!(entries : Array(String?), id : Int32, token : String, location : String) : Nil
        unless 0 <= id < entries.size && !token.empty?
          raise ArgumentError.new("Qwen2.5 tokenizer #{location} has an invalid token id or empty piece")
        end
        raise ArgumentError.new("Qwen2.5 tokenizer has conflicting ids at #{location}: #{id}") if entries[id]
        entries[id] = token
      end

      private def self.field(object : JSON::Any, key : String, location : String) : JSON::Any
        hash = object.as_h?
        raise ArgumentError.new("Qwen2.5 tokenizer #{location} must be an object") unless hash
        hash[key]? || raise ArgumentError.new("Qwen2.5 tokenizer is missing #{location}.#{key}")
      end

      private def self.string_value(value : JSON::Any, location : String) : String
        value.as_s? || raise ArgumentError.new("Qwen2.5 tokenizer #{location} must be a string")
      end

      private def self.int_value(value : JSON::Any, location : String) : Int64
        (value.as_i? || raise ArgumentError.new("Qwen2.5 tokenizer #{location} must be an integer")).to_i64
      end

      private def self.bool_value(value : JSON::Any, location : String, _path : String) : Bool
        raw = value.raw
        raw.is_a?(Bool) ? raw : raise ArgumentError.new("Qwen2.5 tokenizer #{location} must be boolean")
      end
    end

    @tokenizer : Tokenizer
    @crop_start : Int32

    def initialize(checkpoint_root : String)
      root = begin
        File.realpath(checkpoint_root)
      rescue ex
        raise ArgumentError.new("Qwen2.5 checkpoint root is unavailable: #{ex.message}")
      end
      raise ArgumentError.new("Qwen2.5 checkpoint root must be a directory") unless Dir.exists?(root)

      @tokenizer = Tokenizer.new(root)
      @crop_start = derive_crop_start
      unless @crop_start == EXPECTED_CROP_START
        raise ArgumentError.new("Qwen2.5 Hy-Nomas video crop changed: expected #{EXPECTED_CROP_START}, got #{@crop_start}")
      end
    end

    def prepare(prompt : String) : Prepared
      raise ArgumentError.new("Qwen2.5 natural prompt exceeds 64 KiB") if prompt.bytesize > MAX_PROMPT_BYTES
      raise ArgumentError.new("Qwen2.5 natural prompt must not contain NUL bytes") if prompt.includes?('\0')

      user_content = prompt.empty? ? " " : prompt
      rendered = render_prompt(user_content)
      ids = @tokenizer.encode(rendered)
      marker_ids = @tokenizer.encode(USER_MARKER)
      unless marker_ids == USER_MARKER_IDS
        raise ArgumentError.new("Qwen2.5 tokenizer user-header marker IDs changed")
      end

      raw_length = BASE_SEQUENCE_LENGTH + @crop_start
      token_ids = Array(Int32).new(raw_length)
      copied_count = Math.min(ids.size, raw_length)
      token_ids.concat(ids.first(copied_count))
      visible_count = copied_count
      if copied_count < raw_length
        token_ids.concat(Array(Int32).new(raw_length - copied_count, @tokenizer.pad_id))
      end
      mask = Array(Int32).new(raw_length) { |index| index < visible_count ? 1_i32 : 0_i32 }

      marker_start = @crop_start - marker_ids.size
      unless token_ids[marker_start, marker_ids.size] == marker_ids
        raise ArgumentError.new("Qwen2.5 raw prompt is missing the user marker at crop_start")
      end
      unless find_subsequence(token_ids, marker_ids) == marker_start
        raise ArgumentError.new("Qwen2.5 raw prompt user marker is not its first exact occurrence")
      end
      first_padding = mask.index(0_i32) || mask.size
      if mask[first_padding..].any? { |value| value != 0 }
        raise ArgumentError.new("Qwen2.5 tokenizer output is not right padded")
      end
      unless mask[@crop_start] == 1
        raise ArgumentError.new("Qwen2.5 cropped prompt has no visible first token")
      end

      forbidden_ids = FORBIDDEN_MULTIMODAL_IDS.values
      if token_ids.any? { |id| forbidden_ids.includes?(id) }
        raise ArgumentError.new("Qwen2.5 text-only prompt contains a multimodal token")
      end

      cropped_tokens = token_ids[@crop_start, BASE_SEQUENCE_LENGTH]
      cropped_mask = mask[@crop_start, BASE_SEQUENCE_LENGTH]
      unless cropped_tokens.size == BASE_SEQUENCE_LENGTH && cropped_mask.size == BASE_SEQUENCE_LENGTH
        raise ArgumentError.new("Qwen2.5 suffix crop has an invalid width")
      end

      normalized_prompt = prompt
      Prepared.new(
        normalized_prompt,
        Digest::SHA256.hexdigest(normalized_prompt),
        rendered,
        Digest::SHA256.hexdigest(rendered),
        @crop_start,
        raw_length.to_i32,
        token_ids,
        mask,
        Array(Float32).new(raw_length) { |index| index.to_f32 },
        cropped_tokens,
        cropped_mask,
        marker_ids,
        @tokenizer.tokenizer_sha256,
        @tokenizer.config_sha256,
        [
          {role: "system", content: SYSTEM_PROMPT},
          {role: "user", content: user_content},
        ],
      )
    end

    private def derive_crop_start : Int32
      prepass = @tokenizer.encode(render_prompt("a photo of a cat"))
      visible = prepass.first(Math.min(prepass.size, 256))
      marker_ids = @tokenizer.encode(USER_MARKER)
      start = find_subsequence(visible, marker_ids)
      raise ArgumentError.new("Qwen2.5 tokenizer could not find the user marker in the 256-token crop prepass") unless start

      crop_start = start + marker_ids.size
      unless 0 < crop_start <= MAX_CROP_START
        raise ArgumentError.new("Qwen2.5 dynamic crop is outside the supported 1...256 range")
      end
      crop_start.to_i32
    end

    private def render_prompt(user_content : String) : String
      "<|im_start|>system\n#{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n#{user_content}<|im_end|>\n<|im_start|>assistant\n"
    end

    private def find_subsequence(values : Array(Int32), pattern : Array(Int32)) : Int32?
      return nil if pattern.empty? || values.size < pattern.size
      (0..(values.size - pattern.size)).each do |start|
        return start.to_i32 if values[start, pattern.size] == pattern
      end
      nil
    end
  end
end
