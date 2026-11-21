# Shared GPT-2 byte-level encoding and BPE merge logic.
module ML::GGUF::GPT2BPE
  @@byte_encoder : Hash(UInt8, Char)?
  @@byte_decoder : Hash(Char, UInt8)?

  # GPT-2 maps every byte to a printable Unicode scalar. Printable ASCII and
  # Latin-1 bytes keep their code points; the remaining bytes map above U+00FF.
  def self.build_maps : Nil
    bs = [] of UInt8
    cs = [] of Int32
    (0x21_u8..0x7E_u8).each { |byte| bs << byte; cs << byte.to_i32 }
    (0xA1_u8..0xAC_u8).each { |byte| bs << byte; cs << byte.to_i32 }
    (0xAE_u8..0xFF_u8).each { |byte| bs << byte; cs << byte.to_i32 }

    offset = 0
    (0..255).each do |byte|
      value = byte.to_u8
      unless bs.includes?(value)
        bs << value
        cs << 256 + offset
        offset += 1
      end
    end

    encoder = Hash(UInt8, Char).new
    decoder = Hash(Char, UInt8).new
    bs.each_with_index do |byte, index|
      character = cs[index].chr
      encoder[byte] = character
      decoder[character] = byte
    end
    @@byte_encoder = encoder
    @@byte_decoder = decoder
  end

  def self.char_for(byte : UInt8) : Char
    build_maps if @@byte_encoder.nil?
    @@byte_encoder.not_nil![byte]
  end

  def self.byte_for(character : Char) : UInt8?
    build_maps if @@byte_decoder.nil?
    @@byte_decoder.not_nil![character]?
  end

  def self.encode_bytes(text : String) : String
    String.build do |io|
      text.to_slice.each { |byte| io << char_for(byte) }
    end
  end

  def self.decode_piece(piece : String, bytes : Array(UInt8)) : Nil
    piece.each_char do |character|
      byte_for(character).try { |byte| bytes << byte }
    end
  end

  # GPT-2 BPE repeatedly merges the adjacent pair with the smallest rank.
  # Equal-ranked occurrences are merged left-to-right in a single pass.
  def self.bpe(piece : String,
               token_to_id : Hash(String, Int32),
               ranks : Hash(Tuple(String, String), Int32)) : Array(String)
    return [piece] if token_to_id.has_key?(piece)

    word = piece.chars.map(&.to_s)
    return word if word.size <= 1

    loop do
      best_rank = Int32::MAX
      best_pair = nil.as(Tuple(String, String)?)
      (word.size - 1).times do |index|
        pair = {word[index], word[index + 1]}
        if rank = ranks[pair]?
          if rank < best_rank
            best_rank = rank
            best_pair = pair
          end
        end
      end

      pair = best_pair
      break unless pair

      merged = [] of String
      index = 0
      while index < word.size
        if index < word.size - 1 && word[index] == pair[0] && word[index + 1] == pair[1]
          merged << word[index] + word[index + 1]
          index += 2
        else
          merged << word[index]
          index += 1
        end
      end
      word = merged
      break if word.size == 1
    end

    word
  end
end
