module ML::GGUF
  # State machine for a finite set of already-tokenized literal options.
  # Callers own the admissibility contract for the option set.
  struct TokenOptionCorridor
    getter options : Array(Array(Int32))

    def initialize(options : Array(Array(Int32)))
      @options = options.map(&.dup)
    end

    def self.from_options(options : Array(Array(Int32))) : TokenOptionCorridor
      new(options)
    end

    def empty? : Bool
      @options.empty?
    end

    def complete? : Bool
      !@options.empty? && @options.all?(&.empty?)
    end

    def next_ids : Array(Int32)
      seen = Set(Int32).new
      ids = [] of Int32
      @options.each do |option|
        next if option.empty?

        id = option[0]
        next if seen.includes?(id)

        seen << id
        ids << id
      end
      ids
    end

    def advance(emitted_id : Int32) : TokenOptionCorridor
      advanced = [] of Array(Int32)
      @options.each do |option|
        next if option.empty? || option[0] != emitted_id

        advanced << option[1..]
      end
      TokenOptionCorridor.new(advanced)
    end

    def self.selected_literal_index?(full_options : Array(Array(Int32)), emitted_ids : Array(Int32)) : Int32?
      selected_idx = nil.as(Int32?)
      selected_size = -1
      full_options.each_with_index do |ids, idx|
        next unless ids.size <= emitted_ids.size
        next unless emitted_ids[0, ids.size] == ids
        next unless ids.size > selected_size

        selected_idx = idx
        selected_size = ids.size
      end
      selected_idx
    end
  end
end
