{% unless flag?(:cpu_only) %}
  require "../../../core/buffer"

  module ML
    module GGUF
      module Qwen35Metal
        # ── Phase 4.2 scratch pool ─────────────────────────────────────
        # Persistent per-dispatch scratch buffers keyed by (tag, bytes).
        # Each dispatch-site tag names a buffer slot that must not alias
        # another buffer alive in the same command buffer. Sizes vary
        # across call sites and layers, so key includes size; reuse
        # within a key is a cache hit. Buffers live until `clear`.
        module Scratch
          @@pool : Hash({Symbol, Int64}, ML::MetalBuffer) = {} of {Symbol, Int64} => ML::MetalBuffer
          @@pool_s : Hash({String, Int64}, ML::MetalBuffer) = {} of {String, Int64} => ML::MetalBuffer
          @@mutex = Mutex.new
          @@hits   = 0_i64
          @@misses = 0_i64
          @@fresh_collectors = {} of UInt64 => Array(Array(ML::MetalBuffer))
          @@namespaces = {} of UInt64 => Array(String)

          # Private bounded scratch ownership for an asynchronous graph lease.
          # `with_fresh` prevents pool aliases; the arena keeps every buffer
          # alive until its owning lease reaches terminal GPU completion.
          class Arena
            getter namespace : String
            getter buffers : Array(ML::MetalBuffer)

            @released = false

            def initialize(@namespace : String)
              @buffers = [] of ML::MetalBuffer
            end

            def with(&)
              raise ArgumentError.new("scratch arena is released") if @released
              Scratch.with_fresh do |fresh|
                begin
                  Scratch.with_namespace(@namespace) { yield }
                ensure
                  @buffers.concat(fresh)
                end
              end
            end

            def release : Nil
              return if @released
              begin
                @buffers.each do |buffer|
                  begin
                    buffer.release
                  rescue
                    # Release every buffer even if one native handle has
                    # already been invalidated by an earlier failure path.
                  end
                end
              ensure
                @buffers.clear
                @released = true
              end
            end

            def released? : Bool
              @released
            end
          end

          private def self.thread_key : UInt64
            Thread.current.object_id
          end

          private def self.fresh_stack : Array(Array(ML::MetalBuffer))
            key = thread_key
            @@mutex.synchronize { @@fresh_collectors[key] ||= [] of Array(ML::MetalBuffer) }
          end

          private def self.namespace_stack : Array(String)
            key = thread_key
            @@mutex.synchronize { @@namespaces[key] ||= [] of String }
          end

          def self.with_fresh(&)
            collector = [] of ML::MetalBuffer
            fresh_stack << collector
            yield collector
          ensure
            fresh_stack.pop?
          end

          private def self.fresh_buffer(byte_size : Int64) : ML::MetalBuffer
            @@mutex.synchronize { @@misses += 1 }
            buf = ML::MetalBuffer.new(byte_size)
            if collector = fresh_stack.last?
              collector << buf
            end
            buf
          end

          def self.with_namespace(namespace : String, &)
            namespace_stack << namespace
            yield
          ensure
            namespace_stack.pop?
          end

          private def self.get_string_raw(tag : String, byte_size : Int64) : ML::MetalBuffer
            if ENV["QWEN35_SCRATCH_OFF"]? == "1" || !fresh_stack.empty?
              return fresh_buffer(byte_size)
            end
            key = {tag, byte_size}
            @@mutex.synchronize do
              if buf = @@pool_s[key]?
                @@hits += 1
                return buf
              end
              @@misses += 1
              buf = ML::MetalBuffer.new(byte_size)
              @@pool_s[key] = buf
              buf
            end
          end

          def self.get(tag : Symbol, byte_size : Int64) : ML::MetalBuffer
            if namespace = namespace_stack.last?
              return get_string_raw("#{namespace}:#{tag}", byte_size)
            end
            # A/B gate — when set, always allocate fresh (emulates pre-4.2).
            if ENV["QWEN35_SCRATCH_OFF"]? == "1" || !fresh_stack.empty?
              return fresh_buffer(byte_size)
            end
            key = {tag, byte_size}
            @@mutex.synchronize do
              if buf = @@pool[key]?
                @@hits += 1
                return buf
              end
              @@misses += 1
              buf = ML::MetalBuffer.new(byte_size)
              @@pool[key] = buf
              buf
            end
          end

          def self.get(tag : String, byte_size : Int64) : ML::MetalBuffer
            if namespace = namespace_stack.last?
              tag = "#{namespace}:#{tag}"
            end
            get_string_raw(tag, byte_size)
          end

          def self.stats : {Int64, Int64}
            @@mutex.synchronize { {@@hits, @@misses} }
          end

          # Nominal byte lengths retained by the exact-size pools. Fresh arenas
          # and driver allocations are excluded; aliases are not deduplicated.
          def self.memory_inventory
            @@mutex.synchronize do
              by_tag = {} of String => NamedTuple(entries: Int32, bytes: Int64)
              total = 0_i64
              @@pool.each do |key, buffer|
                tag = "symbol:#{key[0]}"
                previous = by_tag[tag]? || {entries: 0, bytes: 0_i64}
                by_tag[tag] = {entries: previous[:entries] + 1, bytes: previous[:bytes] + buffer.size}
                total += buffer.size
              end
              @@pool_s.each do |key, buffer|
                tag = "string:#{key[0]}"
                previous = by_tag[tag]? || {entries: 0, bytes: 0_i64}
                by_tag[tag] = {entries: previous[:entries] + 1, bytes: previous[:bytes] + buffer.size}
                total += buffer.size
              end
              {entries: @@pool.size + @@pool_s.size, retained_bytes: total, by_tag: by_tag}
            end
          end

          def self.clear : Nil
            @@mutex.synchronize do
              @@pool.clear
              @@pool_s.clear
              @@fresh_collectors.clear
              @@namespaces.clear
              @@hits = @@misses = 0_i64
            end
          end
        end
      end
    end
  end
{% end %}
