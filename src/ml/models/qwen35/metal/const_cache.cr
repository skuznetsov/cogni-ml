{% unless flag?(:cpu_only) %}
  require "../../../core/buffer"

  module ML
    module GGUF
      module Qwen35Metal
        module ConstCache
          @@written : Hash(String, Bool) = {} of String => Bool
          @@mutex = Mutex.new

          def self.write_once(tag : String, buf : ML::MetalBuffer, data : Array(Float32)) : Nil
            key = "#{tag}:#{buf.handle.address}:#{buf.size}:#{data.to_unsafe.address}:#{data.size}"
            @@mutex.synchronize do
              return if @@written[key]?
              buf.write(data)
              @@written[key] = true
            end
          end

          def self.write_zero_f32_once(tag : String, buf : ML::MetalBuffer, count : Int32) : Nil
            key = "#{tag}:#{buf.handle.address}:#{buf.size}:zero:#{count}"
            @@mutex.synchronize do
              return if @@written[key]?
              buf.contents.as(Pointer(UInt8)).clear(count.to_i64 * sizeof(Float32))
              @@written[key] = true
            end
          end

          def self.clear : Nil
            @@mutex.synchronize { @@written.clear }
          end
        end
      end
    end
  end
{% end %}
