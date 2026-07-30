require "json"
require "set"

module ML::ThreeD::Trellis2
  class StrictJSONError < Exception
  end

  module StrictJSON
    extend self

    MAX_NESTING_DEPTH = 64

    def parse(source : String) : JSON::Any
      pull = JSON::PullParser.new(source)
      consume(pull, 0)
      unless pull.kind.eof?
        raise StrictJSONError.new("unexpected trailing JSON token")
      end
      JSON.parse(source)
    rescue ex : StrictJSONError
      raise ex
    rescue ex : JSON::ParseException
      raise StrictJSONError.new("invalid JSON: #{ex.message}")
    end

    private def consume(pull : JSON::PullParser, depth : Int32) : Nil
      if depth > MAX_NESTING_DEPTH
        raise StrictJSONError.new(
          "JSON nesting depth exceeds #{MAX_NESTING_DEPTH}"
        )
      end

      case pull.kind
      when .begin_object?
        seen = Set(String).new
        pull.read_object do |key|
          unless seen.add?(key)
            raise StrictJSONError.new("duplicate JSON key #{key.inspect}")
          end
          consume(pull, depth + 1)
        end
      when .begin_array?
        pull.read_array { consume(pull, depth + 1) }
      when .null?
        pull.read_null
      when .bool?
        pull.read_bool
      when .int?
        pull.read_int
      when .float?
        pull.read_float
      when .string?
        pull.read_string
      else
        raise StrictJSONError.new("unexpected JSON token #{pull.kind}")
      end
    end
  end
end
