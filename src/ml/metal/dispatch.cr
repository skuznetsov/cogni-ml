# Metal Dispatch - GPU kernel dispatch utilities
# Stub implementation for cross-platform compatibility

require "./device"

module ML
  module Metal
    module Dispatch
      extend self

      # Get command queue for dispatch
      def queue : CommandQueue?
        return nil unless Device.available?
        @@queue ||= CommandQueue.new
      end

      @@queue : CommandQueue? = nil

      # Dispatch a compute kernel
      def dispatch(pipeline : ComputePipeline, &block : CommandEncoder -> Nil) : Nil
        return unless Device.available?
        q = queue
        return unless q

        buffer = q.command_buffer
        encoder = buffer.encoder
        pipeline.encode(encoder)
        block.call(encoder)
        buffer.commit
        buffer.wait_until_completed
      end
    end
  end
end
