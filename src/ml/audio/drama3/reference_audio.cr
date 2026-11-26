require "./wav"
require "./resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
module ML::Audio::Drama3
  module ReferenceAudio
    extend self
    SOURCE = {{ read_file("#{__DIR__}/kernels/reference_audio.metal") }}

    def upload(input : Wav::Reference, context : Resident::Context) : Resident::Buffer
      raw = context.upload(input.pcm)
      output = context.alloc(input.samples)
      context.dispatch("dr3_reference_pcm16", SOURCE, [raw, output], [input.samples.to_u64], input.samples)
      output
    end
  end
end
{% end %}
