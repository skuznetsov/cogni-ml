(() => {
  "use strict";

  class Drama3VoiceCaptureProcessor extends AudioWorkletProcessor {
    constructor(options) {
      super();
      const requestedLimit = Number(options.processorOptions && options.processorOptions.maxSamples);
      this.maxSamples = Number.isSafeInteger(requestedLimit) && requestedLimit > 0 ? requestedLimit : 1;
      this.capturedSamples = 0;
      this.stopping = false;
      this.finished = false;
      this.port.onmessage = (event) => {
        if (event.data && event.data.type === "stop") this.stopping = true;
      };
    }

    finish(reason) {
      if (this.finished) return;
      this.finished = true;
      this.port.postMessage({ type: "done", reason });
    }

    process(inputs, outputs) {
      for (const output of outputs) {
        for (const channel of output) channel.fill(0);
      }
      if (this.finished) return false;
      if (this.stopping) {
        this.finish("stopped");
        return false;
      }

      const inputChannels = inputs[0];
      if (!inputChannels || inputChannels.length === 0 || inputChannels[0].length === 0) return true;
      const frameCount = inputChannels[0].length;
      const count = Math.min(frameCount, this.maxSamples - this.capturedSamples);
      if (count <= 0) {
        this.finish("limit");
        return false;
      }

      const mono = new Float32Array(count);
      for (let channelIndex = 0; channelIndex < inputChannels.length; channelIndex += 1) {
        const channel = inputChannels[channelIndex];
        for (let frame = 0; frame < count; frame += 1) mono[frame] += channel[frame] / inputChannels.length;
      }
      this.capturedSamples += count;
      this.port.postMessage({ type: "samples", samples: mono }, [mono.buffer]);

      if (this.capturedSamples >= this.maxSamples) {
        this.finish("limit");
        return false;
      }
      return true;
    }
  }

  registerProcessor("drama3-voice-capture", Drama3VoiceCaptureProcessor);
})();
