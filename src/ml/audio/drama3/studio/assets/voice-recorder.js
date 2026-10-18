(function (root, factory) {
  "use strict";
  const VoiceRecorder = factory();
  if (typeof module === "object" && module.exports) module.exports = VoiceRecorder;
  if (root) root.Drama3VoiceRecorder = VoiceRecorder;
})(typeof window === "undefined" ? globalThis : window, function () {
  "use strict";

  const MAX_DURATION_SECONDS = 9.5;
  const MIN_SAMPLE_RATE = 8_000;
  const MAX_SAMPLE_RATE = 192_000;
  const PROCESSOR_NAME = "drama3-voice-capture";
  const WORKLET_URL = "/assets/voice-recorder-worklet.js";

  function wavBlob(samples, sampleRate, BlobConstructor) {
    const dataBytes = samples.length * 2;
    const buffer = new ArrayBuffer(44 + dataBytes);
    const view = new DataView(buffer);
    const writeText = (offset, value) => {
      for (let index = 0; index < value.length; index += 1) view.setUint8(offset + index, value.charCodeAt(index));
    };

    writeText(0, "RIFF");
    view.setUint32(4, 36 + dataBytes, true);
    writeText(8, "WAVE");
    writeText(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeText(36, "data");
    view.setUint32(40, dataBytes, true);

    for (let index = 0; index < samples.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, Number(samples[index]) || 0));
      view.setInt16(44 + index * 2, sample < 0 ? Math.round(sample * 32768) : Math.round(sample * 32767), true);
    }
    return new BlobConstructor([buffer], { type: "audio/wav" });
  }

  function microphoneError(error) {
    const name = error && error.name;
    if (name === "NotAllowedError" || name === "PermissionDeniedError") return new Error("Microphone permission was denied.");
    if (name === "NotFoundError" || name === "DevicesNotFoundError") return new Error("No microphone is available.");
    if (name === "NotReadableError" || name === "TrackStartError") return new Error("The microphone is already in use or unavailable.");
    if (name === "SecurityError") return new Error("Microphone recording requires a secure local page.");
    return error instanceof Error ? error : new Error("Microphone recording could not start.");
  }

  class VoiceRecorder {
    constructor(options) {
      const settings = options || {};
      this.environment = settings.environment || (typeof window === "undefined" ? globalThis : window);
      this.onState = typeof settings.onState === "function" ? settings.onState : () => {};
      this.onComplete = typeof settings.onComplete === "function" ? settings.onComplete : () => {};
      this.onError = typeof settings.onError === "function" ? settings.onError : () => {};
      this.phase = "idle";
      this.token = 0;
      this.stream = null;
      this.track = null;
      this.trackEnded = null;
      this.context = null;
      this.source = null;
      this.node = null;
      this.sampleRate = 0;
      this.maxSamples = 0;
      this.capturedSamples = 0;
      this.chunks = [];
      this.limitTimer = null;
      this.stopGuardTimer = null;
      this.stopPromise = null;
      this.stopResolve = null;
      this.completionTask = null;
    }

    notify(phase) {
      this.phase = phase;
      try { this.onState({ phase, maxDurationSeconds: MAX_DURATION_SECONDS }); }
      catch (_) { /* UI observers must not strand the microphone. */ }
    }

    report(error) {
      try { this.onError(microphoneError(error)); }
      catch (_) { /* Error display must not prevent resource cleanup. */ }
    }

    timer(callback, milliseconds) {
      return this.environment.setTimeout(callback, milliseconds);
    }

    clearTimer(name) {
      if (this[name] !== null) this.environment.clearTimeout(this[name]);
      this[name] = null;
    }

    async start() {
      if (this.phase !== "idle") return false;
      const environment = this.environment;
      const navigator = environment.navigator;
      const mediaDevices = navigator && navigator.mediaDevices;
      const Context = environment.AudioContext || environment.webkitAudioContext;
      const WorkletNode = environment.AudioWorkletNode;
      if (!mediaDevices || typeof mediaDevices.getUserMedia !== "function" || !Context || !WorkletNode || typeof environment.Blob !== "function") {
        this.report(new Error("Microphone recording is not supported in this browser."));
        return false;
      }

      const token = ++this.token;
      this.notify("requesting");
      try {
        const stream = await mediaDevices.getUserMedia({
          audio: {
            channelCount: { ideal: 1 },
            echoCancellation: { ideal: false },
            noiseSuppression: { ideal: false },
            autoGainControl: { ideal: false }
          },
          video: false
        });
        if (token !== this.token) {
          this.stopStream(stream);
          return false;
        }
        this.stream = stream;
        const tracks = typeof stream.getAudioTracks === "function" ? stream.getAudioTracks() : [];
        if (tracks.length === 0) throw new Error("No microphone audio track was available.");
        this.track = tracks[0];
        if (this.track.readyState === "ended") throw new Error("The microphone track has already ended.");
        this.trackEnded = () => {
          if (token !== this.token) return;
          if (this.phase === "recording") this.stop();
          else if (this.phase === "preparing") this.fail(new Error("The microphone track has already ended."), token);
        };
        if (typeof this.track.addEventListener === "function") this.track.addEventListener("ended", this.trackEnded);

        this.notify("preparing");
        this.context = new Context({ latencyHint: "interactive" });
        this.sampleRate = Number(this.context.sampleRate);
        if (!Number.isFinite(this.sampleRate) || this.sampleRate < MIN_SAMPLE_RATE || this.sampleRate > MAX_SAMPLE_RATE) {
          throw new Error("The microphone sample rate is outside the supported range.");
        }
        this.maxSamples = Math.floor(this.sampleRate * MAX_DURATION_SECONDS);
        this.capturedSamples = 0;
        this.chunks = [];
        if (this.context.state === "suspended" && typeof this.context.resume === "function") await this.context.resume();
        if (token !== this.token) return false;
        if (!this.context.audioWorklet || typeof this.context.audioWorklet.addModule !== "function") {
          throw new Error("Uncompressed microphone capture is not supported in this browser.");
        }
        await this.context.audioWorklet.addModule(WORKLET_URL);
        if (token !== this.token) return false;

        this.source = this.context.createMediaStreamSource(stream);
        this.node = new WorkletNode(this.context, PROCESSOR_NAME, {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          outputChannelCount: [1],
          processorOptions: { maxSamples: this.maxSamples }
        });
        this.node.port.onmessage = (event) => this.receive(event.data, token);
        this.node.onprocessorerror = () => this.fail(new Error("The microphone recorder stopped unexpectedly."), token);
        this.source.connect(this.node);
        this.node.connect(this.context.destination);
        this.notify("recording");
        this.limitTimer = this.timer(() => this.stop(), MAX_DURATION_SECONDS * 1000);
        return true;
      } catch (error) {
        if (token === this.token) await this.fail(error, token);
        return false;
      }
    }

    receive(message, token) {
      if (token !== this.token || !message || (this.phase !== "recording" && this.phase !== "stopping")) return;
      try {
        if (message.type === "samples") this.acceptSamples(message.samples);
        else if (message.type === "done") this.complete();
      } catch (error) {
        this.fail(error, token);
      }
    }

    acceptSamples(samples) {
      if (!samples || !ArrayBuffer.isView(samples) || samples.BYTES_PER_ELEMENT !== 4 || typeof samples.length !== "number") {
        throw new Error("The microphone recorder returned invalid audio data.");
      }
      const count = Math.min(samples.length, this.maxSamples - this.capturedSamples);
      if (count <= 0) return;
      const copy = new Float32Array(count);
      copy.set(samples.subarray ? samples.subarray(0, count) : Array.from(samples).slice(0, count));
      this.chunks.push(copy);
      this.capturedSamples += count;
    }

    stop() {
      if (this.phase === "requesting" || this.phase === "preparing") {
        return this.cancel().then(() => null);
      }
      if (this.phase === "stopping") return this.stopPromise || Promise.resolve(null);
      if (this.phase !== "recording") return Promise.resolve(null);

      this.clearTimer("limitTimer");
      this.notify("stopping");
      this.stopPromise = new Promise((resolve) => { this.stopResolve = resolve; });
      if (this.node && this.node.port) this.node.port.postMessage({ type: "stop" });
      else this.complete();
      this.stopGuardTimer = this.timer(() => this.complete(), 1_000);
      return this.stopPromise;
    }

    async complete() {
      if (this.completionTask) return this.completionTask;
      if (!this.stopPromise) this.stopPromise = new Promise((resolve) => { this.stopResolve = resolve; });
      const completionToken = this.token;
      this.clearTimer("limitTimer");
      this.clearTimer("stopGuardTimer");
      this.notify("stopping");
      this.completionTask = (async () => {
        let blob = null;
        let encodingError = null;
        let duration = 0;
        try {
          if (this.capturedSamples > 0) {
            const samples = new Float32Array(this.capturedSamples);
            let offset = 0;
            for (const chunk of this.chunks) {
              samples.set(chunk, offset);
              offset += chunk.length;
            }
            duration = this.capturedSamples / this.sampleRate;
            blob = wavBlob(samples, this.sampleRate, this.environment.Blob);
          }
        } catch (error) {
          encodingError = error;
        }

        await this.cleanup();
        if (completionToken !== this.token) return null;
        this.notify("idle");
        if (blob) {
          try { this.onComplete(blob, duration); }
          catch (_) { /* UI observers must not keep capture resources alive. */ }
        } else {
          this.report(encodingError || new Error("No microphone audio was captured."));
        }
        const resolve = this.stopResolve;
        this.stopResolve = null;
        this.stopPromise = null;
        this.completionTask = null;
        if (resolve) resolve(blob);
        return blob;
      })();
      return this.completionTask;
    }

    async fail(error, token) {
      if (token !== this.token) return null;
      const failureToken = ++this.token;
      this.clearTimer("limitTimer");
      this.clearTimer("stopGuardTimer");
      this.notify("stopping");
      await this.cleanup();
      if (failureToken !== this.token) return null;
      this.notify("idle");
      this.report(error);
      const resolve = this.stopResolve;
      this.stopResolve = null;
      this.stopPromise = null;
      this.completionTask = null;
      if (resolve) resolve(null);
      return null;
    }

    cancel() {
      this.token += 1;
      this.clearTimer("limitTimer");
      this.clearTimer("stopGuardTimer");
      const resolve = this.stopResolve;
      this.stopResolve = null;
      this.stopPromise = null;
      this.completionTask = null;
      const cleanup = this.cleanup();
      this.notify("idle");
      if (resolve) resolve(null);
      return cleanup;
    }

    stopStream(stream) {
      if (!stream || typeof stream.getTracks !== "function") return;
      for (const track of stream.getTracks()) {
        try { track.stop(); }
        catch (_) { /* Best effort across browser track implementations. */ }
      }
    }

    async cleanup() {
      this.clearTimer("limitTimer");
      this.clearTimer("stopGuardTimer");
      const stream = this.stream;
      const track = this.track;
      const trackEnded = this.trackEnded;
      const context = this.context;
      const source = this.source;
      const node = this.node;
      this.stream = null;
      this.track = null;
      this.trackEnded = null;
      this.context = null;
      this.source = null;
      this.node = null;
      this.chunks = [];
      this.capturedSamples = 0;
      this.sampleRate = 0;
      this.maxSamples = 0;

      if (track && trackEnded && typeof track.removeEventListener === "function") track.removeEventListener("ended", trackEnded);
      if (source && typeof source.disconnect === "function") {
        try { source.disconnect(); }
        catch (_) { /* Already disconnected. */ }
      }
      if (node) {
        node.port.onmessage = null;
        node.onprocessorerror = null;
        try { node.disconnect(); }
        catch (_) { /* Already disconnected. */ }
        try { node.port.close(); }
        catch (_) { /* Port already closed. */ }
      }
      this.stopStream(stream);
      if (context && typeof context.close === "function" && context.state !== "closed") {
        try { await context.close(); }
        catch (_) { /* Browser shutdown may close the context first. */ }
      }
    }
  }

  VoiceRecorder.MAX_DURATION_SECONDS = MAX_DURATION_SECONDS;
  VoiceRecorder.WORKLET_URL = WORKLET_URL;
  return VoiceRecorder;
});
