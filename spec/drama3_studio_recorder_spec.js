"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const assets = path.join(__dirname, "../src/ml/audio/drama3/studio/assets");
const html = fs.readFileSync(path.join(assets, "index.html"), "utf8");
const VoiceRecorder = require(path.join(assets, "voice-recorder.js"));
const workletSource = fs.readFileSync(path.join(assets, "voice-recorder-worklet.js"), "utf8");

test("voice form offers explicit recording, preview, discard, and save controls", () => {
  for (const id of [
    "voice-record-button",
    "voice-stop-button",
    "voice-discard-button",
    "voice-preview",
    "voice-save-button",
    "voice-transcript",
    "voice-script-en",
    "voice-script-ru"
  ]) assert.match(html, new RegExp(`id=\"${id}\"`));
  assert.match(html, /id=\"voice-transcript\"[^>]*required/);
});

function fakeTrack() {
  const listeners = new Map();
  return {
    readyState: "live",
    stopCount: 0,
    addEventListener(name, callback) { listeners.set(name, callback); },
    removeEventListener(name, callback) { if (listeners.get(name) === callback) listeners.delete(name); },
    stop() { this.stopCount += 1; this.readyState = "ended"; },
    end() { this.readyState = "ended"; const callback = listeners.get("ended"); if (callback) callback(); }
  };
}

function fakeStream(tracks) {
  return {
    getAudioTracks() { return tracks; },
    getTracks() { return tracks; }
  };
}

function createRecorderHarness(options = {}) {
  const track = options.track || fakeTrack();
  const stream = options.stream || fakeStream([track]);
  const contexts = [];
  const nodes = [];
  const delays = [];
  const errors = [];
  const completions = [];
  const phases = [];
  let constraints = null;
  let permissionCalls = 0;

  class FakePort {
    constructor() { this.onmessage = null; this.closed = false; }
    postMessage(message) {
      if (message.type === "stop") queueMicrotask(() => this.dispatch({ type: "done", reason: "stopped" }));
    }
    dispatch(data) { if (this.onmessage) this.onmessage({ data }); }
    close() { this.closed = true; }
  }

  class FakeAudioWorkletNode {
    constructor(context, name, nodeOptions) {
      this.context = context;
      this.name = name;
      this.options = nodeOptions;
      this.port = new FakePort();
      this.onprocessorerror = null;
      this.disconnected = false;
      nodes.push(this);
    }
    connect() {}
    disconnect() { this.disconnected = true; }
  }

  class FakeAudioContext {
    constructor() {
      this.sampleRate = options.sampleRate || 48_000;
      this.state = "running";
      this.destination = {};
      this.closed = false;
      this.audioWorklet = { addModule: async (url) => {
        assert.equal(url, "/assets/voice-recorder-worklet.js");
        if (options.addModule) return options.addModule();
      } };
      contexts.push(this);
    }
    createMediaStreamSource() { return { connect() {}, disconnect() { this.disconnected = true; } }; }
    async close() { this.closed = true; this.state = "closed"; }
  }

  const environment = {
    navigator: options.noMediaDevices ? {} : {
      mediaDevices: {
        getUserMedia(value) {
          permissionCalls += 1;
          constraints = value;
          return options.getUserMedia ? options.getUserMedia(value) : Promise.resolve(stream);
        }
      }
    },
    AudioContext: FakeAudioContext,
    AudioWorkletNode: FakeAudioWorkletNode,
    Blob,
    setTimeout(callback, delay) { delays.push(delay); return { callback, delay }; },
    clearTimeout() {}
  };
  const recorder = new VoiceRecorder({
    environment,
    onState(snapshot) { phases.push(snapshot.phase); },
    onComplete(blob, duration) { completions.push({ blob, duration }); },
    onError(error) { errors.push(error.message); }
  });
  return { recorder, track, stream, contexts, nodes, delays, errors, completions, phases, get constraints() { return constraints; }, get permissionCalls() { return permissionCalls; }, environment };
}

test("microphone capture becomes bounded mono PCM16 WAV and releases its stream on stop", async () => {
  const harness = createRecorderHarness();
  assert.equal(await harness.recorder.start(), true);
  assert.equal(harness.recorder.phase, "recording");
  assert.deepEqual(harness.constraints.audio.echoCancellation, { ideal: false });
  assert.deepEqual(harness.constraints.audio.noiseSuppression, { ideal: false });
  assert.deepEqual(harness.constraints.audio.autoGainControl, { ideal: false });
  assert.deepEqual(harness.constraints.audio.channelCount, { ideal: 1 });
  assert.equal(harness.nodes[0].options.processorOptions.maxSamples, 48_000 * 9.5);
  assert.ok(harness.delays.includes(9_500));

  harness.nodes[0].port.dispatch({ type: "samples", samples: new Float32Array([0, 0.5, -0.5, 1, -1]) });
  const blob = await harness.recorder.stop();
  assert.equal(blob.type, "audio/wav");
  const bytes = Buffer.from(await blob.arrayBuffer());
  assert.equal(bytes.toString("ascii", 0, 4), "RIFF");
  assert.equal(bytes.toString("ascii", 8, 12), "WAVE");
  assert.equal(bytes.readUInt16LE(22), 1);
  assert.equal(bytes.readUInt32LE(24), 48_000);
  assert.equal(bytes.readUInt16LE(34), 16);
  assert.equal(bytes.readUInt32LE(40), 10);
  assert.deepEqual([0, 1, 2, 3, 4].map((index) => bytes.readInt16LE(44 + index * 2)), [0, 16_384, -16_384, 32_767, -32_768]);
  assert.equal(harness.completions.length, 1);
  assert.equal(harness.completions[0].duration, 5 / 48_000);
  assert.equal(harness.track.stopCount, 1);
  assert.equal(harness.contexts[0].closed, true);
  assert.equal(harness.nodes[0].port.closed, true);
  assert.equal(harness.recorder.phase, "idle");
});

test("cancel during a pending permission prompt stops a late stream without creating a draft", async () => {
  let resolvePermission;
  const pendingPermission = new Promise((resolve) => { resolvePermission = resolve; });
  const harness = createRecorderHarness({ getUserMedia: () => pendingPermission });
  const startup = harness.recorder.start();
  assert.equal(harness.recorder.phase, "requesting");
  assert.equal(harness.permissionCalls, 1);

  await harness.recorder.stop();
  assert.equal(harness.recorder.phase, "idle");
  const lateTrack = fakeTrack();
  resolvePermission(fakeStream([lateTrack]));
  assert.equal(await startup, false);
  assert.equal(lateTrack.stopCount, 1);
  assert.equal(harness.contexts.length, 0);
  assert.equal(harness.completions.length, 0);
  assert.deepEqual(harness.errors, []);
});

test("cancel during worklet setup releases the stream and ignores late initialization", async () => {
  let resolveModule;
  const modulePromise = new Promise((resolve) => { resolveModule = resolve; });
  const harness = createRecorderHarness({ addModule: () => modulePromise });
  const startup = harness.recorder.start();
  await Promise.resolve();
  assert.equal(harness.recorder.phase, "preparing");
  assert.equal(harness.track.stopCount, 0);

  harness.recorder.cancel();
  assert.equal(harness.track.stopCount, 1);
  assert.equal(harness.contexts[0].closed, true);
  resolveModule();
  assert.equal(await startup, false);
  assert.equal(harness.nodes.length, 0);
  assert.equal(harness.completions.length, 0);
  assert.deepEqual(harness.errors, []);
});

test("a late canceled setup cannot clean up a newer recording", async () => {
  const tracks = [fakeTrack(), fakeTrack()];
  let permissionIndex = 0;
  let moduleIndex = 0;
  let resolveFirstModule;
  const firstModule = new Promise((resolve) => { resolveFirstModule = resolve; });
  const harness = createRecorderHarness({
    getUserMedia: () => Promise.resolve(fakeStream([tracks[permissionIndex++]])),
    addModule: () => ++moduleIndex === 1 ? firstModule : Promise.resolve()
  });
  const oldStartup = harness.recorder.start();
  await Promise.resolve();
  assert.equal(harness.recorder.phase, "preparing");

  await harness.recorder.stop();
  assert.equal(tracks[0].stopCount, 1);
  assert.equal(await harness.recorder.start(), true);
  const activeNode = harness.nodes[0];
  const activeContext = harness.contexts[1];
  assert.equal(harness.recorder.phase, "recording");

  resolveFirstModule();
  assert.equal(await oldStartup, false);
  assert.equal(harness.recorder.phase, "recording");
  assert.equal(tracks[1].stopCount, 0);
  assert.equal(activeContext.closed, false);
  assert.equal(harness.recorder.node, activeNode);

  activeNode.port.dispatch({ type: "samples", samples: new Float32Array([0.2]) });
  assert.ok(await harness.recorder.stop());
  assert.equal(tracks[1].stopCount, 1);
});

test("cancel during asynchronous completion permits a fresh recording and drops stale audio", async () => {
  const tracks = [fakeTrack(), fakeTrack()];
  const harness = createRecorderHarness({ getUserMedia: () => Promise.resolve(fakeStream([tracks.shift()])) });
  assert.equal(await harness.recorder.start(), true);
  harness.nodes[0].port.dispatch({ type: "samples", samples: new Float32Array([0.25]) });
  const staleCompletion = harness.recorder.complete();
  harness.recorder.cancel();
  assert.equal(await staleCompletion, null);
  assert.equal(harness.completions.length, 0);

  assert.equal(await harness.recorder.start(), true);
  harness.nodes[1].port.dispatch({ type: "samples", samples: new Float32Array([-0.25]) });
  const blob = await harness.recorder.stop();
  assert.ok(blob);
  assert.equal(harness.completions.length, 1);
  const bytes = Buffer.from(await blob.arrayBuffer());
  assert.equal(bytes.readInt16LE(44), -8_192);
});

test("an ended microphone track stops capture and finalizes samples already received", async () => {
  const track = fakeTrack();
  const harness = createRecorderHarness({ track });
  assert.equal(await harness.recorder.start(), true);
  harness.nodes[0].port.dispatch({ type: "samples", samples: new Float32Array([0.125, -0.125]) });

  track.end();
  const blob = await harness.recorder.stop();
  assert.ok(blob);
  assert.equal(harness.errors.length, 0);
  assert.equal(harness.track.stopCount, 1);
  const bytes = Buffer.from(await blob.arrayBuffer());
  assert.equal(bytes.readInt16LE(44), 4_096);
  assert.equal(bytes.readInt16LE(46), -4_096);
});

test("unsupported browsers fail before requesting microphone permission", async () => {
  const harness = createRecorderHarness({ noMediaDevices: true });
  assert.equal(await harness.recorder.start(), false);
  assert.equal(harness.permissionCalls, 0);
  assert.equal(harness.recorder.phase, "idle");
  assert.deepEqual(harness.errors, ["Microphone recording is not supported in this browser."]);
});

test("empty microphone streams and worklet setup failures report errors and release tracks", async () => {
  const emptyTrack = fakeTrack();
  const empty = createRecorderHarness({ stream: fakeStream([]), track: emptyTrack });
  assert.equal(await empty.recorder.start(), false);
  assert.deepEqual(empty.errors, ["No microphone audio track was available."]);
  assert.equal(emptyTrack.stopCount, 0);

  const track = fakeTrack();
  const broken = createRecorderHarness({ track, addModule: async () => { throw new Error("worklet unavailable"); } });
  assert.equal(await broken.recorder.start(), false);
  assert.deepEqual(broken.errors, ["worklet unavailable"]);
  assert.equal(track.stopCount, 1);
  assert.equal(broken.contexts[0].closed, true);
  assert.equal(broken.recorder.phase, "idle");
});

test("AudioWorklet downmixes to mono and stops exactly at the configured sample bound", () => {
  let Processor;
  const context = {
    AudioWorkletProcessor: class {
      constructor() { this.port = { onmessage: null, messages: [], postMessage(message) { this.messages.push(message); } }; }
    },
    registerProcessor(name, processor) { assert.equal(name, "drama3-voice-capture"); Processor = processor; },
    Float32Array,
    Math,
    Number
  };
  vm.runInNewContext(workletSource, context);
  const processor = new Processor({ processorOptions: { maxSamples: 5 } });
  const output = [[new Float32Array(4)]];
  assert.equal(processor.process([[new Float32Array([0.2, 0.2, 0.2, 0.2]), new Float32Array([0.4, 0.4, 0.4, 0.4])]], output), true);
  assert.deepEqual(Array.from(output[0][0]), [0, 0, 0, 0]);
  assert.equal(processor.process([[new Float32Array([0.8, 0.8, 0.8, 0.8]), new Float32Array([0.2, 0.2, 0.2, 0.2])]], output), false);
  const messages = processor.port.messages;
  for (const sample of messages[0].samples) assert.ok(Math.abs(sample - 0.3) < 1e-6);
  assert.deepEqual(Array.from(messages[1].samples), [0.5]);
  assert.equal(messages[2].type, "done");
  assert.equal(messages[2].reason, "limit");
  assert.equal(processor.process([[new Float32Array([0.9])]], output), false);
  assert.equal(messages.length, 3);
});
