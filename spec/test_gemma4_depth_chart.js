'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const {
  buildChartModel,
  catmullRomSegments,
  commonTimeDomain,
  createShell,
  formatMetric,
  layoutCanvasLegend,
  mount,
  nearestSample,
  panDomain,
  perTokenPoints,
  progressPerTokenPoints,
  presentationOptions,
  sourceDefinitions,
  projectPoint,
  validatePayload,
  valueDomain,
  zoomDomain,
} = require('../scripts/gemma4_depth_chart.js');

function makeRun({ label = 'depth48-before', trial = 1, duration = 1000, gpu = [[0, 100], [500, 150], [duration, 120]], fluke = [[0, 230], [250, 240], [duration, 235]], token_progress } = {}) {
  const layerMatch = label.match(/depth(\d+)/);
  return {
    label,
    decoder_layers: layerMatch ? Number(layerMatch[1]) : 48,
    trial,
    duration_ms: duration,
    color: label === 'depth32' ? '#d39a27' : '#2377b5',
    ...(token_progress === undefined ? {} : { token_progress }),
    gpu: { unit: 'W', integral_unit: 'J', mean: 123, integral: 124, points: gpu },
    fluke: { unit: 'VA', integral_unit: 'VA·s', mean: 230, integral: 231, points: fluke },
  };
}

function makePayload(runs = [makeRun()], requestedTokens = 256) {
  return {
    schema: 'cogni.gemma4.depth_chart.v1',
    campaign_id: 'test-campaign',
    requested_tokens: requestedTokens,
    runs,
    excluded_count: 0,
  };
}

class FakeElement {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.dataset = {};
    this.attributes = {};
    this.classList = { toggle() {} };
    this.style = { setProperty() {} };
    this.textContent = '';
    this.listeners = {};
  }

  append(...nodes) { this.children.push(...nodes); }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  querySelector() { return null; }
  replaceChildren(...nodes) { this.children = nodes; }
  getBoundingClientRect() { return { width: 800, height: 470, left: 0, top: 0 }; }
  getContext() { return this.context; }
  focus() {}
}

function shellFor(payload) {
  const document = { createElement: tagName => new FakeElement(tagName) };
  const section = new FakeElement('section');
  return { section, shell: createShell(document, section, payload, null) };
}

function visibleText(node) {
  return [node.textContent || '', ...node.children.map(visibleText)].join(' ');
}

function mountCompact(payload) {
  const canvasLabels = [];
  const context = new Proxy({
    measureText: text => ({ width: String(text).length * 6 }),
    fillText: text => canvasLabels.push(String(text)),
  }, {
    get(target, key) { return key in target ? target[key] : () => {}; },
  });
  const section = new FakeElement('section');
  const document = {
    defaultView: { devicePixelRatio: 1, addEventListener() {} },
    createElement(tagName) {
      const node = new FakeElement(tagName);
      if (tagName === 'canvas') node.context = context;
      return node;
    },
    getElementById(id) { return id === 'depth-explorer' ? section : null; },
  };
  section.ownerDocument = document;
  return { controller: mount(section, payload), section, canvasLabels };
}

test('accepts the paired GPU-W and Fluke-VA payload contract', () => {
  assert.equal(validatePayload(makePayload()), null);
});

test('accepts UT61B circuit traces and exposes truthful circuit labels', () => {
  const payload = makePayload();
  payload.circuit_meter = 'ut61b';
  payload.measurement_clock = 'gputer';
  assert.equal(validatePayload(payload), null);

  assert.equal(typeof sourceDefinitions, 'function');
  const circuit = sourceDefinitions('ut61b').fluke;
  assert.equal(circuit.label, 'UT61B circuit');
  assert.equal(circuit.shortLabel, 'UT61B circuit');
  assert.equal(circuit.buttonLabel, 'UT61B circuit (VA)');
  assert.equal(circuit.axisLabel, 'UT61B circuit apparent power');
  assert.equal(circuit.unit, 'VA');
  assert.equal(circuit.integralUnit, 'VA·s');
  assert.match(circuit.detail, /AC current.*assumed grid voltage/);
  assert.doesNotMatch(JSON.stringify(circuit), /Fluke|\bW\b|\bJ\b/);

  assert.equal(buildChartModel(payload, 'fluke', 1).perTokenUnit, 'VA·s/token');
  assert.equal(buildChartModel(payload, 'fluke', 1, 'power').perTokenUnit, 'VA/token');
});

test('defaults historical chart payloads to Fluke and rejects unsupported meters', () => {
  assert.equal(sourceDefinitions().fluke.label, 'Fluke circuit');

  const payload = makePayload();
  payload.circuit_meter = 'generic-meter';
  assert.match(validatePayload(payload), /circuit_meter.*fluke.*ut61b/i);

  const invalidClock = makePayload();
  invalidClock.measurement_clock = '  ';
  assert.match(validatePayload(invalidClock), /measurement_clock.*non-empty/i);
});

test('presentation source metadata selects the UT61B compact demo and bounds source IDs', () => {
  const historical = makePayload();
  assert.equal(validatePayload(historical), null);
  assert.deepEqual(presentationOptions(historical), { source: 'gpu', compact: false });

  const demo = makePayload();
  demo.circuit_meter = 'ut61b';
  demo.measurement_clock = 'gputer';
  demo.presentation_source = 'fluke';
  assert.equal(validatePayload(demo), null);
  assert.deepEqual(presentationOptions(demo), { source: 'fluke', compact: true });

  const invalid = makePayload();
  invalid.presentation_source = 'ut61b';
  assert.match(validatePayload(invalid), /presentation_source.*gpu.*fluke/i);
});

test('compact UT61B shell hides source chooser and uses apparent-power labels', () => {
  const payload = makePayload();
  payload.circuit_meter = 'ut61b';
  payload.measurement_clock = 'gputer';
  payload.presentation_source = 'fluke';
  const { section, shell } = shellFor(payload);
  const text = visibleText(section);

  assert.equal(presentationOptions(payload).source, 'fluke');
  assert.equal(shell.energyButton.textContent, 'VA·s/token');
  assert.equal(shell.powerButton.textContent, 'VA/token');
  assert.match(shell.canvas.attributes['aria-label'], /UT61B apparent-power/);
  assert.match(text, /UT61B circuit · selected trial/);
  assert.doesNotMatch(text, /NVIDIA|Fluke|\bW\b|\bJ\b/);

  const legacy = shellFor(makePayload());
  const legacyText = visibleText(legacy.section);
  assert.match(legacyText, /NVIDIA GPU \(W\)/);
  assert.match(legacyText, /Fluke circuit \(VA\)/);
  assert.match(legacyText, /J\/token/);
});

test('mount renders compact UT61B with UT61B selected and no active-energy wording', () => {
  const payload = makePayload();
  payload.circuit_meter = 'ut61b';
  payload.measurement_clock = 'gputer';
  payload.presentation_source = 'fluke';
  payload.runs[0].token_progress = {
    basis: 'received_token_ids',
    points: [[0, 0], [500, 128], [1000, 256]],
  };
  const { controller, section, canvasLabels } = mountCompact(payload);

  assert.ok(controller);
  assert.equal(controller.error, null);
  assert.equal(controller.state.source, 'fluke');
  assert.equal(section.dataset.depthReady, 'true');
  assert.equal(section.dataset.unit, 'VA');
  assert.ok(canvasLabels.some(label => label.includes('APPARENT POWER · VA')));
  assert.ok(canvasLabels.some(label => label.includes('VA·s INTEGRAL / TOKEN · VA·s/token')));
  const text = visibleText(section);
  assert.match(text, /VA·s integral\/N\(t\): uses observed cumulative token IDs.*on gputer/);
  assert.match(text, /the apparent-power integral includes prefill/);
  assert.doesNotMatch(text, /NVIDIA|Fluke|\bW\b|\bJ\b|energy includes prefill/i);
});

test('accepts positive integer token counts and rejects invalid counts', () => {
  assert.equal(validatePayload(makePayload([makeRun()], 192)), null);
  assert.equal(validatePayload(makePayload([makeRun()], 1)), null);

  for (const requestedTokens of [0, -1, 1.5, NaN, Infinity, null]) {
    assert.match(validatePayload(makePayload([makeRun()], requestedTokens)), /positive integer token count/i);
  }
});

test('formats summary metrics to at most two decimal places', () => {
  assert.equal(formatMetric(159.736, 'W'), '159.74 W');
  assert.equal(formatMetric(160, 'VA'), '160 VA');
  assert.equal(formatMetric(null, 'J'), 'Not reported');
});

test('canvas legend wraps labels within the exported plot width', () => {
  const labels = ['depth48-before', 'depth32', 'depth16', 'depth48-after'];
  const legend = layoutCanvasLegend(labels, 190, label => label.length * 7);
  assert.deepEqual(legend.items.map(item => item.label), labels);
  assert.ok(legend.rows > 1);
  for (const item of legend.items) {
    assert.ok(item.x >= 14);
    assert.ok(item.x + 13 + 6 + item.textWidth <= 190 - 14);
  }
});

test('rejects unit drift instead of allowing watts and volt-amperes to mix', () => {
  const payload = makePayload();
  payload.runs[0].fluke.unit = 'W';
  assert.match(validatePayload(payload), /Fluke.*VA/);
});

test('rejects duplicate label/trial pairs and non-monotone samples', () => {
  const duplicate = makePayload([makeRun(), makeRun()]);
  assert.match(validatePayload(duplicate), /duplicate.*label.*trial/i);

  const nonMonotone = makePayload([makeRun({ gpu: [[0, 10], [700, 20], [600, 30]] })]);
  assert.match(validatePayload(nonMonotone), /strictly increasing/);

  const mismatchedWindow = makePayload([makeRun({ duration: 1000, fluke: [[0, 230], [450, 240], [900, 235]] })]);
  assert.match(validatePayload(mismatchedWindow), /duration_ms/);
});

test('uses one actual-time domain across both instruments and all lanes for a trial', () => {
  const runs = [
    makeRun({ label: 'depth48-before', duration: 1000 }),
    makeRun({ label: 'depth32', duration: 1250 }),
    makeRun({ label: 'depth16', trial: 2, duration: 1800 }),
  ];
  assert.deepEqual(commonTimeDomain(runs, 1), { start: 0, end: 1250 });
  assert.deepEqual(commonTimeDomain(runs, 2), { start: 0, end: 1800 });
});

test('chart model keeps samples exact, filters one trial, and never extrapolates a curve', () => {
  const runs = [
    makeRun({ label: 'depth48-before', duration: 1000, gpu: [[0, 100], [400, 140], [1000, 120]] }),
    makeRun({ label: 'depth32', duration: 700, gpu: [[0, 90], [300, 110], [700, 95]] }),
    makeRun({ label: 'depth16', trial: 2, duration: 1500 }),
  ];
  const payload = makePayload(runs, 10);
  const model = buildChartModel(payload, 'gpu', 1);
  assert.deepEqual(model.series.map(series => series.label), ['depth48-before', 'depth32']);
  assert.deepEqual(model.series[0].points.map(point => [point.timeMs, point.value]), [[0, 100], [400, 140], [1000, 120]]);
  assert.deepEqual(model.series[0].perTokenPoints.map(point => point.timeMs), [0, 400, 1000]);
  assert.deepEqual(model.series[0].perTokenPoints.map(point => point.value), [0, 4.8, 12.6]);
  assert.equal(model.series[0].color, payload.runs[0].color);
  assert.equal(model.series[0].normalization, 'final_tokens');
  assert.equal(model.series[1].points.at(-1).timeMs, 700);
  assert.equal(model.timeDomain.end, 1000);
  assert.equal(model.metric, 'energy');
  assert.equal(model.perTokenUnit, 'J/token');
  assert.equal(model.perTokenDomain.start, 0);
  assert.ok(model.perTokenDomain.end >= 12.6);
});

test('per-token energy integrates raw irregular-cadence samples and power divides each sample', () => {
  const points = [
    { timeMs: 0, value: 100 },
    { timeMs: 1000, value: 200 },
    { timeMs: 3000, value: 100 },
  ];
  const energy = perTokenPoints(points, 10, 'energy');
  assert.deepEqual(energy, [
    { timeMs: 0, value: 0 },
    { timeMs: 1000, value: 15 },
    { timeMs: 3000, value: 45 },
  ]);
  assert.ok(energy.every(point => Number.isFinite(point.value) && point.value >= 0));
  assert.ok(energy.every((point, index) => index === 0 || point.value >= energy[index - 1].value));
  assert.deepEqual(perTokenPoints(points, 10, 'power'), [
    { timeMs: 0, value: 10 },
    { timeMs: 1000, value: 20 },
    { timeMs: 3000, value: 10 },
  ]);
});

test('per-token model units follow instrument and metric', () => {
  const run = makeRun({
    duration: 3000,
    gpu: [[0, 100], [1000, 200], [3000, 100]],
    fluke: [[0, 230], [1000, 250], [3000, 240]],
  });
  const payload = makePayload([run], 10);

  const gpuEnergy = buildChartModel(payload, 'gpu', 1);
  const gpuPower = buildChartModel(payload, 'gpu', 1, 'power');
  const flukeEnergy = buildChartModel(payload, 'fluke', 1);
  const flukePower = buildChartModel(payload, 'fluke', 1, 'power');

  assert.equal(gpuEnergy.perTokenUnit, 'J/token');
  assert.equal(gpuEnergy.metric, 'energy');
  assert.equal(gpuPower.perTokenUnit, 'W/token');
  assert.equal(gpuPower.metric, 'power');
  assert.equal(flukeEnergy.perTokenUnit, 'VA·s/token');
  assert.equal(flukePower.perTokenUnit, 'VA/token');
  assert.deepEqual(gpuEnergy.series[0].perTokenPoints.map(point => point.value), [0, 15, 45]);
  assert.deepEqual(gpuPower.series[0].perTokenPoints.map(point => point.value), [10, 20, 10]);
  assert.equal(gpuEnergy.perTokenDomain.start, 0);
  assert.equal(gpuPower.perTokenDomain.start, 0);
});

test('per-token points reject invalid denominators, metrics, and source samples', () => {
  assert.equal(typeof perTokenPoints, 'function');
  const points = [{ timeMs: 0, value: 100 }, { timeMs: 1000, value: 200 }];
  for (const tokenCount of [0, -1, 1.5, NaN, Infinity, null]) {
    assert.throws(() => perTokenPoints(points, tokenCount, 'energy'));
  }
  assert.throws(() => perTokenPoints(points, 10, 'joules'));

  const invalidPointSets = [
    [{ timeMs: NaN, value: 1 }, { timeMs: 1, value: 2 }],
    [{ timeMs: Infinity, value: 1 }, { timeMs: 1, value: 2 }],
    [{ timeMs: 0, value: NaN }, { timeMs: 1, value: 2 }],
    [{ timeMs: 0, value: Infinity }, { timeMs: 1, value: 2 }],
    [{ timeMs: 0, value: -1 }, { timeMs: 1, value: 2 }],
    [{ timeMs: -1, value: 1 }, { timeMs: 1, value: 2 }],
    [{ timeMs: 0, value: 1 }, { timeMs: 0, value: 2 }],
    [{ timeMs: 1, value: 1 }, { timeMs: 0, value: 2 }],
  ];
  for (const invalidPoints of invalidPointSets) {
    assert.throws(() => perTokenPoints(invalidPoints, 10, 'energy'));
  }
});

test('actual-token progress derives at the union of power and received-token times', () => {
  const power = [
    { timeMs: 0, value: 100 },
    { timeMs: 1500, value: 100 },
    { timeMs: 3000, value: 100 },
  ];
  const progress = [[0, 0], [1000, 1], [2000, 2], [3000, 3]];

  assert.deepEqual(progressPerTokenPoints(power, progress, 'energy'), [
    { timeMs: 1000, value: 100 },
    { timeMs: 1500, value: 150 },
    { timeMs: 2000, value: 100 },
    { timeMs: 3000, value: 100 },
  ]);
  assert.deepEqual(progressPerTokenPoints(power, progress, 'power'), [
    { timeMs: 1000, value: 100 },
    { timeMs: 1500, value: 100 },
    { timeMs: 2000, value: 50 },
    { timeMs: 3000, value: 100 / 3 },
  ]);
});

test('actual-token progress linearly interpolates power at counter times and integrates raw power', () => {
  const power = [
    { timeMs: 0, value: 100 },
    { timeMs: 2000, value: 300 },
    { timeMs: 3000, value: 100 },
  ];
  const progress = [[0, 0], [1000, 1], [2000, 2], [3000, 3]];

  assert.deepEqual(progressPerTokenPoints(power, progress, 'power'), [
    { timeMs: 1000, value: 200 },
    { timeMs: 2000, value: 150 },
    { timeMs: 3000, value: 100 / 3 },
  ]);
  assert.deepEqual(progressPerTokenPoints(power, progress, 'energy'), [
    { timeMs: 1000, value: 150 },
    { timeMs: 2000, value: 200 },
    { timeMs: 3000, value: 200 },
  ]);
});

test('chart model uses received-token progress when available and final-token fallback otherwise', () => {
  const run = makeRun({
    duration: 3000,
    gpu: [[0, 100], [1500, 100], [3000, 100]],
    fluke: [[0, 230], [1500, 240], [3000, 235]],
    token_progress: {
      basis: 'received_token_ids',
      points: [[0, 0], [1000, 1], [2000, 2], [3000, 3]],
    },
  });
  const actual = buildChartModel(makePayload([run], 3), 'gpu', 1);
  const historical = buildChartModel(makePayload([makeRun({
    duration: 3000,
    gpu: [[0, 100], [1500, 100], [3000, 100]],
    fluke: [[0, 230], [1500, 240], [3000, 235]],
  })], 3), 'gpu', 1);

  assert.equal(actual.series[0].normalization, 'received_tokens');
  assert.deepEqual(actual.series[0].perTokenPoints.map(point => point.timeMs), [1000, 1500, 2000, 3000]);
  assert.deepEqual(actual.series[0].perTokenPoints.map(point => point.value), [100, 150, 100, 100]);
  assert.deepEqual(actual.series[0].points.map(point => [point.timeMs, point.value]), [[0, 100], [1500, 100], [3000, 100]]);
  assert.equal(historical.series[0].normalization, 'final_tokens');
  assert.deepEqual(historical.series[0].perTokenPoints.map(point => point.value), [0, 50, 100]);
});

test('payload accepts absent token progress and rejects malformed received-token counters', () => {
  assert.equal(validatePayload(makePayload()), null);

  const malformedProgress = [
    ['wrong basis', { basis: 'generated_token_ids', points: [[0, 0], [1000, 256]] }],
    ['missing points', { basis: 'received_token_ids' }],
    ['one point', { basis: 'received_token_ids', points: [[0, 0]] }],
    ['wrong initial sample', { basis: 'received_token_ids', points: [[0, 1], [1000, 256]] }],
    ['wrong final time', { basis: 'received_token_ids', points: [[0, 0], [900, 256]] }],
    ['wrong final count', { basis: 'received_token_ids', points: [[0, 0], [1000, 255]] }],
    ['fractional time', { basis: 'received_token_ids', points: [[0, 0], [500.5, 100], [1000, 256]] }],
    ['negative time', { basis: 'received_token_ids', points: [[0, 0], [-1, 100], [1000, 256]] }],
    ['fractional count', { basis: 'received_token_ids', points: [[0, 0], [500, 1.5], [1000, 256]] }],
    ['negative count', { basis: 'received_token_ids', points: [[0, 0], [500, -1], [1000, 256]] }],
    ['duplicate time', { basis: 'received_token_ids', points: [[0, 0], [500, 1], [500, 2], [1000, 256]] }],
    ['unordered time', { basis: 'received_token_ids', points: [[0, 0], [700, 1], [500, 2], [1000, 256]] }],
    ['decreasing count', { basis: 'received_token_ids', points: [[0, 0], [500, 2], [700, 1], [1000, 256]] }],
    ['count exceeds requested total', { basis: 'received_token_ids', points: [[0, 0], [500, 257], [1000, 256]] }],
  ];

  for (const [description, token_progress] of malformedProgress) {
    const payload = makePayload([makeRun({ token_progress })]);
    assert.ok(validatePayload(payload), `should reject ${description}`);
  }
});

test('Catmull-Rom segments preserve endpoints and constrain controls across irregular cadence', () => {
  const points = [
    { x: 0, y: 0 },
    { x: 1, y: 100 },
    { x: 9, y: 0 },
    { x: 10, y: 80 },
  ];
  const segments = catmullRomSegments(points);
  assert.equal(segments.length, points.length - 1);
  for (let index = 0; index < segments.length; index += 1) {
    const { start, c1, c2, end } = segments[index];
    const lowerY = Math.min(points[index].y, points[index + 1].y);
    const upperY = Math.max(points[index].y, points[index + 1].y);
    assert.deepEqual(start, points[index]);
    assert.deepEqual(end, points[index + 1]);
    for (const control of [c1, c2]) {
      assert.ok(control.x >= start.x && control.x <= end.x);
      assert.ok(control.y >= lowerY && control.y <= upperY);
    }
  }
  assert.deepEqual(catmullRomSegments([]), []);
  assert.deepEqual(catmullRomSegments([{ x: 2, y: 5 }]), []);
});

test('Catmull-Rom segments reject non-finite or non-increasing x coordinates', () => {
  assert.equal(typeof catmullRomSegments, 'function');
  for (const points of [
    [{ x: 0, y: 1 }, { x: 0, y: 2 }],
    [{ x: 1, y: 1 }, { x: 0, y: 2 }],
    [{ x: 0, y: 1 }, { x: Infinity, y: 2 }],
    [{ x: NaN, y: 1 }, { x: 1, y: 2 }],
  ]) {
    assert.throws(() => catmullRomSegments(points));
  }
});

test('source-specific value domains stay zero-based and never combine unlike units', () => {
  const runs = [makeRun({ gpu: [[0, 80], [500, 120], [1000, 100]], fluke: [[0, 200], [500, 250], [1000, 225]] })];
  assert.equal(valueDomain(runs, 'gpu', 1).start, 0);
  assert.ok(valueDomain(runs, 'gpu', 1).end >= 120);
  assert.equal(valueDomain(runs, 'fluke', 1).start, 0);
  assert.ok(valueDomain(runs, 'fluke', 1).end >= 250);
  assert.notEqual(valueDomain(runs, 'gpu', 1).end, valueDomain(runs, 'fluke', 1).end);
});

test('2D projection ignores categorical lane metadata and maps time and value linearly', () => {
  const plot = { left: 60, top: 24, width: 400, height: 320 };
  const timeDomain = { start: 0, end: 1000 };
  const powerDomain = { start: 0, end: 200 };
  const first = projectPoint({ timeMs: 0, value: 0, laneIndex: 0, laneCount: 4 }, plot, timeDomain, powerDomain);
  const midpoint = projectPoint({ timeMs: 500, value: 0, laneIndex: 0, laneCount: 4 }, plot, timeDomain, powerDomain);
  const halfPower = projectPoint({ timeMs: 500, value: 100, laneIndex: 0, laneCount: 4 }, plot, timeDomain, powerDomain);
  const last = projectPoint({ timeMs: 1000, value: 0, laneIndex: 0, laneCount: 4 }, plot, timeDomain, powerDomain);
  const nextLane = projectPoint({ timeMs: 0, value: 0, laneIndex: 3, laneCount: 4 }, plot, timeDomain, powerDomain);

  assert.ok(Math.abs((midpoint.x - first.x) - (last.x - midpoint.x)) < 1e-9);
  assert.equal(first.y, midpoint.y);
  assert.ok(halfPower.y < midpoint.y);
  assert.deepEqual([nextLane.x, nextLane.y], [first.x, first.y]);
  assert.equal(first.y, plot.top + plot.height);
});

test('overlay projection shares a zero baseline and maps time without lane offsets', () => {
  const plot = { left: 50, top: 20, width: 300, height: 240 };
  const a = projectPoint({ timeMs: 0, value: 0 }, plot, { start: 0, end: 600 }, { start: 0, end: 300 });
  const b = projectPoint({ timeMs: 600, value: 300 }, plot, { start: 0, end: 600 }, { start: 0, end: 300 });
  assert.deepEqual([a.x, a.y], [50, 260]);
  assert.deepEqual([b.x, b.y], [350, 20]);
});

test('nearest-sample hover can select source samples only', () => {
  const points = [
    { x: 20, y: 80, label: 'depth48-before', timeMs: 0, value: 100 },
    { x: 60, y: 40, label: 'depth48-before', timeMs: 500, value: 150 },
  ];
  assert.equal(nearestSample(points, 57, 42, 12), points[1]);
  assert.equal(nearestSample(points, 40, 60, 10), null);
});

test('pan and zoom domains remain bounded by the measured campaign interval', () => {
  const full = { start: 0, end: 1000 };
  const zoomed = zoomDomain(full, full, 0.5, 2);
  assert.deepEqual(zoomed, { start: 250, end: 750 });
  assert.deepEqual(panDomain(zoomed, full, -1), { start: 0, end: 500 });
  assert.deepEqual(panDomain(zoomed, full, 1), { start: 500, end: 1000 });
  assert.deepEqual(zoomDomain(zoomed, full, 0.5, 0.1), full);
});


test('explicit PF=1 estimate yields J/token without changing measured VA or raw integrals', () => {
  const payload = makePayload();
  payload.circuit_meter = 'ut61b';
  payload.presentation_source = 'fluke';
  const measured = buildChartModel(payload, 'fluke', 1);
  payload.circuit_energy_estimate = { power_factor: 1, basis: 'assumed' };
  assert.equal(validatePayload(payload), null);
  const estimate = buildChartModel(payload, 'fluke', 1);
  assert.equal(estimate.perTokenUnit, 'J/token');
  assert.equal(estimate.estimatedEnergy, true);
  assert.deepEqual(estimate.series[0].points, measured.series[0].points);
  assert.equal(estimate.series[0].integralUnit, 'VA·s');
  assert.deepEqual(estimate.series[0].perTokenPoints, measured.series[0].perTokenPoints);
  assert.equal(buildChartModel(payload, 'fluke', 1, 'power').perTokenUnit, 'VA/token');
  assert.equal(buildChartModel(payload, 'gpu', 1).perTokenUnit, 'J/token');
  assert.equal(buildChartModel(payload, 'gpu', 1).estimatedEnergy, false);
  const mounted = mountCompact(payload);
  assert.equal(mounted.controller.error, null);
  assert.match(visibleText(mounted.section), /Estimated J\/token/);
  assert.match(visibleText(mounted.section), /PF=1 assumed/);
  assert.ok(mounted.canvasLabels.some(s => s.includes('J/token') && s.includes('PF=1')));
  payload.circuit_energy_estimate.power_factor = 0.9;
  assert.match(validatePayload(payload), /power.factor|PF=1/i);
  payload.circuit_energy_estimate.power_factor = 1;
  delete payload.presentation_source;
  assert.match(validatePayload(payload), /compact|presentation/i);
});
