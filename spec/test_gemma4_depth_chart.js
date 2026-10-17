'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const {
  buildChartModel,
  commonTimeDomain,
  formatMetric,
  layoutCanvasLegend,
  nearestSample,
  panDomain,
  projectPoint,
  validatePayload,
  valueDomain,
  zoomDomain,
} = require('../scripts/gemma4_depth_chart.js');

function makeRun({ label = 'depth48-before', trial = 1, duration = 1000, gpu = [[0, 100], [500, 150], [duration, 120]], fluke = [[0, 230], [250, 240], [duration, 235]] } = {}) {
  const layerMatch = label.match(/depth(\d+)/);
  return {
    label,
    decoder_layers: layerMatch ? Number(layerMatch[1]) : 48,
    trial,
    duration_ms: duration,
    color: label === 'depth32' ? '#d39a27' : '#2377b5',
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

test('accepts the paired GPU-W and Fluke-VA payload contract', () => {
  assert.equal(validatePayload(makePayload()), null);
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
  const model = buildChartModel(makePayload(runs), 'gpu', 1);
  assert.deepEqual(model.series.map(series => series.label), ['depth48-before', 'depth32']);
  assert.deepEqual(model.series[0].points.map(point => [point.timeMs, point.value]), [[0, 100], [400, 140], [1000, 120]]);
  assert.equal(model.series[1].points.at(-1).timeMs, 700);
  assert.equal(model.timeDomain.end, 1000);
  assert.equal(model.valueDomain.start, 0);
  assert.ok(model.valueDomain.end >= 140);
});

test('source-specific value domains stay zero-based and never combine unlike units', () => {
  const runs = [makeRun({ gpu: [[0, 80], [500, 120], [1000, 100]], fluke: [[0, 200], [500, 250], [1000, 225]] })];
  assert.equal(valueDomain(runs, 'gpu', 1).start, 0);
  assert.ok(valueDomain(runs, 'gpu', 1).end >= 120);
  assert.equal(valueDomain(runs, 'fluke', 1).start, 0);
  assert.ok(valueDomain(runs, 'fluke', 1).end >= 250);
  assert.notEqual(valueDomain(runs, 'gpu', 1).end, valueDomain(runs, 'fluke', 1).end);
});

test('3D projection separates categorical lanes while preserving linear elapsed time', () => {
  const plot = { left: 60, top: 24, width: 400, height: 320 };
  const timeDomain = { start: 0, end: 1000 };
  const powerDomain = { start: 0, end: 200 };
  const first = projectPoint({ timeMs: 0, value: 0, laneIndex: 0, laneCount: 4 }, plot, timeDomain, powerDomain, 'lanes');
  const midpoint = projectPoint({ timeMs: 500, value: 0, laneIndex: 0, laneCount: 4 }, plot, timeDomain, powerDomain, 'lanes');
  const halfPower = projectPoint({ timeMs: 500, value: 100, laneIndex: 0, laneCount: 4 }, plot, timeDomain, powerDomain, 'lanes');
  const last = projectPoint({ timeMs: 1000, value: 0, laneIndex: 0, laneCount: 4 }, plot, timeDomain, powerDomain, 'lanes');
  const nextLane = projectPoint({ timeMs: 0, value: 0, laneIndex: 1, laneCount: 4 }, plot, timeDomain, powerDomain, 'lanes');

  assert.ok(Math.abs((midpoint.x - first.x) - (last.x - midpoint.x)) < 1e-9);
  assert.equal(first.y, midpoint.y);
  assert.ok(halfPower.y < midpoint.y);
  assert.ok(nextLane.x > first.x);
  assert.ok(nextLane.y < first.y);
  assert.equal(first.y, plot.top + plot.height);
});

test('overlay projection shares a zero baseline and maps time without lane offsets', () => {
  const plot = { left: 50, top: 20, width: 300, height: 240 };
  const a = projectPoint({ timeMs: 0, value: 0, laneIndex: 0, laneCount: 4 }, plot, { start: 0, end: 600 }, { start: 0, end: 300 }, 'overlay');
  const b = projectPoint({ timeMs: 600, value: 300, laneIndex: 3, laneCount: 4 }, plot, { start: 0, end: 600 }, { start: 0, end: 300 }, 'overlay');
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
