/*
 * ISC License
 *
 * Copyright (c) 2026 Sergey Kuznetsov
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE.
 */

(function depthChartModule(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root && root.document) api.autoMount(root.document);
})(typeof globalThis === 'object' ? globalThis : this, function createDepthChartApi() {
  'use strict';

  const SCHEMA = 'cogni.gemma4.depth_chart.v1';
  const SOURCE = {
    gpu: {
      label: 'NVIDIA GPU',
      shortLabel: 'NVIDIA GPU',
      unit: 'W',
      integralUnit: 'J',
      axisLabel: 'Board power',
      detail: 'Active NVIDIA board · watts (W)',
    },
    fluke: {
      label: 'Fluke circuit',
      shortLabel: 'Fluke circuit',
      unit: 'VA',
      integralUnit: 'VA·s',
      axisLabel: 'Circuit apparent power',
      detail: 'Whole circuit · volt-amperes (VA) · assumed grid voltage',
    },
  };

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function isPositiveInteger(value) {
    return Number.isInteger(value) && value > 0;
  }

  function requestedTokenLabel(payload) {
    return isPositiveInteger(payload?.requested_tokens) ? String(payload.requested_tokens) : 'Unavailable';
  }

  function validatePayload(payload) {
    if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return 'Power trace data is not an object.';
    if (payload.schema !== SCHEMA) return 'Power trace schema is missing or unsupported.';
    if (typeof payload.campaign_id !== 'string' || !payload.campaign_id.trim()) return 'Power trace campaign ID is missing.';
    if (!isPositiveInteger(payload.requested_tokens)) return 'Power trace requested_tokens must be a positive integer token count.';
    if (!Array.isArray(payload.runs)) return 'Power trace runs must be an array.';
    if (!Number.isInteger(payload.excluded_count) || payload.excluded_count < 0) return 'Excluded count must be a nonnegative integer.';

    const seen = new Set();
    const laneMetadata = new Map();
    for (let index = 0; index < payload.runs.length; index += 1) {
      const run = payload.runs[index];
      const prefix = `Run ${index + 1}`;
      if (!run || typeof run !== 'object' || Array.isArray(run)) return `${prefix} is not an object.`;
      if (typeof run.label !== 'string' || !run.label.trim()) return `${prefix} label is missing.`;
      if (!Number.isInteger(run.decoder_layers) || run.decoder_layers <= 0) return `${prefix} decoder layer count must be positive.`;
      if (!Number.isInteger(run.trial) || run.trial <= 0) return `${prefix} trial number must be positive.`;
      if (!Number.isFinite(run.duration_ms) || run.duration_ms <= 0) return `${prefix} duration_ms must be a positive measured duration.`;
      if (typeof run.color !== 'string' || !/^#[0-9a-f]{6}$/i.test(run.color)) return `${prefix} color must be a six-digit hex color.`;

      const pairKey = `${run.label}\u0000${run.trial}`;
      if (seen.has(pairKey)) return `Duplicate label/trial pair: ${run.label}, trial ${run.trial}.`;
      seen.add(pairKey);

      const knownLane = laneMetadata.get(run.label);
      if (knownLane && (knownLane.color.toLowerCase() !== run.color.toLowerCase() || knownLane.layers !== run.decoder_layers)) {
        return `Runs in lane ${run.label} must keep one color and decoder layer count.`;
      }
      laneMetadata.set(run.label, { color: run.color, layers: run.decoder_layers });

      for (const sourceKey of ['gpu', 'fluke']) {
        const source = SOURCE[sourceKey];
        const series = run[sourceKey];
        if (!series || typeof series !== 'object' || Array.isArray(series)) return `${prefix} ${source.label} series is missing.`;
        if (series.unit !== source.unit) return `${prefix} ${source.label} unit must be ${source.unit}.`;
        if (series.integral_unit !== source.integralUnit) return `${prefix} ${source.label} integral unit must be ${source.integralUnit}.`;
        if (!nullableNonnegativeNumber(series.mean) || !nullableNonnegativeNumber(series.integral)) {
          return `${prefix} ${source.label} mean and integral must be nonnegative numbers or null.`;
        }
        if (!Array.isArray(series.points) || series.points.length < 2) return `${prefix} ${source.label} needs at least two source samples.`;
        let previousTime = -Infinity;
        for (const point of series.points) {
          if (!Array.isArray(point) || point.length !== 2) return `${prefix} ${source.label} samples must be [timeMs, value] pairs.`;
          const [timeMs, value] = point;
          if (!Number.isFinite(timeMs) || timeMs < 0 || !Number.isFinite(value) || value < 0) {
          return `${prefix} ${source.label} source samples must contain finite nonnegative time and value.`;
          }
          if (timeMs <= previousTime) return `${prefix} ${source.label} sample times must be strictly increasing.`;
          previousTime = timeMs;
        }
        if (series.points[0][0] !== 0 || series.points.at(-1)[0] <= 0) {
          return `${prefix} ${source.label} source sample times must begin at 0 ms and have a positive duration.`;
        }
        if (series.points.at(-1)[0] !== run.duration_ms) {
          return `${prefix} ${source.label} source points must end at run duration_ms.`;
        }
      }
    }
    return null;
  }

  function nullableNonnegativeNumber(value) {
    return value === null || (Number.isFinite(value) && value >= 0);
  }

  function runsForTrial(runs, trial) {
    return (Array.isArray(runs) ? runs : []).filter(run => run && run.trial === trial);
  }

  function commonTimeDomain(runs, trial) {
    let end = 0;
    for (const run of runsForTrial(runs, trial)) {
      for (const sourceKey of ['gpu', 'fluke']) {
        const points = run[sourceKey]?.points;
        if (Array.isArray(points) && points.length) end = Math.max(end, points[points.length - 1][0]);
      }
    }
    return { start: 0, end };
  }

  function valueDomain(runs, sourceKey, trial) {
    if (!SOURCE[sourceKey]) throw new TypeError(`Unknown power trace source: ${sourceKey}`);
    let end = 0;
    for (const run of runsForTrial(runs, trial)) {
      for (const point of run[sourceKey]?.points || []) end = Math.max(end, point[1]);
    }
    return { start: 0, end: end || 1 };
  }

  function buildChartModel(payload, sourceKey, trial) {
    const error = validatePayload(payload);
    if (error) throw new TypeError(error);
    if (!SOURCE[sourceKey]) throw new TypeError(`Unknown power trace source: ${sourceKey}`);
    const labels = [...new Set(payload.runs.map(run => run.label))];
    const laneByLabel = new Map(labels.map((label, index) => [label, index]));
    const series = runsForTrial(payload.runs, trial).map(run => ({
      label: run.label,
      decoderLayers: run.decoder_layers,
      trial: run.trial,
      color: run.color,
      laneIndex: laneByLabel.get(run.label),
      unit: SOURCE[sourceKey].unit,
      integralUnit: SOURCE[sourceKey].integralUnit,
      mean: run[sourceKey].mean,
      integral: run[sourceKey].integral,
      durationMs: run[sourceKey].points[run[sourceKey].points.length - 1][0],
      points: run[sourceKey].points.map(([timeMs, value]) => ({ timeMs, value })),
    }));
    return {
      source: sourceKey,
      trial,
      lanes: labels,
      series,
      timeDomain: commonTimeDomain(payload.runs, trial),
      valueDomain: valueDomain(payload.runs, sourceKey, trial),
    };
  }

  function projectPoint(sample, plot, timeDomain, powerDomain, layout = 'lanes') {
    const laneCount = Math.max(1, sample.laneCount || 1);
    const laneIndex = clamp(Number(sample.laneIndex) || 0, 0, laneCount - 1);
    const timeSpan = Math.max(1e-12, timeDomain.end - timeDomain.start);
    const powerSpan = Math.max(1e-12, powerDomain.end - powerDomain.start);
    const laneLayout = layout === 'lanes';
    const laneStepX = laneLayout ? Math.min(32, Math.max(14, plot.width * 0.075)) : 0;
    const laneStepY = laneLayout ? Math.min(26, Math.max(12, plot.height * 0.07)) : 0;
    const usableTimeWidth = Math.max(1, plot.width - laneStepX * (laneCount - 1));
    const valueHeight = Math.max(1, plot.height - laneStepY * (laneCount - 1));
    const timeRatio = (sample.timeMs - timeDomain.start) / timeSpan;
    const powerRatio = (sample.value - powerDomain.start) / powerSpan;
    const x = plot.left + laneIndex * laneStepX + timeRatio * usableTimeWidth;
    const baselineY = plot.top + plot.height - laneIndex * laneStepY;
    const y = baselineY - powerRatio * valueHeight;
    return { x, y, baselineY, laneStepX, laneStepY, usableTimeWidth, valueHeight };
  }

  function nearestSample(points, x, y, maxDistance = 24) {
    const maxDistanceSquared = maxDistance * maxDistance;
    let nearest = null;
    let nearestDistance = maxDistanceSquared;
    for (const point of points || []) {
      const dx = point.x - x;
      const dy = point.y - y;
      const distance = dx * dx + dy * dy;
      if (distance <= nearestDistance) {
        nearest = point;
        nearestDistance = distance;
      }
    }
    return nearest;
  }

  function layoutCanvasLegend(labels, width, measureText) {
    const left = 14;
    const right = 14;
    const swatchWidth = 13;
    const labelGap = 6;
    const itemGap = 14;
    const innerWidth = Math.max(1, width - left - right);
    let x = left;
    let row = 0;
    const items = [];
    for (const label of labels) {
      const labelWidth = Math.max(0, measureText(label));
      const itemWidth = Math.min(innerWidth, swatchWidth + labelGap + labelWidth);
      if (x > left && x + itemWidth > width - right) {
        x = left;
        row += 1;
      }
      const textWidth = Math.max(0, Math.min(labelWidth, width - right - x - swatchWidth - labelGap));
      items.push({ label, x, y: 13 + row * 19, textWidth, row });
      x += itemWidth + itemGap;
    }
    return { items, rows: items.length ? row + 1 : 0, height: items.length ? 17 + row * 19 : 0 };
  }

  function panDomain(domain, fullDomain, fraction) {
    const fullSpan = Math.max(0, fullDomain.end - fullDomain.start);
    const span = clamp(domain.end - domain.start, 0, fullSpan);
    if (span === 0 || span >= fullSpan) return { ...fullDomain };
    const maxStart = fullDomain.end - span;
    const start = clamp(domain.start + fraction * span, fullDomain.start, maxStart);
    return { start, end: start + span };
  }

  function zoomDomain(domain, fullDomain, ratio, factor) {
    const fullSpan = Math.max(0, fullDomain.end - fullDomain.start);
    const currentSpan = clamp(domain.end - domain.start, 0, fullSpan);
    if (fullSpan === 0 || currentSpan === 0 || !Number.isFinite(factor) || factor <= 0) return { ...fullDomain };
    const nextSpan = clamp(currentSpan / factor, Math.min(fullSpan, Math.max(fullSpan / 1000, 1e-9)), fullSpan);
    const anchor = domain.start + clamp(ratio, 0, 1) * currentSpan;
    const anchorRatio = clamp(ratio, 0, 1);
    let start = anchor - anchorRatio * nextSpan;
    let end = start + nextSpan;
    if (start < fullDomain.start) {
      end += fullDomain.start - start;
      start = fullDomain.start;
    }
    if (end > fullDomain.end) {
      start -= end - fullDomain.end;
      end = fullDomain.end;
    }
    return { start: clamp(start, fullDomain.start, fullDomain.end), end: clamp(end, fullDomain.start, fullDomain.end) };
  }

  function element(document, tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function setPressed(button, pressed) {
    button.classList.toggle('active', pressed);
    button.setAttribute('aria-pressed', String(pressed));
  }

  function safeText(value) {
    return value == null ? '' : String(value);
  }

  function displayNumber(value) {
    if (!Number.isFinite(value)) return '—';
    return String(value);
  }

  function axisNumber(value) {
    if (!Number.isFinite(value)) return '—';
    return new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(value);
  }

  function displayTime(value) {
    return `${axisNumber(value)} ms`;
  }

  function formatMetric(value, unit) {
    return value === null || value === undefined ? 'Not reported' : `${axisNumber(value)} ${unit}`;
  }

  function dataAttribute(section, key, value) {
    section.dataset[key] = safeText(value);
  }

  function createShell(document, section, payload, error) {
    const root = element(document, 'div', 'depth-explorer-root');
    root.dataset.depthChartApp = 'true';

    const hero = element(document, 'header', 'depth-hero');
    const heroCopy = element(document, 'div', 'depth-hero-copy');
    heroCopy.append(element(document, 'p', 'depth-eyebrow', 'MEASURED ON GPUTER'));
    heroCopy.append(element(document, 'h2', 'depth-title', 'Fewer layers. A shorter power trace.'));
    heroCopy.append(element(document, 'p', 'depth-subtitle', 'Compare each measured run over its actual token-generation time.'));
    const heroBadges = element(document, 'div', 'depth-hero-badges');
    heroBadges.append(element(document, 'span', 'depth-hero-badge', `${requestedTokenLabel(payload)} TOKENS`));
    heroBadges.append(element(document, 'span', 'depth-hero-badge depth-hero-badge-muted', 'QUALITY UNVALIDATED'));
    hero.append(heroCopy, heroBadges);
    root.append(hero);

    const controls = element(document, 'div', 'depth-controls');
    const sourceControl = element(document, 'div', 'depth-control-block');
    sourceControl.append(element(document, 'span', 'depth-control-label', 'Measurement source'));
    const sourceButtons = element(document, 'div', 'depth-segmented');
    sourceButtons.setAttribute('role', 'group');
    sourceButtons.setAttribute('aria-label', 'Measurement source and unit');
    const gpuButton = element(document, 'button', 'depth-segment-button', 'NVIDIA GPU (W)');
    gpuButton.type = 'button';
    gpuButton.dataset.depthSource = 'gpu';
    const flukeButton = element(document, 'button', 'depth-segment-button', 'Fluke circuit (VA)');
    flukeButton.type = 'button';
    flukeButton.dataset.depthSource = 'fluke';
    sourceButtons.append(gpuButton, flukeButton);
    sourceControl.append(sourceButtons);

    const trialControl = element(document, 'label', 'depth-control-block depth-trial-control');
    trialControl.append(element(document, 'span', 'depth-control-label', 'Trial'));
    const trialSelect = element(document, 'select', 'depth-select');
    trialSelect.setAttribute('aria-label', 'Trial');
    trialSelect.dataset.depthTrial = 'true';
    trialControl.append(trialSelect);

    const layoutControl = element(document, 'div', 'depth-control-block');
    layoutControl.append(element(document, 'span', 'depth-control-label', 'View'));
    const layoutButtons = element(document, 'div', 'depth-segmented');
    layoutButtons.setAttribute('role', 'group');
    layoutButtons.setAttribute('aria-label', 'Chart layout');
    const lanesButton = element(document, 'button', 'depth-segment-button', '3D lanes');
    lanesButton.type = 'button';
    lanesButton.dataset.depthLayout = 'lanes';
    const overlayButton = element(document, 'button', 'depth-segment-button', '2D overlay');
    overlayButton.type = 'button';
    overlayButton.dataset.depthLayout = 'overlay';
    layoutButtons.append(lanesButton, overlayButton);
    layoutControl.append(layoutButtons);

    const actions = element(document, 'div', 'depth-control-actions');
    const resetButton = element(document, 'button', 'depth-action-button', 'Reset view');
    resetButton.type = 'button';
    resetButton.dataset.depthReset = 'true';
    const downloadButton = element(document, 'button', 'depth-action-button depth-action-primary', 'Download PNG');
    downloadButton.type = 'button';
    downloadButton.dataset.depthDownload = 'true';
    actions.append(resetButton, downloadButton);
    controls.append(sourceControl, trialControl, layoutControl, actions);
    root.append(controls);

    const current = element(document, 'div', 'depth-current-selection');
    current.dataset.depthCurrent = 'true';
    root.append(current);

    const chartPanel = element(document, 'div', 'depth-chart-panel');
    const chartHeading = element(document, 'div', 'depth-chart-heading');
    const chartHeadingCopy = element(document, 'div', 'depth-chart-heading-copy');
    chartHeadingCopy.append(element(document, 'strong', 'depth-chart-title', 'Measured power over elapsed time'));
    chartHeadingCopy.append(element(document, 'span', 'depth-chart-subtitle', 'Each colored path is one decoder-depth run for the selected trial.'));
    const chartUnit = element(document, 'span', 'depth-unit-chip');
    chartUnit.dataset.depthUnit = 'true';
    chartHeading.append(chartHeadingCopy, chartUnit);
    chartPanel.append(chartHeading);

    const canvasWrap = element(document, 'div', 'depth-canvas-wrap');
    const canvas = element(document, 'canvas', 'depth-canvas');
    canvas.dataset.depthCanvas = 'true';
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('tabindex', '0');
    canvas.setAttribute('aria-label', 'Interactive power trace chart. Use arrow keys to inspect source samples, plus and minus to zoom, and R to reset.');
    canvas.textContent = 'Interactive measured power chart. Use the controls above to inspect the traces.';
    const tooltip = element(document, 'div', 'depth-tooltip');
    tooltip.dataset.depthTooltip = 'true';
    tooltip.setAttribute('role', 'tooltip');
    tooltip.hidden = true;
    canvasWrap.append(canvas, tooltip);
    chartPanel.append(canvasWrap);

    const chartFooter = element(document, 'div', 'depth-chart-footer');
    chartFooter.append(element(document, 'span', 'depth-interaction-hint', 'Hover a source point · drag to pan · scroll to zoom · R to reset'));
    const chartStatus = element(document, 'span', 'depth-chart-status', '');
    chartStatus.dataset.depthStatus = 'true';
    chartStatus.setAttribute('role', 'status');
    chartStatus.setAttribute('aria-live', 'polite');
    chartFooter.append(chartStatus);
    chartPanel.append(chartFooter);
    root.append(chartPanel);

    const legend = element(document, 'div', 'depth-legend');
    legend.dataset.depthLegend = 'true';
    legend.setAttribute('aria-label', 'Decoder-depth runs');
    root.append(legend);

    const batchHeading = element(document, 'div', 'depth-batch-heading');
    batchHeading.append(element(document, 'h3', 'depth-batch-title', 'Batch readings'));
    batchHeading.append(element(document, 'span', 'depth-batch-caption', 'Selected source and trial · reported summaries'));
    root.append(batchHeading);
    const cards = element(document, 'div', 'depth-batch-grid');
    cards.dataset.depthCards = 'true';
    root.append(cards);

    const tokenLabel = requestedTokenLabel(payload);
    const scope = element(document, 'p', 'depth-scope-note', `Historical measured gputer traces · ${tokenLabel} tokens per trial · experimental quality not validated. NVIDIA is active-board W; Fluke is whole-circuit VA using assumed grid voltage. Power integration linearly interpolates bracketing source samples at request boundaries. Curves stop at their reported endpoints; elapsed time is not normalized.`);
    scope.dataset.depthScope = 'true';
    root.append(scope);
    const unavailable = element(document, 'p', 'depth-unavailable', error || '');
    unavailable.dataset.depthUnavailable = 'true';
    unavailable.hidden = !error;
    root.append(unavailable);

    const dataNode = section.querySelector('#depth-chart-data');
    if (dataNode) section.replaceChildren(dataNode, root);
    else section.replaceChildren(root);

    return {
      root,
      controls,
      gpuButton,
      flukeButton,
      trialSelect,
      lanesButton,
      overlayButton,
      resetButton,
      downloadButton,
      current,
      canvas,
      tooltip,
      chartStatus,
      chartUnit,
      legend,
      cards,
      unavailable,
    };
  }

  function sizeCanvas(canvas, windowObject) {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width || 1));
    const height = Math.max(360, Math.floor(rect.height || 470));
    const dpr = Math.max(1, windowObject && Number(windowObject.devicePixelRatio) || 1);
    if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
  }

  function colorWithAlpha(color, alpha) {
    const value = color.slice(1);
    const red = parseInt(value.slice(0, 2), 16);
    const green = parseInt(value.slice(2, 4), 16);
    const blue = parseInt(value.slice(4, 6), 16);
    return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
  }

  function drawLine(ctx, points, color, width = 1) {
    if (!points.length) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let index = 1; index < points.length; index += 1) ctx.lineTo(points[index].x, points[index].y);
    ctx.stroke();
  }

  function drawEmptyState(ctx, width, message) {
    ctx.save();
    ctx.fillStyle = '#5f675f';
    ctx.font = '600 15px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(message, width / 2, 76, Math.max(140, width - 36));
    ctx.restore();
  }

  function buildChartDom(document, payload, error) {
    const section = document.getElementById('depth-explorer');
    if (!section) return null;
    const elements = createShell(document, section, payload, error);
    const trials = payload && Array.isArray(payload.runs)
      ? [...new Set(payload.runs.map(run => run && run.trial).filter(Number.isInteger))].sort((a, b) => a - b)
      : [];
    const trialSelect = elements.trialSelect;
    for (const trial of trials) {
      const option = element(document, 'option', '', `Trial ${trial}`);
      option.value = String(trial);
      trialSelect.append(option);
    }
    trialSelect.disabled = Boolean(error) || trials.length === 0;
    elements.gpuButton.disabled = Boolean(error) || trials.length === 0;
    elements.flukeButton.disabled = Boolean(error) || trials.length === 0;
    elements.lanesButton.disabled = Boolean(error) || trials.length === 0;
    elements.overlayButton.disabled = Boolean(error) || trials.length === 0;
    elements.resetButton.disabled = Boolean(error) || trials.length === 0;
    elements.downloadButton.disabled = Boolean(error) || trials.length === 0;
    if (!error && trials.length === 0) elements.unavailable.textContent = 'No eligible measured power traces are available for this campaign.';
    if (!error && trials.length === 0) elements.unavailable.hidden = false;
    return { section, elements, trials };
  }

  function mount(section, providedPayload, providedError) {
    if (!section || !section.ownerDocument) return null;
    if (section.dataset.depthMounted === 'true') return section.__depthChartController || null;
    const document = section.ownerDocument;
    let payload = providedPayload;
    let error = providedError || null;
    if (payload === undefined) {
      const dataNode = section.querySelector('#depth-chart-data');
      if (!dataNode) error = error || 'Power trace data is unavailable.';
      else {
        try {
          payload = JSON.parse(dataNode.textContent || '');
        } catch {
          error = error || 'Power trace data could not be read.';
        }
      }
    }
    if (!error) error = validatePayload(payload);
    const dom = buildChartDom(document, payload, error);
    if (!dom) return null;
    const { elements } = dom;
    const windowObject = document.defaultView || (typeof window === 'object' ? window : null);
    const trials = dom.trials;
    const initialTrial = trials.includes(1) ? 1 : trials[0] ?? null;
    const state = {
      source: 'gpu',
      layout: 'lanes',
      trial: initialTrial,
      viewDomain: null,
      fullDomain: null,
      plot: null,
      visibleNodes: [],
      hovered: null,
      dragStartX: null,
      dragStartDomain: null,
      dragging: false,
    };

    if (initialTrial !== null) elements.trialSelect.value = String(initialTrial);
    if (!error && payload && payload.runs.length) {
      state.fullDomain = commonTimeDomain(payload.runs, initialTrial);
      state.viewDomain = { ...state.fullDomain };
    }

    function updateSelectionLabels(model) {
      const source = SOURCE[state.source];
      elements.current.textContent = `${source.shortLabel} · ${source.unit} · trial ${state.trial} · ${state.layout === 'lanes' ? '3D lanes' : '2D overlay'}`;
      elements.chartUnit.textContent = `${source.axisLabel} (${source.unit})`;
      elements.canvas.setAttribute('aria-label', `Interactive ${source.axisLabel.toLowerCase()} chart in ${source.unit}. Time is measured in milliseconds. ${state.layout === 'lanes' ? 'Curves are separated into categorical depth lanes.' : 'Curves are overlaid.'} Selected trial ${state.trial}. Use arrow keys to inspect source samples, plus and minus to zoom, and R to reset.`);
      dataAttribute(dom.section, 'source', state.source);
      dataAttribute(dom.section, 'unit', source.unit);
      dataAttribute(dom.section, 'trial', state.trial);
      dataAttribute(dom.section, 'layout', state.layout);
      dataAttribute(dom.section, 'viewStartMs', state.viewDomain?.start ?? '');
      dataAttribute(dom.section, 'viewEndMs', state.viewDomain?.end ?? '');
      dataAttribute(dom.section, 'hoverLabel', state.hovered?.label ?? '');
      dataAttribute(dom.section, 'depthReady', error ? 'false' : 'true');
      dom.section.dataset.seriesCount = String(model ? model.series.length : 0);
    }

    function clearTooltip() {
      elements.tooltip.hidden = true;
      elements.tooltip.replaceChildren();
      state.hovered = null;
      dataAttribute(dom.section, 'hoverLabel', '');
    }

    function renderLegend(model) {
      elements.legend.replaceChildren();
      if (!model || !model.lanes.length) {
        elements.legend.append(element(document, 'span', 'depth-legend-empty', 'No decoder-depth runs to display'));
        return;
      }
      for (let index = 0; index < model.lanes.length; index += 1) {
        const label = model.lanes[index];
        const run = payload.runs.find(candidate => candidate.label === label);
        const item = element(document, 'div', 'depth-legend-item');
        item.setAttribute('role', 'listitem');
        const swatch = element(document, 'span', 'depth-legend-swatch');
        swatch.style.backgroundColor = run.color;
        swatch.setAttribute('aria-hidden', 'true');
        const number = element(document, 'span', 'depth-legend-number', String(index + 1).padStart(2, '0'));
        const copy = element(document, 'span', 'depth-legend-copy');
        copy.append(element(document, 'strong', 'depth-legend-name', label));
        copy.append(element(document, 'span', 'depth-legend-detail', `${run.decoder_layers} decoder layers`));
        item.append(swatch, number, copy);
        elements.legend.append(item);
      }
    }

    function renderCards(model) {
      elements.cards.replaceChildren();
      for (const label of model.lanes) {
        const run = model.series.find(candidate => candidate.label === label);
        const sourceRun = payload.runs.find(candidate => candidate.label === label);
        const card = element(document, 'article', 'depth-batch-card');
        card.style.setProperty('--depth-card-color', sourceRun.color);
        const head = element(document, 'div', 'depth-batch-card-head');
        const title = element(document, 'div', 'depth-batch-card-title');
        title.append(element(document, 'strong', 'depth-batch-name', label));
        title.append(element(document, 'span', 'depth-batch-layer', `${sourceRun.decoder_layers} layers · trial ${state.trial}`));
        const badge = element(document, 'span', 'depth-batch-unit', SOURCE[state.source].unit);
        head.append(title, badge);
        card.append(head);
        if (!run) {
          card.append(element(document, 'p', 'depth-card-missing', `No measured data for trial ${state.trial}.`));
        } else {
          const metrics = element(document, 'dl', 'depth-metrics');
          const rows = [
            ['Duration', displayTime(run.durationMs)],
            ['Mean', formatMetric(run.mean, run.unit)],
            ['Integral', formatMetric(run.integral, run.integralUnit)],
          ];
          for (const [labelText, valueText] of rows) {
            const row = element(document, 'div', 'depth-metric-row');
            row.append(element(document, 'dt', '', labelText));
            row.append(element(document, 'dd', '', valueText));
            metrics.append(row);
          }
          card.append(metrics);
        }
        elements.cards.append(card);
      }
    }

    function setTooltip(point, canvasWidth, canvasHeight) {
      if (!point) {
        clearTooltip();
        elements.chartStatus.textContent = 'Hover a source sample to inspect its value.';
        return;
      }
      state.hovered = point;
      const source = SOURCE[state.source];
      elements.tooltip.replaceChildren();
      elements.tooltip.append(element(document, 'strong', 'depth-tooltip-title', point.label));
      elements.tooltip.append(element(document, 'span', 'depth-tooltip-meta', `Trial ${point.trial} · ${source.shortLabel}`));
      elements.tooltip.append(element(document, 'span', 'depth-tooltip-value', `${displayNumber(point.value)} ${source.unit} at ${displayTime(point.timeMs)}`));
      elements.tooltip.hidden = false;
      const left = clamp(point.x + 12, 8, Math.max(8, canvasWidth - 240));
      const top = clamp(point.y - 18, 8, Math.max(8, canvasHeight - 94));
      elements.tooltip.style.left = `${left}px`;
      elements.tooltip.style.top = `${top}px`;
      elements.chartStatus.textContent = `${point.label}, trial ${point.trial}: ${displayNumber(point.value)} ${source.unit} at ${displayTime(point.timeMs)}.`;
      dataAttribute(dom.section, 'hoverLabel', point.label);
      dataAttribute(dom.section, 'hoverTimeMs', point.timeMs);
      dataAttribute(dom.section, 'hoverValue', point.value);
    }

    function render() {
      const sized = sizeCanvas(elements.canvas, windowObject);
      if (!sized) return;
      const { ctx, width, height } = sized;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#fffefa';
      ctx.fillRect(0, 0, width, height);
      state.visibleNodes = [];
      const model = !error && payload ? buildChartModel(payload, state.source, state.trial) : null;
      updateSelectionLabels(model);
      renderLegend(model);
      if (model) renderCards(model);
      else elements.cards.replaceChildren();

      if (error) {
        drawEmptyState(ctx, width, 'Measured power traces are unavailable.');
        elements.chartStatus.textContent = 'Power trace data is unavailable.';
        clearTooltip();
        return;
      }
      if (!model || model.series.length === 0) {
        drawEmptyState(ctx, width, 'No eligible measured power traces are available.');
        elements.chartStatus.textContent = 'No measured traces for this trial.';
        clearTooltip();
        return;
      }

      const source = SOURCE[state.source];
      const laneCount = Math.max(1, model.lanes.length);
      ctx.save();
      ctx.font = '600 10px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      const canvasLegend = layoutCanvasLegend(model.lanes, width, label => ctx.measureText(label).width);
      for (const item of canvasLegend.items) {
        const run = payload.runs.find(candidate => candidate.label === item.label);
        ctx.fillStyle = run.color;
        ctx.fillRect(item.x, item.y - 2, 13, 3);
        ctx.fillStyle = '#445044';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(item.label, item.x + 19, item.y, item.textWidth);
      }
      ctx.restore();
      const margin = { top: Math.max(30, canvasLegend.height + 17), right: 22, bottom: 58, left: width < 460 ? 52 : 72 };
      const plot = {
        left: margin.left,
        top: margin.top,
        width: Math.max(1, width - margin.left - margin.right),
        height: Math.max(1, height - margin.top - margin.bottom),
      };
      state.plot = plot;
      const timeDomain = state.viewDomain || model.timeDomain;
      const powerDomain = model.valueDomain;
      const project = (timeMs, value, laneIndex) => projectPoint(
        { timeMs, value, laneIndex, laneCount }, plot, timeDomain, powerDomain, state.layout,
      );
      const frontStart = project(timeDomain.start, 0, 0);
      const frontEnd = project(timeDomain.end, 0, 0);
      const stepX = frontStart.laneStepX;
      const stepY = frontStart.laneStepY;

      ctx.save();
      ctx.font = '11px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.textBaseline = 'middle';
      ctx.lineWidth = 1;
      const tickFractions = [0, 0.25, 0.5, 0.75, 1];
      for (const fraction of tickFractions) {
        const value = powerDomain.end * fraction;
        for (let laneIndex = 0; laneIndex < laneCount; laneIndex += 1) {
          const start = project(timeDomain.start, value, laneIndex);
          const end = project(timeDomain.end, value, laneIndex);
          drawLine(ctx, [start, end], fraction === 0 ? '#d2d5cd' : '#e9ebe4', fraction === 0 ? 1.2 : 1);
        }
        const labelPoint = project(timeDomain.start, value, 0);
        ctx.fillStyle = '#687267';
        ctx.textAlign = 'right';
        ctx.fillText(axisNumber(value), plot.left - 9, labelPoint.y);
      }

      if (state.layout === 'lanes' && laneCount > 1) {
        const timeFractions = [0, 0.25, 0.5, 0.75, 1];
        for (const fraction of timeFractions) {
          const timeMs = timeDomain.start + (timeDomain.end - timeDomain.start) * fraction;
          const depthPoints = [];
          for (let laneIndex = 0; laneIndex < laneCount; laneIndex += 1) depthPoints.push(project(timeMs, 0, laneIndex));
          drawLine(ctx, depthPoints, '#e2e5dd', 1);
        }
      }

      const yTop = project(timeDomain.start, powerDomain.end, 0);
      drawLine(ctx, [frontStart, frontEnd], '#687267', 1.35);
      drawLine(ctx, [frontStart, yTop], '#687267', 1.35);
      if (state.layout === 'lanes' && laneCount > 1) {
        const rearStart = project(timeDomain.start, 0, laneCount - 1);
        drawLine(ctx, [frontStart, rearStart], '#687267', 1.2);
      }

      const timeSpan = Math.max(1e-12, timeDomain.end - timeDomain.start);
      const axisTimeWidth = Math.max(1, plot.width - stepX * (laneCount - 1));
      const timeTickFractions = axisTimeWidth < 300 ? [0, 0.5, 1] : tickFractions;
      for (const fraction of timeTickFractions) {
        const timeMs = timeDomain.start + timeSpan * fraction;
        const point = project(timeMs, 0, 0);
        ctx.strokeStyle = '#9da59b';
        ctx.beginPath();
        ctx.moveTo(point.x, point.y);
        ctx.lineTo(point.x, point.y + 5);
        ctx.stroke();
        ctx.fillStyle = '#687267';
        ctx.textAlign = 'center';
        ctx.fillText(axisNumber(timeMs), point.x, point.y + 16);
      }
      ctx.fillStyle = '#445044';
      ctx.font = '600 11px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('TIME (MS)', plot.left + axisTimeWidth / 2, height - 9);
      ctx.save();
      ctx.translate(14, plot.top + plot.height / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(`${source.axisLabel.toUpperCase()} (${source.unit})`, 0, 0);
      ctx.restore();

      ctx.save();
      ctx.beginPath();
      ctx.rect(plot.left, plot.top, plot.width, plot.height);
      ctx.clip();
      for (const series of model.series) {
        const measured = series.points.map(point => ({
          ...project(point.timeMs, point.value, series.laneIndex),
          label: series.label,
          trial: series.trial,
          color: series.color,
          timeMs: point.timeMs,
          value: point.value,
          laneIndex: series.laneIndex,
        }));
        if (measured.length > 1) drawLine(ctx, measured, series.color, state.layout === 'overlay' ? 2.4 : 2.7);
        for (const point of measured) {
          ctx.fillStyle = series.color;
          ctx.beginPath();
          ctx.arc(point.x, point.y, measured.length > 1400 ? 1.05 : 1.75, 0, Math.PI * 2);
          ctx.fill();
          if (point.timeMs >= timeDomain.start && point.timeMs <= timeDomain.end) state.visibleNodes.push(point);
        }
      }
      if (state.hovered) {
        const current = state.visibleNodes.find(point => point.label === state.hovered.label && point.timeMs === state.hovered.timeMs);
        if (current) {
          ctx.strokeStyle = '#fffefa';
          ctx.lineWidth = 4;
          ctx.beginPath();
          ctx.arc(current.x, current.y, 5, 0, Math.PI * 2);
          ctx.stroke();
          ctx.fillStyle = current.color;
          ctx.beginPath();
          ctx.arc(current.x, current.y, 4, 0, Math.PI * 2);
          ctx.fill();
          state.hovered = current;
        }
      }
      ctx.restore();
      ctx.restore();

      if (state.fullDomain && state.viewDomain) {
        dataAttribute(dom.section, 'viewStartMs', state.viewDomain.start);
        dataAttribute(dom.section, 'viewEndMs', state.viewDomain.end);
      }
      if (!state.hovered) {
        elements.tooltip.hidden = true;
        elements.tooltip.replaceChildren();
        elements.chartStatus.textContent = 'Hover a source sample to inspect its value.';
        dataAttribute(dom.section, 'hoverLabel', '');
        delete dom.section.dataset.hoverTimeMs;
        delete dom.section.dataset.hoverValue;
      } else setTooltip(state.hovered, width, height);
    }

    function updateSource(source) {
      if (!SOURCE[source] || source === state.source || error) return;
      state.source = source;
      clearTooltip();
      setPressed(elements.gpuButton, source === 'gpu');
      setPressed(elements.flukeButton, source === 'fluke');
      render();
    }

    function updateLayout(layout) {
      if (!['lanes', 'overlay'].includes(layout) || layout === state.layout || error) return;
      state.layout = layout;
      clearTooltip();
      setPressed(elements.lanesButton, layout === 'lanes');
      setPressed(elements.overlayButton, layout === 'overlay');
      render();
    }

    function resetView() {
      if (!state.fullDomain) return;
      state.viewDomain = { ...state.fullDomain };
      clearTooltip();
      render();
      elements.canvas.focus();
    }

    function zoomAt(ratio, factor) {
      if (!state.viewDomain || !state.fullDomain) return;
      state.viewDomain = zoomDomain(state.viewDomain, state.fullDomain, ratio, factor);
      clearTooltip();
      render();
    }

    function canvasPosition(event) {
      const rect = elements.canvas.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    }

    function panByPixels(deltaPixels) {
      if (!state.plot || !state.viewDomain || !state.fullDomain) return;
      const laneCount = Math.max(1, new Set(payload.runs.map(run => run.label)).size);
      const laneStepX = state.layout === 'lanes' ? Math.min(32, Math.max(14, state.plot.width * 0.075)) : 0;
      const width = Math.max(1, state.plot.width - laneStepX * (laneCount - 1));
      state.viewDomain = panDomain(state.viewDomain, state.fullDomain, -deltaPixels / width);
      clearTooltip();
      render();
    }

    elements.gpuButton.addEventListener('click', () => updateSource('gpu'));
    elements.flukeButton.addEventListener('click', () => updateSource('fluke'));
    elements.trialSelect.addEventListener('change', () => {
      if (error) return;
      const trial = Number(elements.trialSelect.value);
      if (!Number.isInteger(trial) || trial === state.trial) return;
      state.trial = trial;
      state.fullDomain = commonTimeDomain(payload.runs, trial);
      state.viewDomain = { ...state.fullDomain };
      clearTooltip();
      render();
    });
    elements.lanesButton.addEventListener('click', () => updateLayout('lanes'));
    elements.overlayButton.addEventListener('click', () => updateLayout('overlay'));
    elements.resetButton.addEventListener('click', resetView);
    elements.downloadButton.addEventListener('click', () => {
      try {
        const anchor = element(document, 'a');
        const campaign = payload.campaign_id.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 48) || 'campaign';
        anchor.href = elements.canvas.toDataURL('image/png');
        anchor.download = `gemma4-power-trace-${campaign}-trial-${state.trial}-${state.source}.png`;
        anchor.hidden = true;
        section.append(anchor);
        anchor.click();
        anchor.remove();
        elements.chartStatus.textContent = 'PNG download prepared for the current chart view.';
      } catch {
        elements.chartStatus.textContent = 'This browser could not export the current chart as PNG.';
      }
    });

    elements.canvas.addEventListener('pointerdown', event => {
      if (!state.plot || event.button !== 0) return;
      const position = canvasPosition(event);
      if (position.x < state.plot.left || position.x > state.plot.left + state.plot.width || position.y < state.plot.top || position.y > state.plot.top + state.plot.height) return;
      state.dragStartX = position.x;
      state.dragStartDomain = state.viewDomain ? { ...state.viewDomain } : null;
      state.dragging = false;
      if (elements.canvas.setPointerCapture) elements.canvas.setPointerCapture(event.pointerId);
    });
    elements.canvas.addEventListener('pointermove', event => {
      const position = canvasPosition(event);
      if (state.dragStartX !== null && state.dragStartDomain) {
        const delta = position.x - state.dragStartX;
        if (Math.abs(delta) > 3) state.dragging = true;
        if (state.dragging) {
          const laneCount = Math.max(1, new Set(payload.runs.map(run => run.label)).size);
          const laneStepX = state.layout === 'lanes' ? Math.min(32, Math.max(14, state.plot.width * 0.075)) : 0;
          const width = Math.max(1, state.plot.width - laneStepX * (laneCount - 1));
          state.viewDomain = panDomain(state.dragStartDomain, state.fullDomain, -delta / width);
          clearTooltip();
          render();
          return;
        }
      }
      if (!state.plot || position.x < state.plot.left || position.x > state.plot.left + state.plot.width || position.y < state.plot.top || position.y > state.plot.top + state.plot.height) {
        if (state.hovered) {
          clearTooltip();
          render();
        }
        return;
      }
      const point = nearestSample(state.visibleNodes, position.x, position.y, 22);
      if (point !== state.hovered) {
        state.hovered = point;
        render();
      }
    });
    function finishPointer(event) {
      if (state.dragStartX === null) return;
      const dragged = state.dragging;
      state.dragStartX = null;
      state.dragStartDomain = null;
      state.dragging = false;
      if (!dragged) {
        const position = canvasPosition(event);
        const point = nearestSample(state.visibleNodes, position.x, position.y, 22);
        state.hovered = point;
      }
      render();
    }
    elements.canvas.addEventListener('pointerup', finishPointer);
    elements.canvas.addEventListener('pointercancel', finishPointer);
    elements.canvas.addEventListener('pointerleave', () => {
      if (state.dragStartX !== null) return;
      if (state.hovered) {
        clearTooltip();
        render();
      }
    });
    elements.canvas.addEventListener('wheel', event => {
      if (!state.viewDomain || !state.plot) return;
      event.preventDefault();
      const position = canvasPosition(event);
      const laneCount = Math.max(1, new Set(payload.runs.map(run => run.label)).size);
      const laneStepX = state.layout === 'lanes' ? Math.min(32, Math.max(14, state.plot.width * 0.075)) : 0;
      const width = Math.max(1, state.plot.width - laneStepX * (laneCount - 1));
      const nearest = nearestSample(state.visibleNodes, position.x, position.y, 40);
      const laneIndex = nearest ? nearest.laneIndex : 0;
      const ratio = clamp((position.x - state.plot.left - laneIndex * laneStepX) / width, 0, 1);
      zoomAt(ratio, event.deltaY > 0 ? 1 / 1.2 : 1.2);
    }, { passive: false });
    elements.canvas.addEventListener('dblclick', resetView);
    elements.canvas.addEventListener('keydown', event => {
      if (['ArrowLeft', 'ArrowRight', '+', '=', '-', '0', 'r', 'R', 'Escape'].includes(event.key)) event.preventDefault();
      if (event.key === '0' || event.key === 'r' || event.key === 'R') {
        resetView();
        return;
      }
      if (event.key === '+' || event.key === '=') {
        zoomAt(0.5, 1.5);
        return;
      }
      if (event.key === '-') {
        zoomAt(0.5, 1 / 1.5);
        return;
      }
      if (event.key === 'Escape') {
        clearTooltip();
        render();
        return;
      }
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        if (!state.visibleNodes.length) return;
        const currentIndex = state.hovered ? state.visibleNodes.indexOf(state.hovered) : -1;
        const direction = event.key === 'ArrowRight' ? 1 : -1;
        const nextIndex = currentIndex < 0
          ? (direction > 0 ? 0 : state.visibleNodes.length - 1)
          : clamp(currentIndex + direction, 0, state.visibleNodes.length - 1);
        state.hovered = state.visibleNodes[nextIndex];
        render();
      }
    });

    setPressed(elements.gpuButton, true);
    setPressed(elements.flukeButton, false);
    setPressed(elements.lanesButton, true);
    setPressed(elements.overlayButton, false);
    dataAttribute(dom.section, 'depthMounted', 'true');
    render();
    if (windowObject && typeof windowObject.ResizeObserver === 'function') {
      const observer = new windowObject.ResizeObserver(() => render());
      observer.observe(elements.canvas);
    } else if (windowObject) {
      windowObject.addEventListener('resize', render, { passive: true });
    }
    const controller = { render, resetView, updateSource, updateLayout, state, payload, error };
    section.__depthChartController = controller;
    return controller;
  }

  function autoMount(document) {
    if (!document || !document.getElementById || !document.getElementById('depth-explorer')) return null;
    const section = document.getElementById('depth-explorer');
    if (section.dataset.depthMounted === 'true') return section.__depthChartController || null;
    return mount(section);
  }

  return {
    SCHEMA,
    SOURCE,
    buildChartModel,
    commonTimeDomain,
    formatMetric,
    layoutCanvasLegend,
    mount,
    nearestSample,
    panDomain,
    projectPoint,
    validatePayload,
    valueDomain,
    zoomDomain,
    autoMount,
  };
});
