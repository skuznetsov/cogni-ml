(() => {
  "use strict";

  const i18n = window.StudioI18n;
  const t = i18n.t;
  const segmentCountText = i18n.segmentCount;
  const formatSeconds = i18n.formatSeconds;
  const formatStamp = i18n.formatStamp;

  const byId = (id) => document.getElementById(id);
  const state = { sessions: [], voices: [], assets: [], jobs: [], capabilities: {} };
  const drafts = new Map();
  const composerDrafts = new Map();
  const settingsDisclosure = new Map();
  const historyDisclosure = new Map();
  const locks = new Set();
  let activeSessionId = null;
  let activeSegmentId = null;
  let showArchivedSessions = false;
  let libraryTab = "voices";
  let assetFilter = "current";
  let newSessionName = "";
  let renameEditing = false;
  let renameDraft = "";
  let voiceForm = { name: "", transcript: "", file: null, fileName: "", previewUrl: null, duration: null };
  let voiceDraftVersion = 0;
  let voiceSaving = false;
  let recorderPhase = "idle";
  let voiceRecorder = null;
  let exportFormat = null;
  let renderedEditorSessionId = null;
  let lastKeys = {};
  let noticeTimer = null;
  let pollTimer = null;
  let refreshPromise = null;
  let stateRequestSequence = 0;
  let connectionState = { connected: false, message: "Connecting…" };
  let noticeMessage = null;

  class LocalizedError extends Error {
    constructor(message, parameters) {
      super(message);
      this.parameters = parameters;
    }
  }

  const node = (tag, className, text) => {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text !== undefined && text !== null) item.textContent = String(text);
    return item;
  };

  const makeButton = (label, className, action, attrs) => {
    const item = node("button", className, label);
    item.type = "button";
    if (action) item.dataset.action = action;
    if (attrs) {
      Object.keys(attrs).forEach((key) => {
        if (key === "disabled") item.disabled = Boolean(attrs[key]);
        else if (key === "title") item.title = attrs[key];
        else if (key === "ariaLabel") item.setAttribute("aria-label", attrs[key]);
        else if (key === "pressed") item.setAttribute("aria-pressed", String(attrs[key]));
      });
    }
    return item;
  };

  const append = (parent, ...children) => children.forEach((child) => child && parent.append(child));
  const sameId = (left, right) => left !== null && left !== undefined && String(left) === String(right);
  const idPart = (value) => encodeURIComponent(String(value));
  const selectedSession = () => state.sessions.find((session) => sameId(session.id, activeSessionId)) || null;
  const sessionSegments = (session) => (session && Array.isArray(session.segments) ? session.segments : []);
  const currentComposerText = () => composerDrafts.get(String(activeSessionId)) || "";
  const setComposerText = (value) => {
    if (activeSessionId !== null) composerDrafts.set(String(activeSessionId), value);
  };
  const activeJobs = () => state.jobs.filter((job) => job.status === "queued" || job.status === "running");

  function showNotice(message, kind) {
    const target = byId("notice");
    noticeMessage = message;
    target.textContent = typeof message === "function" ? message() : message;
    target.className = "notice" + (kind ? " is-" + kind : "");
    target.hidden = !message;
    if (noticeTimer) window.clearTimeout(noticeTimer);
    if (message && kind !== "error") {
      noticeTimer = window.setTimeout(() => { target.hidden = true; }, 6500);
    }
  }

  function updateConnection(connected, message) {
    connectionState = { connected, message };
    const status = byId("connection-status");
    const group = status.parentElement;
    status.textContent = t(message);
    group.classList.toggle("is-online", connected);
    group.classList.toggle("is-error", !connected);
  }

  function errorText(error) {
    const message = error instanceof LocalizedError ? t(error.message, error.parameters) : error && error.message ? error.message : String(error || t("Unknown error"));
    return message.length > 700 ? message.slice(0, 697) + "…" : message;
  }

  async function request(path, options) {
    const settings = options || {};
    const headers = Object.assign({ Accept: "application/json" }, settings.headers || {});
    const response = await fetch(path, Object.assign({}, settings, {
      headers,
      cache: "no-store",
      credentials: "same-origin"
    }));
    const raw = await response.text();
    let payload = null;
    if (raw) {
      try { payload = JSON.parse(raw); }
      catch (_) { payload = { message: raw }; }
    }
    if (!response.ok) {
      if (payload && (payload.error || payload.message)) throw new Error(String(payload.error || payload.message));
      throw new LocalizedError("Request failed ({status})", { status: response.status });
    }
    return payload;
  }

  function jsonRequest(method, body) {
    return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  }

  function draftFromSegment(segment) {
    return {
      id: segment.id,
      text: segment.text || "",
      voice_id: segment.voice_id || null,
      seed: segment.seed === undefined ? 0 : segment.seed,
      max_frames: segment.max_frames === undefined ? 512 : segment.max_frames,
      auto_frames: segment.auto_frames === true,
      temperature: segment.temperature === undefined ? 0.8 : segment.temperature,
      top_k: segment.top_k === undefined ? 30 : segment.top_k,
      top_p: segment.top_p === undefined ? 0.9 : segment.top_p,
      baseRevision: segment.revision,
      dirty: false,
      saving: false,
      conflict: false,
      version: 0,
      savePromise: null
    };
  }

  function ingestState(payload) {
    if (!payload || typeof payload !== "object") throw new LocalizedError("The server returned invalid studio state.");
    const sessions = Array.isArray(payload.sessions) ? payload.sessions : [];
    const voices = Array.isArray(payload.voices) ? payload.voices : [];
    const assets = Array.isArray(payload.assets) ? payload.assets : [];
    const jobs = Array.isArray(payload.jobs) ? payload.jobs : [];
    const seen = new Set();

    state.sessions = sessions;
    state.voices = voices;
    state.assets = assets;
    state.jobs = jobs;
    state.capabilities = payload.capabilities && typeof payload.capabilities === "object" ? payload.capabilities : {};

    sessions.forEach((session) => sessionSegments(session).forEach((segment) => {
      const key = String(segment.id);
      seen.add(key);
      const draft = drafts.get(key);
      if (!draft) {
        drafts.set(key, draftFromSegment(segment));
      } else if (!draft.dirty && !draft.saving) {
        const replacement = draftFromSegment(segment);
        replacement.version = draft.version;
        drafts.set(key, replacement);
      } else if (!draft.saving && draft.baseRevision !== segment.revision) {
        draft.conflict = true;
      }
    }));

    Array.from(drafts.keys()).forEach((key) => { if (!seen.has(key)) drafts.delete(key); });

    if (!sessions.some((session) => sameId(session.id, activeSessionId))) {
      const visible = sessions.find((session) => !session.archived) || sessions[0] || null;
      activeSessionId = visible ? visible.id : null;
      activeSegmentId = null;
      renameEditing = false;
    }
    const current = selectedSession();
    if (current && !sessionSegments(current).some((segment) => sameId(segment.id, activeSegmentId))) {
      activeSegmentId = sessionSegments(current)[0] ? sessionSegments(current)[0].id : null;
    }
    renderState();
  }

  function sessionsKey() {
    return JSON.stringify({
      archived: showArchivedSessions,
      newName: newSessionName,
      active: activeSessionId,
      sessions: state.sessions.map((session) => [session.id, session.name, Boolean(session.archived), sessionSegments(session).length])
    });
  }

  function editorKey() {
    const session = selectedSession();
    const segments = sessionSegments(session);
    return JSON.stringify({
      active: activeSessionId,
      activeSegment: activeSegmentId,
      name: session && session.name,
      archived: session && Boolean(session.archived),
      renameEditing,
      renameDraft: renameEditing ? renameDraft : "",
      composer: currentComposerText(),
      formats: supportedFormats(),
      voices: state.voices.map((voice) => [voice.id, voice.name, Boolean(voice.archived)]),
      pending: segments.map((segment) => {
        const pending = state.jobs.find((job) => sameId(job.segment_id, segment.id) && (job.status === "queued" || job.status === "running"));
        return [segment.id, pending ? pending.status : null];
      }),
      segments: segments.map((segment) => {
        const draft = drafts.get(String(segment.id));
        return {
          id: segment.id,
          revision: segment.revision,
          selected: segment.selected_take_id,
          draft: draft ? [draft.text, draft.voice_id, draft.seed, draft.max_frames, draft.auto_frames, draft.temperature, draft.top_k, draft.top_p, draft.dirty, draft.saving, draft.conflict, draft.version] : null
        };
      }),
      assets: state.assets.filter((asset) => sameId(asset.session_id, activeSessionId)).map((asset) => [asset.id, asset.kind, asset.filename, asset.session_id, asset.segment_id, asset.revision, asset.seconds, asset.created_at, asset.stopped_on_eos])
    });
  }

  function jobsKey() {
    return JSON.stringify(state.jobs.map((job) => [job.id, job.kind, job.status, job.session_id, job.segment_id, job.progress, job.detail, job.error]));
  }

  function libraryKey() {
    return JSON.stringify({
      tab: libraryTab,
      filter: assetFilter,
      active: activeSessionId,
      voiceForm: [voiceForm.name, voiceForm.transcript, voiceForm.fileName, voiceForm.file && voiceForm.file.size, voiceDraftVersion, voiceForm.duration, recorderPhase, voiceSaving],
      voices: state.voices.map((voice) => [voice.id, voice.name, voice.transcript, voice.filename, voice.seconds, Boolean(voice.archived)]),
      assets: state.assets.map((asset) => [asset.id, asset.kind, asset.filename, asset.format, asset.session_id, asset.segment_id, asset.revision, asset.seconds, asset.created_at, asset.stopped_on_eos])
    });
  }

  function renderState(force) {
    const keys = {
      sessions: sessionsKey(),
      editor: editorKey(),
      jobs: jobsKey(),
      library: libraryKey(),
      queuePaused: queuePaused()
    };
    if (force || keys.sessions !== lastKeys.sessions) renderSessions();
    if (force || keys.editor !== lastKeys.editor) renderEditor();
    if (force || keys.jobs !== lastKeys.jobs) renderJobs();
    if (force || keys.library !== lastKeys.library) renderLibrary();
    if (force || keys.queuePaused !== lastKeys.queuePaused) renderQueueWarning();
    lastKeys = keys;
  }

  function recordRenderKeys() {
    lastKeys = { sessions: sessionsKey(), editor: editorKey(), jobs: jobsKey(), library: libraryKey(), queuePaused: queuePaused() };
  }

  function queuePaused() {
    const error = state.capabilities && state.capabilities.queue_error;
    return error !== null && error !== undefined;
  }

  function renderQueueWarning() {
    const warning = byId("queue-paused-warning");
    warning.hidden = !queuePaused();
  }

  function captureEditorViewState() {
    byId("segment-list").querySelectorAll("[data-segment-card]").forEach((card) => {
      const segmentId = String(card.dataset.segmentId);
      const settings = card.querySelector(".settings-disclosure");
      const history = card.querySelector(".take-history");
      if (settings) settingsDisclosure.set(segmentId, settings.open);
      if (history) historyDisclosure.set(segmentId, history.open);
    });

    const active = document.activeElement;
    if (!active) return null;
    let focus = null;
    if (active.id === "new-segment-text") {
      focus = { kind: "composer" };
    } else if (active.dataset && active.dataset.segmentId !== undefined && active.dataset.field) {
      focus = { kind: "segment", segmentId: active.dataset.segmentId, field: active.dataset.field };
    }
    if (!focus) return null;

    try {
      if (typeof active.selectionStart === "number" && typeof active.selectionEnd === "number") {
        focus.selectionStart = active.selectionStart;
        focus.selectionEnd = active.selectionEnd;
        focus.selectionDirection = active.selectionDirection;
      }
    } catch (_) {
      // Some input types do not expose a text selection range.
    }
    focus.sessionId = renderedEditorSessionId;
    return focus;
  }

  function restoreEditorFocus(focus, sessionId) {
    if (!focus || !sameId(focus.sessionId, sessionId)) return;
    const target = focus.kind === "composer"
      ? byId("new-segment-text")
      : Array.from(byId("segment-list").querySelectorAll("[data-segment-id][data-field]")).find((field) => sameId(field.dataset.segmentId, focus.segmentId) && field.dataset.field === focus.field);
    if (!target || target.disabled) return;
    try { target.focus({ preventScroll: true }); }
    catch (_) { target.focus(); }
    if (focus.selectionStart !== undefined && typeof target.setSelectionRange === "function") {
      try { target.setSelectionRange(focus.selectionStart, focus.selectionEnd, focus.selectionDirection || "none"); }
      catch (_) { /* Ignore selection restoration for unsupported input types. */ }
    }
  }

  function refreshLocalEditorState(segmentId) {
    const card = document.querySelector('[data-segment-card="' + CSS.escape(String(segmentId)) + '"]');
    if (!card) return;
    const draft = drafts.get(String(segmentId));
    if (!draft) return;
    const badge = card.querySelector(".save-state");
    if (badge) {
      badge.textContent = draft.conflict ? t("Conflict") : draft.saving ? t("Saving…") : draft.dirty ? t("Unsaved") : t("Saved");
      badge.classList.toggle("is-dirty", draft.dirty || draft.conflict);
    }
    const save = card.querySelector('[data-action="save-segment"]');
    if (save) save.disabled = !draft.dirty || draft.saving || draft.conflict || Boolean(selectedSession() && selectedSession().archived);
  }

  function renderSessions() {
    const list = byId("sessions-list");
    list.replaceChildren();
    const sessions = state.sessions.filter((session) => showArchivedSessions || !session.archived);
    byId("session-count").textContent = String(sessions.length);
    byId("sessions-empty").hidden = sessions.length > 0;
    byId("archive-toggle").textContent = showArchivedSessions ? t("Hide archive") : t("Show archive");
    byId("archive-toggle").setAttribute("aria-pressed", String(showArchivedSessions));
    byId("new-session-name").value = newSessionName;

    sessions.forEach((session, index) => {
      const row = node("li", "session-item" + (sameId(session.id, activeSessionId) ? " is-active" : ""));
      const select = makeButton("", "session-select", "select-session");
      select.dataset.id = session.id;
      select.setAttribute("aria-current", sameId(session.id, activeSessionId) ? "page" : "false");
      const glyph = node("span", "session-glyph", String(index + 1).padStart(2, "0"));
      glyph.setAttribute("aria-hidden", "true");
      const copy = node("span", "session-copy");
      const name = node("span", "session-name", session.name || t("Untitled"));
      const metaText = session.archived ? t("Archived") : segmentCountText(sessionSegments(session).length);
      const meta = node("span", "session-meta", metaText);
      append(copy, name, meta);
      append(select, glyph, copy);
      const actions = node("span", "session-row-actions");
      actions.append(makeButton(session.archived ? "↶" : "⌑", "mini-action", session.archived ? "restore-session" : "archive-session", {
        title: session.archived ? t("Restore from archive") : t("Move to archive"),
        ariaLabel: session.archived ? t("Restore session from archive") : t("Move session to archive")
      }));
      actions.lastElementChild.dataset.id = session.id;
      append(row, select, actions);
      list.append(row);
    });
  }

  function renderEditor() {
    const viewState = captureEditorViewState();
    const session = selectedSession();
    const toolbar = byId("session-toolbar");
    const composer = byId("composer-card");
    const segmentList = byId("segment-list");
    const empty = byId("editor-empty");
    toolbar.replaceChildren();
    segmentList.replaceChildren();

    if (!session) {
      composer.hidden = true;
      empty.hidden = false;
      const heading = node("h2", "", t("Start with a new session"));
      const copy = node("p", "", t("Create a session in the left panel to add text and audio."));
      empty.replaceChildren(node("div", "empty-orbit"), node("p", "eyebrow", t("Your workspace")), heading, copy);
      renderedEditorSessionId = null;
      return;
    }

    empty.hidden = true;
    composer.hidden = false;
    const titleBlock = node("div", "session-title-block");
    const eyebrow = node("p", "eyebrow", t("Text session"));
    if (renameEditing) {
      const form = node("form", "rename-form");
      form.id = "rename-session-form";
      const input = node("input");
      input.id = "rename-session-input";
      input.type = "text";
      input.maxLength = 120;
      input.required = true;
      input.value = renameDraft;
      input.setAttribute("aria-label", t("New name for this session"));
      const saveName = makeButton(t("Done"), "button button-secondary");
      saveName.type = "submit";
      const cancelName = makeButton(t("Cancel"), "button button-quiet", "cancel-rename");
      append(form, input, saveName, cancelName);
      append(titleBlock, eyebrow, form);
    } else {
      const line = node("div", "session-title-line");
      const title = node("h1", "session-title", session.name || t("Untitled"));
      if (session.archived) line.append(node("span", "archive-pill", t("Archived")));
      const rename = makeButton("✎", "rename-button", "rename-session", { title: t("Rename session"), ariaLabel: t("Rename session") });
      append(line, title, rename);
      const subtitle = node("p", "session-subtitle", t("{count} · exports use selected takes in order", { count: segmentCountText(sessionSegments(session).length) }));
      append(titleBlock, eyebrow, line, subtitle);
    }

    const toolbarActions = node("div", "toolbar-actions");
    const format = node("select");
    format.id = "export-format";
    format.setAttribute("aria-label", t("Export format"));
    const formats = supportedFormats();
    formats.forEach((value) => {
      const option = node("option", "", value.toUpperCase());
      option.value = value;
      format.append(option);
    });
    exportFormat = formats.includes(exportFormat) ? exportFormat : formats[0];
    format.value = exportFormat;
    const exportButton = makeButton(t("Export session"), "button button-secondary", "export-session", { disabled: Boolean(session.archived) || sessionSegments(session).length === 0 });
    exportButton.dataset.id = session.id;
    const missing = sessionSegments(session).filter((segment) => {
      if (!segment.selected_take_id) return true;
      return !state.assets.some((asset) => asset.kind === "take" && sameId(asset.id, segment.selected_take_id) && sameId(asset.session_id, session.id) && sameId(asset.segment_id, segment.id));
    }).length;
    if (missing > 0 && sessionSegments(session).length > 0) {
      exportButton.disabled = true;
      exportButton.title = t("Select audio for every segment before exporting");
    }
    append(toolbarActions, format, exportButton);
    append(toolbar, titleBlock, toolbarActions);

    document.querySelectorAll(".export-hint").forEach((item) => item.remove());
    const input = byId("new-segment-text");
    input.value = currentComposerText();
    input.disabled = Boolean(session.archived);
    const submit = byId("new-segment-form").querySelector('button[type="submit"]');
    submit.disabled = Boolean(session.archived);
    if (missing > 0 && sessionSegments(session).length > 0) {
      const hint = node("p", "field-hint export-hint", t("Exports include all segments in order. Generate or select a take for each first ({count} not ready).", { count: missing }));
      const footer = document.querySelector(".composer-footer");
      if (footer) footer.append(hint);
    }

    sessionSegments(session).forEach((segment, index) => segmentList.append(renderSegmentCard(session, segment, index)));
    if (sessionSegments(session).length === 0) {
      const blank = node("div", "editor-empty");
      blank.append(node("div", "empty-orbit"), node("p", "eyebrow", t("Session start")), node("h2", "", t("Your story starts here")), node("p", "", t("Add text above, then choose a voice and generate your first audio segment.")));
      segmentList.append(blank);
    }
    if (renameEditing) window.setTimeout(() => byId("rename-session-input") && byId("rename-session-input").focus(), 0);
    restoreEditorFocus(viewState, session.id);
    renderedEditorSessionId = session.id;
  }

  function renderSegmentCard(session, segment, index) {
    const draft = drafts.get(String(segment.id)) || draftFromSegment(segment);
    if (!drafts.has(String(segment.id))) drafts.set(String(segment.id), draft);
    const card = node("article", "segment-card" + (sameId(segment.id, activeSegmentId) ? " is-selected" : ""));
    card.dataset.segmentCard = segment.id;
    card.dataset.segmentId = segment.id;
    const head = node("div", "segment-head");
    const titleGroup = node("div", "segment-title");
    const number = node("span", "segment-index", String(index + 1).padStart(2, "0"));
    const heading = node("h3", "", t("Segment {number}", { number: String(index + 1).padStart(2, "0") }));
    const pending = state.jobs.find((job) => sameId(job.segment_id, segment.id) && (job.status === "queued" || job.status === "running"));
    const pendingLabel = pending ? node("span", "success-pill", pending.status === "queued" ? t("Queued") : t("Generating…")) : null;
    append(titleGroup, number, heading, pendingLabel);
    const headActions = node("div", "segment-actions");
    const badge = node("span", "save-state" + (draft.dirty || draft.conflict ? " is-dirty" : ""), draft.conflict ? t("Conflict") : draft.saving ? t("Saving…") : draft.dirty ? t("Unsaved") : t("Saved"));
    badge.setAttribute("aria-live", "polite");
    const save = makeButton(t("Save"), "button button-quiet", "save-segment", { disabled: !draft.dirty || draft.saving || draft.conflict || Boolean(session.archived) });
    save.dataset.id = segment.id;
    const remove = makeButton(t("Delete"), "button button-quiet", "delete-segment", { disabled: Boolean(session.archived), title: t("Delete segment") });
    remove.dataset.id = segment.id;
    append(headActions, badge, save, remove);
    append(head, titleGroup, headActions);

    const body = node("div", "segment-body");
    const textLabel = node("label", "segment-label", t("Text"));
    const textArea = node("textarea", "segment-text");
    textArea.dataset.segmentId = segment.id;
    textArea.dataset.field = "text";
    textArea.rows = 4;
    textArea.maxLength = 12000;
    textArea.value = String(draft.text || "");
    textArea.disabled = Boolean(session.archived);
    textLabel.htmlFor = "segment-text-" + String(segment.id);
    textArea.id = textLabel.htmlFor;
    append(body, textLabel, textArea);

    const controls = node("div", "segment-controls");
    const voiceWrap = node("div", "voice-select-wrap");
    const voiceLabel = node("label", "field-label", t("Reference voice"));
    voiceLabel.htmlFor = "segment-voice-" + String(segment.id);
    const voiceSelect = node("select");
    voiceSelect.id = voiceLabel.htmlFor;
    voiceSelect.dataset.segmentId = segment.id;
    voiceSelect.dataset.field = "voice_id";
    voiceSelect.disabled = Boolean(session.archived);
    const noVoice = node("option", "", t("No reference"));
    noVoice.value = "";
    voiceSelect.append(noVoice);
    state.voices.filter((voice) => !voice.archived || sameId(voice.id, draft.voice_id)).forEach((voice) => {
      const option = node("option", "", voice.archived ? t("[archived] {name}", { name: voice.name || t("Voice") }) : (voice.name || t("Voice")));
      option.value = voice.id;
      option.selected = sameId(voice.id, draft.voice_id);
      voiceSelect.append(option);
    });
    if (draft.voice_id && !state.voices.some((voice) => sameId(voice.id, draft.voice_id))) {
      const unavailable = node("option", "", t("Saved voice unavailable"));
      unavailable.value = draft.voice_id;
      unavailable.selected = true;
      voiceSelect.append(unavailable);
    }
    voiceSelect.value = draft.voice_id || "";
    append(voiceWrap, voiceLabel, voiceSelect);
    const generate = makeButton(t("Generate audio"), "button button-primary", "generate-segment", { disabled: Boolean(session.archived) || Boolean(pending) });
    generate.dataset.id = segment.id;
    append(controls, voiceWrap, generate);
    body.append(controls);

    const settings = node("details", "settings-disclosure");
    settings.open = settingsDisclosure.get(String(segment.id)) === true;
    settings.addEventListener("toggle", () => settingsDisclosure.set(String(segment.id), settings.open));
    const settingsSummary = node("summary", "", t("Generation settings"));
    const grid = node("div", "settings-grid");
    const windowLabel = node("label", "", t("Window size"));
    const windowSelect = node("select");
    windowSelect.dataset.segmentId = segment.id;
    windowSelect.dataset.field = "auto_frames";
    windowSelect.disabled = Boolean(session.archived);
    const manualOption = node("option", "", t("Manual"));
    manualOption.value = "false";
    const autoOption = node("option", "", t("Automatic (memory budget)"));
    autoOption.value = "true";
    autoOption.disabled = Boolean(draft.voice_id) || !state.capabilities.auto_window_available;
    append(windowSelect, manualOption, autoOption);
    windowSelect.value = String(draft.auto_frames);
    windowLabel.append(windowSelect);
    grid.append(windowLabel);
    const settingsSpec = [
      ["seed", "Seed", "number", "1"],
      ["max_frames", t("Frame limit"), "number", "1"],
      ["temperature", "Temperature", "number", "0.05"],
      ["top_k", "Top K", "number", "1"],
      ["top_p", "Top P", "number", "0.05"]
    ];
    settingsSpec.forEach(([field, labelText, type, step]) => {
      const label = node("label", "", labelText);
      const input = node("input");
      input.type = type;
      input.step = step;
      input.dataset.segmentId = segment.id;
      input.dataset.field = field;
      input.value = draft[field] === null || draft[field] === undefined ? "" : String(draft[field]);
      input.disabled = Boolean(session.archived);
      label.append(input);
      grid.append(label);
    });
    append(settings, settingsSummary, grid);
    settings.append(node("p", "field-hint", t("Automatic sizing uses available memory up to the frame limit. Speech ends at EOS; duration cannot be predicted exactly.")));
    if (autoOption.disabled) settings.append(node("p", "field-hint", t("Automatic windows are unavailable on this server or with a voice reference.")));
    body.append(settings);

    if (draft.conflict) {
      const conflict = node("div", "notice is-error", t("This segment changed on the server after your edits. Reload it or copy your text before continuing."));
      const reload = makeButton(t("Load server version"), "button button-quiet", "reload-segment");
      reload.dataset.id = segment.id;
      append(body, conflict, reload);
    }
    body.append(renderSelectedTake(session, segment));
    card.append(head, body);
    return card;
  }

  function renderSelectedTake(session, segment) {
    const wrapper = node("section", "selected-take");
    wrapper.setAttribute("aria-label", t("Selected audio take"));
    const take = state.assets.find((asset) => sameId(asset.id, segment.selected_take_id));
    if (segment.selected_take_id && !take) {
      wrapper.append(node("div", "history-empty", t("The selected take is unavailable in the library.")));
    } else if (!take) {
      wrapper.append(node("div", "history-empty", t("No audio for this segment yet. Save the text, then generate your first take.")));
    } else {
      const heading = node("div", "take-heading");
      const copy = node("div", "");
      const title = node("strong", "", take.filename || t("Selected take"));
      const meta = node("small", "", t("Version {revision}", { revision: take.revision === undefined ? "—" : take.revision }) + " · " + formatSeconds(take.seconds) + stampSuffix(take.created_at));
      append(copy, title, meta);
      const tags = node("div", "take-tags");
      tags.append(node("span", "take-tag", t("Selected")));
      if (take.stopped_on_eos === false) tags.append(node("span", "take-tag is-warning", t("Possibly truncated")));
      append(heading, copy, tags);
      const player = node("audio");
      player.controls = true;
      player.preload = "none";
      player.src = "/api/audio/" + idPart(take.id);
      player.setAttribute("aria-label", t("Play selected take"));
      append(wrapper, heading, player);
      if (take.stopped_on_eos === false) {
        wrapper.append(node("p", "field-hint", t("The model did not signal the end of this segment. Check the take before exporting.")));
      }
    }

    const history = node("details", "take-history");
    history.open = historyDisclosure.get(String(segment.id)) === true;
    history.addEventListener("toggle", () => historyDisclosure.set(String(segment.id), history.open));
    history.append(node("summary", "", t("Take history")));
    const list = node("div", "history-list");
    const takes = state.assets.filter((asset) => asset.kind === "take" && sameId(asset.session_id, session.id) && sameId(asset.segment_id, segment.id)).sort((a, b) => assetTime(b) - assetTime(a));
    if (takes.length === 0) {
      list.append(node("div", "history-empty", t("No takes yet.")));
    } else {
      takes.forEach((asset) => {
        const row = node("div", "history-item");
        const copy = node("div", "history-copy");
        const title = node("strong", "", t(sameId(asset.id, segment.selected_take_id) ? "Version {revision} · selected" : "Version {revision}", { revision: asset.revision === undefined ? "—" : asset.revision }));
        const meta = node("span", "", formatSeconds(asset.seconds) + stampSuffix(asset.created_at) + (asset.stopped_on_eos === false ? " · " + t("possibly truncated") : ""));
        append(copy, title, meta);
        const actions = node("div", "history-actions");
        const listen = node("a", "", t("Listen"));
        listen.href = "/api/audio/" + idPart(asset.id);
        listen.textContent = t("Listen");
        listen.setAttribute("aria-label", t("Listen to version {revision}", { revision: asset.revision === undefined ? "" : asset.revision }));
        const select = makeButton(sameId(asset.id, segment.selected_take_id) ? t("Selected") : t("Select"), "button button-quiet", "select-take", { disabled: sameId(asset.id, segment.selected_take_id) || Boolean(session.archived) });
        select.dataset.id = segment.id;
        select.dataset.assetId = asset.id;
        append(actions, listen, select);
        append(row, copy, actions);
        list.append(row);
      });
    }
    history.append(list);
    wrapper.append(history);
    return wrapper;
  }

  function stampSuffix(value) {
    const stamp = formatStamp(value);
    return stamp ? " · " + stamp : "";
  }

  function assetTime(asset) {
    const value = new Date(asset.created_at || 0).getTime();
    return Number.isNaN(value) ? 0 : value;
  }

  function renderJobs() {
    const region = byId("jobs-region");
    region.replaceChildren();
    const ordered = state.jobs.slice().sort((a, b) => {
      const weight = (job) => job.status === "running" ? 4 : job.status === "queued" ? 3 : job.status === "failed" ? 2 : 1;
      return weight(b) - weight(a);
    }).slice(0, 6);
    ordered.forEach((job) => {
      const card = node("article", "job-card");
      const top = node("div", "job-topline");
      const session = state.sessions.find((item) => sameId(item.id, job.session_id));
      const segment = session && sessionSegments(session).find((item) => sameId(item.id, job.segment_id));
      const titleText = job.kind === "export" ? t("Export") : job.kind === "generate" ? t("Audio generation") : String(job.kind || t("Job"));
      const title = node("span", "job-title", segment ? t("{title} · Segment {number}", { title: titleText, number: sessionSegments(session).indexOf(segment) + 1 }) : titleText);
      const status = node("span", "job-state", jobStatus(job.status));
      append(top, title, status);
      const detail = node("div", "job-detail");
      const context = session ? session.name : t("Local queue");
      const detailText = job.status === "queued" ? t("Waiting to run") : job.status === "running" ? jobDetailText(job.detail) || t("Running") : job.status === "failed" ? t("Job failed") : t("Done");
      append(detail, node("span", "", context), node("span", "", detailText));
      append(card, top, detail);
      if (job.status === "running" || job.status === "queued") {
        const track = node("div", "progress-track");
        const fill = node("div", "progress-fill");
        const progress = Number(job.progress);
        if (job.status === "running" && Number.isFinite(progress) && progress > 0) {
          const percent = Math.max(0, Math.min(100, progress <= 1 ? progress * 100 : progress));
          fill.style.width = percent + "%";
        } else {
          fill.classList.add("is-indeterminate");
        }
        track.append(fill);
        card.append(track);
      }
      if (job.error) card.append(node("p", "job-error", errorText({ message: job.error })));
      region.append(card);
    });
  }

  function jobDetailText(value) {
    if (typeof value !== "string") return "";
    const detail = value.trim();
    if (!detail) return "";
    const counter = detail.match(/\b(prefill|frame)\s+(\d+)\s*\/\s*(\d+)\b/i);
    if (counter) {
      const phase = counter[1].toLowerCase() === "prefill" ? t("Prefill") : t("Frame");
      return phase + " · " + counter[2] + " / " + counter[3];
    }
    return detail;
  }

  function jobStatus(status) {
    if (status === "queued") return t("Queued");
    if (status === "running") return t("In progress");
    if (status === "complete") return t("Done");
    if (status === "failed") return t("Error");
    return String(status || t("Unknown status"));
  }

  function renderLibrary() {
    const voiceList = byId("voice-list");
    const assetList = byId("asset-list");
    voiceList.replaceChildren();
    assetList.replaceChildren();
    byId("voice-name").value = voiceForm.name;
    byId("voice-transcript").value = voiceForm.transcript;
    byId("voice-file-label").textContent = voiceForm.file ? voiceForm.fileName : t("Choose an audio file");
    renderVoiceRecorder();
    byId("asset-filter").value = assetFilter;
    byId("voices-panel").hidden = libraryTab !== "voices";
    byId("audio-panel").hidden = libraryTab !== "audio";
    byId("voices-tab").classList.toggle("is-active", libraryTab === "voices");
    byId("audio-tab").classList.toggle("is-active", libraryTab === "audio");
    byId("voices-tab").setAttribute("aria-selected", String(libraryTab === "voices"));
    byId("audio-tab").setAttribute("aria-selected", String(libraryTab === "audio"));

    const voices = state.voices.filter((voice) => !voice.archived);
    byId("voice-count").textContent = String(voices.length);
    byId("voices-empty").hidden = voices.length > 0;
    voices.forEach((voice) => {
      const card = node("article", "voice-card");
      const top = node("div", "voice-card-top");
      const info = node("div", "voice-info");
      append(info,
        node("strong", "", voice.name || t("Unnamed voice")),
        node("small", "", formatSeconds(voice.seconds) + (voice.filename ? " · " + voice.filename : ""))
      );
      const remove = makeButton("⌫", "mini-action", "delete-voice", { title: t("Move voice to archive"), ariaLabel: t("Move voice to archive") });
      remove.dataset.id = voice.id;
      append(top, info, remove);
      card.append(top);
      if (voice.transcript) card.append(node("p", "voice-transcript", voice.transcript));
      const player = node("audio");
      player.controls = true;
      player.preload = "none";
      player.src = "/api/voice-audio/" + idPart(voice.id);
      player.setAttribute("aria-label", t("Play reference {name}", { name: voice.name || "" }));
      card.append(player);
      voiceList.append(card);
    });

    const visibleAssets = state.assets.filter((asset) => assetFilter === "all" || sameId(asset.session_id, activeSessionId)).slice().sort((a, b) => assetTime(b) - assetTime(a));
    byId("asset-count").textContent = String(visibleAssets.length);
    byId("assets-empty").hidden = visibleAssets.length > 0;
    visibleAssets.forEach((asset) => assetList.append(renderAssetCard(asset)));

    const formats = supportedFormats().map((value) => value.toUpperCase()).join(" · ");
    const maxReference = Number(state.capabilities.max_reference_seconds);
    const limit = Number.isFinite(maxReference) && maxReference > 0 ? maxReference : 10;
    const ffmpegNote = state.capabilities.ffmpeg === false ? t(" Export conversion is unavailable.") : "";
    byId("capability-note").replaceChildren(node("strong", "", t("Formats: {formats}", { formats: formats || "WAV" })), document.createTextNode(t(" · reference up to {seconds} sec.", { seconds: limit }) + ffmpegNote));
  }

  function renderVoiceRecorder() {
    const busy = recorderPhase !== "idle";
    const hasSample = Boolean(voiceForm.file);
    const record = byId("voice-record-button");
    const stop = byId("voice-stop-button");
    const rerecord = byId("voice-rerecord-button");
    const discard = byId("voice-discard-button");
    const status = byId("voice-record-status");
    const previewWrap = byId("voice-preview-wrap");
    const preview = byId("voice-preview");
    const fileInput = byId("voice-file");
    const nameInput = byId("voice-name");
    const transcriptInput = byId("voice-transcript");
    const save = byId("voice-save-button");

    if (hasSample && !voiceForm.previewUrl) createVoicePreviewUrl();
    const desiredUrl = voiceForm.previewUrl || "";
    if (preview.getAttribute("src") !== desiredUrl) {
      preview.pause();
      preview.removeAttribute("src");
      if (desiredUrl) preview.setAttribute("src", desiredUrl);
      preview.load();
    }
    previewWrap.hidden = !hasSample;
    byId("voice-sample-duration").textContent = voiceForm.duration ? t("Recording length: {seconds} sec", { seconds: voiceForm.duration.toFixed(1).replace(".", i18n.getLanguage() === "ru" ? "," : ".") }) : "";

    record.hidden = busy || hasSample;
    rerecord.hidden = busy || !hasSample;
    discard.hidden = !hasSample;
    stop.hidden = !busy;
    stop.disabled = recorderPhase === "stopping";
    status.hidden = !busy;
    status.textContent = recorderPhase === "requesting" ? t("Waiting for microphone permission…")
      : recorderPhase === "preparing" ? t("Preparing microphone…")
      : recorderPhase === "recording" ? t("Recording. Stop when finished; capture ends at 9.5 seconds.")
      : t("Finishing recording…");
    stop.textContent = recorderPhase === "requesting" || recorderPhase === "preparing" ? t("Cancel recording setup") : t("Stop recording");

    fileInput.disabled = voiceSaving || busy;
    nameInput.disabled = voiceSaving;
    transcriptInput.disabled = voiceSaving;
    byId("voice-script-en").disabled = voiceSaving;
    byId("voice-script-ru").disabled = voiceSaving;
    record.disabled = voiceSaving;
    rerecord.disabled = voiceSaving;
    discard.disabled = voiceSaving || busy;
    save.disabled = voiceSaving || busy || !hasSample || !voiceForm.name.trim() || !voiceForm.transcript.trim() || !voiceForm.previewUrl;
    save.textContent = voiceSaving ? t("Saving…") : t("Save voice");
  }

  function createVoicePreviewUrl() {
    if (!voiceForm.file || voiceForm.previewUrl) return;
    if (!window.URL || typeof window.URL.createObjectURL !== "function") throw new LocalizedError("Audio preview is not available in this browser.");
    voiceForm.previewUrl = window.URL.createObjectURL(voiceForm.file);
  }

  function revokeVoicePreviewUrl(url) {
    if (!url || !window.URL || typeof window.URL.revokeObjectURL !== "function") return;
    try { window.URL.revokeObjectURL(url); }
    catch (_) { /* The browser may already have released this preview URL. */ }
  }

  function setVoiceSample(file, fileName, duration) {
    let previewUrl = null;
    if (file) {
      if (!window.URL || typeof window.URL.createObjectURL !== "function") throw new LocalizedError("Audio preview is not available in this browser.");
      previewUrl = window.URL.createObjectURL(file);
    }
    const previousUrl = voiceForm.previewUrl;
    voiceForm.file = file || null;
    voiceForm.fileName = file ? String(fileName || file.name || "microphone-reference.wav") : "";
    voiceForm.previewUrl = previewUrl;
    voiceForm.duration = Number.isFinite(duration) && duration > 0 ? duration : null;
    voiceDraftVersion += 1;
    revokeVoicePreviewUrl(previousUrl);
    byId("voice-file").value = "";
    recordRenderKeys();
  }

  function discardVoiceSample() {
    if (recorderPhase !== "idle") voiceRecorder.cancel();
    setVoiceSample(null, "", null);
    showNotice(() => t("Sample discarded."), "success");
    renderState(true);
  }

  function startVoiceRecording() {
    if (voiceSaving || recorderPhase !== "idle" || !voiceRecorder) return;
    voiceRecorder.start();
  }

  async function stopVoiceRecording() {
    if (!voiceRecorder || recorderPhase === "idle") return;
    await voiceRecorder.stop();
  }

  function renderAssetCard(asset) {
    const card = node("article", "asset-card");
    const top = node("div", "asset-card-top");
    const info = node("div", "asset-info");
    const session = state.sessions.find((item) => sameId(item.id, asset.session_id));
    const label = asset.kind === "export" ? t("Export") : t("Take");
    const metaParts = [String(asset.format || "audio").toUpperCase(), formatSeconds(asset.seconds)];
    if (asset.revision !== undefined && asset.revision !== null) metaParts.push(t("version {revision}", { revision: asset.revision }));
    if (session && assetFilter === "all") metaParts.push(session.name || t("Untitled"));
    if (asset.created_at) metaParts.push(formatStamp(asset.created_at));
    append(info, node("strong", "", asset.filename || label), node("small", "", metaParts.join(" · ")));
    const kind = node("span", "asset-kind" + (asset.kind === "export" ? " is-export" : ""), label);
    append(top, info, kind);
    const player = node("audio");
    player.controls = true;
    player.preload = "none";
    player.src = "/api/audio/" + idPart(asset.id);
    player.setAttribute("aria-label", t("Play {name}", { name: asset.filename || label }));
    const actions = node("div", "asset-actions");
    const download = node("a", "", t("Download"));
    download.href = "/api/audio/" + idPart(asset.id) + "?download=1";
    download.setAttribute("download", "");
    const remove = makeButton(t("Delete"), "", "delete-asset", { title: t("Delete audio file") });
    remove.dataset.id = asset.id;
    append(actions, download, remove);
    if (asset.stopped_on_eos === false) card.append(node("p", "field-hint", t("Possibly truncated: recording stopped at the frame limit.")));
    append(card, top, player, actions);
    return card;
  }

  function supportedFormats() {
    const available = state.capabilities.formats;
    if (Array.isArray(available)) {
      const parsed = available.map((item) => String(item).toLowerCase().replace(/^\./, "")).filter((item) => ["wav", "mp3", "opus"].includes(item));
      return parsed.length ? parsed : ["wav"];
    }
    if (typeof available === "string") {
      const parsed = available.toLowerCase().split(/[\s,|]+/).map((item) => item.replace(/^\./, "")).filter((item) => ["wav", "mp3", "opus"].includes(item));
      return parsed.length ? parsed : ["wav"];
    }
    if (available && typeof available === "object") {
      const parsed = Object.keys(available).filter((key) => available[key] && ["wav", "mp3", "opus"].includes(key.toLowerCase())).map((key) => key.toLowerCase());
      return parsed.length ? parsed : ["wav"];
    }
    return ["wav", "mp3", "opus"];
  }

  function segmentPayload(draft) {
    const values = {
      text: String(draft.text || ""),
      voice_id: draft.voice_id ? String(draft.voice_id) : null,
      seed: Number(draft.seed),
      max_frames: Number(draft.max_frames),
      auto_frames: draft.auto_frames === true,
      temperature: Number(draft.temperature),
      top_k: Number(draft.top_k),
      top_p: Number(draft.top_p)
    };
    if (!values.text.trim()) throw new LocalizedError("Add text before saving.");
    if (![values.seed, values.max_frames, values.temperature, values.top_k, values.top_p].every(Number.isFinite)) throw new LocalizedError("Check the numeric generation settings.");
    if (!Number.isInteger(values.seed) || !Number.isInteger(values.max_frames) || !Number.isInteger(values.top_k)) throw new LocalizedError("Seed, frame limit and Top K must be integers.");
    return values;
  }

  async function saveDraft(segmentId) {
    const key = String(segmentId);
    const draft = drafts.get(key);
    const session = selectedSession();
    if (!draft || !session) return false;
    if (!draft.dirty) return true;
    if (session.archived) throw new LocalizedError("Archived sessions cannot be edited.");
    if (draft.conflict) throw new LocalizedError("Load the current segment version first.");
    if (draft.saving && draft.savePromise) return draft.savePromise;

    const payload = segmentPayload(draft);
    const sentVersion = draft.version;
    const savedRevision = draft.baseRevision;
    draft.saving = true;
    refreshLocalEditorState(segmentId);
    const operation = (async () => {
      try {
        await request("/api/sessions/" + idPart(session.id) + "/segments/" + idPart(segmentId), jsonRequest("PATCH", Object.assign({ revision: savedRevision }, payload)));
        const refreshed = await refreshState(false, true);
        if (!refreshed) throw new LocalizedError("Changes were sent, but studio state could not be refreshed. Check the connection and refresh the studio.");
        const live = sessionSegments(selectedSession()).find((segment) => sameId(segment.id, segmentId));
        if (live) {
          draft.baseRevision = live.revision;
          if (draft.version === sentVersion) {
            Object.assign(draft, draftFromSegment(live));
          } else {
            draft.dirty = true;
            draft.conflict = false;
            window.setTimeout(() => saveDraft(segmentId).catch((error) => showNotice(() => errorText(error), "error")), 0);
          }
        } else if (draft.version === sentVersion) {
          draft.dirty = false;
        }
        return true;
      } finally {
        draft.saving = false;
        draft.savePromise = null;
        refreshLocalEditorState(segmentId);
        renderState();
      }
    })();
    draft.savePromise = operation;
    try {
      return await operation;
    } catch (error) {
      await refreshState(false, true);
      draft.dirty = true;
      const live = sessionSegments(selectedSession()).find((segment) => sameId(segment.id, segmentId));
      if (live && live.revision !== savedRevision) draft.conflict = true;
      refreshLocalEditorState(segmentId);
      renderState();
      throw error;
    }
  }

  async function refreshState(isManual, force) {
    if (refreshPromise && !force) {
      if (isManual) byId("refresh-button").classList.add("is-loading");
      return refreshPromise;
    }
    const requestSequence = ++stateRequestSequence;
    const refreshButton = byId("refresh-button");
    const task = (async () => {
      if (isManual) refreshButton.classList.add("is-loading");
      try {
        const payload = await request("/api/state", { method: "GET" });
        if (requestSequence !== stateRequestSequence) return false;
        ingestState(payload);
        updateConnection(true, activeJobs().length ? "Queue running" : "Connected locally");
        return true;
      } catch (error) {
        if (requestSequence === stateRequestSequence) {
          updateConnection(false, "Disconnected");
          showNotice(() => t("Could not refresh studio state: {error}", { error: errorText(error) }), "error");
        }
        return false;
      } finally {
        if (requestSequence === stateRequestSequence) {
          refreshButton.classList.remove("is-loading");
          if (refreshPromise === task) refreshPromise = null;
          schedulePoll();
        }
      }
    })();
    refreshPromise = task;
    return task;
  }

  function schedulePoll() {
    if (pollTimer) window.clearTimeout(pollTimer);
    const delay = activeJobs().length ? 1600 : 12000;
    pollTimer = window.setTimeout(() => refreshState(false), delay);
  }

  function changeDraft(target) {
    const draft = drafts.get(String(target.dataset.segmentId));
    if (!draft) return;
    const field = target.dataset.field;
    if (field === "voice_id") {
      draft.voice_id = target.value || null;
      if (draft.voice_id) draft.auto_frames = false;
    }
    else if (field === "auto_frames") draft.auto_frames = target.value === "true";
    else if (field) draft[field] = target.value;
    draft.dirty = true;
    draft.version += 1;
    if (field === "voice_id") renderState(true);
    else refreshLocalEditorState(draft.id);
    recordRenderKeys();
  }

  function renderTab(tab) {
    libraryTab = tab;
    byId("voices-panel").hidden = tab !== "voices";
    byId("audio-panel").hidden = tab !== "audio";
    byId("voices-tab").classList.toggle("is-active", tab === "voices");
    byId("audio-tab").classList.toggle("is-active", tab === "audio");
    byId("voices-tab").setAttribute("aria-selected", String(tab === "voices"));
    byId("audio-tab").setAttribute("aria-selected", String(tab === "audio"));
    recordRenderKeys();
  }

  function activateSession(id) {
    activeSessionId = id;
    const session = selectedSession();
    activeSegmentId = session && sessionSegments(session)[0] ? sessionSegments(session)[0].id : null;
    renameEditing = false;
    renderState(true);
  }

  async function withLock(key, operation) {
    if (locks.has(key)) return;
    locks.add(key);
    try { await operation(); }
    catch (error) { showNotice(() => errorText(error), "error"); }
    finally { locks.delete(key); }
  }

  async function createSession(event) {
    event.preventDefault();
    const name = newSessionName.trim();
    if (!name) { showNotice(() => t("Enter a session name."), "error"); return; }
    await withLock("create-session", async () => {
      const previous = new Set(state.sessions.map((session) => String(session.id)));
      const result = await request("/api/sessions", jsonRequest("POST", { name }));
      newSessionName = "";
      if (result && result.session && result.session.id !== undefined) activeSessionId = result.session.id;
      else if (result && result.id !== undefined) activeSessionId = result.id;
      await refreshState(false, true);
      const created = state.sessions.find((session) => !previous.has(String(session.id)) && session.name === name);
      if (created) activeSessionId = created.id;
      activeSegmentId = null;
      renderState(true);
      showNotice(() => t("Session created."), "success");
    });
  }

  async function createSegment(event) {
    event.preventDefault();
    const session = selectedSession();
    if (!session) { showNotice(() => t("Create a session first."), "error"); return; }
    const text = currentComposerText();
    if (!text.trim()) { showNotice(() => t("Enter segment text."), "error"); return; }
    if (session.archived) { showNotice(() => t("Archived sessions cannot be edited."), "error"); return; }
    await withLock("create-segment", async () => {
      const before = new Set(sessionSegments(session).map((segment) => String(segment.id)));
      const result = await request("/api/sessions/" + idPart(session.id) + "/segments", jsonRequest("POST", {
        text,
        voice_id: null,
        seed: 0,
        max_frames: 512,
        auto_frames: false,
        temperature: 0.8,
        top_k: 30,
        top_p: 0.9
      }));
      setComposerText("");
      await refreshState(false, true);
      const latest = selectedSession();
      const created = result && result.segment ? result.segment : result;
      const found = created && created.id !== undefined ? sessionSegments(latest).find((segment) => sameId(segment.id, created.id)) : sessionSegments(latest).find((segment) => !before.has(String(segment.id)));
      activeSegmentId = found ? found.id : null;
      renderState(true);
      showNotice(() => t("Segment added."), "success");
    });
  }

  async function renameSession(event) {
    event.preventDefault();
    const session = selectedSession();
    if (!session) return;
    const name = renameDraft.trim();
    if (!name) { showNotice(() => t("The name cannot be empty."), "error"); return; }
    await withLock("rename-session", async () => {
      await request("/api/sessions/" + idPart(session.id), jsonRequest("PATCH", { name }));
      renameEditing = false;
      await refreshState(false, true);
      showNotice(() => t("Name updated."), "success");
    });
  }

  async function archiveSession(id, archived) {
    const session = state.sessions.find((item) => sameId(item.id, id));
    if (!session) return;
    await withLock("session-" + id, async () => {
      await request("/api/sessions/" + idPart(id), jsonRequest("PATCH", { archived }));
      await refreshState(false, true);
      if (archived && sameId(activeSessionId, id) && !showArchivedSessions) {
        const next = state.sessions.find((item) => !item.archived);
        if (next) activateSession(next.id);
      }
      showNotice(() => archived ? t("Session moved to archive.") : t("Session restored from archive."), "success");
    });
  }

  async function deleteSegment(id) {
    const session = selectedSession();
    const segment = sessionSegments(session).find((item) => sameId(item.id, id));
    if (!session || !segment) return;
    if (!window.confirm(t("Delete this text segment? Its takes may remain in the audio library."))) return;
    await withLock("segment-" + id, async () => {
      const draft = drafts.get(String(id));
      await request("/api/sessions/" + idPart(session.id) + "/segments/" + idPart(id) + "?revision=" + encodeURIComponent(String(draft ? draft.baseRevision : segment.revision)), { method: "DELETE" });
      drafts.delete(String(id));
      await refreshState(false, true);
      showNotice(() => t("Segment deleted."), "success");
    });
  }

  async function reloadSegment(id) {
    const session = selectedSession();
    const live = sessionSegments(session).find((segment) => sameId(segment.id, id));
    if (!live) return;
    drafts.set(String(id), draftFromSegment(live));
    renderState(true);
    showNotice(() => t("Loaded the current segment version."), "success");
  }

  async function generateSegment(id) {
    const session = selectedSession();
    const draft = drafts.get(String(id));
    if (!session || !draft) return;
    await withLock("generate-" + id, async () => {
      if (draft.dirty) await saveDraft(id);
      if (draft.conflict) throw new LocalizedError("Resolve the version conflict before generating.");
      await request("/api/sessions/" + idPart(session.id) + "/segments/" + idPart(id) + "/generate", jsonRequest("POST", {}));
      await refreshState(false, true);
      showNotice(() => t("Job added to the queue. The previous take remains available."), "success");
    });
  }

  async function selectTake(segmentId, assetId) {
    const session = selectedSession();
    if (!session) return;
    await withLock("take-" + segmentId, async () => {
      await request("/api/sessions/" + idPart(session.id) + "/segments/" + idPart(segmentId) + "/select", jsonRequest("POST", { asset_id: assetId }));
      await refreshState(false, true);
      showNotice(() => t("Selected a different take."), "success");
    });
  }

  async function exportSession(id) {
    const session = state.sessions.find((item) => sameId(item.id, id));
    if (!session) return;
    const format = byId("export-format").value;
    await withLock("export-" + id, async () => {
      await request("/api/sessions/" + idPart(id) + "/export", jsonRequest("POST", { format }));
      await refreshState(false, true);
      renderTab("audio");
      showNotice(() => t("Export added to the queue. The result will appear in the audio library."), "success");
    });
  }

  async function uploadVoice(event) {
    event.preventDefault();
    if (voiceSaving || recorderPhase !== "idle") return;
    const file = voiceForm.file;
    if (!file) { showNotice(() => t("Choose or record an audio sample first."), "error"); return; }
    if (!voiceForm.previewUrl) { showNotice(() => t("Preview the sample before saving."), "error"); return; }
    if (file.size > 20 * 1024 * 1024) { showNotice(() => t("The file exceeds the 20 MB limit."), "error"); return; }
    const name = voiceForm.name.trim();
    if (!name) { showNotice(() => t("Enter a voice name."), "error"); return; }
    const transcript = voiceForm.transcript.trim();
    if (!transcript) { showNotice(() => t("Enter the exact transcript before saving."), "error"); return; }
    const sentVersion = voiceDraftVersion;
    const sentFile = file;
    voiceSaving = true;
    renderState(true);
    try {
      const query = new URLSearchParams({ name, transcript });
      await request("/api/voices?" + query.toString(), {
        method: "POST",
        headers: { "Content-Type": file.type || "application/octet-stream" },
        body: file
      });
      const unchanged = voiceDraftVersion === sentVersion && voiceForm.file === sentFile && voiceForm.name.trim() === name && voiceForm.transcript.trim() === transcript;
      if (unchanged) {
        const previewUrl = voiceForm.previewUrl;
        voiceForm = { name: "", transcript: "", file: null, fileName: "", previewUrl: null, duration: null };
        voiceDraftVersion += 1;
        revokeVoicePreviewUrl(previewUrl);
        byId("voice-upload-form").reset();
      }
      const refreshed = await refreshState(false, true);
      renderState(true);
      showNotice(() => refreshed ? t("Voice added to the library.") : t("Voice saved, but the library could not be refreshed."), refreshed ? "success" : "error");
    } catch (error) {
      showNotice(() => errorText(error), "error");
    } finally {
      voiceSaving = false;
      renderState(true);
    }
  }

  async function archiveVoice(id) {
    const voice = state.voices.find((item) => sameId(item.id, id));
    if (!voice) return;
    if (!window.confirm(t("Move voice “{name}” to archive?", { name: voice.name || t("Untitled") }))) return;
    await withLock("voice-" + id, async () => {
      await request("/api/voices/" + idPart(id), { method: "DELETE" });
      await refreshState(false, true);
      showNotice(() => t("Voice moved to archive."), "success");
    });
  }

  async function deleteAsset(id) {
    const asset = state.assets.find((item) => sameId(item.id, id));
    if (!asset) return;
    if (!window.confirm(t("Delete audio file “{name}”?", { name: asset.filename || t("untitled") }))) return;
    await withLock("asset-" + id, async () => {
      await request("/api/assets/" + idPart(id), { method: "DELETE" });
      await refreshState(false, true);
      showNotice(() => t("Audio file deleted."), "success");
    });
  }

  function handleAction(action, target) {
    const id = target.dataset.id;
    if (action === "record-voice") startVoiceRecording();
    else if (action === "rerecord-voice") startVoiceRecording();
    else if (action === "stop-voice") stopVoiceRecording().catch((error) => showNotice(() => errorText(error), "error"));
    else if (action === "discard-voice") discardVoiceSample();
    else if (action === "select-session") activateSession(id);
    else if (action === "archive-session") archiveSession(id, true).catch((error) => showNotice(() => errorText(error), "error"));
    else if (action === "restore-session") archiveSession(id, false).catch((error) => showNotice(() => errorText(error), "error"));
    else if (action === "rename-session") {
      const session = selectedSession();
      if (session && !session.archived) { renameEditing = true; renameDraft = session.name || ""; renderState(true); }
    } else if (action === "cancel-rename") { renameEditing = false; renderState(true); }
    else if (action === "save-segment") withLock("save-" + id, async () => { await saveDraft(id); showNotice(() => t("Changes saved."), "success"); }).catch((error) => showNotice(() => errorText(error), "error"));
    else if (action === "delete-segment") deleteSegment(id).catch((error) => showNotice(() => errorText(error), "error"));
    else if (action === "reload-segment") reloadSegment(id);
    else if (action === "generate-segment") generateSegment(id).catch((error) => showNotice(() => errorText(error), "error"));
    else if (action === "select-take") selectTake(id, target.dataset.assetId).catch((error) => showNotice(() => errorText(error), "error"));
    else if (action === "export-session") exportSession(id).catch((error) => showNotice(() => errorText(error), "error"));
    else if (action === "delete-voice") archiveVoice(id).catch((error) => showNotice(() => errorText(error), "error"));
    else if (action === "delete-asset") deleteAsset(id).catch((error) => showNotice(() => errorText(error), "error"));
  }

  document.addEventListener("click", (event) => {
    const actionTarget = event.target.closest("[data-action]");
    if (actionTarget) {
      event.preventDefault();
      handleAction(actionTarget.dataset.action, actionTarget);
      return;
    }
    const scriptButton = event.target.closest("[data-voice-script]");
    if (scriptButton) {
      event.preventDefault();
      const scripts = {
        en: "Bright skies, three quiet friends, and warm sunlight. Please bring the blue box, fresh flowers, and a small cup of tea.",
        ru: "Щенок шёл по узкой дорожке. Вчера я купил пять свежих яблок, чашку чая и жёлтую шаль."
      };
      voiceForm.transcript = scripts[scriptButton.dataset.voiceScript] || "";
      voiceDraftVersion += 1;
      byId("voice-transcript").value = voiceForm.transcript;
      recordRenderKeys();
      byId("voice-transcript").focus();
      return;
    }
    const tab = event.target.closest("[data-library-tab]");
    if (tab) renderTab(tab.dataset.libraryTab);
  });

  document.addEventListener("input", (event) => {
    const target = event.target;
    if (target.matches("[data-segment-id][data-field]")) {
      changeDraft(target);
    } else if (target.id === "new-session-name") {
      newSessionName = target.value;
      recordRenderKeys();
    } else if (target.id === "new-segment-text") {
      setComposerText(target.value);
      recordRenderKeys();
    } else if (target.id === "rename-session-input") {
      renameDraft = target.value;
      recordRenderKeys();
    } else if (target.id === "voice-name") {
      voiceForm.name = target.value;
      voiceDraftVersion += 1;
      recordRenderKeys();
    } else if (target.id === "voice-transcript") {
      voiceForm.transcript = target.value;
      voiceDraftVersion += 1;
      recordRenderKeys();
    }
  });

  document.addEventListener("change", (event) => {
    const target = event.target;
    if (target.id === "language-select") {
      i18n.setLanguage(target.value);
      i18n.apply();
      renderState(true);
      updateConnection(connectionState.connected, connectionState.message);
      if (typeof noticeMessage === "function") byId("notice").textContent = noticeMessage();
    } else if (target.id === "voice-file") {
      const file = target.files && target.files[0] ? target.files[0] : null;
      if (file) {
        try {
          setVoiceSample(file, file.name, null);
          showNotice(() => t("Sample ready. Preview it and check the exact transcript before saving."), "success");
          renderState(true);
        } catch (error) {
          showNotice(() => errorText(error), "error");
          target.value = "";
        }
      }
    } else if (target.id === "asset-filter") {
      assetFilter = target.value;
      renderLibrary();
      recordRenderKeys();
    } else if (target.id === "export-format") {
      exportFormat = target.value;
      recordRenderKeys();
    }
  });

  document.addEventListener("focusin", (event) => {
    const card = event.target.closest("[data-segment-card]");
    if (card && !sameId(activeSegmentId, card.dataset.segmentId)) {
      const previous = document.querySelector(".segment-card.is-selected");
      if (previous) previous.classList.remove("is-selected");
      card.classList.add("is-selected");
      activeSegmentId = card.dataset.segmentId;
      recordRenderKeys();
    }
  });

  byId("new-session-form").addEventListener("submit", createSession);
  byId("new-segment-form").addEventListener("submit", createSegment);
  byId("voice-upload-form").addEventListener("submit", uploadVoice);
  byId("voice-record-button").dataset.action = "record-voice";
  byId("voice-rerecord-button").dataset.action = "rerecord-voice";
  byId("voice-stop-button").dataset.action = "stop-voice";
  byId("voice-discard-button").dataset.action = "discard-voice";
  byId("refresh-button").addEventListener("click", () => refreshState(true, true));
  byId("archive-toggle").addEventListener("click", () => {
    showArchivedSessions = !showArchivedSessions;
    renderSessions();
    recordRenderKeys();
  });
  byId("session-toolbar").addEventListener("submit", (event) => {
    if (event.target.id === "rename-session-form") renameSession(event).catch((error) => showNotice(() => errorText(error), "error"));
  });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) refreshState(false, true);
    else if (recorderPhase !== "idle") stopVoiceRecording();
  });
  window.addEventListener("pagehide", () => {
    if (voiceRecorder) voiceRecorder.cancel();
    const previewUrl = voiceForm.previewUrl;
    voiceForm.previewUrl = null;
    revokeVoicePreviewUrl(previewUrl);
    const preview = byId("voice-preview");
    preview.pause();
    preview.removeAttribute("src");
    preview.load();
  });
  window.addEventListener("pageshow", () => {
    if (!voiceForm.file) return;
    try { createVoicePreviewUrl(); renderState(true); }
    catch (error) { showNotice(() => errorText(error), "error"); }
  });
  document.addEventListener("error", (event) => {
    if (event.target instanceof HTMLMediaElement) showNotice(() => t("Could not open the audio file. Refresh the library and try again."), "error");
  }, true);

  i18n.apply();
  byId("language-select").value = i18n.getLanguage();
  voiceRecorder = new window.Drama3VoiceRecorder({
    environment: window,
    onState: (snapshot) => {
      recorderPhase = snapshot.phase;
      renderState(true);
    },
    onComplete: (blob, duration) => {
      try {
        setVoiceSample(blob, "microphone-reference.wav", duration);
        showNotice(() => t("Recording ready. Preview it and check the exact transcript before saving."), "success");
        renderState(true);
      } catch (error) {
        showNotice(() => errorText(error), "error");
      }
    },
    onError: (error) => {
      showNotice(() => t(error && error.message ? error.message : "Microphone recording could not start."), "error");
      renderState(true);
    }
  });
  renderState(true);
  updateConnection(false, "Connecting…");
  refreshState(false, true);
})();
