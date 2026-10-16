(() => {
  "use strict";

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
  let voiceForm = { name: "", transcript: "", file: null };
  let exportFormat = null;
  let renderedEditorSessionId = null;
  let lastKeys = {};
  let noticeTimer = null;
  let pollTimer = null;
  let refreshPromise = null;
  let stateRequestSequence = 0;

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
    target.textContent = message;
    target.className = "notice" + (kind ? " is-" + kind : "");
    target.hidden = !message;
    if (noticeTimer) window.clearTimeout(noticeTimer);
    if (message && kind !== "error") {
      noticeTimer = window.setTimeout(() => { target.hidden = true; }, 6500);
    }
  }

  function updateConnection(connected, message) {
    const status = byId("connection-status");
    const group = status.parentElement;
    status.textContent = message;
    group.classList.toggle("is-online", connected);
    group.classList.toggle("is-error", !connected);
  }

  function errorText(error) {
    const message = error && error.message ? error.message : String(error || "Неизвестная ошибка");
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
      const message = payload && (payload.error || payload.message) ? payload.error || payload.message : "Запрос не выполнен (" + response.status + ")";
      throw new Error(String(message));
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
    if (!payload || typeof payload !== "object") throw new Error("Сервер вернул некорректное состояние студии.");
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
          draft: draft ? [draft.text, draft.voice_id, draft.seed, draft.max_frames, draft.temperature, draft.top_k, draft.top_p, draft.dirty, draft.saving, draft.conflict, draft.version] : null
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
      voiceForm: [voiceForm.name, voiceForm.transcript, voiceForm.file && voiceForm.file.name, voiceForm.file && voiceForm.file.size],
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
      badge.textContent = draft.conflict ? "Конфликт" : draft.saving ? "Сохранение…" : draft.dirty ? "Не сохранено" : "Сохранено";
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
    byId("archive-toggle").textContent = showArchivedSessions ? "Скрыть архив" : "Показать архив";
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
      const name = node("span", "session-name", session.name || "Без названия");
      const metaText = session.archived ? "В архиве" : segmentCountText(sessionSegments(session).length);
      const meta = node("span", "session-meta", metaText);
      append(copy, name, meta);
      append(select, glyph, copy);
      const actions = node("span", "session-row-actions");
      actions.append(makeButton(session.archived ? "↶" : "⌑", "mini-action", session.archived ? "restore-session" : "archive-session", {
        title: session.archived ? "Вернуть из архива" : "Переместить в архив",
        ariaLabel: session.archived ? "Вернуть сессию из архива" : "Переместить сессию в архив"
      }));
      actions.lastElementChild.dataset.id = session.id;
      append(row, select, actions);
      list.append(row);
    });
  }

  function segmentCountText(count) {
    if (count === 1) return "1 фрагмент";
    if (count > 1 && count < 5) return count + " фрагмента";
    return count + " фрагментов";
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
      const heading = node("h2", "", "Начните с новой сессии");
      const copy = node("p", "", "Создайте рабочую сессию в левой панели, чтобы добавить текст и аудио.");
      empty.replaceChildren(node("div", "empty-orbit"), node("p", "eyebrow", "Ваше пространство"), heading, copy);
      renderedEditorSessionId = null;
      return;
    }

    empty.hidden = true;
    composer.hidden = false;
    const titleBlock = node("div", "session-title-block");
    const eyebrow = node("p", "eyebrow", "Текстовая сессия");
    if (renameEditing) {
      const form = node("form", "rename-form");
      form.id = "rename-session-form";
      const input = node("input");
      input.id = "rename-session-input";
      input.type = "text";
      input.maxLength = 120;
      input.required = true;
      input.value = renameDraft;
      input.setAttribute("aria-label", "Новое название сессии");
      const saveName = makeButton("Готово", "button button-secondary");
      saveName.type = "submit";
      const cancelName = makeButton("Отмена", "button button-quiet", "cancel-rename");
      append(form, input, saveName, cancelName);
      append(titleBlock, eyebrow, form);
    } else {
      const line = node("div", "session-title-line");
      const title = node("h1", "session-title", session.name || "Без названия");
      if (session.archived) line.append(node("span", "archive-pill", "В архиве"));
      const rename = makeButton("✎", "rename-button", "rename-session", { title: "Переименовать сессию", ariaLabel: "Переименовать сессию" });
      append(line, title, rename);
      const subtitle = node("p", "session-subtitle", segmentCountText(sessionSegments(session).length) + " · экспорт использует выбранные записи по порядку");
      append(titleBlock, eyebrow, line, subtitle);
    }

    const toolbarActions = node("div", "toolbar-actions");
    const format = node("select");
    format.id = "export-format";
    format.setAttribute("aria-label", "Формат экспорта");
    const formats = supportedFormats();
    formats.forEach((value) => {
      const option = node("option", "", value.toUpperCase());
      option.value = value;
      format.append(option);
    });
    exportFormat = formats.includes(exportFormat) ? exportFormat : formats[0];
    format.value = exportFormat;
    const exportButton = makeButton("Экспорт сессии", "button button-secondary", "export-session", { disabled: Boolean(session.archived) || sessionSegments(session).length === 0 });
    exportButton.dataset.id = session.id;
    const missing = sessionSegments(session).filter((segment) => {
      if (!segment.selected_take_id) return true;
      return !state.assets.some((asset) => asset.kind === "take" && sameId(asset.id, segment.selected_take_id) && sameId(asset.session_id, session.id) && sameId(asset.segment_id, segment.id));
    }).length;
    if (missing > 0 && sessionSegments(session).length > 0) {
      exportButton.disabled = true;
      exportButton.title = "Для экспорта нужно выбрать аудио для каждого фрагмента";
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
      const hint = node("p", "field-hint export-hint", "Экспорт включает все фрагменты по порядку. Сначала создайте или выберите запись для каждого (не готово: " + missing + ").");
      const footer = document.querySelector(".composer-footer");
      if (footer) footer.append(hint);
    }

    sessionSegments(session).forEach((segment, index) => segmentList.append(renderSegmentCard(session, segment, index)));
    if (sessionSegments(session).length === 0) {
      const blank = node("div", "editor-empty");
      blank.append(node("div", "empty-orbit"), node("p", "eyebrow", "Начало сессии"), node("h2", "", "Здесь будет ваша история"), node("p", "", "Добавьте текст выше — затем выберите голос и создайте первый аудиофрагмент."));
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
    const heading = node("h3", "", "Фрагмент " + String(index + 1).padStart(2, "0"));
    const pending = state.jobs.find((job) => sameId(job.segment_id, segment.id) && (job.status === "queued" || job.status === "running"));
    const pendingLabel = pending ? node("span", "success-pill", pending.status === "queued" ? "В очереди" : "Создание…") : null;
    append(titleGroup, number, heading, pendingLabel);
    const headActions = node("div", "segment-actions");
    const badge = node("span", "save-state" + (draft.dirty || draft.conflict ? " is-dirty" : ""), draft.conflict ? "Конфликт" : draft.saving ? "Сохранение…" : draft.dirty ? "Не сохранено" : "Сохранено");
    badge.setAttribute("aria-live", "polite");
    const save = makeButton("Сохранить", "button button-quiet", "save-segment", { disabled: !draft.dirty || draft.saving || draft.conflict || Boolean(session.archived) });
    save.dataset.id = segment.id;
    const remove = makeButton("Удалить", "button button-quiet", "delete-segment", { disabled: Boolean(session.archived), title: "Удалить фрагмент" });
    remove.dataset.id = segment.id;
    append(headActions, badge, save, remove);
    append(head, titleGroup, headActions);

    const body = node("div", "segment-body");
    const textLabel = node("label", "segment-label", "Текст");
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
    const voiceLabel = node("label", "field-label", "Голос-референс");
    voiceLabel.htmlFor = "segment-voice-" + String(segment.id);
    const voiceSelect = node("select");
    voiceSelect.id = voiceLabel.htmlFor;
    voiceSelect.dataset.segmentId = segment.id;
    voiceSelect.dataset.field = "voice_id";
    voiceSelect.disabled = Boolean(session.archived);
    const noVoice = node("option", "", "Без референса");
    noVoice.value = "";
    voiceSelect.append(noVoice);
    state.voices.filter((voice) => !voice.archived || sameId(voice.id, draft.voice_id)).forEach((voice) => {
      const option = node("option", "", (voice.archived ? "[архив] " : "") + (voice.name || "Голос"));
      option.value = voice.id;
      option.selected = sameId(voice.id, draft.voice_id);
      voiceSelect.append(option);
    });
    if (draft.voice_id && !state.voices.some((voice) => sameId(voice.id, draft.voice_id))) {
      const unavailable = node("option", "", "Сохранённый голос недоступен");
      unavailable.value = draft.voice_id;
      unavailable.selected = true;
      voiceSelect.append(unavailable);
    }
    voiceSelect.value = draft.voice_id || "";
    append(voiceWrap, voiceLabel, voiceSelect);
    const generate = makeButton("Создать аудио", "button button-primary", "generate-segment", { disabled: Boolean(session.archived) || Boolean(pending) });
    generate.dataset.id = segment.id;
    append(controls, voiceWrap, generate);
    body.append(controls);

    const settings = node("details", "settings-disclosure");
    settings.open = settingsDisclosure.get(String(segment.id)) === true;
    settings.addEventListener("toggle", () => settingsDisclosure.set(String(segment.id), settings.open));
    const settingsSummary = node("summary", "", "Параметры генерации");
    const grid = node("div", "settings-grid");
    const settingsSpec = [
      ["seed", "Seed", "number", "1"],
      ["max_frames", "Лимит кадров", "number", "1"],
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
    body.append(settings);

    if (draft.conflict) {
      const conflict = node("div", "notice is-error", "Сегмент изменился на сервере после ваших правок. Перезагрузите версию или скопируйте текст перед продолжением.");
      const reload = makeButton("Загрузить версию сервера", "button button-quiet", "reload-segment");
      reload.dataset.id = segment.id;
      append(body, conflict, reload);
    }
    body.append(renderSelectedTake(session, segment));
    card.append(head, body);
    return card;
  }

  function renderSelectedTake(session, segment) {
    const wrapper = node("section", "selected-take");
    wrapper.setAttribute("aria-label", "Выбранная аудиозапись");
    const take = state.assets.find((asset) => sameId(asset.id, segment.selected_take_id));
    if (segment.selected_take_id && !take) {
      wrapper.append(node("div", "history-empty", "Выбранная аудиозапись недоступна в библиотеке."));
    } else if (!take) {
      wrapper.append(node("div", "history-empty", "Для этого фрагмента пока нет аудио. Создайте первую запись после сохранения текста."));
    } else {
      const heading = node("div", "take-heading");
      const copy = node("div", "");
      const title = node("strong", "", take.filename || "Выбранная запись");
      const meta = node("small", "", "Версия " + (take.revision === undefined ? "—" : take.revision) + " · " + formatSeconds(take.seconds) + stampSuffix(take.created_at));
      append(copy, title, meta);
      const tags = node("div", "take-tags");
      tags.append(node("span", "take-tag", "Выбрано"));
      if (take.stopped_on_eos === false) tags.append(node("span", "take-tag is-warning", "Возможна обрезка"));
      append(heading, copy, tags);
      const player = node("audio");
      player.controls = true;
      player.preload = "none";
      player.src = "/api/audio/" + idPart(take.id);
      player.setAttribute("aria-label", "Прослушать выбранную запись");
      append(wrapper, heading, player);
      if (take.stopped_on_eos === false) {
        wrapper.append(node("p", "field-hint", "Модель не сообщила о завершении фрагмента: проверьте запись перед экспортом."));
      }
    }

    const history = node("details", "take-history");
    history.open = historyDisclosure.get(String(segment.id)) === true;
    history.addEventListener("toggle", () => historyDisclosure.set(String(segment.id), history.open));
    history.append(node("summary", "", "История записей"));
    const list = node("div", "history-list");
    const takes = state.assets.filter((asset) => asset.kind === "take" && sameId(asset.session_id, session.id) && sameId(asset.segment_id, segment.id)).sort((a, b) => assetTime(b) - assetTime(a));
    if (takes.length === 0) {
      list.append(node("div", "history-empty", "Записей пока нет."));
    } else {
      takes.forEach((asset) => {
        const row = node("div", "history-item");
        const copy = node("div", "history-copy");
        const title = node("strong", "", "Версия " + (asset.revision === undefined ? "—" : asset.revision) + (sameId(asset.id, segment.selected_take_id) ? " · выбрана" : ""));
        const meta = node("span", "", formatSeconds(asset.seconds) + stampSuffix(asset.created_at) + (asset.stopped_on_eos === false ? " · возможна обрезка" : ""));
        append(copy, title, meta);
        const actions = node("div", "history-actions");
        const listen = node("a", "", "Слушать");
        listen.href = "/api/audio/" + idPart(asset.id);
        listen.textContent = "Слушать";
        listen.setAttribute("aria-label", "Слушать версию " + (asset.revision === undefined ? "" : asset.revision));
        const select = makeButton(sameId(asset.id, segment.selected_take_id) ? "Выбрано" : "Выбрать", "button button-quiet", "select-take", { disabled: sameId(asset.id, segment.selected_take_id) || Boolean(session.archived) });
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

  function formatSeconds(value) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds) || seconds < 0) return "длительность неизвестна";
    if (seconds < 10) return seconds.toFixed(1).replace(".", ",") + " сек";
    const whole = Math.floor(seconds);
    const minutes = Math.floor(whole / 60);
    const remainder = whole % 60;
    return minutes ? minutes + ":" + String(remainder).padStart(2, "0") : whole + " сек";
  }

  function stampSuffix(value) {
    const stamp = formatStamp(value);
    return stamp ? " · " + stamp : "";
  }

  function formatStamp(value) {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    return new Intl.DateTimeFormat("ru-RU", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }).format(date);
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
      const titleText = job.kind === "export" ? "Экспорт" : job.kind === "generate" ? "Генерация аудио" : String(job.kind || "Задача");
      const title = node("span", "job-title", titleText + (segment ? " · Фрагмент " + (sessionSegments(session).indexOf(segment) + 1) : ""));
      const status = node("span", "job-state", jobStatus(job.status));
      append(top, title, status);
      const detail = node("div", "job-detail");
      const context = session ? session.name : "Локальная очередь";
      const detailText = job.status === "queued" ? "Ожидает выполнения" : job.status === "running" ? jobDetailText(job.detail) || "Выполняется" : job.status === "failed" ? "Задача завершилась с ошибкой" : "Готово";
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
      const phase = counter[1].toLowerCase() === "prefill" ? "Подготовка" : "Кадр";
      return phase + " · " + counter[2] + " / " + counter[3];
    }
    return detail;
  }

  function jobStatus(status) {
    if (status === "queued") return "В очереди";
    if (status === "running") return "В работе";
    if (status === "complete") return "Готово";
    if (status === "failed") return "Ошибка";
    return String(status || "Статус неизвестен");
  }

  function renderLibrary() {
    const voiceList = byId("voice-list");
    const assetList = byId("asset-list");
    voiceList.replaceChildren();
    assetList.replaceChildren();
    byId("voice-name").value = voiceForm.name;
    byId("voice-transcript").value = voiceForm.transcript;
    byId("voice-file-label").textContent = voiceForm.file ? voiceForm.file.name : "Выберите аудиофайл";
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
        node("strong", "", voice.name || "Голос без названия"),
        node("small", "", formatSeconds(voice.seconds) + (voice.filename ? " · " + voice.filename : ""))
      );
      const remove = makeButton("⌫", "mini-action", "delete-voice", { title: "Переместить голос в архив", ariaLabel: "Переместить голос в архив" });
      remove.dataset.id = voice.id;
      append(top, info, remove);
      card.append(top);
      if (voice.transcript) card.append(node("p", "voice-transcript", voice.transcript));
      const player = node("audio");
      player.controls = true;
      player.preload = "none";
      player.src = "/api/voice-audio/" + idPart(voice.id);
      player.setAttribute("aria-label", "Прослушать референс " + (voice.name || ""));
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
    const ffmpegNote = state.capabilities.ffmpeg === false ? " Конвертация экспорта недоступна." : "";
    byId("capability-note").replaceChildren(node("strong", "", "Форматы: " + (formats || "WAV")), document.createTextNode(" · референс до " + limit + " сек." + ffmpegNote));
  }

  function renderAssetCard(asset) {
    const card = node("article", "asset-card");
    const top = node("div", "asset-card-top");
    const info = node("div", "asset-info");
    const session = state.sessions.find((item) => sameId(item.id, asset.session_id));
    const label = asset.kind === "export" ? "Экспорт" : "Запись";
    const metaParts = [String(asset.format || "audio").toUpperCase(), formatSeconds(asset.seconds)];
    if (asset.revision !== undefined && asset.revision !== null) metaParts.push("версия " + asset.revision);
    if (session && assetFilter === "all") metaParts.push(session.name || "Без названия");
    if (asset.created_at) metaParts.push(formatStamp(asset.created_at));
    append(info, node("strong", "", asset.filename || label), node("small", "", metaParts.join(" · ")));
    const kind = node("span", "asset-kind" + (asset.kind === "export" ? " is-export" : ""), label);
    append(top, info, kind);
    const player = node("audio");
    player.controls = true;
    player.preload = "none";
    player.src = "/api/audio/" + idPart(asset.id);
    player.setAttribute("aria-label", "Прослушать " + (asset.filename || label));
    const actions = node("div", "asset-actions");
    const download = node("a", "", "Скачать");
    download.href = "/api/audio/" + idPart(asset.id) + "?download=1";
    download.setAttribute("download", "");
    const remove = makeButton("Удалить", "", "delete-asset", { title: "Удалить аудиофайл" });
    remove.dataset.id = asset.id;
    append(actions, download, remove);
    if (asset.stopped_on_eos === false) card.append(node("p", "field-hint", "Возможна обрезка: запись остановлена на лимите."));
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
      temperature: Number(draft.temperature),
      top_k: Number(draft.top_k),
      top_p: Number(draft.top_p)
    };
    if (!values.text.trim()) throw new Error("Добавьте текст перед сохранением.");
    if (![values.seed, values.max_frames, values.temperature, values.top_k, values.top_p].every(Number.isFinite)) throw new Error("Проверьте числовые параметры генерации.");
    if (!Number.isInteger(values.seed) || !Number.isInteger(values.max_frames) || !Number.isInteger(values.top_k)) throw new Error("Seed, лимит кадров и Top K должны быть целыми числами.");
    return values;
  }

  async function saveDraft(segmentId) {
    const key = String(segmentId);
    const draft = drafts.get(key);
    const session = selectedSession();
    if (!draft || !session) return false;
    if (!draft.dirty) return true;
    if (session.archived) throw new Error("Архивную сессию нельзя изменять.");
    if (draft.conflict) throw new Error("Сначала загрузите актуальную версию сегмента.");
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
        if (!refreshed) throw new Error("Изменения отправлены, но состояние не обновилось. Проверьте соединение и обновите студию.");
        const live = sessionSegments(selectedSession()).find((segment) => sameId(segment.id, segmentId));
        if (live) {
          draft.baseRevision = live.revision;
          if (draft.version === sentVersion) {
            Object.assign(draft, draftFromSegment(live));
          } else {
            draft.dirty = true;
            draft.conflict = false;
            window.setTimeout(() => saveDraft(segmentId).catch((error) => showNotice(errorText(error), "error")), 0);
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
        updateConnection(true, activeJobs().length ? "Очередь работает" : "Локально подключено");
        return true;
      } catch (error) {
        if (requestSequence === stateRequestSequence) {
          updateConnection(false, "Нет соединения");
          showNotice("Не удалось обновить состояние студии: " + errorText(error), "error");
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
    if (field === "voice_id") draft.voice_id = target.value || null;
    else if (field) draft[field] = target.value;
    draft.dirty = true;
    draft.version += 1;
    refreshLocalEditorState(draft.id);
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
    catch (error) { showNotice(errorText(error), "error"); }
    finally { locks.delete(key); }
  }

  async function createSession(event) {
    event.preventDefault();
    const name = newSessionName.trim();
    if (!name) { showNotice("Введите название сессии.", "error"); return; }
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
      showNotice("Сессия создана.", "success");
    });
  }

  async function createSegment(event) {
    event.preventDefault();
    const session = selectedSession();
    if (!session) { showNotice("Сначала создайте сессию.", "error"); return; }
    const text = currentComposerText();
    if (!text.trim()) { showNotice("Введите текст фрагмента.", "error"); return; }
    if (session.archived) { showNotice("Архивную сессию нельзя изменять.", "error"); return; }
    await withLock("create-segment", async () => {
      const before = new Set(sessionSegments(session).map((segment) => String(segment.id)));
      const result = await request("/api/sessions/" + idPart(session.id) + "/segments", jsonRequest("POST", {
        text,
        voice_id: null,
        seed: 0,
        max_frames: 512,
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
      showNotice("Фрагмент добавлен.", "success");
    });
  }

  async function renameSession(event) {
    event.preventDefault();
    const session = selectedSession();
    if (!session) return;
    const name = renameDraft.trim();
    if (!name) { showNotice("Название не может быть пустым.", "error"); return; }
    await withLock("rename-session", async () => {
      await request("/api/sessions/" + idPart(session.id), jsonRequest("PATCH", { name }));
      renameEditing = false;
      await refreshState(false, true);
      showNotice("Название обновлено.", "success");
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
      showNotice(archived ? "Сессия перемещена в архив." : "Сессия возвращена из архива.", "success");
    });
  }

  async function deleteSegment(id) {
    const session = selectedSession();
    const segment = sessionSegments(session).find((item) => sameId(item.id, id));
    if (!session || !segment) return;
    if (!window.confirm("Удалить этот текстовый фрагмент? Связанные записи могут остаться в аудиобиблиотеке.")) return;
    await withLock("segment-" + id, async () => {
      const draft = drafts.get(String(id));
      await request("/api/sessions/" + idPart(session.id) + "/segments/" + idPart(id) + "?revision=" + encodeURIComponent(String(draft ? draft.baseRevision : segment.revision)), { method: "DELETE" });
      drafts.delete(String(id));
      await refreshState(false, true);
      showNotice("Фрагмент удалён.", "success");
    });
  }

  async function reloadSegment(id) {
    const session = selectedSession();
    const live = sessionSegments(session).find((segment) => sameId(segment.id, id));
    if (!live) return;
    drafts.set(String(id), draftFromSegment(live));
    renderState(true);
    showNotice("Загружена актуальная версия сегмента.", "success");
  }

  async function generateSegment(id) {
    const session = selectedSession();
    const draft = drafts.get(String(id));
    if (!session || !draft) return;
    await withLock("generate-" + id, async () => {
      if (draft.dirty) await saveDraft(id);
      if (draft.conflict) throw new Error("Разрешите конфликт версии перед генерацией.");
      await request("/api/sessions/" + idPart(session.id) + "/segments/" + idPart(id) + "/generate", jsonRequest("POST", {}));
      await refreshState(false, true);
      showNotice("Задача добавлена в очередь. Предыдущая запись останется доступна.", "success");
    });
  }

  async function selectTake(segmentId, assetId) {
    const session = selectedSession();
    if (!session) return;
    await withLock("take-" + segmentId, async () => {
      await request("/api/sessions/" + idPart(session.id) + "/segments/" + idPart(segmentId) + "/select", jsonRequest("POST", { asset_id: assetId }));
      await refreshState(false, true);
      showNotice("Выбрана другая запись.", "success");
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
      showNotice("Экспорт добавлен в очередь. Результат появится в библиотеке аудио.", "success");
    });
  }

  async function uploadVoice(event) {
    event.preventDefault();
    const file = voiceForm.file;
    if (!file) { showNotice("Выберите аудиофайл для референса.", "error"); return; }
    if (file.size > 20 * 1024 * 1024) { showNotice("Файл превышает ограничение 20 МБ.", "error"); return; }
    const name = voiceForm.name.trim();
    if (!name) { showNotice("Укажите название голоса.", "error"); return; }
    const query = new URLSearchParams({ name, transcript: voiceForm.transcript });
    await withLock("upload-voice", async () => {
      await request("/api/voices?" + query.toString(), {
        method: "POST",
        headers: { "Content-Type": file.type || "application/octet-stream" },
        body: file
      });
      voiceForm = { name: "", transcript: "", file: null };
      byId("voice-upload-form").reset();
      await refreshState(false, true);
      renderState(true);
      showNotice("Голос добавлен в библиотеку.", "success");
    });
  }

  async function archiveVoice(id) {
    const voice = state.voices.find((item) => sameId(item.id, id));
    if (!voice) return;
    if (!window.confirm("Переместить голос «" + (voice.name || "Без названия") + "» в архив?")) return;
    await withLock("voice-" + id, async () => {
      await request("/api/voices/" + idPart(id), { method: "DELETE" });
      await refreshState(false, true);
      showNotice("Голос перемещён в архив.", "success");
    });
  }

  async function deleteAsset(id) {
    const asset = state.assets.find((item) => sameId(item.id, id));
    if (!asset) return;
    if (!window.confirm("Удалить аудиофайл «" + (asset.filename || "без названия") + "»?")) return;
    await withLock("asset-" + id, async () => {
      await request("/api/assets/" + idPart(id), { method: "DELETE" });
      await refreshState(false, true);
      showNotice("Аудиофайл удалён.", "success");
    });
  }

  function handleAction(action, target) {
    const id = target.dataset.id;
    if (action === "select-session") activateSession(id);
    else if (action === "archive-session") archiveSession(id, true).catch((error) => showNotice(errorText(error), "error"));
    else if (action === "restore-session") archiveSession(id, false).catch((error) => showNotice(errorText(error), "error"));
    else if (action === "rename-session") {
      const session = selectedSession();
      if (session && !session.archived) { renameEditing = true; renameDraft = session.name || ""; renderState(true); }
    } else if (action === "cancel-rename") { renameEditing = false; renderState(true); }
    else if (action === "save-segment") withLock("save-" + id, async () => { await saveDraft(id); showNotice("Изменения сохранены.", "success"); }).catch((error) => showNotice(errorText(error), "error"));
    else if (action === "delete-segment") deleteSegment(id).catch((error) => showNotice(errorText(error), "error"));
    else if (action === "reload-segment") reloadSegment(id);
    else if (action === "generate-segment") generateSegment(id).catch((error) => showNotice(errorText(error), "error"));
    else if (action === "select-take") selectTake(id, target.dataset.assetId).catch((error) => showNotice(errorText(error), "error"));
    else if (action === "export-session") exportSession(id).catch((error) => showNotice(errorText(error), "error"));
    else if (action === "delete-voice") archiveVoice(id).catch((error) => showNotice(errorText(error), "error"));
    else if (action === "delete-asset") deleteAsset(id).catch((error) => showNotice(errorText(error), "error"));
  }

  document.addEventListener("click", (event) => {
    const actionTarget = event.target.closest("[data-action]");
    if (actionTarget) {
      event.preventDefault();
      handleAction(actionTarget.dataset.action, actionTarget);
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
      recordRenderKeys();
    } else if (target.id === "voice-transcript") {
      voiceForm.transcript = target.value;
      recordRenderKeys();
    }
  });

  document.addEventListener("change", (event) => {
    const target = event.target;
    if (target.id === "voice-file") {
      voiceForm.file = target.files && target.files[0] ? target.files[0] : null;
      byId("voice-file-label").textContent = voiceForm.file ? voiceForm.file.name : "Выберите аудиофайл";
      recordRenderKeys();
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
  byId("refresh-button").addEventListener("click", () => refreshState(true, true));
  byId("archive-toggle").addEventListener("click", () => {
    showArchivedSessions = !showArchivedSessions;
    renderSessions();
    recordRenderKeys();
  });
  byId("session-toolbar").addEventListener("submit", (event) => {
    if (event.target.id === "rename-session-form") renameSession(event).catch((error) => showNotice(errorText(error), "error"));
  });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) refreshState(false, true); });
  document.addEventListener("error", (event) => {
    if (event.target instanceof HTMLMediaElement) showNotice("Не удалось открыть аудиофайл. Обновите библиотеку и попробуйте ещё раз.", "error");
  }, true);

  renderState(true);
  updateConnection(false, "Подключение…");
  refreshState(false, true);
})();
