(() => {
  "use strict";

  // English message IDs are the fallback copy. No browser locale detection:
  // English stays the default until a supported choice is explicitly saved.
  const storageKey = "drama3-studio-language";
  const russian = {
    "DRAMA · Audio studio": "DRAMA · Аудиостудия",
    "Interface language": "Язык интерфейса",
    "DRAMA, home": "DRAMA, главная",
    "audio studio": "аудиостудия",
    "Connecting…": "Подключение…",
    "Refresh library": "Обновить библиотеку",
    "Sessions": "Сессии",
    "Workspace": "Рабочее пространство",
    "New session name": "Название новой сессии",
    "Project name": "Название проекта",
    "Create session": "Создать сессию",
    "Show archive": "Показать архив",
    "Session list": "Список сессий",
    "No sessions yet. Create your first one to get started.": "Здесь пока нет сессий. Создайте первую, чтобы начать.",
    "Each segment is generated separately. Assign a reference voice for a similar sound.": "Каждый фрагмент создаётся отдельно. Для похожего звучания назначьте ему голос-референс.",
    "Session editor": "Редактор сессии",
    "The job queue is paused.": "Очередь задач приостановлена.",
    "Studio state could not be saved. The server is retrying; new jobs will wait.": "Не удалось сохранить состояние студии. Сервер повторяет попытку; новые задачи пока не запускаются.",
    "Job queue": "Очередь задач",
    "New segment": "Новый фрагмент",
    "Add text to turn into speech": "Добавьте текст для озвучивания",
    "New segment text": "Текст нового фрагмента",
    "Type or paste text…": "Введите или вставьте текст…",
    "The new segment will be added to the end of this session.": "Новый фрагмент появится в конце сессии.",
    "Add segment": "Добавить фрагмент",
    "Session start": "Начало сессии",
    "Your story starts here": "Здесь будет ваша история",
    "Add text above, then choose a voice and generate your first audio segment.": "Добавьте текст выше — затем выберите голос и создайте первый аудиофрагмент.",
    "Library": "Библиотека",
    "Materials": "Материалы",
    "Library sections": "Разделы библиотеки",
    "Voices": "Голоса",
    "Audio": "Аудио",
    "Add a voice": "Добавить голос",
    "up to 10 sec": "до 10 сек",
    "Choose an audio file": "Выберите аудиофайл",
    "WAV, MP3 or another supported format · up to 20 MB": "WAV, MP3 или другой поддерживаемый формат · до 20 МБ",
    "Name": "Название",
    "For example, Quiet narrator": "Например, Тихий рассказчик",
    "What is said in the recording": "Что произнесено в записи",
    "An exact transcript helps match the voice": "Точная расшифровка помогает сопоставить голос",
    "Use a prepared reading or your own short text. Speak in a neutral, natural style at a comfortable volume. A short sample can guide a similar sound, but results vary.": "Прочитайте предложенный текст или выберите свой короткий. Говорите нейтрально и естественно, с комфортной громкостью. Короткая запись может помочь приблизить звучание, но результат зависит от текста и записи.",
    "English sample": "Текст на английском",
    "Russian sample": "Текст на русском",
    "Use this text": "Использовать этот текст",
    "Microphone recording": "Запись с микрофона",
    "Record with microphone": "Записать с микрофона",
    "Stop recording": "Остановить запись",
    "Record again": "Записать ещё раз",
    "Discard sample": "Удалить образец",
    "Preview before saving": "Прослушайте перед сохранением",
    "Save voice": "Сохранить голос",
    "Waiting for microphone permission…": "Ожидаем разрешение на микрофон…",
    "Preparing microphone…": "Подготовка микрофона…",
    "Recording. Stop when finished; capture ends at 9.5 seconds.": "Идёт запись. Остановите её по окончании; запись завершится через 9,5 секунды.",
    "Finishing recording…": "Завершаем запись…",
    "Cancel recording setup": "Отменить подключение микрофона",
    "Recording length: {seconds} sec": "Длительность записи: {seconds} сек",
    "Audio preview is not available in this browser.": "В этом браузере недоступно прослушивание образца.",
    "Sample discarded.": "Образец удалён.",
    "Choose or record an audio sample first.": "Сначала выберите аудио или запишите образец.",
    "Preview the sample before saving.": "Прослушайте образец перед сохранением.",
    "Enter the exact transcript before saving.": "Введите точную расшифровку перед сохранением.",
    "Sample ready. Preview it and check the exact transcript before saving.": "Образец готов. Прослушайте его и проверьте точную расшифровку перед сохранением.",
    "Recording ready. Preview it and check the exact transcript before saving.": "Запись готова. Прослушайте её и проверьте точную расшифровку перед сохранением.",
    "Voice saved, but the library could not be refreshed.": "Голос сохранён, но библиотеку не удалось обновить.",
    "Microphone recording is not supported in this browser.": "Этот браузер не поддерживает запись с микрофона.",
    "Microphone permission was denied.": "В разрешении на использование микрофона отказано.",
    "No microphone is available.": "Микрофон не найден.",
    "The microphone is already in use or unavailable.": "Микрофон уже занят или недоступен.",
    "Microphone recording requires a secure local page.": "Для записи с микрофона нужна защищённая локальная страница.",
    "No microphone audio track was available.": "Аудиодорожка микрофона недоступна.",
    "The microphone track has already ended.": "Аудиодорожка микрофона уже завершилась.",
    "The microphone sample rate is outside the supported range.": "Частота дискретизации микрофона не поддерживается.",
    "Uncompressed microphone capture is not supported in this browser.": "Этот браузер не поддерживает запись микрофона без сжатия.",
    "The microphone recorder stopped unexpectedly.": "Запись микрофона неожиданно остановилась.",
    "The microphone recorder returned invalid audio data.": "Записывающее устройство вернуло некорректные аудиоданные.",
    "No microphone audio was captured.": "Звук с микрофона не записан.",
    "Microphone recording could not start.": "Не удалось начать запись с микрофона.",
    "Upload voice": "Загрузить голос",
    "Saved voices": "Сохранённые голоса",
    "Add a short recording to use as a reference.": "Добавьте короткую запись, чтобы использовать её как референс.",
    "Audio filter": "Фильтр аудио",
    "This session": "Эта сессия",
    "All sessions": "Все сессии",
    "Saved audio will appear here after generation or export.": "Сохранённые аудио появятся здесь после генерации или экспорта.",
    "Unknown error": "Неизвестная ошибка",
    "Request failed ({status})": "Запрос не выполнен ({status})",
    "The server returned invalid studio state.": "Сервер вернул некорректное состояние студии.",
    "Conflict": "Конфликт",
    "Saving…": "Сохранение…",
    "Unsaved": "Не сохранено",
    "Saved": "Сохранено",
    "Hide archive": "Скрыть архив",
    "Untitled": "Без названия",
    "Archived": "В архиве",
    "Restore from archive": "Вернуть из архива",
    "Move to archive": "Переместить в архив",
    "Restore session from archive": "Вернуть сессию из архива",
    "Move session to archive": "Переместить сессию в архив",
    "{count} segment": "{count} фрагмент",
    "{count} segments": "{count} фрагментов",
    "Start with a new session": "Начните с новой сессии",
    "Create a session in the left panel to add text and audio.": "Создайте рабочую сессию в левой панели, чтобы добавить текст и аудио.",
    "Your workspace": "Ваше пространство",
    "Text session": "Текстовая сессия",
    "New name for this session": "Новое название сессии",
    "Done": "Готово",
    "Cancel": "Отмена",
    "Rename session": "Переименовать сессию",
    "{count} · exports use selected takes in order": "{count} · экспорт использует выбранные записи по порядку",
    "Export format": "Формат экспорта",
    "Export session": "Экспорт сессии",
    "Select audio for every segment before exporting": "Для экспорта нужно выбрать аудио для каждого фрагмента",
    "Exports include all segments in order. Generate or select a take for each first ({count} not ready).": "Экспорт включает все фрагменты по порядку. Сначала создайте или выберите запись для каждого (не готово: {count}).",
    "Segment {number}": "Фрагмент {number}",
    "Queued": "В очереди",
    "Generating…": "Создание…",
    "Save": "Сохранить",
    "Delete": "Удалить",
    "Delete segment": "Удалить фрагмент",
    "Text": "Текст",
    "Reference voice": "Голос-референс",
    "No reference": "Без референса",
    "[archived] {name}": "[архив] {name}",
    "Voice": "Голос",
    "Saved voice unavailable": "Сохранённый голос недоступен",
    "Generate audio": "Создать аудио",
    "Generation settings": "Параметры генерации",
    "Window size": "Размер окна",
    "Manual": "Вручную",
    "Automatic (memory budget)": "Автоматически (по памяти)",
    "Automatic mode grows the window during generation within the memory budget and frame limit. Speech ends at EOS; duration cannot be predicted exactly.": "Автоматический режим расширяет окно во время генерации в пределах бюджета памяти и лимита кадров. Речь заканчивается на EOS; точно предсказать длительность нельзя.",
    "Automatic windows are unavailable on this server or with a voice reference.": "Автоматическое окно недоступно на этом сервере или при использовании голоса-референса.",
    "Frame limit": "Лимит кадров",
    "This segment changed on the server after your edits. Reload it or copy your text before continuing.": "Сегмент изменился на сервере после ваших правок. Перезагрузите версию или скопируйте текст перед продолжением.",
    "Load server version": "Загрузить версию сервера",
    "Selected audio take": "Выбранная аудиозапись",
    "The selected take is unavailable in the library.": "Выбранная аудиозапись недоступна в библиотеке.",
    "No audio for this segment yet. Save the text, then generate your first take.": "Для этого фрагмента пока нет аудио. Создайте первую запись после сохранения текста.",
    "Selected take": "Выбранная запись",
    "Version {revision}": "Версия {revision}",
    "Selected": "Выбрано",
    "Possibly truncated": "Возможна обрезка",
    "Play selected take": "Прослушать выбранную запись",
    "The model did not signal the end of this segment. Check the take before exporting.": "Модель не сообщила о завершении фрагмента: проверьте запись перед экспортом.",
    "Take history": "История записей",
    "No takes yet.": "Записей пока нет.",
    "Version {revision} · selected": "Версия {revision} · выбрана",
    "possibly truncated": "возможна обрезка",
    "Listen": "Слушать",
    "Listen to version {revision}": "Слушать версию {revision}",
    "Select": "Выбрать",
    "duration unknown": "длительность неизвестна",
    "{seconds} sec": "{seconds} сек",
    "Export": "Экспорт",
    "Audio generation": "Генерация аудио",
    "Job": "Задача",
    "{title} · Segment {number}": "{title} · Фрагмент {number}",
    "Local queue": "Локальная очередь",
    "Waiting to run": "Ожидает выполнения",
    "Running": "Выполняется",
    "Job failed": "Задача завершилась с ошибкой",
    "Prefill": "Подготовка",
    "Frame": "Кадр",
    "In progress": "В работе",
    "Error": "Ошибка",
    "Unknown status": "Статус неизвестен",
    "Unnamed voice": "Голос без названия",
    "Move voice to archive": "Переместить голос в архив",
    "Play reference {name}": "Прослушать референс {name}",
    " Export conversion is unavailable.": " Конвертация экспорта недоступна.",
    "Formats: {formats}": "Форматы: {formats}",
    " · reference up to {seconds} sec.": " · референс до {seconds} сек.",
    "Take": "Запись",
    "version {revision}": "версия {revision}",
    "Play {name}": "Прослушать {name}",
    "Download": "Скачать",
    "Delete audio file": "Удалить аудиофайл",
    "Possibly truncated: recording stopped at the frame limit.": "Возможна обрезка: запись остановлена на лимите.",
    "Add text before saving.": "Добавьте текст перед сохранением.",
    "Check the numeric generation settings.": "Проверьте числовые параметры генерации.",
    "Seed, frame limit and Top K must be integers.": "Seed, лимит кадров и Top K должны быть целыми числами.",
    "Archived sessions cannot be edited.": "Архивную сессию нельзя изменять.",
    "Load the current segment version first.": "Сначала загрузите актуальную версию сегмента.",
    "Changes were sent, but studio state could not be refreshed. Check the connection and refresh the studio.": "Изменения отправлены, но состояние не обновилось. Проверьте соединение и обновите студию.",
    "Queue running": "Очередь работает",
    "Connected locally": "Локально подключено",
    "Disconnected": "Нет соединения",
    "Could not refresh studio state: {error}": "Не удалось обновить состояние студии: {error}",
    "Enter a session name.": "Введите название сессии.",
    "Session created.": "Сессия создана.",
    "Create a session first.": "Сначала создайте сессию.",
    "Enter segment text.": "Введите текст фрагмента.",
    "Segment added.": "Фрагмент добавлен.",
    "The name cannot be empty.": "Название не может быть пустым.",
    "Name updated.": "Название обновлено.",
    "Session moved to archive.": "Сессия перемещена в архив.",
    "Session restored from archive.": "Сессия возвращена из архива.",
    "Delete this text segment? Its takes may remain in the audio library.": "Удалить этот текстовый фрагмент? Связанные записи могут остаться в аудиобиблиотеке.",
    "Segment deleted.": "Фрагмент удалён.",
    "Loaded the current segment version.": "Загружена актуальная версия сегмента.",
    "Resolve the version conflict before generating.": "Разрешите конфликт версии перед генерацией.",
    "Job added to the queue. The previous take remains available.": "Задача добавлена в очередь. Предыдущая запись останется доступна.",
    "Selected a different take.": "Выбрана другая запись.",
    "Export added to the queue. The result will appear in the audio library.": "Экспорт добавлен в очередь. Результат появится в библиотеке аудио.",
    "Choose an audio file for the reference.": "Выберите аудиофайл для референса.",
    "The file exceeds the 20 MB limit.": "Файл превышает ограничение 20 МБ.",
    "Enter a voice name.": "Укажите название голоса.",
    "Voice added to the library.": "Голос добавлен в библиотеку.",
    "Move voice “{name}” to archive?": "Переместить голос «{name}» в архив?",
    "Voice moved to archive.": "Голос перемещён в архив.",
    "Delete audio file “{name}”?": "Удалить аудиофайл «{name}»?",
    "untitled": "без названия",
    "Audio file deleted.": "Аудиофайл удалён.",
    "Changes saved.": "Изменения сохранены.",
    "Could not open the audio file. Refresh the library and try again.": "Не удалось открыть аудиофайл. Обновите библиотеку и попробуйте ещё раз."
  };
  let language = "en";
  try {
    if (window.localStorage.getItem(storageKey) === "ru") language = "ru";
  } catch (_) { /* Storage may be blocked; the page still works. */ }

  function t(message, parameters = {}) {
    const text = language === "ru" && Object.hasOwn(russian, message) ? russian[message] : message;
    return text.replace(/\{(\w+)\}/g, (placeholder, key) => Object.hasOwn(parameters, key) ? String(parameters[key]) : placeholder);
  }

  function setLanguage(value) {
    language = value === "ru" ? "ru" : "en";
    try { window.localStorage.setItem(storageKey, language); }
    catch (_) { /* Keep the current page's choice when storage is blocked. */ }
  }

  function apply(root = document) {
    document.documentElement.lang = language;
    document.title = t("DRAMA · Audio studio");
    root.querySelectorAll("[data-i18n], [data-i18n-placeholder], [data-i18n-title], [data-i18n-aria-label]").forEach((element) => {
      if (element.dataset.i18n) element.textContent = t(element.dataset.i18n);
      for (const [field, attribute] of [["i18nPlaceholder", "placeholder"], ["i18nTitle", "title"], ["i18nAriaLabel", "aria-label"]]) {
        if (element.dataset[field]) element.setAttribute(attribute, t(element.dataset[field]));
      }
    });
  }

  function segmentCount(count) {
    if (language === "ru") {
      const plural = new Intl.PluralRules("ru").select(count);
      const word = plural === "one" ? "фрагмент" : plural === "few" ? "фрагмента" : "фрагментов";
      return count + " " + word;
    }
    return t(count === 1 ? "{count} segment" : "{count} segments", { count });
  }

  function formatSeconds(value) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds) || seconds < 0) return t("duration unknown");
    if (seconds < 10) return t("{seconds} sec", { seconds: seconds.toFixed(1).replace(".", language === "ru" ? "," : ".") });
    const whole = Math.floor(seconds);
    const minutes = Math.floor(whole / 60);
    return minutes ? minutes + ":" + String(whole % 60).padStart(2, "0") : t("{seconds} sec", { seconds: whole });
  }

  function formatStamp(value) {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    return new Intl.DateTimeFormat(language === "ru" ? "ru-RU" : "en-US", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }).format(date);
  }

  window.StudioI18n = Object.freeze({ t, getLanguage: () => language, setLanguage, apply, segmentCount, formatSeconds, formatStamp });
})();
