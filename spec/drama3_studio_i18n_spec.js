"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const assets = path.join(__dirname, "../src/ml/audio/drama3/studio/assets");
const source = fs.readFileSync(path.join(assets, "i18n.js"), "utf8");

function load(saved, blocked = false) {
  const values = new Map(saved === undefined ? [] : [["drama3-studio-language", saved]]);
  const document = { documentElement: {}, querySelectorAll: () => [] };
  const window = {
    navigator: { language: "ru-RU" },
    localStorage: {
      getItem(key) { if (blocked) throw new Error("Storage denied"); return values.get(key) || null; },
      setItem(key, value) { if (blocked) throw new Error("Storage denied"); values.set(key, value); }
    }
  };
  vm.runInNewContext(source, { window, document, Intl });
  return { i18n: window.StudioI18n, document, values };
}

test("English is the default even with a Russian browser; only supported saved languages apply", () => {
  for (const saved of [undefined, "", "de", "__proto__"]) {
    const { i18n, document } = load(saved);
    assert.equal(i18n.getLanguage(), "en");
    i18n.apply();
    assert.equal(document.documentElement.lang, "en");
    assert.equal(document.title, "DRAMA · Audio studio");
    assert.equal(i18n.t("Sessions"), "Sessions");
  }
  assert.equal(load("ru").i18n.t("Sessions"), "Сессии");
});

test("switching persists the choice and updates the document language and title", () => {
  const { i18n, document, values } = load();
  i18n.setLanguage("ru");
  i18n.apply();
  assert.equal(values.get("drama3-studio-language"), "ru");
  assert.equal(document.documentElement.lang, "ru");
  assert.equal(document.title, "DRAMA · Аудиостудия");
  assert.equal(load(values.get("drama3-studio-language")).i18n.getLanguage(), "ru");
  i18n.setLanguage("en");
  assert.equal(i18n.t("Sessions"), "Sessions");
});

test("blocked browser storage still allows language changes for the current page", () => {
  const { i18n } = load("ru", true);
  assert.equal(i18n.getLanguage(), "en");
  assert.doesNotThrow(() => i18n.setLanguage("ru"));
  assert.equal(i18n.t("Sessions"), "Сессии");
});

test("interpolation preserves user content literally, including placeholder syntax", () => {
  const { i18n } = load();
  const name = '<script>{revision}</script> $& "Тест"';
  assert.equal(i18n.t('Delete audio file “{name}”?', { name }), `Delete audio file “${name}”?`);
  i18n.setLanguage("ru");
  assert.equal(i18n.t('Delete audio file “{name}”?', { name }), `Удалить аудиофайл «${name}»?`);
});

test("segment counts and audio duration follow the selected language", () => {
  const { i18n } = load();
  assert.equal(i18n.segmentCount(1), "1 segment");
  assert.equal(i18n.segmentCount(21), "21 segments");
  assert.equal(i18n.formatSeconds(1.5), "1.5 sec");
  assert.equal(i18n.formatSeconds(65), "1:05");
  assert.match(i18n.formatStamp("2026-01-02T12:00:00Z"), /Jan/);
  i18n.setLanguage("ru");
  for (const [count, text] of [[0, "0 фрагментов"], [1, "1 фрагмент"], [2, "2 фрагмента"], [11, "11 фрагментов"], [21, "21 фрагмент"], [22, "22 фрагмента"], [112, "112 фрагментов"]]) {
    assert.equal(i18n.segmentCount(count), text);
  }
  assert.equal(i18n.formatSeconds(1.5), "1,5 сек");
  assert.equal(i18n.formatSeconds(NaN), "длительность неизвестна");
  assert.match(i18n.formatStamp("2026-01-02T12:00:00Z"), /янв/);
  assert.equal(i18n.formatStamp("invalid"), "");
});

test("static translation changes labels and attributes without touching input values", () => {
  const { i18n, document } = load("ru");
  const input = { dataset: { i18nPlaceholder: "Project name" }, value: "Мой проект", setAttribute(key, value) { this[key] = value; } };
  const label = { dataset: { i18n: "Sessions" }, textContent: "Sessions" };
  document.querySelectorAll = () => [input, label];
  i18n.apply();
  assert.equal(input.placeholder, "Название проекта");
  assert.equal(input.value, "Мой проект");
  assert.equal(label.textContent, "Сессии");
});
