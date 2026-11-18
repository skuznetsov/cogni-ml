"""Run and audit the frozen native QLoRA truth-training arm."""

from array import array
from contextlib import redirect_stdout
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


MODEL_SHA = "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223"
INITIAL_SHA = "f5715a8fa3555954871f1a47bc2c2b164ae0565a80708769ef256d9138fbb074"
TRAINER_SHA = "fb97af3d0c7e45de65e2401388fca7d7db4c13fe144b466aded72e07ba827b74"
UPDATES = 50
SEED = 42
FULL_BLOCKS = 4
LAYERS = [11, 15, 19, 23]
RANK = 8
PROJECTIONS = (
    ("Q", 2048, 4096), ("K", 2048, 512), ("V", 2048, 512),
    ("O", 2048, 2048), ("gate", 2048, 6144), ("up", 2048, 6144),
    ("down", 6144, 2048),
)
CONFIG = {
    "updates": UPDATES,
    "seed": SEED,
    "full_blocks": FULL_BLOCKS,
    "layers": LAYERS,
    "rank": RANK,
    "projections": [name for name, _, _ in PROJECTIONS],
    "early_stopping": False,
}


def _digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8") as target:
            json.dump(value, target, indent=2, sort_keys=True, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _rows(path: Path):
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path.name}:{number}") from error
        messages = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"{path.name}:{number} must contain three SFT messages")
        if [m.get("role") for m in messages if isinstance(m, dict)] != ["system", "user", "assistant"]:
            raise ValueError(f"{path.name}:{number} must be system/user/assistant SFT")
        if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
            raise ValueError(f"{path.name}:{number} has an empty or invalid message")
        rows.append(row)
    return rows


def _checker(repo: Path):
    path = repo / "build/qlora/scale-2b/qlora-20/check_adapter.py"
    spec = importlib.util.spec_from_file_location("truth_training_check_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load adapter checker: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (module.GGUF_SHA != MODEL_SHA or module.LAYERS != LAYERS or
            module.RANK != RANK or tuple(module.PROJECTIONS) != PROJECTIONS):
        raise RuntimeError("adapter checker does not match the pinned QLoRA layout")
    return path, module


def _protocol_file_hash(protocol, repo: Path, path: Path):
    files = protocol.get("files")
    if not isinstance(files, dict):
        raise ValueError("protocol.json is missing its sealed code and binary hashes")
    path = path.resolve()
    matches = []
    for name, digest in files.items():
        candidate = Path(name)
        if not candidate.is_absolute():
            candidate = repo / candidate
        if candidate.resolve() == path:
            matches.append(digest)
    if len(matches) != 1:
        raise ValueError(f"protocol.json must pin {path}")
    return matches[0]


def _events(log: Path):
    result = []
    if not log.exists():
        return result
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("malformed trainer event in train.log") from error
        if isinstance(event, dict) and "event" in event:
            result.append(event)
    return result


def _partial(events):
    updates = [event for event in events if event.get("event") == "update"]
    visits = [event.get("row") for event in updates if isinstance(event.get("row"), int)]
    losses = [event.get("loss") for event in updates]
    losses = [loss if not isinstance(loss, float) or math.isfinite(loss) else str(loss) for loss in losses]
    tokens = [event for event in events if event.get("event") == "prefill"]
    return {
        "updates_observed": len(updates),
        "visits": visits,
        "losses": losses,
        "prefill_target_tokens": [
            {"file": event.get("file"), "row": event.get("row"), "target_tokens": event.get("target_tokens")}
            for event in tokens
        ],
    }


def _load_factors(path: Path):
    with path.open("rb") as source:
        source.readline()
        source.readline()
        source.readline()
        values = array("f")
        values.frombytes(source.read())
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _audit(attempt: Path, inputs, config, train_rows, instrument_rows, token_counts_file,
           checker_path, checker):
    log = attempt / "train.log"
    adapter = attempt / "adapter.bin"
    if not log.is_file() or not adapter.is_file():
        raise ValueError("trainer did not produce both train.log and adapter.bin")
    events = _events(log)
    starts = [event for event in events if event.get("event") == "start"]
    saves = [event for event in events if event.get("event") == "saved"]
    updates = [event for event in events if event.get("event") == "update"]
    if len(starts) != 1 or len(saves) != 1:
        raise ValueError("trainer log must contain one start and one saved event")
    start, saved = starts[0], saves[0]
    if events[-1].get("event") != "saved":
        raise ValueError("trainer log does not end with a saved event")
    if [event.get("step") for event in updates] != list(range(1, UPDATES + 1)):
        raise ValueError("trainer did not report exactly 50 ordered updates")
    visits = [event.get("row") for event in updates]
    if any(not isinstance(row, int) or not 1 <= row <= len(train_rows) for row in visits):
        raise ValueError("trainer reported a visit outside train.jsonl")
    losses = [event.get("loss") for event in updates]
    if any(not isinstance(loss, (int, float)) or not math.isfinite(loss) for loss in losses):
        raise ValueError("trainer reported a nonfinite update loss")
    for field, expected in (("model_sha256", MODEL_SHA), ("seed", SEED),
                            ("updates", UPDATES), ("examples", len(train_rows)),
                            ("input_adapter", inputs["initial"]["path"]),
                            ("blocks", LAYERS)):
        if start.get(field) != expected:
            raise ValueError(f"trainer start event has wrong {field}")
    initial_loss = start.get("validation_loss")
    final_loss = saved.get("validation_loss")
    if any(not isinstance(loss, (int, float)) or not math.isfinite(loss)
           for loss in (initial_loss, final_loss)):
        raise ValueError("trainer reported a nonfinite validation loss")
    if saved.get("adapter") != str(adapter.resolve()):
        raise ValueError("trainer saved to an unexpected adapter path")
    if any(not isinstance(saved.get(field), (int, float)) or not math.isfinite(saved[field])
           for field in ("seconds",)):
        raise ValueError("trainer reported invalid elapsed time")

    expected_prefills = {"train.jsonl": len(train_rows), "instrument.jsonl": len(instrument_rows)}
    prefills = [event for event in events if event.get("event") == "prefill"]
    token_counts = {"train": {}, "instrument": {}}
    native_counts = _json(token_counts_file)
    if not isinstance(native_counts, list) or len(native_counts) != len(train_rows):
        raise ValueError("train-counts.json row count does not match train.jsonl")
    for number, count in enumerate(native_counts, 1):
        if (not isinstance(count, dict) or count.get("row") != number or
                count.get("id") != train_rows[number - 1].get("id") or
                not isinstance(count.get("prefix_tokens"), int) or
                not isinstance(count.get("target_tokens"), int)):
            raise ValueError(f"invalid native token count record at row {number}")
    for filename, count in expected_prefills.items():
        matching = [event for event in prefills if event.get("file") == filename]
        if len(matching) != count:
            raise ValueError(f"trainer prefill count mismatch for {filename}")
        for event in matching:
            row, tokens = event.get("row"), event.get("target_tokens")
            lane = "train" if filename == "train.jsonl" else "instrument"
            if (not isinstance(row, int) or not 1 <= row <= count or str(row) in token_counts[lane] or
                    not isinstance(tokens, int) or tokens <= 0):
                raise ValueError(f"invalid prefill row or target token count for {filename}")
            if lane == "train":
                expected = native_counts[row - 1]
                if (event.get("prefix_tokens") != expected["prefix_tokens"] or
                        tokens != expected["target_tokens"]):
                    raise ValueError(f"trainer token counts differ from native counter at train row {row}")
            token_counts[lane][str(row)] = tokens
    if len(prefills) != len(train_rows) + len(instrument_rows):
        raise ValueError("trainer logged unexpected prefill inputs")

    with open(os.devnull, "w") as sink, redirect_stdout(sink):
        adapter_sha = checker.check_adapter(adapter)
        checker.check_adapter(Path(inputs["initial"]["path"]))
    before, after = _load_factors(Path(inputs["initial"]["path"])), _load_factors(adapter)
    if len(before) != len(after):
        raise ValueError("initial and trained adapter factor counts differ")
    offset, factor_updates = 0, []
    for layer in LAYERS:
        for name, width, height in PROJECTIONS:
            a_end = offset + RANK * width
            b_end = a_end + RANK * height
            changed_a = sum(a != b for a, b in zip(before[offset:a_end], after[offset:a_end]))
            changed_b = sum(a != b for a, b in zip(before[a_end:b_end], after[a_end:b_end]))
            if not changed_a or not changed_b:
                raise ValueError(f"LoRA factors did not both update: layer {layer} {name}")
            factor_updates.append({
                "layer": layer, "projection": name, "rank": RANK,
                "a_elements": a_end - offset, "a_changed_elements": changed_a,
                "b_elements": b_end - a_end, "b_changed_elements": changed_b,
            })
            offset = b_end
    if offset != len(before) or offset != len(after):
        raise ValueError("adapter factor layout has trailing or missing values")
    if _digest(checker_path) != inputs["adapter_checker"]["sha256"]:
        raise ValueError("adapter checker changed during training")
    for key, record in inputs.items():
        if _digest(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"training input changed during run: {key}")

    visit_counts = {str(row): visits.count(row) for row in range(1, len(train_rows) + 1)}
    visited = {row for row in visits}
    return {
        "status": "complete",
        "inputs": inputs,
        "config": config,
        "train_rows": len(train_rows),
        "instrument_rows": len(instrument_rows),
        "holdout_supplied": False,
        "losses": {"initial_validation": initial_loss, "updates": losses,
                   "final_validation": final_loss},
        "updates_observed": len(updates),
        "visits": visits,
        "visit_counts": visit_counts,
        "unvisited_rows": [row for row in range(1, len(train_rows) + 1) if row not in visited],
        "coverage_complete": len(visited) == len(train_rows),
        "target_tokens_by_train_row": token_counts["train"],
        "instrument_target_tokens": token_counts["instrument"]["1"],
        "supervised_target_tokens_for_updates": sum(token_counts["train"][str(row)] for row in visits),
        "factor_updates": factor_updates,
        "updated_projections": len(factor_updates),
        "updated_factors": len(factor_updates) * 2,
        "adapter_sha256": adapter_sha,
        "train_log_sha256": _digest(log),
        "elapsed_seconds": saved["seconds"],
    }


def _archive_root_outputs(training: Path) -> None:
    existing = [training / name for name in ("adapter.bin", "train.log") if (training / name).exists()]
    if not existing:
        return
    archive = training / "attempts" / ("legacy-" + uuid.uuid4().hex)
    archive.mkdir(parents=True)
    for path in existing:
        shutil.move(str(path), str(archive / path.name))


def _publish(training: Path, attempt: Path, audit) -> None:
    if (training / "audit.json").exists():
        raise FileExistsError("refusing to replace existing training audit")
    _archive_root_outputs(training)
    for name in ("adapter.bin", "train.log"):
        os.link(attempt / name, training / name)
    _write_new(training / "audit.json", audit)


def fit(out: Path, initial: Path, model: Path, trainer: Path) -> Path:
    """Train from the pinned adapter and preserve every complete or failed attempt."""
    out = Path(out).resolve(strict=True)
    if not out.is_dir():
        raise NotADirectoryError(out)
    training = out / "training"
    training.mkdir(exist_ok=True)
    train_path, instrument_path, protocol_path = (
        out / "train.jsonl", out / "instrument.jsonl", out / "protocol.json")
    for path in (train_path, instrument_path, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol = _json(protocol_path)
    if not isinstance(protocol, dict):
        raise ValueError("protocol.json must contain the coordinator's sealed object")
    train_rows, instrument_rows = _rows(train_path), _rows(instrument_path)
    if len(train_rows) != 32:
        raise ValueError(f"expected 32 frozen train rows, found {len(train_rows)}")
    train_ids = [row.get("id") for row in train_rows]
    if any(not isinstance(identifier, str) or not identifier for identifier in train_ids) or len(set(train_ids)) != 32:
        raise ValueError("train.jsonl must have 32 uniquely identified rows")
    if len(instrument_rows) != 1:
        raise ValueError("expected one unrelated instrument row")
    if instrument_rows[0]["messages"] in [row["messages"] for row in train_rows]:
        raise ValueError("instrument row duplicates a training example")
    if instrument_rows[0].get("id") in set(train_ids):
        raise ValueError("instrument row id collides with a training row")

    initial, model, trainer = (Path(path).resolve(strict=True) for path in (initial, model, trainer))
    if _digest(initial) != INITIAL_SHA:
        raise ValueError("initial adapter is not the pinned tool-four adapter")
    if _digest(model) != MODEL_SHA:
        raise ValueError("model is not the pinned Qwen3.5-2B Q4_K_M GGUF")
    if _digest(trainer) != TRAINER_SHA or not os.access(trainer, os.X_OK):
        raise ValueError("trainer is not the verified native-recovery QLoRA binary")
    repo = Path(__file__).resolve().parents[4]
    checker_path, checker = _checker(repo)
    token_counter = repo / "build/qlora/four-repair/qlora_token_counts"
    for path, expected in ((Path(__file__).resolve(), _digest(Path(__file__).resolve())),
                          (initial, INITIAL_SHA), (trainer, TRAINER_SHA),
                          (token_counter, _digest(token_counter))):
        if _protocol_file_hash(protocol, repo, path) != expected:
            raise ValueError(f"protocol file hash mismatch: {path.name}")
    if not token_counter.is_file() or not os.access(token_counter, os.X_OK):
        raise ValueError("native Qwen3.5 token counter is unavailable")
    if protocol.get("pins", {}).get("model") != MODEL_SHA:
        raise ValueError("protocol does not pin the expected Qwen3.5 model")
    counts_path = out / "train-counts.json"
    if not counts_path.is_file():
        raise FileNotFoundError(counts_path)
    count_rows = _json(counts_path)
    if not isinstance(count_rows, list) or len(count_rows) != len(train_rows):
        raise ValueError("train-counts.json must contain one native token count per train row")
    protocol_inputs = protocol.get("inputs")
    if not isinstance(protocol_inputs, dict):
        raise ValueError("protocol.json is missing its sealed input hashes")
    for path in (train_path, instrument_path, counts_path):
        if protocol_inputs.get(path.name) != _digest(path):
            raise ValueError(f"protocol input hash mismatch: {path.name}")
    protocol_config = {
        "train_seed": SEED, "rank": RANK, "blocks": LAYERS, "steps": UPDATES,
    }
    if any(protocol.get(field) != expected for field, expected in protocol_config.items()):
        raise ValueError("protocol training configuration does not match fit()")
    with open(os.devnull, "w") as sink, redirect_stdout(sink):
        checker.check_adapter(initial)
    inputs = {
        "train": {"path": str(train_path), "sha256": _digest(train_path)},
        "instrument": {"path": str(instrument_path), "sha256": _digest(instrument_path)},
        "train_counts": {"path": str(counts_path), "sha256": _digest(counts_path)},
        "protocol": {"path": str(protocol_path), "sha256": _digest(protocol_path)},
        "fit": {"path": str(Path(__file__).resolve()), "sha256": _digest(Path(__file__).resolve())},
        "initial": {"path": str(initial), "sha256": _digest(initial)},
        "model": {"path": str(model), "sha256": _digest(model)},
        "trainer": {"path": str(trainer), "sha256": _digest(trainer)},
        "adapter_checker": {"path": str(checker_path), "sha256": _digest(checker_path)},
    }
    config = dict(CONFIG)

    root_audit = training / "audit.json"
    if root_audit.exists():
        prior = _json(root_audit)
        if prior.get("status") != "complete" or prior.get("inputs") != inputs or prior.get("config") != config:
            raise RuntimeError("existing training audit does not match the frozen inputs")
        attempt = training / "attempts" / prior.get("attempt", "")
        if not attempt.is_dir():
            raise RuntimeError("completed training attempt evidence is missing")
        audited = _audit(attempt, inputs, config, train_rows, instrument_rows,
                         counts_path, checker_path, checker)
        if audited != {key: value for key, value in prior.items() if key != "attempt"}:
            raise RuntimeError("completed training evidence no longer matches its audit")
        if _digest(training / "adapter.bin") != prior["adapter_sha256"] or _digest(
                training / "train.log") != prior["train_log_sha256"]:
            raise RuntimeError("published training artifacts changed")
        return training / "adapter.bin"

    attempts = training / "attempts"
    attempts.mkdir(exist_ok=True)
    # Recover a fully saved attempt if the prior process stopped before publishing it.
    for attempt in sorted(attempts.glob("attempt-*")):
        meta_path = attempt / "inputs.json"
        if not meta_path.is_file() or _json(meta_path) != {"inputs": inputs, "config": config}:
            continue
        prior_attempt_audit = attempt / "audit.json"
        saved_attempt_audit = _json(prior_attempt_audit) if prior_attempt_audit.exists() else None
        if saved_attempt_audit is not None and saved_attempt_audit.get("status") != "complete":
            continue
        try:
            audited = _audit(attempt, inputs, config, train_rows, instrument_rows,
                             counts_path, checker_path, checker)
            if saved_attempt_audit is not None and {
                    key: value for key, value in saved_attempt_audit.items() if key != "attempt"} != audited:
                raise RuntimeError("attempt audit disagrees with saved trainer evidence")
            if not prior_attempt_audit.exists():
                _write_new(prior_attempt_audit, audited)
            _publish(training, attempt, dict(audited, attempt=attempt.name))
            return training / "adapter.bin"
        except (OSError, ValueError, RuntimeError) as error:
            if saved_attempt_audit is not None:
                raise RuntimeError(f"completed training attempt evidence is invalid: {attempt}") from error
            try:
                observed = _partial(_events(attempt / "train.log"))
            except ValueError as parse_error:
                observed = {"parse_error": str(parse_error)}
            _write_new(prior_attempt_audit, {
                "status": "invalid", "inputs": inputs, "config": config,
                "error": str(error), "observed": observed,
            })

    _archive_root_outputs(training)
    indices = [int(p.name.removeprefix("attempt-")) for p in attempts.glob("attempt-[0-9]*")
               if p.name.removeprefix("attempt-").isdigit()]
    attempt = attempts / f"attempt-{max(indices, default=0) + 1:03d}"
    attempt.mkdir()
    _write_new(attempt / "inputs.json", {"inputs": inputs, "config": config})
    adapter = attempt / "adapter.bin"
    command = [str(trainer), "train", str(model), str(train_path), str(instrument_path),
               str(adapter), str(UPDATES), str(SEED), str(FULL_BLOCKS), str(initial)]
    try:
        with (attempt / "train.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        audited = _audit(attempt, inputs, config, train_rows, instrument_rows,
                         counts_path, checker_path, checker)
    except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError) as error:
        try:
            observed = _partial(_events(attempt / "train.log"))
        except ValueError as parse_error:
            observed = {"parse_error": str(parse_error)}
        _write_new(attempt / "audit.json", {
            "status": "incomplete" if isinstance(error, (OSError, subprocess.CalledProcessError)) else "invalid",
            "inputs": inputs, "config": config, "error": str(error), "observed": observed,
        })
        raise RuntimeError(f"truth training attempt preserved at {attempt}: {error}") from error
    _write_new(attempt / "audit.json", audited)
    _publish(training, attempt, dict(audited, attempt=attempt.name))
    return training / "adapter.bin"

