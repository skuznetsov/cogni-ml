"""CPU-only fail-closed tests for the terminal MLX prefill fixture exporter."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "drama3" / "prefill_reference.py"
SPEC = importlib.util.spec_from_file_location("drama3_prefill_reference", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
prefill_reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prefill_reference)


UPSTREAM_REVISION = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
MODEL_REVISION = "5331ab509359c0e4a7e353a19e255ddf3e8152a8"
PROMPT_SOURCE_SHA256 = "c2f9c80ed2e4ba6c73a3806e3fdbd94f5cea80c75cce450a003e4e71d17f25c2"
MODEL_FILES = [
    "codec.safetensors",
    "config.json",
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class PrefillReferenceSpec(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        self.report_path = self.root / "report.json"
        self.checkpoint_path = self.root / "checkpoint.json"
        self.capture_path = self.root / "capture.npz"
        self.output_dir = self.root / "prefill"
        self.hidden = np.linspace(-1, 1, 2048, dtype=np.float32).reshape(1, 2048)
        self.main = np.linspace(-2, 2, 4097, dtype=np.float32).reshape(1, 4097)
        np.savez(self.capture_path,
                 step_00000_prompt_backbone_hidden=self.hidden,
                 step_00000_main=self.main)
        self._make_model_and_checkpoint()
        self.report = self._report()
        self._write_report()

    def tearDown(self):
        self.temp.cleanup()

    def _make_model_and_checkpoint(self, config=None):
        if config is None:
            config = {
                "architectures": ["FishOmniModel"],
                "model_type": "fish_omni",
                "audio_decoder_config": {"num_codebooks": 10, "vocab_size": 4096},
                "semantic_start_token_id": 248099,
                "eos_token_id": 248046,
                "sample_rate": 44100,
                "text_config": {"hidden_size": 2048},
            }
        (self.model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        weight_map = {
            f"tensor_{i}": f"model-{i + 1:05d}-of-00005.safetensors"
            for i in range(5)
        }
        (self.model_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf-8"
        )
        for name in MODEL_FILES:
            path = self.model_dir / name
            if not path.exists():
                path.write_bytes(("fixture:" + name).encode("ascii"))
        entries = []
        for name in MODEL_FILES:
            path = self.model_dir / name
            content = path.read_bytes()
            stat = path.stat()
            entries.append({
                "file": name,
                "bytes": len(content),
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256(content),
            })
        self.checkpoint = {
            "scope": "Hashes of actual cached local checkpoint bytes",
            "model_dir": str(self.model_dir),
            "recorded_at_utc": "2026-10-15T20:19:00.651262+00:00",
            "files": entries,
            "rechecked_from": str(self.root / "prior-checkpoint.json"),
        }
        self.checkpoint_path.write_text(json.dumps(self.checkpoint), encoding="utf-8")

    def _report(self):
        text = "Привет, мир."
        transcript = "Reference transcript."
        return {
            "format": "drama3-mlx-control-v1",
            "created_utc": "2026-10-15T20:31:46.342691+00:00",
            "upstream": {
                "repository": "/cache/mlx-audio",
                "revision": UPSTREAM_REVISION,
                "worktree_clean": True,
            },
            "source_hashes": {
                "mlx_audio/tts/models/fish_omni/prompt.py": PROMPT_SOURCE_SHA256,
                "scripts/drama3/generate_mlx_control.py": "1" * 64,
                "scripts/drama3/replay_mlx_generation.py": "2" * 64,
            },
            "model": {
                "repository": "cguldogan/drama-3-mlx-q4",
                "revision": MODEL_REVISION,
                "path": str(self.model_dir),
                "model_type": "fish_omni",
                "config_sha256": self.checkpoint["files"][1]["sha256"],
                "index_sha256": self.checkpoint["files"][7]["sha256"],
                "indexed_shards": MODEL_FILES[2:7],
                "num_codebooks": 10,
                "codebook_size": 4096,
                "semantic_begin": 248099,
                "eos_id": 248046,
                "sample_rate": 44100,
            },
            "input": {
                "text": text,
                "text_utf8_sha256": sha256(text.encode("utf-8")),
                "speaker_id": 0,
                "speaker_tag": "<|speaker:0|>",
            },
            "settings": {"capture_steps": [0]},
            "termination": {
                "status": "eos",
                "eos_detected": True,
                "frame_cap_reached": False,
                "returned_frames": 2,
            },
            "sampler_capture": {
                "path": self.capture_path.name,
                "sha256": sha256(self.capture_path.read_bytes()),
                "entries": [
                    {"key": "step_00000_main", "step": 0, "sampler": "main",
                     "shape": [1, 4097], "dtype": "float32"},
                    {"key": "step_00000_prompt_backbone_hidden", "step": 0,
                     "kind": "prompt.backbone.hidden", "shape": [1, 2048],
                     "dtype": "float32"},
                ],
            },
            "reference_voice": {
                "supplied": True,
                "complete_conditioning": True,
                "audio": {"wav_sha256": "a" * 64},
                "transcript": transcript,
                "transcript_utf8_sha256": sha256(transcript.encode("utf-8")),
            },
        }

    def _write_report(self):
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")

    def _export(self):
        return prefill_reference.export_prefill_reference(
            self.report_path, self.checkpoint_path, self.output_dir
        )

    def test_exports_exact_step_zero_arrays_and_all_checkpoint_hashes(self):
        manifest_path = self._export()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["format"], "drama3-prefill-reference-v1")
        self.assertEqual(manifest["upstream_revision"], UPSTREAM_REVISION)
        self.assertEqual(manifest["source_identity"]["revision"], UPSTREAM_REVISION)
        self.assertEqual(manifest["source_identity"]["source_hashes"], self.report["source_hashes"])
        self.assertEqual(len(manifest["model_files"]), 9)
        self.assertEqual(
            {entry["file"] for entry in manifest["model_files"]}, set(MODEL_FILES)
        )
        self.assertEqual(manifest["report_sha256"], sha256(self.report_path.read_bytes()))
        self.assertEqual(manifest["checkpoint_manifest_sha256"], sha256(self.checkpoint_path.read_bytes()))
        self.assertEqual(manifest["capture_sha256"], sha256(self.capture_path.read_bytes()))
        self.assertEqual(
            [stage["name"] for stage in manifest["stages"]],
            ["prompt.backbone.hidden", "prompt.main_logits"],
        )
        for stage, values in zip(manifest["stages"], [self.hidden, self.main]):
            raw = (self.output_dir / stage["file"]).read_bytes()
            expected = np.asarray(values, dtype="<f4", order="C").tobytes(order="C")
            self.assertEqual(raw, expected)
            self.assertEqual(stage["sha256"], sha256(expected))
            self.assertEqual(stage["capture_key"],
                             "step_00000_prompt_backbone_hidden" if values.shape[1] == 2048
                             else "step_00000_main")

    def test_cli_accepts_explicit_report_checkpoint_and_output_flags(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--report", str(self.report_path),
             "--checkpoint-manifest", str(self.checkpoint_path), "--output-dir",
             str(self.output_dir)],
            check=False, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["manifest"],
                         str(self.output_dir.resolve() / "manifest.json"))
        self.assertTrue((self.output_dir / "manifest.json").is_file())

    def test_rejects_unpinned_upstream_and_does_not_create_output(self):
        self.report["upstream"]["revision"] = "0" * 40
        self._write_report()
        with self.assertRaises(prefill_reference.ExportError):
            self._export()
        self.assertFalse(self.output_dir.exists())

    def test_rejects_report_to_checkpoint_config_hash_mismatch(self):
        self.report["model"]["config_sha256"] = "0" * 64
        self._write_report()
        with self.assertRaises(prefill_reference.ExportError):
            self._export()
        self.assertFalse(self.output_dir.exists())

    def test_rejects_checkpoint_manifest_stale_file_metadata(self):
        self.checkpoint["files"][0]["bytes"] += 1
        self.checkpoint_path.write_text(json.dumps(self.checkpoint), encoding="utf-8")
        with self.assertRaises(prefill_reference.ExportError):
            self._export()
        self.assertFalse(self.output_dir.exists())

    def test_rejects_wrong_step_zero_capture_name_or_shape_metadata(self):
        self.report["sampler_capture"]["entries"][0]["key"] = "step_00000_main_logits"
        self._write_report()
        with self.assertRaises(prefill_reference.ExportError):
            self._export()
        self.assertFalse(self.output_dir.exists())

    def test_rejects_capture_bytes_not_linked_by_report_hash(self):
        self.report["sampler_capture"]["sha256"] = "0" * 64
        self._write_report()
        with self.assertRaises(prefill_reference.ExportError):
            self._export()
        self.assertFalse(self.output_dir.exists())

    def test_rejects_npz_array_shape_mismatch_even_if_report_claims_expected_shape(self):
        np.savez(self.capture_path,
                 step_00000_prompt_backbone_hidden=self.hidden[:, :-1],
                 step_00000_main=self.main)
        self.report["sampler_capture"]["sha256"] = sha256(self.capture_path.read_bytes())
        self._write_report()
        with self.assertRaises(prefill_reference.ExportError):
            self._export()
        self.assertFalse(self.output_dir.exists())

    def test_rejects_non_finite_step_zero_array(self):
        bad_main = self.main.copy()
        bad_main[0, 0] = np.nan
        np.savez(self.capture_path,
                 step_00000_prompt_backbone_hidden=self.hidden,
                 step_00000_main=bad_main)
        self.report["sampler_capture"]["sha256"] = sha256(self.capture_path.read_bytes())
        self._write_report()
        with self.assertRaises(prefill_reference.ExportError):
            self._export()
        self.assertFalse(self.output_dir.exists())

    def test_rejects_dirty_source_identity(self):
        self.report["upstream"]["worktree_clean"] = False
        self._write_report()
        with self.assertRaises(prefill_reference.ExportError):
            self._export()
        self.assertFalse(self.output_dir.exists())

    def test_rejects_checkpoint_config_with_wrong_hidden_width(self):
        config = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        config["text_config"]["hidden_size"] = 1024
        self._make_model_and_checkpoint(config)
        config_hash = next(item["sha256"] for item in self.checkpoint["files"]
                           if item["file"] == "config.json")
        self.report["model"]["config_sha256"] = config_hash
        self._write_report()
        with self.assertRaises(prefill_reference.ExportError):
            self._export()
        self.assertFalse(self.output_dir.exists())

    def test_refuses_to_overwrite_any_existing_output(self):
        self.output_dir.mkdir()
        marker = self.output_dir / "keep.txt"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaises(prefill_reference.ExportError):
            self._export()
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
