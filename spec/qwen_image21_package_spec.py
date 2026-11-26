from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen_image21_package.py"
SPEC = importlib.util.spec_from_file_location("qwen_image21_package", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def _rgba_png(width: int, height: int) -> bytes:
    raw = b"".join(b"\x00" + bytes((32, 64, 96, 255)) * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 6, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )


class QwenImage21PackageSpec(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="qwen-image21-package-spec-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model_dir = self.root / "source-model"
        self._make_model(self.model_dir)

        self.gguf = self.root / "source.gguf"
        self.gguf.write_bytes(b"fixture mixed-quant GGUF")
        self.denoiser = self.root / "source-denoiser"
        self.denoiser.write_bytes(b"\xcf\xfa\xed\xfe" + struct.pack("<I", 0x0100000C) + b"fixture")
        self.denoiser.chmod(0o755)
        self.artifact_patcher = patch.multiple(
            PACKAGE,
            GGUF_SHA256=hashlib.sha256(self.gguf.read_bytes()).hexdigest(),
            GGUF_SIZE_BYTES=self.gguf.stat().st_size,
        )
        self.artifact_patcher.start()
        self.addCleanup(self.artifact_patcher.stop)

    def _make_model(self, model_dir: Path) -> None:
        processor = model_dir / "processor"
        encoder = model_dir / "text_encoder"
        vae = model_dir / "vae"
        scheduler = model_dir / "scheduler"
        for directory in (processor, encoder, vae, scheduler):
            directory.mkdir(parents=True, exist_ok=True)

        (model_dir / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "QwenImage21Pipeline",
                    "processor": ["transformers", "Qwen3VLProcessor"],
                    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                    "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
                    "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
                    "vae": ["diffusers", "AutoencoderKLQwenImage21"],
                }
            ),
            encoding="utf-8",
        )
        for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.jinja"):
            (processor / name).write_text("{}", encoding="utf-8")
        (encoder / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_vl",
                    "architectures": ["Qwen3VLForConditionalGeneration"],
                    "text_config": {"hidden_size": 4096},
                }
            ),
            encoding="utf-8",
        )
        weight_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        (encoder / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight.a": weight_names[0], "weight.b": weight_names[1]}}),
            encoding="utf-8",
        )
        for name in weight_names:
            (encoder / name).write_bytes(b"weights")
        (vae / "config.json").write_text(
            json.dumps(
                {
                    "_class_name": "AutoencoderKLQwenImage21",
                    "z_dim": 64,
                    "out_channels": 4,
                    "scale_factor_spatial": 16,
                }
            ),
            encoding="utf-8",
        )
        (vae / "diffusion_pytorch_model.safetensors").write_bytes(b"vae weights")
        (scheduler / "scheduler_config.json").write_text(
            json.dumps({"_class_name": "FlowMatchEulerDiscreteScheduler"}), encoding="utf-8"
        )
        metadata = model_dir / ".cache" / "huggingface" / "download" / "model_index.json.metadata"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(PACKAGE.MODEL_REVISION + "\n", encoding="utf-8")

    def _init(self, target: Path | None = None) -> Path:
        target = target or self.root / "package"
        return PACKAGE.initialize_package(
            package_dir=target,
            model_dir=self.model_dir,
            gguf_path=self.gguf,
            denoiser_path=self.denoiser,
            revision=PACKAGE.MODEL_REVISION,
        )

    def test_init_creates_self_contained_manifest_with_pinned_mixed_gguf_and_hardlinks(self) -> None:
        manifest_path = self._init()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        package_dir = manifest_path.parent

        self.assertEqual("qwen-image21-package", manifest["schema"])
        self.assertEqual(1, manifest["schema_version"])
        self.assertEqual(PACKAGE.MODEL_REVISION, manifest["model"]["revision"])
        self.assertEqual("hybrid", manifest["runtime"]["kind"])
        self.assertEqual("darwin-arm64-metal", manifest["runtime"]["target"])
        self.assertEqual(PACKAGE.GGUF_SHA256, manifest["gguf"]["sha256"])
        self.assertEqual(PACKAGE.GGUF_SIZE_BYTES, manifest["gguf"]["size_bytes"])
        self.assertEqual(PACKAGE.GGUF_TENSOR_POLICY, manifest["gguf"]["tensor_policy"])
        self.assertEqual(
            ["torch", "transformers", "diffusers", "Pillow", "NumPy"],
            manifest["runtime"]["python_packages"],
        )
        self.assertIn("not bundled", manifest["runtime"]["python_interpreter"])
        self.assertIn("not bundled", manifest["runtime"]["native_build_dependencies"])
        self.assertEqual(3, len(manifest["runtime"]["external_native_libraries"]))

        gguf_path = package_dir / manifest["components"]["dit_gguf"]
        encoder_path = package_dir / manifest["components"]["model_dir"] / "text_encoder" / "model-00001-of-00002.safetensors"
        self.assertEqual(self.gguf.stat().st_ino, gguf_path.stat().st_ino)
        self.assertEqual(
            (self.model_dir / "text_encoder" / "model-00001-of-00002.safetensors").stat().st_ino,
            encoder_path.stat().st_ino,
        )
        self.assertTrue((package_dir / "bin" / "qwen_image21_package.py").is_file())
        self.assertTrue((package_dir / "bin" / "qwen_image21_prepare_conditioning.py").is_file())
        self.assertTrue((package_dir / "bin" / "qwen_image21_vae_decode.py").is_file())
        self.assertTrue((package_dir / "src" / "qwen_image21_generate_latents.cr").is_file())
        self.assertEqual(
            {
                "dit_gguf",
                "denoiser",
                "conditioning_script",
                "decoder_script",
                "launcher_script",
                "denoiser_entrypoint",
                "model_files",
            },
            set(manifest["artifacts"]),
        )
        for key in (
            "conditioning_script",
            "decoder_script",
            "launcher_script",
            "denoiser_entrypoint",
        ):
            relative_path = manifest["components"][key]
            record = manifest["artifacts"][key]
            staged_path = package_dir / relative_path
            self.assertEqual(staged_path.stat().st_size, record["size_bytes"])
            self.assertEqual(hashlib.sha256(staged_path.read_bytes()).hexdigest(), record["sha256"])
        model_files = manifest["artifacts"]["model_files"]
        self.assertIn("text_encoder/model-00001-of-00002.safetensors", model_files)
        self.assertEqual(
            [
                "text_encoder/model-00001-of-00002.safetensors",
                "text_encoder/model-00002-of-00002.safetensors",
                "vae/diffusion_pytorch_model.safetensors",
            ],
            manifest["model_source_attestation"]["unavailable_safetensors"],
        )
        self.assertIn("not a cryptographic signature", manifest["model_source_attestation"]["limitation"])
        for relative_path, record in model_files.items():
            staged_path = package_dir / "model" / relative_path
            self.assertEqual(staged_path.stat().st_size, record["size_bytes"])
            self.assertEqual(hashlib.sha256(staged_path.read_bytes()).hexdigest(), record["sha256"])
        self.assertFalse(any(path.is_symlink() for path in package_dir.rglob("*")))
        self.assertTrue(PACKAGE.validate_package(package_dir))

    def test_init_rejects_existing_package_without_touching_it(self) -> None:
        target = self.root / "already-there"
        target.mkdir()
        sentinel = target / "keep.txt"
        sentinel.write_text("user data", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            self._init(target)

        self.assertEqual("user data", sentinel.read_text(encoding="utf-8"))

    def test_init_rejects_unknown_revision_and_missing_component_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "revision"):
            PACKAGE.initialize_package(
                package_dir=self.root / "bad-revision",
                model_dir=self.model_dir,
                gguf_path=self.gguf,
                denoiser_path=self.denoiser,
                revision="a" * 40,
            )

        (self.model_dir / "vae" / "diffusion_pytorch_model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "VAE|vae|weight"):
            self._init(self.root / "missing-component")

    def test_init_rejects_a_wrong_gguf_before_creating_the_package(self) -> None:
        wrong = self.root / "wrong.gguf"
        wrong.write_bytes(b"not the pinned mixed-quant model")
        target = self.root / "wrong-package"

        with self.assertRaisesRegex(ValueError, "GGUF|checksum|SHA"):
            PACKAGE.initialize_package(
                package_dir=target,
                model_dir=self.model_dir,
                gguf_path=wrong,
                denoiser_path=self.denoiser,
                revision=PACKAGE.MODEL_REVISION,
            )

        self.assertFalse(target.exists())

    def test_init_rejects_safetensors_that_disagree_with_available_hugging_face_sidecar(self) -> None:
        metadata = (
            self.model_dir
            / ".cache"
            / "huggingface"
            / "download"
            / "text_encoder"
            / "model-00001-of-00002.safetensors.metadata"
        )
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(f"{PACKAGE.MODEL_REVISION}\n{'0' * 64}\n", encoding="utf-8")
        target = self.root / "wrong-sidecar-package"

        with self.assertRaisesRegex(ValueError, "LFS SHA-256 sidecar"):
            self._init(target)

        self.assertFalse(target.exists())

    def test_init_rejects_model_file_symlink_that_escapes_source_root(self) -> None:
        outside = self.root / "outside-tokenizer-asset.json"
        outside.write_text("{}", encoding="utf-8")
        escaping = self.model_dir / "processor" / "outside.json"
        escaping.symlink_to(outside)
        target = self.root / "escaping-symlink-package"

        with self.assertRaisesRegex(ValueError, "symlink escapes"):
            self._init(target)

        self.assertFalse(target.exists())

    def test_empty_hugging_face_lock_files_are_excluded_from_package_inventory(self) -> None:
        lock = (
            self.model_dir
            / ".cache"
            / "huggingface"
            / "download"
            / "model_index.json.lock"
        )
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_bytes(b"")

        manifest_path = self._init()
        package_dir = manifest_path.parent
        model_lock = package_dir / "model" / ".cache" / "huggingface" / "download" / "model_index.json.lock"

        self.assertTrue(PACKAGE.validate_package(package_dir))
        self.assertFalse(model_lock.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertNotIn(
            ".cache/huggingface/download/model_index.json.lock",
            manifest["artifacts"]["model_files"],
        )

    def test_generate_runs_three_offline_stages_and_publishes_validated_png(self) -> None:
        package_dir = self._init().parent
        output = self.root / "result.png"
        calls: list[tuple[list[str], dict[str, str]]] = []

        def run(command: list[str], *, check: bool, env: dict[str, str]) -> None:
            self.assertTrue(check)
            calls.append((command, env))
            if command[0] == os.sys.executable and command[1].endswith("qwen_image21_prepare_conditioning.py"):
                output_dir = Path(command[command.index("--output-dir") + 1])
                output_dir.mkdir()
                conditioning = {
                    "schema": "qwen-image21-conditioning",
                    "schema_version": 1,
                    "model": {"repo": PACKAGE.MODEL_REPO, "revision": PACKAGE.MODEL_REVISION},
                    "prompt": "red cube",
                    "image": {"width": 64, "height": 64},
                    "noise": {"seed": 7},
                    "payload_file": "qwen_image21_conditioning.bin",
                    "payload_nbytes": 4,
                    "payload_sha256": hashlib.sha256(b"cond").hexdigest(),
                    "tensors": {
                        "initial_target_latents": {
                            "dtype": "float32-le",
                            "shape": [16, 64],
                            "nbytes": 16 * 64 * 4,
                        }
                    },
                }
                (output_dir / conditioning["payload_file"]).write_bytes(b"cond")
                (output_dir / "qwen_image21_conditioning.json").write_text(
                    json.dumps(conditioning), encoding="utf-8"
                )
            elif Path(command[0]).name == "qwen_image21_denoiser":
                latent_dir = Path(command[3])
                latent_dir.mkdir()
                condition = json.loads(Path(command[2]).read_text(encoding="utf-8"))
                payload = bytes(4 * 4 * 64 * 4)
                latent = {
                    "format": "qwen-image21-latents-v1",
                    "model_id": PACKAGE.MODEL_REPO,
                    "layout": "tokens_hwc",
                    "channels": 64,
                    "latent_height": 4,
                    "latent_width": 4,
                    "image_height": 64,
                    "image_width": 64,
                    "dtype": "float32-le",
                    "scaling": "diffusers_normalized",
                    "payload": "qwen_image21_latents.bin",
                    "payload_bytes": len(payload),
                    "model_revision": PACKAGE.MODEL_REVISION,
                    "seed": 7,
                    "prompt": condition["prompt"],
                    "denoising_steps": 40,
                    "dit_gguf": command[1],
                }
                (latent_dir / latent["payload"]).write_bytes(payload)
                (latent_dir / "qwen_image21_latents.json").write_text(
                    json.dumps(latent), encoding="utf-8"
                )
            elif command[1].endswith("qwen_image21_vae_decode.py"):
                image_path = Path(command[command.index("--output") + 1])
                image_path.write_bytes(_rgba_png(64, 64))

        with patch.object(PACKAGE.subprocess, "run", side_effect=run), patch.object(
            PACKAGE, "require_supported_runtime"
        ):
            PACKAGE.generate_image(
                package_dir=package_dir,
                prompt="red cube",
                width=64,
                height=64,
                seed=7,
                steps=40,
                output_path=output,
            )

        self.assertEqual(3, len(calls))
        self.assertIn("qwen_image21_prepare_conditioning.py", calls[0][0][1])
        self.assertEqual(package_dir / "bin" / "qwen_image21_denoiser", Path(calls[1][0][0]))
        self.assertIn("qwen_image21_vae_decode.py", calls[2][0][1])
        self.assertEqual("--device", calls[0][0][calls[0][0].index("--device")])
        self.assertEqual("cpu", calls[0][0][calls[0][0].index("--device") + 1])
        self.assertEqual("cpu", calls[2][0][calls[2][0].index("--device") + 1])
        self.assertEqual("float32", calls[2][0][calls[2][0].index("--dtype") + 1])
        for _, env in calls:
            self.assertEqual("1", env["HF_HUB_OFFLINE"])
            self.assertEqual("1", env["TRANSFORMERS_OFFLINE"])
            self.assertEqual("1", env["HF_DATASETS_OFFLINE"])
            self.assertEqual("1", env["DIFFUSERS_OFFLINE"])
        self.assertEqual(_rgba_png(64, 64), output.read_bytes())
        self.assertFalse(any(path.name.startswith("qwen-image21-run-") for path in self.root.iterdir()))

    def test_generate_does_not_trust_success_exit_codes_without_stage_bundles(self) -> None:
        package_dir = self._init().parent
        output = self.root / "false-success.png"

        with patch.object(PACKAGE.subprocess, "run"), patch.object(PACKAGE, "require_supported_runtime"):
            with self.assertRaisesRegex(RuntimeError, "without writing its manifest"):
                PACKAGE.generate_image(
                    package_dir=package_dir,
                    prompt="red cube",
                    width=64,
                    height=64,
                    seed=7,
                    steps=40,
                    output_path=output,
                )

        self.assertFalse(output.exists())

    def test_generate_rejects_package_root_swap_after_validation_before_any_stage(self) -> None:
        package_dir = self._init().parent
        output = self.root / "swapped-root.png"
        preserved_root = self.root / "validated-package"
        outside_root = self.root / "outside-package"
        outside_root.mkdir()
        (outside_root / "manifest.json").write_text("outside", encoding="utf-8")
        validate = PACKAGE.validate_package

        def validate_then_swap_root(path: str | Path) -> dict[str, object]:
            manifest = validate(path)
            Path(path).rename(preserved_root)
            Path(path).symlink_to(outside_root, target_is_directory=True)
            return manifest

        try:
            with patch.object(PACKAGE, "validate_package", side_effect=validate_then_swap_root), patch.object(
                PACKAGE.subprocess, "run"
            ) as run, patch.object(PACKAGE, "require_supported_runtime"):
                with self.assertRaisesRegex(RuntimeError, "package root.*changed|symlink"):
                    PACKAGE.generate_image(
                        package_dir=package_dir,
                        prompt="red cube",
                        width=64,
                        height=64,
                        seed=7,
                        steps=40,
                        output_path=output,
                    )
                run.assert_not_called()
        finally:
            if package_dir.is_symlink():
                package_dir.unlink()
            if preserved_root.exists():
                preserved_root.rename(package_dir)

        self.assertFalse(output.exists())

    def test_generate_failure_keeps_output_absent_and_existing_output_untouched(self) -> None:
        package_dir = self._init().parent
        output = self.root / "failed.png"

        with patch.object(PACKAGE.subprocess, "run", side_effect=RuntimeError("stage failed")), patch.object(
            PACKAGE, "require_supported_runtime"
        ):
            with self.assertRaisesRegex(RuntimeError, "stage failed"):
                PACKAGE.generate_image(
                    package_dir=package_dir,
                    prompt="red cube",
                    width=64,
                    height=64,
                    seed=7,
                    steps=40,
                    output_path=output,
                )
        self.assertFalse(output.exists())

        output.write_bytes(b"existing image")
        with patch.object(PACKAGE.subprocess, "run") as run, patch.object(PACKAGE, "require_supported_runtime"):
            with self.assertRaises(FileExistsError):
                PACKAGE.generate_image(
                    package_dir=package_dir,
                    prompt="red cube",
                    width=64,
                    height=64,
                    seed=7,
                    steps=40,
                    output_path=output,
                )
            run.assert_not_called()
        self.assertEqual(b"existing image", output.read_bytes())

    def test_generate_rejects_bad_manifest_path_and_changed_gguf_before_running_stages(self) -> None:
        package_dir = self._init().parent
        manifest_path = package_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["components"]["dit_gguf"] = "../outside.gguf"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with patch.object(PACKAGE.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "path|package|contained"):
                PACKAGE.generate_image(
                    package_dir=package_dir,
                    prompt="red cube",
                    width=64,
                    height=64,
                    seed=7,
                    steps=40,
                    output_path=self.root / "bad-path.png",
                )
            run.assert_not_called()

        manifest["components"]["dit_gguf"] = "weights/Qwen-Image-2.1-Q4.gguf"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (package_dir / manifest["components"]["dit_gguf"]).write_bytes(b"tampered")
        with patch.object(PACKAGE.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "GGUF|checksum|SHA"):
                PACKAGE.generate_image(
                    package_dir=package_dir,
                    prompt="red cube",
                    width=64,
                    height=64,
                    seed=7,
                    steps=40,
                    output_path=self.root / "tampered.png",
                )
            run.assert_not_called()

    def test_generate_rejects_changed_model_weights_or_packaged_scripts_before_running(self) -> None:
        package_dir = self._init().parent
        manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
        model_shard = package_dir / "model" / "text_encoder" / "model-00001-of-00002.safetensors"
        # Replace hard links to avoid changing the test fixture source inode.
        model_shard.unlink()
        model_shard.write_bytes(b"modified model shard")
        with patch.object(PACKAGE.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "model.*checksum|checksum.*model"):
                PACKAGE.generate_image(
                    package_dir=package_dir,
                    prompt="red cube",
                    width=64,
                    height=64,
                    seed=7,
                    steps=40,
                    output_path=self.root / "changed-model.png",
                )
            run.assert_not_called()

        # Restore the source model and package from a fresh init, then tamper with a script.
        package_dir = self._init(self.root / "script-package").parent
        manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
        script = package_dir / manifest["components"]["conditioning_script"]
        script.write_text("# changed", encoding="utf-8")
        with patch.object(PACKAGE.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "script.*checksum|checksum.*script"):
                PACKAGE.generate_image(
                    package_dir=package_dir,
                    prompt="red cube",
                    width=64,
                    height=64,
                    seed=7,
                    steps=40,
                    output_path=self.root / "changed-script.png",
                )
            run.assert_not_called()

    def test_packaged_launcher_rejects_init_with_repository_source_requirement(self) -> None:
        package_dir = self._init().parent
        packaged_launcher = package_dir / "bin" / "qwen_image21_package.py"
        spec = importlib.util.spec_from_file_location("qwen_image21_packaged_launcher", packaged_launcher)
        assert spec is not None and spec.loader is not None
        packaged = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(packaged)

        with self.assertRaisesRegex(ValueError, "repository launcher.*init|init.*repository launcher"):
            packaged.initialize_package(
                package_dir=self.root / "nested-init",
                model_dir=self.model_dir,
                gguf_path=self.gguf,
                denoiser_path=self.denoiser,
                revision=PACKAGE.MODEL_REVISION,
            )

if __name__ == "__main__":
    unittest.main()
