#!/usr/bin/env python3
"""CPU-only falsifiers for the opt-in conditioning batch boundary."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen_image21_prepare_conditioning.py"
SPEC = importlib.util.spec_from_file_location("qwen_image21_prepare_conditioning_batch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

PINNED_MODEL_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"


class FakeGenerator:
    def __init__(self, device: str) -> None:
        self.device = device
        self.seed = 0

    def manual_seed(self, seed: int) -> "FakeGenerator":
        self.seed = seed
        return self


class FakeTorch:
    __version__ = "stub"

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()

    @staticmethod
    def device(name: str) -> str:
        return name

    @staticmethod
    def Generator(device: str) -> FakeGenerator:
        return FakeGenerator(device)


class FakePipeline:
    def __init__(self) -> None:
        self.prompt_calls: list[tuple[str, object, object]] = []
        self.latent_calls: list[tuple[int, int, int, int]] = []

    def _get_qwen_prompt_embeds(self, prompt, image=None, device=None):
        self.prompt_calls.append((prompt, image, device))
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        value = int.from_bytes(prompt_digest[:4], "little") / (2**32)
        embeddings = np.full((1, 3, MODULE.CONTEXT_DIM), value, dtype=np.float32)
        attention_mask = np.ones((1, 3), dtype=np.bool_)
        image_mask = np.zeros((1, 3), dtype=np.bool_)
        return embeddings, attention_mask, image_mask

    def prepare_latents(
        self,
        images,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        self.latent_calls.append((height, width, generator.seed, id(generator)))
        latent_height = height // MODULE.VAE_SCALE_FACTOR
        latent_width = width // MODULE.VAE_SCALE_FACTOR
        shape = (batch_size, latent_height * latent_width, num_channels_latents)
        target = np.random.default_rng(generator.seed).standard_normal(shape, dtype=np.float32)
        return target, None


def read_tensor(manifest_path: Path, name: str) -> np.ndarray:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptor = manifest["tensors"][name]
    payload = (manifest_path.parent / manifest["payload_file"]).read_bytes()
    start = descriptor["offset_bytes"]
    end = start + descriptor["nbytes"]
    dtype = {"float32-le": "<f4", "uint8": "u1"}[descriptor["dtype"]]
    return np.frombuffer(payload[start:end], dtype=dtype).reshape(descriptor["shape"])


class QwenImage21ConditioningBatchSpec(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="qwen-image21-conditioning-batch-spec-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model_dir = self.root / "model"
        (self.model_dir / "processor").mkdir(parents=True)
        (self.model_dir / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": MODULE.PIPELINE_CLASS,
                    "text_encoder": ["transformers", MODULE.TEXT_ENCODER_CLASS],
                }
            ),
            encoding="utf-8",
        )
        (self.model_dir / "text_encoder").mkdir()
        (self.model_dir / "text_encoder" / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_vl",
                    "architectures": [MODULE.TEXT_ENCODER_CLASS],
                    "text_config": {"hidden_size": MODULE.CONTEXT_DIM},
                }
            ),
            encoding="utf-8",
        )
        self.pipeline = FakePipeline()
        self.diffusers = SimpleNamespace(__version__="stub")
        self.output_root = self.root / "conditioning"

    def test_batch_loads_and_encodes_once_then_writes_one_seeded_v1_bundle_per_seed(self) -> None:
        loader = mock.Mock(return_value=(FakeTorch, self.diffusers, self.pipeline, "cpu"))
        generators: list[FakeGenerator] = []
        original_generator_factory = FakeTorch.Generator

        def make_generator(device: str) -> FakeGenerator:
            generator = original_generator_factory(device)
            generators.append(generator)
            return generator

        generator_factory = mock.Mock(side_effect=make_generator)
        with mock.patch.object(MODULE, "_load_local_pipeline", loader), mock.patch.object(
            MODULE, "_installed_diffusers_commit", return_value=None
        ), mock.patch.object(FakeTorch, "Generator", generator_factory):
            manifests = MODULE.prepare_conditioning_batch(
                model_dir=self.model_dir,
                output_dir=self.output_root,
                prompt="A glass observatory above a pine forest",
                width=64,
                height=32,
                seeds=[7, 8, 11],
                revision=PINNED_MODEL_REVISION,
                device="cpu",
            )

        loader.assert_called_once()
        self.assertEqual(
            [("A glass observatory above a pine forest", None, "cpu")],
            self.pipeline.prompt_calls,
        )
        self.assertEqual([mock.call(device="cpu")] * 3, generator_factory.call_args_list)
        self.assertEqual(3, len({id(generator) for generator in generators}))
        self.assertEqual(
            [(32, 64, 7), (32, 64, 8), (32, 64, 11)],
            [call[:3] for call in self.pipeline.latent_calls],
        )
        self.assertEqual(
            [self.output_root / f"seed-{seed}" / MODULE.MANIFEST_NAME for seed in (7, 8, 11)],
            manifests,
        )

        hidden: list[np.ndarray] = []
        noise: list[np.ndarray] = []
        for seed, manifest_path in zip((7, 8, 11), manifests, strict=True):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("qwen-image21-conditioning", manifest["schema"])
            self.assertEqual(1, manifest["schema_version"])
            self.assertEqual(PINNED_MODEL_REVISION, manifest["model"]["revision"])
            self.assertEqual("A glass observatory above a pine forest", manifest["prompt"])
            self.assertEqual(
                {
                    "width": 64,
                    "height": 32,
                    "vae_scale_factor": 16,
                    "latent_height": 2,
                    "latent_width": 4,
                    "img_shapes": [[1, 2, 4]],
                },
                manifest["image"],
            )
            self.assertEqual(seed, manifest["noise"]["seed"])
            payload_path = manifest_path.parent / MODULE.PAYLOAD_NAME
            payload = payload_path.read_bytes()
            self.assertEqual(manifest["payload_nbytes"], len(payload))
            self.assertEqual(manifest["payload_sha256"], hashlib.sha256(payload).hexdigest())
            hidden.append(read_tensor(manifest_path, "encoder_hidden_states"))
            self.assertTrue(
                np.array_equal(
                    read_tensor(manifest_path, "encoder_hidden_states_mask"),
                    np.ones(3, dtype=np.uint8),
                )
            )
            self.assertTrue(
                np.array_equal(
                    read_tensor(manifest_path, "encoder_img_mask"),
                    np.zeros(3, dtype=np.uint8),
                )
            )
            target_latents = read_tensor(manifest_path, "initial_target_latents")
            self.assertEqual((8, 64), target_latents.shape)
            noise.append(target_latents)

        self.assertTrue(all(np.array_equal(hidden[0], value) for value in hidden[1:]))
        self.assertTrue(all(not np.array_equal(noise[0], value) for value in noise[1:]))

    def test_batch_bundle_matches_unchanged_single_request_bytes_for_same_seed(self) -> None:
        single_pipeline = FakePipeline()
        batch_pipeline = FakePipeline()
        single_dir = self.root / "single"
        batch_dir = self.root / "batch"
        fake_diffusers = self.diffusers

        def loader_for(pipeline: FakePipeline):
            return mock.Mock(return_value=(FakeTorch, fake_diffusers, pipeline, "cpu"))

        with mock.patch.object(MODULE, "_load_local_pipeline", loader_for(single_pipeline)), mock.patch.object(
            MODULE, "_installed_diffusers_commit", return_value=None
        ):
            single_manifest = MODULE.prepare_conditioning(
                model_dir=self.model_dir,
                output_dir=single_dir,
                prompt="A glass observatory above a pine forest",
                width=64,
                height=32,
                seed=8,
                revision=PINNED_MODEL_REVISION,
                device="cpu",
            )

        with mock.patch.object(MODULE, "_load_local_pipeline", loader_for(batch_pipeline)), mock.patch.object(
            MODULE, "_installed_diffusers_commit", return_value=None
        ):
            batch_manifests = MODULE.prepare_conditioning_batch(
                model_dir=self.model_dir,
                output_dir=batch_dir,
                prompt="A glass observatory above a pine forest",
                width=64,
                height=32,
                seeds=[7, 8, 11],
                revision=PINNED_MODEL_REVISION,
                device="cpu",
            )

        batch_manifest = batch_manifests[1]
        self.assertEqual(single_manifest.read_bytes(), batch_manifest.read_bytes())
        self.assertEqual(
            (single_dir / MODULE.PAYLOAD_NAME).read_bytes(),
            (batch_manifest.parent / MODULE.PAYLOAD_NAME).read_bytes(),
        )

    def test_duplicate_or_invalid_seeds_fail_before_model_load(self) -> None:
        invalid_seed_sets = (
            [],
            [7, 7],
            [-1],
            [2**63],
            [True],
            ["7"],
        )
        for seeds in invalid_seed_sets:
            with self.subTest(seeds=seeds):
                loader = mock.Mock()
                with mock.patch.object(MODULE, "_load_local_pipeline", loader):
                    with self.assertRaises(ValueError):
                        MODULE.prepare_conditioning_batch(
                            model_dir=self.model_dir,
                            output_dir=self.output_root,
                            prompt="A glass observatory",
                            seeds=seeds,
                            revision=PINNED_MODEL_REVISION,
                            device="cpu",
                        )
                loader.assert_not_called()
                self.assertFalse(self.output_root.exists())

    def test_existing_seed_directory_fails_before_model_load_and_is_preserved(self) -> None:
        collision = self.output_root / "seed-8"
        collision.mkdir(parents=True)
        sentinel = collision / "keep.txt"
        sentinel.write_text("user data", encoding="utf-8")
        loader = mock.Mock()

        with mock.patch.object(MODULE, "_load_local_pipeline", loader):
            with self.assertRaises(FileExistsError):
                MODULE.prepare_conditioning_batch(
                    model_dir=self.model_dir,
                    output_dir=self.output_root,
                    prompt="A glass observatory",
                    seeds=[7, 8, 11],
                    revision=PINNED_MODEL_REVISION,
                    device="cpu",
                )

        loader.assert_not_called()
        self.assertEqual("user data", sentinel.read_text(encoding="utf-8"))
        self.assertFalse((self.output_root / "seed-7").exists())
        self.assertFalse((self.output_root / "seed-11").exists())

    def test_seed_symlink_collision_fails_before_model_load(self) -> None:
        self.output_root.mkdir()
        target = self.root / "outside"
        target.mkdir()
        sentinel = target / "keep.txt"
        sentinel.write_text("user data", encoding="utf-8")
        collision = self.output_root / "seed-8"
        collision.symlink_to(target, target_is_directory=True)
        loader = mock.Mock()

        with mock.patch.object(MODULE, "_load_local_pipeline", loader):
            with self.assertRaises(FileExistsError):
                MODULE.prepare_conditioning_batch(
                    model_dir=self.model_dir,
                    output_dir=self.output_root,
                    prompt="A glass observatory",
                    seeds=[7, 8, 11],
                    revision=PINNED_MODEL_REVISION,
                    device="cpu",
                )

        loader.assert_not_called()
        self.assertTrue(collision.is_symlink())
        self.assertEqual("user data", sentinel.read_text(encoding="utf-8"))
        self.assertFalse((self.output_root / "seed-7").exists())
        self.assertFalse((self.output_root / "seed-11").exists())

    def test_mid_batch_failure_cleans_owned_bundles_but_preserves_root_sentinel(self) -> None:
        self.output_root.mkdir()
        sentinel = self.output_root / "keep.txt"
        sentinel.write_text("user data", encoding="utf-8")
        pipeline = FakePipeline()
        prepare_latents = pipeline.prepare_latents

        def fail_on_second_seed(*args, **kwargs):
            generator = args[7]
            if generator.seed == 8:
                raise ValueError("synthetic seed-8 failure")
            return prepare_latents(*args, **kwargs)

        pipeline.prepare_latents = fail_on_second_seed
        loader = mock.Mock(return_value=(FakeTorch, self.diffusers, pipeline, "cpu"))

        with mock.patch.object(MODULE, "_load_local_pipeline", loader), mock.patch.object(
            MODULE, "_installed_diffusers_commit", return_value=None
        ):
            with self.assertRaisesRegex(RuntimeError, "seed 8"):
                MODULE.prepare_conditioning_batch(
                    model_dir=self.model_dir,
                    output_dir=self.output_root,
                    prompt="A glass observatory",
                    width=64,
                    height=32,
                    seeds=[7, 8, 11],
                    revision=PINNED_MODEL_REVISION,
                    device="cpu",
                )

        loader.assert_called_once()
        self.assertEqual("user data", sentinel.read_text(encoding="utf-8"))
        for seed in (7, 8, 11):
            self.assertFalse((self.output_root / f"seed-{seed}").exists())
        self.assertEqual(["keep.txt"], [path.name for path in self.output_root.iterdir()])

    def test_rollback_preserves_unowned_file_in_reserved_seed_directory(self) -> None:
        self.output_root.mkdir()
        pipeline = FakePipeline()
        prepare_latents = pipeline.prepare_latents

        def fail_on_second_seed(*args, **kwargs):
            generator = args[7]
            if generator.seed == 8:
                raise ValueError("synthetic seed-8 failure")
            return prepare_latents(*args, **kwargs)

        pipeline.prepare_latents = fail_on_second_seed
        loader = mock.Mock(return_value=(FakeTorch, self.diffusers, pipeline, "cpu"))
        write_bundle = MODULE.write_conditioning_bundle

        def write_bundle_then_inject_unowned_file(**kwargs):
            manifest = write_bundle(**kwargs)
            (manifest.parent / "foreign.txt").write_text("leave me", encoding="utf-8")
            return manifest

        with mock.patch.object(MODULE, "_load_local_pipeline", loader), mock.patch.object(
            MODULE, "_installed_diffusers_commit", return_value=None
        ), mock.patch.object(
            MODULE, "write_conditioning_bundle", side_effect=write_bundle_then_inject_unowned_file
        ):
            with self.assertRaisesRegex(RuntimeError, "seed 8"):
                MODULE.prepare_conditioning_batch(
                    model_dir=self.model_dir,
                    output_dir=self.output_root,
                    prompt="A glass observatory",
                    width=64,
                    height=32,
                    seeds=[7, 8, 11],
                    revision=PINNED_MODEL_REVISION,
                    device="cpu",
                )

        loader.assert_called_once()
        seed_seven_dir = self.output_root / "seed-7"
        self.assertEqual(["foreign.txt"], [path.name for path in seed_seven_dir.iterdir()])
        self.assertEqual("leave me", (seed_seven_dir / "foreign.txt").read_text(encoding="utf-8"))
        self.assertFalse((self.output_root / "seed-8").exists())
        self.assertFalse((self.output_root / "seed-11").exists())

    def test_batch_does_not_adopt_files_inserted_after_collision_check(self) -> None:
        for colliding_name in (MODULE.PAYLOAD_NAME, MODULE.MANIFEST_NAME):
            with self.subTest(file=colliding_name):
                output_root = self.root / f"collision-{colliding_name.rsplit('.', 1)[-1]}"
                pipeline = FakePipeline()
                loader = mock.Mock(return_value=(FakeTorch, self.diffusers, pipeline, "cpu"))
                collision_path = output_root / "seed-7" / colliding_name
                original_exists = Path.exists
                original_write_bytes = Path.write_bytes
                injected: list[Path] = []

                def exists_then_inject(path: Path) -> bool:
                    existed = original_exists(path)
                    if path == collision_path and not existed and not injected:
                        original_write_bytes(path, b"created by concurrent writer")
                        injected.append(path)
                    return existed

                with mock.patch.object(MODULE, "_load_local_pipeline", loader), mock.patch.object(
                    MODULE, "_installed_diffusers_commit", return_value=None
                ), mock.patch.object(Path, "exists", exists_then_inject):
                    with self.assertRaises(FileExistsError):
                        MODULE.prepare_conditioning_batch(
                            model_dir=self.model_dir,
                            output_dir=output_root,
                            prompt="A glass observatory",
                            width=64,
                            height=32,
                            seeds=[7, 8, 11],
                            revision=PINNED_MODEL_REVISION,
                            device="cpu",
                        )

                self.assertEqual([collision_path], injected)
                self.assertEqual(b"created by concurrent writer", collision_path.read_bytes())
                self.assertEqual([colliding_name], [path.name for path in collision_path.parent.iterdir()])
                self.assertFalse((output_root / "seed-8").exists())
                self.assertFalse((output_root / "seed-11").exists())

    def test_partial_exclusive_write_is_cleaned_without_touching_root_sentinel(self) -> None:
        self.output_root.mkdir()
        sentinel = self.output_root / "keep.txt"
        sentinel.write_text("user data", encoding="utf-8")
        pipeline = FakePipeline()
        loader = mock.Mock(return_value=(FakeTorch, self.diffusers, pipeline, "cpu"))
        original_write = os.write
        write_calls: list[int] = []

        def fail_after_prefix(fd: int, data) -> int:
            write_calls.append(fd)
            if len(write_calls) == 1:
                return original_write(fd, data[:19])
            raise OSError("synthetic partial write failure")

        with mock.patch.object(MODULE, "_load_local_pipeline", loader), mock.patch.object(
            MODULE, "_installed_diffusers_commit", return_value=None
        ), mock.patch.object(os, "write", fail_after_prefix):
            with self.assertRaisesRegex(OSError, "synthetic partial write"):
                MODULE.prepare_conditioning_batch(
                    model_dir=self.model_dir,
                    output_dir=self.output_root,
                    prompt="A glass observatory",
                    width=64,
                    height=32,
                    seeds=[7, 8, 11],
                    revision=PINNED_MODEL_REVISION,
                    device="cpu",
                )

        self.assertEqual(2, len(write_calls))
        self.assertEqual("user data", sentinel.read_text(encoding="utf-8"))
        for seed in (7, 8, 11):
            self.assertFalse((self.output_root / f"seed-{seed}").exists())
        self.assertEqual(["keep.txt"], [path.name for path in self.output_root.iterdir()])

    def test_partial_manifest_write_cleans_full_payload_and_owned_seed_dirs(self) -> None:
        self.output_root.mkdir()
        sentinel = self.output_root / "keep.txt"
        sentinel.write_text("user data", encoding="utf-8")
        loader = mock.Mock(return_value=(FakeTorch, self.diffusers, self.pipeline, "cpu"))
        original_write = os.write
        write_events: list[str] = []

        def finish_payload_then_fail_manifest(fd: int, data) -> int:
            if not write_events:
                view = memoryview(data)
                written = 0
                while written < len(view):
                    count = original_write(fd, view[written:])
                    if count <= 0:
                        raise OSError("synthetic payload write made no progress")
                    written += count
                write_events.append("payload-complete")
                return written
            if len(write_events) == 1:
                write_events.append("manifest-prefix")
                return original_write(fd, data[:19])
            write_events.append("manifest-failure")
            raise OSError("synthetic partial manifest write failure")

        with mock.patch.object(MODULE, "_load_local_pipeline", loader), mock.patch.object(
            MODULE, "_installed_diffusers_commit", return_value=None
        ), mock.patch.object(os, "write", finish_payload_then_fail_manifest):
            with self.assertRaisesRegex(OSError, "partial manifest write"):
                MODULE.prepare_conditioning_batch(
                    model_dir=self.model_dir,
                    output_dir=self.output_root,
                    prompt="A glass observatory",
                    width=64,
                    height=32,
                    seeds=[7, 8, 11],
                    revision=PINNED_MODEL_REVISION,
                    device="cpu",
                )

        loader.assert_called_once()
        self.assertEqual(
            ["payload-complete", "manifest-prefix", "manifest-failure"], write_events
        )
        self.assertEqual("user data", sentinel.read_text(encoding="utf-8"))
        for seed in (7, 8, 11):
            self.assertFalse((self.output_root / f"seed-{seed}").exists())
        self.assertEqual(["keep.txt"], [path.name for path in self.output_root.iterdir()])

    def test_cli_keeps_default_and_single_seed_on_the_existing_single_request_api(self) -> None:
        single = mock.Mock(return_value=self.root / "one.json")
        batch = mock.Mock(return_value=[self.root / "seed-7.json"])
        common = [
            "--model-dir", str(self.model_dir),
            "--output-dir", str(self.output_root),
            "--prompt", "A glass observatory",
            "--width", "64",
            "--height", "32",
        ]

        with mock.patch.object(MODULE, "prepare_conditioning", single), mock.patch.object(
            MODULE, "prepare_conditioning_batch", batch
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(0, MODULE.main(common))
            single.assert_called_once()
            self.assertEqual(0, single.call_args.kwargs["seed"])
            batch.assert_not_called()

            single.reset_mock()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(0, MODULE.main(common + ["--seed", "19"]))
            single.assert_called_once()
            self.assertEqual(19, single.call_args.kwargs["seed"])
            batch.assert_not_called()

            single.reset_mock()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(0, MODULE.main(common + ["--seeds", "7", "8"]))
            single.assert_not_called()
            self.assertEqual([7, 8], batch.call_args.kwargs["seeds"])


if __name__ == "__main__":
    unittest.main()
