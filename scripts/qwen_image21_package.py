#!/usr/bin/env python3
"""Package and run the local, hybrid Qwen-Image 2.1 text-to-image stages.

Initialization packages already-present model components, the pinned mixed-quant
GGUF, a macOS arm64 Metal denoiser, and the two small Python bridge stages. It
does not download model files. Large model files and the GGUF are hard-linked,
so in-place changes through either hard link affect both paths until the package
is copied to another filesystem.

The generated directory is a local-use package, not a redistribution bundle or
standalone executable. It requires an external Python environment with Torch,
Transformers, Diffusers, Pillow, and NumPy, plus the denoiser's Crystal/Homebrew
dynamic libraries. Its runtime is intentionally platform-specific and hybrid:
CPU Qwen3-VL prompt conditioning, Crystal's native Metal GGUF denoiser, and CPU
Diffusers VAE decode. The manifest hashes detect package self-consistency and
later drift; they are not a signature or independent authenticity guarantee.
Root identity checks reduce package-path replacement between validation and
stage launches, but do not eliminate races with malicious concurrent mutation.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "qwen-image21-package"
SCHEMA_VERSION = 1
MODEL_REPO = "Qwen/Qwen-Image-2.1"
MODEL_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
PIPELINE_CLASS = "QwenImage21Pipeline"
GGUF_REPO = "realrebelai/Qwen-Image-2.1_GGUFs"
GGUF_REVISION = "8d393b750593a72ee040fe7d40611f479ccee679"
GGUF_FILENAME = "Qwen-Image-2.1-Q4.gguf"
GGUF_SIZE_BYTES = 5_959_127_264
GGUF_SHA256 = "51998ad7c068ce7d68e233237537900ffe874ab4d5c72e20758f5f18ceb15b8a"
GGUF_TENSOR_POLICY = "BF16:8,F32:65,Q8_0:96,Q6_K:64,Q5_K:32"
TARGET_RUNTIME = "darwin-arm64-metal"
REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
MAX_IMAGE_SIDE = 4096

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONDITIONER = REPO_ROOT / "scripts" / "qwen_image21_prepare_conditioning.py"
SOURCE_DECODER = REPO_ROOT / "scripts" / "qwen_image21_vae_decode.py"
SOURCE_DENOISER = REPO_ROOT / "scripts" / "qwen_image21_generate_latents.cr"
SOURCE_LAUNCHER = Path(__file__).resolve()

MANIFEST_NAME = "manifest.json"
COMPONENT_PATHS = {
    "model_dir": "model",
    "dit_gguf": f"weights/{GGUF_FILENAME}",
    "denoiser": "bin/qwen_image21_denoiser",
    "conditioning_script": "bin/qwen_image21_prepare_conditioning.py",
    "decoder_script": "bin/qwen_image21_vae_decode.py",
    "launcher_script": "bin/qwen_image21_package.py",
    "denoiser_entrypoint": "src/qwen_image21_generate_latents.cr",
}
OFFLINE_ENVIRONMENT = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "DIFFUSERS_OFFLINE",
)


class _ValidatedPackageManifest(dict[str, Any]):
    """Manifest mapping plus the root identity verified with its contents."""

    def __init__(
        self,
        manifest: dict[str, Any],
        *,
        requested_root: Path,
        package_root: Path,
        root_identity: tuple[int, int],
    ) -> None:
        super().__init__(manifest)
        self._requested_root = requested_root
        self._package_root = package_root
        self._root_identity = root_identity


def _fail(message: str) -> ValueError:
    return ValueError(message)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _fail(f"cannot read {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise _fail(f"{description} must be a JSON object")
    return value


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise _fail(f"cannot checksum {path}: {exc}") from exc
    return digest.hexdigest()


def _record_file(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise _fail(f"cannot stat package artifact {path}: {exc}") from exc
    return {"size_bytes": size, "sha256": _sha256(path)}


def _include_model_file(relative_path: Path) -> bool:
    """Keep only HF download metadata under `.cache`; discard transient cache state."""
    parts = relative_path.parts
    if not parts or parts[0] != ".cache":
        return True
    return (
        len(parts) >= 4
        and parts[:3] == (".cache", "huggingface", "download")
        and parts[-1].endswith(".metadata")
    )


def _include_model_directory(relative_path: Path) -> bool:
    """Allow only ancestors and descendants of the retained HF metadata tree."""
    parts = relative_path.parts
    if not parts or parts[0] != ".cache":
        return True
    retained_root = (".cache", "huggingface", "download")
    return parts == retained_root[: len(parts)] or parts[: len(retained_root)] == retained_root


def _record_model_files(
    model_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    checked_sidecars: dict[str, dict[str, str]] = {}
    unavailable_sidecars: list[str] = []
    metadata_root = model_dir / ".cache" / "huggingface" / "download"
    for current, directory_names, file_names in os.walk(model_dir, followlinks=False):
        current_path = Path(current)
        for directory_name in list(directory_names):
            source_dir = current_path / directory_name
            if source_dir.is_symlink():
                raise _fail(f"model directory symlink is not allowed: {source_dir}")
            if not _include_model_directory(source_dir.relative_to(model_dir)):
                directory_names.remove(directory_name)
        for file_name in file_names:
            source_file = current_path / file_name
            if source_file.is_symlink():
                resolved = source_file.resolve(strict=True)
                if not _within(resolved, model_dir.resolve(strict=True)):
                    raise _fail(f"model component symlink escapes --model-dir: {source_file}")
            if not _include_model_file(source_file.relative_to(model_dir)):
                continue
            if not source_file.is_file():
                raise _fail(f"model package contains a non-regular file: {source_file}")
            relative = source_file.relative_to(model_dir).as_posix()
            record = _record_file(source_file)
            records[relative] = record
            if source_file.suffix == ".safetensors":
                sidecar = metadata_root / f"{relative}.metadata"
                if not sidecar.is_file():
                    unavailable_sidecars.append(relative)
                    continue
                try:
                    lines = sidecar.read_text(encoding="utf-8").splitlines()
                except OSError as exc:
                    raise _fail(f"cannot read model weight metadata {sidecar}: {exc}") from exc
                if (
                    len(lines) < 2
                    or not REVISION_RE.fullmatch(lines[0])
                    or lines[0].lower() != MODEL_REVISION
                    or not re.fullmatch(r"[0-9a-fA-F]{64}", lines[1])
                ):
                    raise _fail(f"invalid pinned revision/LFS SHA-256 metadata for {relative}")
                etag_sha256 = lines[1].lower()
                if record["sha256"] != etag_sha256:
                    raise _fail(
                        f"model weight does not match its Hugging Face LFS SHA-256 sidecar: {relative}"
                    )
                checked_sidecars[relative] = {"revision": lines[0].lower(), "sha256": etag_sha256}
    if not records:
        raise _fail("local Qwen-Image 2.1 model directory contains no files")
    return dict(sorted(records.items())), {
        "mechanism": "Hugging Face cache revision and LFS SHA-256 sidecars, when present",
        "checked_safetensors": dict(sorted(checked_sidecars.items())),
        "unavailable_safetensors": sorted(unavailable_sidecars),
        "limitation": (
            "Sidecars corroborate only the listed files and are not a cryptographic signature; "
            "all package model files are separately SHA-256-pinned at initialization."
        ),
    }


def _check_revision(revision: str) -> str:
    if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
        raise _fail("model revision must be a full 40-character commit SHA")
    normalized = revision.lower()
    if normalized != MODEL_REVISION:
        raise _fail(f"unknown Qwen-Image 2.1 model revision: expected {MODEL_REVISION}")
    return normalized


def _metadata_revision(model_dir: Path) -> str | None:
    relative_paths = (
        "model_index.json",
        "processor/tokenizer.json",
        "text_encoder/config.json",
    )
    found: set[str] = set()
    metadata_root = model_dir / ".cache" / "huggingface" / "download"
    for relative in relative_paths:
        metadata = metadata_root / f"{relative}.metadata"
        if not metadata.exists():
            continue
        try:
            lines = metadata.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise _fail(f"cannot read local model revision metadata {metadata}: {exc}") from exc
        if not lines or not REVISION_RE.fullmatch(lines[0]):
            raise _fail(f"invalid local model revision metadata at {metadata}")
        found.add(lines[0].lower())
    if len(found) > 1:
        raise _fail(f"local model files come from different revisions: {sorted(found)}")
    return next(iter(found), None)


def _require_regular_file(path: Path, description: str) -> Path:
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            raise _fail(f"missing or empty {description}: {path}")
    except OSError as exc:
        raise _fail(f"cannot inspect {description} {path}: {exc}") from exc
    return path


def _validate_model_directory(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise _fail(f"local Qwen-Image 2.1 components directory does not exist: {model_dir}")

    index_path = _require_regular_file(model_dir / "model_index.json", "model_index.json")
    index = _read_json(index_path, "model_index.json")
    expected_index = {
        "_class_name": PIPELINE_CLASS,
        "processor": ["transformers", "Qwen3VLProcessor"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
        # The official config identifies this component, while the package uses
        # its inspected GGUF counterpart instead of transformer/ weights.
        "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLQwenImage21"],
    }
    for key, value in expected_index.items():
        if index.get(key) != value:
            raise _fail(f"model_index.json {key!r} must identify the pinned Qwen-Image 2.1 pipeline")

    processor_dir = model_dir / "processor"
    if not processor_dir.is_dir():
        raise _fail("missing local Qwen3-VL processor directory")
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "chat_template.jinja",
    ):
        _require_regular_file(processor_dir / name, f"processor/{name}")

    encoder_dir = model_dir / "text_encoder"
    encoder_config = _read_json(
        _require_regular_file(encoder_dir / "config.json", "text_encoder/config.json"),
        "text_encoder/config.json",
    )
    text_config = encoder_config.get("text_config")
    if (
        encoder_config.get("model_type") != "qwen3_vl"
        or encoder_config.get("architectures") != ["Qwen3VLForConditionalGeneration"]
        or not isinstance(text_config, dict)
        or text_config.get("hidden_size") != 4096
    ):
        raise _fail("text_encoder/config.json does not identify the expected Qwen3-VL model")
    weights_index = _read_json(
        _require_regular_file(
            encoder_dir / "model.safetensors.index.json",
            "text_encoder/model.safetensors.index.json",
        ),
        "text_encoder/model.safetensors.index.json",
    )
    weight_map = weights_index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise _fail("text_encoder weight index must contain a non-empty weight_map")
    shard_names = set(weight_map.values())
    for shard_name in shard_names:
        if not isinstance(shard_name, str):
            raise _fail("text_encoder weight index contains an invalid shard path")
        shard_relative = PurePosixPath(shard_name)
        if shard_relative.is_absolute() or ".." in shard_relative.parts or "\\" in shard_name:
            raise _fail("text_encoder weight index contains a path outside text_encoder/")
        shard_path = encoder_dir.joinpath(*shard_relative.parts)
        if not _within(shard_path.resolve(), encoder_dir.resolve()):
            raise _fail("text_encoder weight index contains a path outside text_encoder/")
        _require_regular_file(shard_path, f"text_encoder weight shard {shard_name}")

    scheduler = _read_json(
        _require_regular_file(model_dir / "scheduler" / "scheduler_config.json", "scheduler/scheduler_config.json"),
        "scheduler/scheduler_config.json",
    )
    if not scheduler:
        raise _fail("scheduler/scheduler_config.json must not be empty")

    vae_config = _read_json(
        _require_regular_file(model_dir / "vae" / "config.json", "vae/config.json"),
        "vae/config.json",
    )
    if (
        vae_config.get("_class_name") != "AutoencoderKLQwenImage21"
        or vae_config.get("z_dim") != 64
        or vae_config.get("out_channels") != 4
        or vae_config.get("scale_factor_spatial") != 16
    ):
        raise _fail("vae/config.json does not identify the expected Qwen-Image 2.1 VAE")
    vae_weights = (
        list((model_dir / "vae").glob("*.safetensors"))
        + list((model_dir / "vae").glob("*.bin"))
        + list((model_dir / "vae").glob("*.pt"))
    )
    if not vae_weights or not any(path.is_file() and path.stat().st_size > 0 for path in vae_weights):
        raise _fail("missing local Qwen-Image 2.1 VAE weight files")

    cached_revision = _metadata_revision(model_dir)
    if cached_revision is not None and cached_revision != MODEL_REVISION:
        raise _fail(
            f"local model cache revision {cached_revision} differs from pinned revision {MODEL_REVISION}"
        )


def _validate_macho_arm64(path: Path) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise _fail(f"native denoiser must be an executable file: {path}")
    try:
        with path.open("rb") as stream:
            header = stream.read(8)
    except OSError as exc:
        raise _fail(f"cannot read native denoiser header: {exc}") from exc
    if len(header) < 8 or header[:4] != b"\xcf\xfa\xed\xfe":
        raise _fail("native denoiser must be a 64-bit little-endian Mach-O executable")
    if struct.unpack("<I", header[4:8])[0] != 0x0100000C:
        raise _fail("native denoiser must target Apple arm64")


def require_supported_runtime() -> None:
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise RuntimeError(
            "this package requires macOS on Apple arm64 with the native Metal runtime"
        )


def _source_path(value: str | Path, description: str) -> Path:
    path = Path(value).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise _fail(f"{description} does not exist: {path}") from exc
    return resolved


def _paths_overlap(left: Path, right: Path) -> bool:
    return _within(left, right) or _within(right, left)


def _hardlink_file(source: Path, destination: Path, *, model_root: Path | None = None) -> None:
    source_file = source.resolve(strict=True)
    if not source_file.is_file():
        raise _fail(f"hard-link source must be a regular file: {source}")
    if model_root is not None and not _within(source_file, model_root):
        raise _fail(f"model component symlink escapes --model-dir: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source_file, destination)
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            raise _fail(
                f"cannot hard-link {source} into the package across filesystems; "
                "place the package on the source filesystem"
            ) from exc
        raise _fail(f"cannot hard-link {source} into the package: {exc}") from exc


def _hardlink_tree(source_root: Path, destination_root: Path) -> None:
    source_root = source_root.resolve(strict=True)
    destination_root.mkdir(parents=True)
    for current, directory_names, file_names in os.walk(source_root, followlinks=False):
        current_path = Path(current)
        for directory_name in list(directory_names):
            source_dir = current_path / directory_name
            if source_dir.is_symlink():
                raise _fail(f"model directory symlink is not allowed: {source_dir}")
            if not _include_model_directory(source_dir.relative_to(source_root)):
                directory_names.remove(directory_name)
        for file_name in file_names:
            source_file = current_path / file_name
            relative = source_file.relative_to(source_root)
            if source_file.is_symlink():
                resolved = source_file.resolve(strict=True)
                if not _within(resolved, source_root):
                    raise _fail(f"model component symlink escapes --model-dir: {source_file}")
            if not _include_model_file(relative):
                continue
            destination_file = destination_root / relative
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            _hardlink_file(source_file, destination_file, model_root=source_root)


def _no_replace_rename(source: Path, destination: Path) -> None:
    """Atomically publish a directory only if the destination is absent."""
    encoded_source = os.fsencode(source)
    encoded_destination = os.fsencode(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise OSError("safe no-overwrite directory publication is unavailable on this macOS")
        renamex_np.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex_np.restype = ctypes.c_int
        result = renamex_np(encoded_source, encoded_destination, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError("safe no-overwrite directory publication is unavailable on this Linux")
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, encoded_source, -100, encoded_destination, 1)  # RENAME_NOREPLACE
    elif os.name == "nt":
        os.rename(source, destination)
        return
    else:
        raise OSError(f"safe no-overwrite publication is unsupported on {sys.platform}")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), str(destination))
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _manifest_for(
    artifact_records: dict[str, Any], model_source_attestation: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scope": "local-use-only",
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "pipeline_class": PIPELINE_CLASS,
        },
        "runtime": {
            "kind": "hybrid",
            "target": TARGET_RUNTIME,
            "conditioning_device": "cpu",
            "denoiser": "crystal-metal",
            "decode_device": "cpu",
            "python_interpreter": "not bundled; invoke with a compatible external Python environment",
            "python_packages": ["torch", "transformers", "diffusers", "Pillow", "NumPy"],
            "external_native_libraries": [
                "libcrypto.3.dylib (Homebrew openssl@3)",
                "libpcre2-8.0.dylib (Homebrew pcre2)",
                "libgc.1.dylib (Homebrew bdw-gc)",
            ],
            "native_build_dependencies": "not bundled; included Crystal file is the CLI entrypoint source copy only",
        },
        "gguf": {
            "repo": GGUF_REPO,
            "revision": GGUF_REVISION,
            "filename": GGUF_FILENAME,
            "size_bytes": GGUF_SIZE_BYTES,
            "sha256": GGUF_SHA256,
            "tensor_policy": GGUF_TENSOR_POLICY,
        },
        "components": dict(COMPONENT_PATHS),
        "artifacts": {
            "dit_gguf": {"size_bytes": GGUF_SIZE_BYTES, "sha256": GGUF_SHA256},
            **artifact_records,
        },
        "model_source_attestation": model_source_attestation,
        "storage": {
            "large_files": "hardlinked",
            "hardlink_notice": "In-place changes through package or source paths affect both until copied to another filesystem.",
        },
    }


def _prepare_package_root(package_dir: str | Path) -> tuple[Path, Path]:
    requested = Path(package_dir).expanduser()
    if not requested.name or requested.name in {".", ".."}:
        raise _fail("package directory must name a new directory")
    parent = requested.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        parent_real = parent.resolve(strict=True)
    except OSError as exc:
        raise _fail(f"cannot prepare package parent directory {parent}: {exc}") from exc
    final = parent_real / requested.name
    if os.path.lexists(final):
        raise FileExistsError(f"package directory already exists: {final}")
    return parent_real, final


def initialize_package(
    *,
    package_dir: str | Path,
    model_dir: str | Path,
    gguf_path: str | Path,
    denoiser_path: str | Path,
    revision: str,
) -> Path:
    """Create a new package without downloading, copying large weights, or overwriting."""
    source_files = (SOURCE_CONDITIONER, SOURCE_DECODER, SOURCE_DENOISER, SOURCE_LAUNCHER)
    if any(not source.is_file() for source in source_files):
        raise _fail(
            "init must be run from the repository launcher; a packaged launcher supports generate only"
        )
    normalized_revision = _check_revision(revision)
    if normalized_revision != MODEL_REVISION:
        raise _fail("package model revision does not match the pinned official snapshot")

    model_source = _source_path(model_dir, "model components directory")
    gguf_source = _source_path(gguf_path, "GGUF file")
    denoiser_source = _source_path(denoiser_path, "native denoiser")
    if not model_source.is_dir():
        raise _fail(f"model components source must be a directory: {model_source}")
    _validate_model_directory(model_source)

    if not gguf_source.is_file():
        raise _fail(f"GGUF source must be a regular file: {gguf_source}")
    gguf_record = _record_file(gguf_source)
    if gguf_record["size_bytes"] != GGUF_SIZE_BYTES or gguf_record["sha256"] != GGUF_SHA256:
        raise _fail(
            "GGUF identity/checksum mismatch; expected the pinned mixed-quant Qwen-Image 2.1 artifact "
            f"({GGUF_SIZE_BYTES} bytes, SHA-256 {GGUF_SHA256})"
        )
    _validate_macho_arm64(denoiser_source)

    parent, final_root = _prepare_package_root(package_dir)
    for source in (model_source, gguf_source, denoiser_source):
        if _paths_overlap(final_root, source):
            raise _fail("package directory must be separate from all input paths")

    stage_root = Path(tempfile.mkdtemp(prefix=f".{final_root.name}.qwen-image21-", dir=parent))
    try:
        _hardlink_tree(model_source, stage_root / COMPONENT_PATHS["model_dir"])
        _hardlink_file(gguf_source, stage_root / COMPONENT_PATHS["dit_gguf"])
        denoiser_dest = stage_root / COMPONENT_PATHS["denoiser"]
        _hardlink_file(denoiser_source, denoiser_dest)
        (stage_root / "bin").mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_CONDITIONER, stage_root / COMPONENT_PATHS["conditioning_script"])
        shutil.copy2(SOURCE_DECODER, stage_root / COMPONENT_PATHS["decoder_script"])
        shutil.copy2(SOURCE_LAUNCHER, stage_root / COMPONENT_PATHS["launcher_script"])
        (stage_root / "src").mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_DENOISER, stage_root / COMPONENT_PATHS["denoiser_entrypoint"])

        # Recheck staged inputs, then write the manifest last. The final path is
        # published atomically only after all content and identities validate.
        staged_model = stage_root / COMPONENT_PATHS["model_dir"]
        _validate_model_directory(staged_model)
        staged_gguf = stage_root / COMPONENT_PATHS["dit_gguf"]
        if _record_file(staged_gguf) != {"size_bytes": GGUF_SIZE_BYTES, "sha256": GGUF_SHA256}:
            raise _fail("staged GGUF changed during package initialization")
        _validate_macho_arm64(denoiser_dest)
        denoiser_record = _record_file(denoiser_dest)
        model_files, model_source_attestation = _record_model_files(staged_model)
        artifact_records: dict[str, Any] = {
            "denoiser": denoiser_record,
            "model_files": model_files,
        }
        for key in (
            "conditioning_script",
            "decoder_script",
            "launcher_script",
            "denoiser_entrypoint",
        ):
            artifact_records[key] = _record_file(stage_root / COMPONENT_PATHS[key])
        manifest = _manifest_for(artifact_records, model_source_attestation)
        (stage_root / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _no_replace_rename(stage_root, final_root)
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise
    return final_root / MANIFEST_NAME


def _reject_package_symlinks(package_root: Path) -> None:
    for current, directory_names, file_names in os.walk(package_root, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            if (current_path / name).is_symlink():
                raise _fail(f"package contains a symlinked directory: {current_path / name}")
        for name in file_names:
            if (current_path / name).is_symlink():
                raise _fail(f"package contains a symlinked file: {current_path / name}")


def _package_path(package_root: Path, raw: Any, description: str, *, directory: bool = False) -> Path:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise _fail(f"manifest {description} must be a package-relative path")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise _fail(f"manifest {description} must stay inside the package")
    path = package_root.joinpath(*relative.parts)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise _fail(f"manifest {description} does not exist: {raw}") from exc
    if not _within(resolved, package_root):
        raise _fail(f"manifest {description} escapes the package directory")
    if directory:
        if not resolved.is_dir():
            raise _fail(f"manifest {description} must name a directory")
    elif not resolved.is_file():
        raise _fail(f"manifest {description} must name a regular file")
    return resolved


def _validate_manifest_shape(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != SCHEMA or manifest.get("schema_version") != SCHEMA_VERSION:
        raise _fail("unsupported Qwen-Image 2.1 package manifest schema/version")
    if manifest.get("scope") != "local-use-only":
        raise _fail("package manifest must identify a local-use-only package")
    if manifest.get("model") != {
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "pipeline_class": PIPELINE_CLASS,
    }:
        raise _fail("package model identity/revision does not match the pinned Qwen-Image 2.1 snapshot")
    if manifest.get("runtime") != {
        "kind": "hybrid",
        "target": TARGET_RUNTIME,
        "conditioning_device": "cpu",
        "denoiser": "crystal-metal",
        "decode_device": "cpu",
        "python_interpreter": "not bundled; invoke with a compatible external Python environment",
        "python_packages": ["torch", "transformers", "diffusers", "Pillow", "NumPy"],
        "external_native_libraries": [
            "libcrypto.3.dylib (Homebrew openssl@3)",
            "libpcre2-8.0.dylib (Homebrew pcre2)",
            "libgc.1.dylib (Homebrew bdw-gc)",
        ],
        "native_build_dependencies": "not bundled; included Crystal file is the CLI entrypoint source copy only",
    }:
        raise _fail("package runtime must declare the supported hybrid CPU/Metal route")
    if manifest.get("gguf") != {
        "repo": GGUF_REPO,
        "revision": GGUF_REVISION,
        "filename": GGUF_FILENAME,
        "size_bytes": GGUF_SIZE_BYTES,
        "sha256": GGUF_SHA256,
        "tensor_policy": GGUF_TENSOR_POLICY,
    }:
        raise _fail("package GGUF identity does not match the pinned mixed-quant artifact")
    if manifest.get("components") != COMPONENT_PATHS:
        raise _fail("package component paths do not match manifest schema version 1")
    artifacts = manifest.get("artifacts")
    expected_artifacts = {
        "dit_gguf",
        "denoiser",
        "conditioning_script",
        "decoder_script",
        "launcher_script",
        "denoiser_entrypoint",
        "model_files",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise _fail("package manifest is missing artifact integrity records")
    if artifacts["dit_gguf"] != {"size_bytes": GGUF_SIZE_BYTES, "sha256": GGUF_SHA256}:
        raise _fail("package GGUF checksum record does not match the pinned artifact")

    def valid_record(record: Any) -> bool:
        return (
            isinstance(record, dict)
            and set(record) == {"size_bytes", "sha256"}
            and isinstance(record.get("size_bytes"), int)
            and record["size_bytes"] > 0
            and isinstance(record.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None
        )

    for key in (
        "denoiser",
        "conditioning_script",
        "decoder_script",
        "launcher_script",
        "denoiser_entrypoint",
    ):
        if not valid_record(artifacts[key]):
            raise _fail(f"package {key} integrity record is invalid")
    model_records = artifacts["model_files"]
    if not isinstance(model_records, dict) or not model_records:
        raise _fail("package model file integrity records are missing")
    for relative, record in model_records.items():
        if not isinstance(relative, str) or "\\" in relative:
            raise _fail("package model integrity record contains an invalid relative path")
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or relative_path.as_posix() != relative or any(
            part in {"", ".", ".."} for part in relative_path.parts
        ):
            raise _fail("package model integrity record path is not package-relative")
        if not valid_record(record):
            raise _fail(f"package model integrity record is invalid for {relative}")
    source_attestation = manifest.get("model_source_attestation")
    if (
        not isinstance(source_attestation, dict)
        or set(source_attestation)
        != {"mechanism", "checked_safetensors", "unavailable_safetensors", "limitation"}
        or source_attestation.get("mechanism")
        != "Hugging Face cache revision and LFS SHA-256 sidecars, when present"
        or source_attestation.get("limitation")
        != (
            "Sidecars corroborate only the listed files and are not a cryptographic signature; "
            "all package model files are separately SHA-256-pinned at initialization."
        )
        or not isinstance(source_attestation.get("checked_safetensors"), dict)
        or not isinstance(source_attestation.get("unavailable_safetensors"), list)
    ):
        raise _fail("package model source integrity disclosure is invalid")
    for relative, record in source_attestation["checked_safetensors"].items():
        if (
            relative not in model_records
            or not isinstance(record, dict)
            or record != {"revision": MODEL_REVISION, "sha256": model_records[relative]["sha256"]}
        ):
            raise _fail(f"package model source sidecar record is invalid for {relative}")
    if any(not isinstance(relative, str) or relative not in model_records for relative in source_attestation["unavailable_safetensors"]):
        raise _fail("package model source disclosure lists an invalid missing sidecar")
    if set(source_attestation["checked_safetensors"]).intersection(
        source_attestation["unavailable_safetensors"]
    ):
        raise _fail("package model source sidecar cannot be both checked and unavailable")
    storage = manifest.get("storage")
    if storage != {
        "large_files": "hardlinked",
        "hardlink_notice": "In-place changes through package or source paths affect both until copied to another filesystem.",
    }:
        raise _fail("package storage/hard-link notice is missing")


def _assert_package_root_identity(
    requested_root: Path,
    package_root: Path,
    expected_identity: tuple[int, int],
    *,
    before_stage: str,
) -> None:
    """Reject a replaced, symlinked, or otherwise changed package root."""
    try:
        root_stat = requested_root.lstat()
        resolved_root = requested_root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            f"package root changed after validation before {before_stage}: {exc}"
        ) from exc
    identity = (root_stat.st_dev, root_stat.st_ino)
    if (
        stat.S_ISLNK(root_stat.st_mode)
        or not stat.S_ISDIR(root_stat.st_mode)
        or resolved_root != package_root
        or identity != expected_identity
    ):
        raise RuntimeError(
            f"package root changed after validation before {before_stage}; refusing to launch stage"
        )


def validate_package(package_dir: str | Path) -> dict[str, Any]:
    """Validate package contents and retain the exact root identity checked.

    Manifest checks detect internal inconsistency or later drift; they do not
    authenticate who produced the package or guarantee resistance to concurrent
    filesystem mutation.
    """
    requested_root = Path(package_dir).expanduser().absolute()
    if requested_root.is_symlink():
        raise _fail("package root must not be a symlink")
    try:
        package_root = requested_root.resolve(strict=True)
    except OSError as exc:
        raise _fail(f"package directory does not exist: {requested_root}") from exc
    if not package_root.is_dir():
        raise _fail(f"package directory does not exist: {package_root}")
    try:
        root_stat = requested_root.lstat()
        canonical_stat = package_root.lstat()
    except OSError as exc:
        raise _fail(f"cannot inspect package root: {exc}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise _fail("package root must be a real directory, not a symlink")
    root_identity = (root_stat.st_dev, root_stat.st_ino)
    if root_identity != (canonical_stat.st_dev, canonical_stat.st_ino):
        raise _fail("package root changed while resolving its path")
    _reject_package_symlinks(package_root)
    manifest_path = _package_path(package_root, MANIFEST_NAME, "manifest path")
    manifest = _read_json(manifest_path, "package manifest")
    _validate_manifest_shape(manifest)

    components = manifest["components"]
    model_dir = _package_path(package_root, components["model_dir"], "model_dir", directory=True)
    _validate_model_directory(model_dir)
    actual_model_files, actual_model_source_attestation = _record_model_files(model_dir)
    expected_model_files = manifest["artifacts"]["model_files"]
    if actual_model_source_attestation != manifest["model_source_attestation"]:
        raise _fail("package model source sidecar attestation differs from initialization")
    if set(actual_model_files) != set(expected_model_files):
        raise _fail("package model file set differs from its integrity manifest")
    for relative, actual_record in actual_model_files.items():
        if actual_record != expected_model_files[relative]:
            raise _fail(f"package model file size/SHA-256 checksum mismatch: {relative}")
    gguf_path = _package_path(package_root, components["dit_gguf"], "dit_gguf")
    if _record_file(gguf_path) != manifest["artifacts"]["dit_gguf"]:
        raise _fail("package GGUF size or SHA-256 checksum mismatch")
    denoiser_path = _package_path(package_root, components["denoiser"], "denoiser")
    _validate_macho_arm64(denoiser_path)
    if _record_file(denoiser_path) != manifest["artifacts"]["denoiser"]:
        raise _fail("package native denoiser size or SHA-256 checksum mismatch")
    for key in ("conditioning_script", "decoder_script", "launcher_script", "denoiser_entrypoint"):
        artifact_path = _package_path(package_root, components[key], key)
        if _record_file(artifact_path) != manifest["artifacts"][key]:
            raise _fail(f"package {key} size/SHA-256 checksum mismatch")
    _assert_package_root_identity(
        requested_root,
        package_root,
        root_identity,
        before_stage="generation",
    )
    return _ValidatedPackageManifest(
        manifest,
        requested_root=requested_root,
        package_root=package_root,
        root_identity=root_identity,
    )


def _validate_generation_args(
    prompt: str, width: int, height: int, seed: int, steps: int, output_path: str | Path
) -> Path:
    if not isinstance(prompt, str) or not prompt.strip():
        raise _fail("prompt must be a non-empty string")
    for name, dimension in (("width", width), ("height", height)):
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise _fail(f"image {name} must be a positive integer")
        if dimension > MAX_IMAGE_SIDE:
            raise _fail(f"image {name} must not exceed {MAX_IMAGE_SIDE} pixels")
        if dimension % 32:
            raise _fail("image width and height must be multiples of 32")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= (2**63 - 1):
        raise _fail("seed must be an integer in [0, 2^63-1]")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 2 <= steps <= 100:
        raise _fail("steps must be an integer in [2, 100]")

    requested = Path(output_path).expanduser()
    if requested.suffix.lower() != ".png" or not requested.name:
        raise _fail("output path must name a .png file")
    try:
        output_parent = requested.parent.resolve(strict=True)
    except OSError as exc:
        raise _fail(f"output parent directory does not exist: {requested.parent}") from exc
    if not output_parent.is_dir():
        raise _fail(f"output parent is not a directory: {output_parent}")
    output = output_parent / requested.name
    if os.path.lexists(output):
        raise FileExistsError(f"output already exists; refusing to overwrite: {output}")
    return output


def _offline_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in OFFLINE_ENVIRONMENT:
        environment[name] = "1"
    return environment


def _validate_png(path: Path, width: int, height: int) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to validate the final PNG") from exc
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGBA" or image.size != (width, height):
                raise _fail(
                    f"decoded output must be a {width}x{height} RGBA PNG; "
                    f"got {image.format} {image.mode} {image.size}"
                )
            image.load()
    except (OSError, ValueError) as exc:
        raise _fail(f"decoded output is not a valid PNG: {exc}") from exc


def _bundle_file(directory: Path, raw_name: Any, expected_name: str, description: str) -> Path:
    if raw_name != expected_name:
        raise _fail(f"{description} must name {expected_name!r}")
    relative = PurePosixPath(raw_name)
    if relative.is_absolute() or len(relative.parts) != 1 or ".." in relative.parts:
        raise _fail(f"{description} must stay in its stage output directory")
    path = directory / expected_name
    if path.is_symlink() or not path.is_file():
        raise _fail(f"{description} payload is missing or not a regular file")
    return path


def _validate_conditioning_output(
    manifest_path: Path,
    *,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    revision: str,
) -> None:
    manifest = _read_json(manifest_path, "conditioning stage manifest")
    if manifest.get("schema") != "qwen-image21-conditioning" or manifest.get("schema_version") != 1:
        raise _fail("conditioning stage wrote an unsupported manifest")
    model = manifest.get("model")
    if not isinstance(model, dict) or model.get("repo") != MODEL_REPO or model.get("revision") != revision:
        raise _fail("conditioning stage model identity/revision does not match the package")
    if manifest.get("prompt") != prompt:
        raise _fail("conditioning stage prompt does not match the request")
    image = manifest.get("image")
    if not isinstance(image, dict) or image.get("width") != width or image.get("height") != height:
        raise _fail("conditioning stage image dimensions do not match the request")
    noise = manifest.get("noise")
    if not isinstance(noise, dict) or noise.get("seed") != seed:
        raise _fail("conditioning stage seed does not match the request")

    payload_path = _bundle_file(
        manifest_path.parent,
        manifest.get("payload_file"),
        "qwen_image21_conditioning.bin",
        "conditioning payload",
    )
    try:
        payload_size = payload_path.stat().st_size
    except OSError as exc:
        raise _fail(f"cannot stat conditioning payload: {exc}") from exc
    if manifest.get("payload_nbytes") != payload_size:
        raise _fail("conditioning payload length does not match its manifest")
    payload_hash = manifest.get("payload_sha256")
    if not isinstance(payload_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", payload_hash):
        raise _fail("conditioning payload manifest has an invalid SHA-256")
    if _sha256(payload_path) != payload_hash:
        raise _fail("conditioning payload checksum does not match its manifest")

    latent_height = height // 16
    latent_width = width // 16
    latent_tokens = latent_height * latent_width
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict):
        raise _fail("conditioning manifest is missing tensor descriptors")
    latent_desc = tensors.get("initial_target_latents")
    if (
        not isinstance(latent_desc, dict)
        or latent_desc.get("dtype") != "float32-le"
        or latent_desc.get("shape") != [latent_tokens, 64]
        or latent_desc.get("nbytes") != latent_tokens * 64 * 4
    ):
        raise _fail("conditioning target-noise tensor does not match the requested image shape")


def _validate_latent_output(
    manifest_path: Path,
    *,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    revision: str,
    gguf_path: Path,
) -> None:
    manifest = _read_json(manifest_path, "native denoiser manifest")
    expected_fields = {
        "format": "qwen-image21-latents-v1",
        "model_id": MODEL_REPO,
        "layout": "tokens_hwc",
        "channels": 64,
        "latent_height": height // 16,
        "latent_width": width // 16,
        "image_height": height,
        "image_width": width,
        "dtype": "float32-le",
        "scaling": "diffusers_normalized",
        "model_revision": revision,
        "seed": seed,
        "prompt": prompt,
        "denoising_steps": steps,
    }
    for key, expected in expected_fields.items():
        if manifest.get(key) != expected:
            raise _fail(f"native denoiser manifest {key!r} does not match the request/package")
    if manifest.get("dit_gguf") != str(gguf_path.resolve()):
        raise _fail("native denoiser manifest does not identify this package GGUF")
    payload_path = _bundle_file(
        manifest_path.parent,
        manifest.get("payload"),
        "qwen_image21_latents.bin",
        "latent payload",
    )
    expected_bytes = (height // 16) * (width // 16) * 64 * 4
    if manifest.get("payload_bytes") != expected_bytes or payload_path.stat().st_size != expected_bytes:
        raise _fail("native denoiser latent payload length does not match the requested image shape")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to validate the native denoiser output") from exc
    values = np.fromfile(payload_path, dtype="<f4")
    if values.size != expected_bytes // 4 or not bool(np.isfinite(values).all()):
        raise _fail("native denoiser payload contains invalid or non-finite latents")


def generate_image(
    *,
    package_dir: str | Path,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    output_path: str | Path,
) -> Path:
    """Run all three packaged stages offline and atomically publish one PNG."""
    output = _validate_generation_args(prompt, width, height, seed, steps, output_path)
    manifest = validate_package(package_dir)
    require_supported_runtime()
    package_root = manifest._package_root
    requested_root = manifest._requested_root
    root_identity = manifest._root_identity
    components = manifest["components"]
    model_dir = package_root / components["model_dir"]
    gguf_path = package_root / components["dit_gguf"]
    denoiser = package_root / components["denoiser"]
    conditioner = package_root / components["conditioning_script"]
    decoder = package_root / components["decoder_script"]
    offline_env = _offline_environment()

    with tempfile.TemporaryDirectory(prefix="qwen-image21-run-", dir=output.parent) as work_name:
        work_dir = Path(work_name)
        conditioning_dir = work_dir / "conditioning"
        latent_dir = work_dir / "latents"
        staged_png = work_dir / "decoded.png"
        conditioning_manifest = conditioning_dir / "qwen_image21_conditioning.json"
        latent_manifest = latent_dir / "qwen_image21_latents.json"

        conditioning_command = [
            sys.executable,
            str(conditioner),
            "--model-dir",
            str(model_dir),
            "--output-dir",
            str(conditioning_dir),
            "--prompt",
            prompt,
            "--width",
            str(width),
            "--height",
            str(height),
            "--seed",
            str(seed),
            "--revision",
            manifest["model"]["revision"],
            "--device",
            "cpu",
            "--dtype",
            "bfloat16",
        ]
        _assert_package_root_identity(
            requested_root,
            package_root,
            root_identity,
            before_stage="conditioning",
        )
        subprocess.run(conditioning_command, check=True, env=offline_env)
        if not conditioning_manifest.is_file():
            raise RuntimeError("conditioning stage succeeded without writing its manifest")
        _validate_conditioning_output(
            conditioning_manifest,
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
            revision=manifest["model"]["revision"],
        )

        denoiser_command = [
            str(denoiser),
            str(gguf_path),
            str(conditioning_manifest),
            str(latent_dir),
            str(steps),
        ]
        _assert_package_root_identity(
            requested_root,
            package_root,
            root_identity,
            before_stage="native denoiser",
        )
        subprocess.run(denoiser_command, check=True, env=offline_env)
        if not latent_manifest.is_file():
            raise RuntimeError("native denoiser succeeded without writing its latent manifest")
        _validate_latent_output(
            latent_manifest,
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            revision=manifest["model"]["revision"],
            gguf_path=gguf_path,
        )

        decoder_command = [
            sys.executable,
            str(decoder),
            "--manifest",
            str(latent_manifest),
            "--model-dir",
            str(model_dir),
            "--output",
            str(staged_png),
            "--device",
            "cpu",
            "--dtype",
            "float32",
        ]
        _assert_package_root_identity(
            requested_root,
            package_root,
            root_identity,
            before_stage="VAE decode",
        )
        subprocess.run(decoder_command, check=True, env=offline_env)
        if not staged_png.is_file():
            raise RuntimeError("VAE decode stage succeeded without writing a PNG")
        _validate_png(staged_png, width, height)

        # `link` is an atomic no-clobber publish when staging and destination
        # share a filesystem. It leaves a pre-existing output untouched, even
        # if another process creates it during generation.
        try:
            os.link(staged_png, output)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise FileExistsError(f"output appeared during generation; refusing to overwrite: {output}") from exc
            raise RuntimeError(f"could not atomically publish final PNG: {exc}") from exc
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="package local components and native binaries")
    init_parser.add_argument("--package-dir", required=True, help="new destination directory")
    init_parser.add_argument("--model-dir", required=True, help="complete local Qwen-Image 2.1 component directory")
    init_parser.add_argument("--gguf", required=True, help="pinned mixed-quant Qwen-Image 2.1 GGUF")
    init_parser.add_argument("--denoiser", required=True, help="compiled macOS arm64 Metal denoiser")
    init_parser.add_argument("--revision", required=True, help="official full 40-character model commit SHA")

    generate_parser = subparsers.add_parser(
        "generate",
        help="generate one prompt-conditioned PNG offline (requires external Python and native libraries)",
    )
    generate_parser.add_argument("--package-dir", required=True, help="initialized local package directory")
    generate_parser.add_argument("--prompt", required=True, help="text prompt; image editing/guidance are not supported")
    generate_parser.add_argument("--width", type=int, required=True, help="image width, a multiple of 32")
    generate_parser.add_argument("--height", type=int, required=True, help="image height, a multiple of 32")
    generate_parser.add_argument("--seed", type=int, required=True, help="CPU noise seed")
    generate_parser.add_argument("--steps", type=int, required=True, help="native denoising steps in [2, 100]")
    generate_parser.add_argument("--output", required=True, help="new output .png path; existing files are preserved")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            manifest_path = initialize_package(
                package_dir=args.package_dir,
                model_dir=args.model_dir,
                gguf_path=args.gguf,
                denoiser_path=args.denoiser,
                revision=args.revision,
            )
            print(f"initialized local-use package: {manifest_path}")
            return 0
        output = generate_image(
            package_dir=args.package_dir,
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            seed=args.seed,
            steps=args.steps,
            output_path=args.output,
        )
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"qwen_image21_package: error: {exc}", file=sys.stderr)
        return 1
    print(f"saved {args.width}x{args.height} RGBA PNG to {output}")
    print("runtime: hybrid CPU Qwen3-VL + Crystal Metal GGUF DiT + CPU Diffusers VAE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
