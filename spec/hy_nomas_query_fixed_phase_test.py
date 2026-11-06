"""CPU falsifiers for the fixed-destination-phase content null."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parent / "support" / "hy_nomas_query_fixed_phase.py"
SPEC = importlib.util.spec_from_file_location("hy_nomas_query_fixed_phase", SOURCE)
fixed_phase = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixed_phase)
ref = fixed_phase.ref


def _bf16_expanded(shape, value):
    return np.full(shape, np.float32(value), dtype="<f4")


def _qk_fixture(directory):
    query = _bf16_expanded(ref.SHAPE, 1.0)
    key = _bf16_expanded(ref.SHAPE, 0.5)
    path, digest = ref.publish_input(directory / "qk.json", query, key,
                                     {"fixture": "cpu-test", "frame": 14})
    return {"path": str(path.resolve()), "sha256": digest}, query, key


def _phase_fixture():
    values = np.arange(np.prod(fixed_phase.PHASE_SHAPE), dtype=np.float32)
    cosine = (np.float32(0.1) + values * np.float32(0.000123)).reshape(fixed_phase.PHASE_SHAPE)
    sine = (-np.float32(0.2) + values * np.float32(0.000071)).reshape(fixed_phase.PHASE_SHAPE)
    return np.ascontiguousarray(cosine, dtype="<f4"), np.ascontiguousarray(sine, dtype="<f4")


def _publish_fixture(directory, name="valid", qk=None):
    if qk is None:
        qk, _, _ = _qk_fixture(directory)
    cosine, sine = _phase_fixture()
    record = fixed_phase.publish_case(directory / f"{name}.json", qk, cosine, sine,
                                      fixed_phase.case_specs()[4],
                                      {"fixture": "cpu-test", "phase": "stored-F32"})
    return Path(record["path"]), record["sha256"], qk, cosine, sine


def _clone_and_mutate(path, directory, name, mutation):
    document = json.loads(path.read_text())
    payload_name = document["payload"]["path"]
    raw = bytearray((path.parent / payload_name).read_bytes())
    document["payload"]["path"] = f"{name}.f32le"
    mutation(document, raw)
    payload_path = directory / document["payload"]["path"]
    payload_path.write_bytes(raw)
    manifest = directory / f"{name}.json"
    manifest.write_text(json.dumps(document, indent=2) + "\n")
    return manifest, ref.file_sha256(manifest)


class FixedPhaseTests(unittest.TestCase):
    def test_frozen_nine_case_matrix_rejects_head_frame_arm_roll_and_bool_changes(self):
        self.assertEqual(fixed_phase.policy(), {
            "heads": [12, 15], "frames": [1, 14, 29], "key_roll": 17,
            "positions": 144, "atol": ref.ATOL, "rtol": ref.RTOL,
            "independent_saved_states": 1, "transport_gate": False,
        })
        cases = fixed_phase.case_specs()
        self.assertEqual(len(cases), 9)
        self.assertEqual(
            [(case["frame"], case["arm"], case["key_roll"]) for case in cases],
            [(frame, arm, roll) for frame in (1, 14, 29)
             for arm, roll in (("identity", 0), ("moving", 17), ("fixed", 17))],
        )
        for case in cases:
            fixed_phase.validate_case(case)
        invalid = [
            dict(cases[0], frame=0),
            dict(cases[0], frame=True),
            dict(cases[0], arm="moving"),
            dict(cases[1], key_roll=0),
            dict(cases[0], key_roll=True),
            {**cases[0], "head": 12},
        ]
        for case in invalid:
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, "case differs"):
                fixed_phase.validate_case(case)

    def test_content_targets_roll_by_seventeen_with_wrap_for_both_control_arms(self):
        identity = fixed_phase.content_targets("identity")
        expected = (np.arange(144) + 17) % 144
        moving = fixed_phase.content_targets("moving")
        fixed = fixed_phase.content_targets("fixed")
        np.testing.assert_array_equal(identity, np.arange(144))
        np.testing.assert_array_equal(moving, expected)
        np.testing.assert_array_equal(fixed, expected)
        self.assertEqual(int(moving[127]), 0)
        self.assertEqual(int(moving[-1]), 16)
        with self.assertRaisesRegex(ValueError, "unknown arm"):
            fixed_phase.content_targets("transport")

    def test_summary_separates_diagonal_and_relocated_content_winners_and_ties(self):
        rows = np.arange(144)
        scores = {}
        metrics = {}
        for arm in ("identity", "moving", "fixed"):
            targets = fixed_phase.content_targets(arm)
            values = np.zeros((16, 144, 144), dtype="<f4")
            values[:, rows, targets] = 2.0
            diagonal = values[:, rows, rows]
            others = values.copy()
            others[:, rows, targets] = -np.inf
            margin = diagonal - others.max(axis=-1)
            result_metrics = np.zeros((16, 144, 6), dtype="<f4")
            result_metrics[:, :, 2] = margin
            result_metrics[:, :, 4] = np.argmax(values, axis=-1)
            scores[arm], metrics[arm] = values, result_metrics

        identity = fixed_phase.summary(scores["identity"], metrics["identity"], "identity")
        moving = fixed_phase.summary(scores["moving"], metrics["moving"], "moving")
        fixed = fixed_phase.summary(scores["fixed"], metrics["fixed"], "fixed")
        for head in ("12", "15"):
            self.assertEqual(identity[head]["strict_diagonal_count"], 144)
            self.assertEqual(identity[head]["strict_relocated_content_count"], 144)
            for summary in (moving, fixed):
                self.assertEqual(summary[head]["strict_diagonal_count"], 0)
                self.assertEqual(summary[head]["strict_relocated_content_count"], 144)
                self.assertEqual(summary[head]["content_argmax_count"], 144)

        tied = np.zeros((16, 144, 144), dtype="<f4")
        tied_metrics = np.zeros((16, 144, 6), dtype="<f4")
        tied_metrics[:, :, 4] = np.argmax(tied, axis=-1)  # Stable first-index winner.
        tied_summary = fixed_phase.summary(tied, tied_metrics, "identity")
        for head in ("12", "15"):
            self.assertEqual(tied_summary[head]["strict_diagonal_count"], 0)
            self.assertEqual(tied_summary[head]["content_argmax_count"], 1)
            self.assertTrue(np.all(tied_metrics[int(head), :, 4] == 0))

    def test_round_trip_preserves_exact_f32_phase_bytes_and_requires_bf16_expanded_qk(self):
        with tempfile.TemporaryDirectory(prefix="fixed-phase-test-", dir=Path(tempfile.gettempdir()).resolve()) as raw_dir:
            directory = Path(raw_dir)
            manifest, digest, qk, cosine, sine = _publish_fixture(directory)
            admitted = fixed_phase.load_input(manifest, digest)
            self.assertEqual(admitted["cosine"].dtype, np.dtype("<f4"))
            self.assertEqual(admitted["sine"].dtype, np.dtype("<f4"))
            self.assertEqual(admitted["cosine"].tobytes(), cosine.tobytes())
            self.assertEqual(admitted["sine"].tobytes(), sine.tobytes())
            self.assertEqual(admitted["qk"]["query"].tobytes(), _bf16_expanded(ref.SHAPE, 1.0).tobytes())
            self.assertEqual(admitted["qk"]["key"].tobytes(), _bf16_expanded(ref.SHAPE, 0.5).tobytes())
            self.assertEqual(admitted["document"]["qk"], qk)
            self.assertTrue(np.any(cosine.view("<u4") & 0xFFFF))  # Input has bits BF16 rounding would drop.

            bad_dir = directory / "bad-qk"
            bad_dir.mkdir()
            bad_qk, _, _ = _qk_fixture(bad_dir)
            qk_path = Path(bad_qk["path"])
            qk_doc = json.loads(qk_path.read_text())
            qk_payload = qk_path.parent / qk_doc["payload"]["path"]
            qk_raw = bytearray(qk_payload.read_bytes())
            query_record = qk_doc["tensors"]["query"]
            query_bytes = bytearray(qk_raw[query_record["offset_bytes"]:
                                            query_record["offset_bytes"] + query_record["nbytes"]])
            query_words = np.frombuffer(query_bytes, dtype="<u4").copy()
            query_words[0] |= 1  # Preserve the F32 value while violating BF16-expanded low bits.
            query_bytes[:] = query_words.astype("<u4").tobytes()
            start = query_record["offset_bytes"]
            qk_raw[start:start + query_record["nbytes"]] = query_bytes
            query_record["sha256"] = ref.sha256(query_bytes)
            qk_doc["payload"]["sha256"] = ref.sha256(qk_raw)
            qk_payload.write_bytes(qk_raw)
            qk_path.write_text(json.dumps(qk_doc, indent=2) + "\n")
            bad_qk = {"path": str(qk_path.resolve()), "sha256": ref.file_sha256(qk_path)}
            bad_manifest, bad_digest, _, _, _ = _publish_fixture(
                bad_dir, "bad-expanded-qk", qk=bad_qk)
            with self.assertRaisesRegex(ValueError, "lossless BF16 expansion"):
                fixed_phase.load_input(bad_manifest, bad_digest)

    def test_malformed_case_payload_hash_range_dtype_nan_and_unknown_field_reject(self):
        with tempfile.TemporaryDirectory(prefix="fixed-phase-tamper-", dir=Path(tempfile.gettempdir()).resolve()) as raw_dir:
            root = Path(raw_dir)
            _, _, qk, _, _ = _publish_fixture(root, "base")

            def valid_case(name):
                directory = root / name
                directory.mkdir()
                return _publish_fixture(directory, "input", qk=qk)

            mutations = [
                ("case", lambda doc, raw: doc["case"].update(frame=True)),
                ("range", lambda doc, raw: doc["tensors"]["sine"].update(offset_bytes=4)),
                ("dtype", lambda doc, raw: doc["tensors"]["cosine"].update(dtype="BF16")),
                ("unknown", lambda doc, raw: doc.update(unexpected=True)),
            ]
            for name, mutation in mutations:
                with self.subTest(kind=name):
                    original, _, _, _, _ = valid_case(name)
                    directory = root / name
                    altered, digest = _clone_and_mutate(original, directory, f"altered-{name}", mutation)
                    with self.assertRaises(ValueError):
                        fixed_phase.load_input(altered, digest)

            original, _, _, _, _ = valid_case("payload-hash")
            directory = root / "payload-hash"

            def corrupt_payload(doc, raw):
                raw[0] ^= 1

            altered, digest = _clone_and_mutate(original, directory, "altered-payload", corrupt_payload)
            with self.assertRaisesRegex(ValueError, "payload length or digest mismatch"):
                fixed_phase.load_input(altered, digest)

            original, _, _, _, _ = valid_case("nan")
            directory = root / "nan"

            def authenticated_nan(doc, raw):
                cosine = np.frombuffer(raw[:fixed_phase.PHASE_SHAPE[0] * fixed_phase.PHASE_SHAPE[1] * 4],
                                       dtype="<f4").copy()
                cosine[0] = np.nan
                data = cosine.astype("<f4").tobytes()
                raw[:len(data)] = data
                doc["tensors"]["cosine"]["sha256"] = ref.sha256(data)
                doc["payload"]["sha256"] = ref.sha256(raw)

            altered, digest = _clone_and_mutate(original, directory, "altered-nan", authenticated_nan)
            with self.assertRaisesRegex(ValueError, "tensor must be finite"):
                fixed_phase.load_input(altered, digest)


if __name__ == "__main__":
    unittest.main()
