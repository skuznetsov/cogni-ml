"""Offline contract tests for the optional Hy-Nomas Luna planner."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.error import HTTPError


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "examples" / "data" / "hy_nomas_photorealistic_drafts.json"
SPEC = importlib.util.spec_from_file_location(
    "hy_nomas_plan", ROOT / "examples" / "hy_nomas_plan.py")
assert SPEC is not None and SPEC.loader is not None
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)
sys.path.insert(0, str(ROOT / "examples" / "support"))
import hy_nomas_draft_results as draft_results


def load_suite() -> dict:
    return json.loads(SUITE_PATH.read_text(encoding="utf-8"))


def model_plan(suite: dict | None = None) -> dict:
    suite = suite or load_suite()
    return {
        "scenes": [
            {
                "id": scene["id"],
                "title": scene["title"],
                "style": planner.STYLE,
                "perspective": planner.PERSPECTIVE,
                "subjects": [{"name": "Main subject", "description": "Clearly visible."}],
                "visual_description": (
                    "Photorealistic continuous footage of the main subject in one "
                    "steady, physically plausible shot."
                ),
                "checks": ["The main subject remains visible."],
            }
            for scene in suite["scenes"]
        ]
    }


class FakeResponse:
    status = 200

    def __init__(self, body: bytes):
        self.body = body
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.body) - self.offset
        result = self.body[self.offset:self.offset + size]
        self.offset += len(result)
        return result


def api_response(plan: dict | None = None, *, status: str = "completed") -> bytes:
    suite = load_suite()
    text = json.dumps(plan if plan is not None else model_plan(suite), ensure_ascii=False)
    return json.dumps({
        "id": "resp_offline_test",
        "model": "gpt-6-luna-2026-10-30",
        "status": status,
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }],
        "usage": {"input_tokens": 12, "output_tokens": 34, "total_tokens": 46},
    }).encode("utf-8")


class PlannerRequestTests(unittest.TestCase):
    def test_dry_run_builds_bounded_request_without_auth_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "fresh.json"
            result = planner.run(SUITE_PATH, str(out), None, "medium", False)
            self.assertEqual(result["status"], "request_validated_not_sent")
            self.assertFalse(result["api_call"])
            self.assertEqual(result["model"], "gpt-6-luna")
            self.assertEqual(result["scene_count"], 6)
            self.assertFalse(out.exists())
            self.assertFalse((root / "fresh.receipt.json").exists())

        suite = load_suite()
        payload = planner.build_request(suite, suite["scenes"], "medium")
        self.assertEqual(payload["store"], False)
        self.assertEqual(payload["reasoning"], {"effort": "medium"})
        self.assertLessEqual(payload["max_output_tokens"], 12_000)
        self.assertNotIn("tools", payload)
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertTrue(payload["text"]["format"]["strict"])

    def test_input_profile_and_scene_ids_are_frozen(self):
        suite = load_suite()
        scenes = planner.validate_input_suite(suite)
        plan = model_plan(suite)
        changed = copy.deepcopy(plan)
        changed["scenes"][0]["id"] = "different"
        with self.assertRaisesRegex(planner.PlannerError, "id/order"):
            planner.validate_model_plan(changed, scenes)
        changed = copy.deepcopy(suite)
        changed["settings"]["fps"] = 30
        with self.assertRaisesRegex(planner.PlannerError, "setting 'fps'"):
            planner.validate_input_suite(changed)
        changed = model_plan(suite)
        changed["scenes"].pop()
        with self.assertRaisesRegex(planner.PlannerError, "scene count"):
            planner.validate_model_plan(changed, scenes)

    def test_reserved_tokens_and_noncanonical_style_are_rejected(self):
        suite = load_suite()
        scenes = planner.validate_input_suite(suite)
        for mutation, message in (
            (lambda scene: scene.update(visual_description="Photorealistic continuous <|Subject1|>"),
             "reserved model token"),
            (lambda scene: scene.update(style="cinematic"), "canonical style"),
        ):
            with self.subTest(message=message):
                plan = model_plan(suite)
                mutation(plan["scenes"][0])
                with self.assertRaisesRegex(planner.PlannerError, message):
                    planner.validate_model_plan(plan, scenes)

    def test_style_and_continuity_are_not_inferred_from_literal_keywords(self):
        suite = load_suite()
        plan = model_plan(suite)
        plan["scenes"][0]["visual_description"] = (
            "One unbroken ten-second camera take of a woman turning toward a window."
        )
        validated = planner.validate_model_plan(plan, suite["scenes"])
        self.assertEqual(validated[0]["visual_description"],
                         plan["scenes"][0]["visual_description"])

    def test_native_prompt_matches_existing_results_contract(self):
        suite = load_suite()
        validated = planner.validate_model_plan(model_plan(suite), suite["scenes"])
        output = planner.make_output_suite(suite, validated)
        self.assertEqual(output["settings"], suite["settings"])
        self.assertEqual([scene["id"] for scene in output["scenes"]],
                         [scene["id"] for scene in suite["scenes"]])
        for scene in output["scenes"]:
            draft_results.validate_native_prompt(scene["native_prompt"], scene["prompt"])
            native = json.loads(scene["native_prompt"])
            self.assertEqual(native["subject_list"][0]["subject_id"], "<|Subject1|>")


class PlannerExecutionTests(unittest.TestCase):
    def test_redirect_cannot_forward_authorization(self):
        request = planner.Request(planner.API_URL,
                                  headers={"Authorization": "Bearer test-secret"})
        with self.assertRaisesRegex(planner.PlannerError, "redirect rejected"):
            planner.NoRedirect().redirect_request(
                request, None, 307, "redirect", {}, "https://other.example/responses")

    def test_one_mocked_http_call_writes_suite_and_redacted_receipt(self):
        captured = {}
        secret = "unit-test-secret-never-output"

        def opener(request, *, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["authorization"] = (
                request.get_header("Authorization")
                or request.headers.get("Authorization")
            )
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(api_response())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "refined.json"
            receipt = root / "receipt.json"
            result = planner.run(SUITE_PATH, str(output), str(receipt), "medium", True,
                                 api_key=secret, opener=opener)
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["api_call"])
            self.assertEqual(captured["url"], planner.API_URL)
            self.assertEqual(captured["timeout"], planner.REQUEST_TIMEOUT_SECONDS)
            self.assertEqual(captured["authorization"], f"Bearer {secret}")
            self.assertEqual(captured["payload"]["model"], "gpt-6-luna")
            self.assertFalse(captured["payload"]["store"])
            refined = json.loads(output.read_text(encoding="utf-8"))
            receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(refined["settings"], load_suite()["settings"])
            self.assertEqual(receipt_data["response_id"], "resp_offline_test")
            self.assertEqual(receipt_data["usage"]["total_tokens"], 46)
            self.assertNotIn(secret, output.read_text(encoding="utf-8"))
            self.assertNotIn(secret, receipt.read_text(encoding="utf-8"))
            self.assertNotIn(secret, json.dumps(result))
            response_path = Path(result["response"])
            self.assertTrue(response_path.is_file())
            self.assertNotIn(secret, response_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt_data["response_sha256"],
                             planner.hashlib.sha256(response_path.read_bytes()).hexdigest())

    def test_existing_outputs_fail_before_api_call(self):
        calls = 0

        def opener(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            return FakeResponse(api_response())

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "already.json"
            output.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(planner.PlannerError, "already exists"):
                planner.run(SUITE_PATH, str(output), None, "medium", True,
                            api_key="test", opener=opener)
            self.assertEqual(calls, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")

    def test_incomplete_refusal_http_and_invalid_plan_fail_without_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out.json"
            receipt = root / "receipt.json"

            def incomplete(_request, **_kwargs):
                return FakeResponse(api_response(status="incomplete"))

            with self.assertRaisesRegex(planner.PlannerError, "did not complete"):
                planner.run(SUITE_PATH, str(output), str(receipt), "medium", True,
                            api_key="test", opener=incomplete)
            self.assertFalse(output.exists())
            self.assertFalse(receipt.exists())
            response_path = root / "receipt.response.json"
            self.assertEqual(json.loads(response_path.read_text())["status"], "incomplete")
            output = root / "refusal-out.json"
            receipt = root / "refusal-receipt.json"

            def refusal(_request, **_kwargs):
                raw = json.dumps({
                    "id": "resp_refusal", "model": "gpt-6-luna", "status": "completed",
                    "output": [{"type": "refusal", "refusal": "not allowed"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                }).encode()
                return FakeResponse(raw)

            with self.assertRaisesRegex(planner.PlannerError, "refused"):
                planner.run(SUITE_PATH, str(output), str(receipt), "medium", True,
                            api_key="test", opener=refusal)
            self.assertFalse(output.exists())
            self.assertFalse(receipt.exists())
            self.assertTrue((root / "refusal-receipt.response.json").is_file())
            output = root / "invalid-plan-out.json"
            receipt = root / "invalid-plan-receipt.json"

            def bad_plan(_request, **_kwargs):
                changed = model_plan()
                changed["scenes"][0]["id"] = "wrong-id"
                return FakeResponse(api_response(changed))

            with self.assertRaisesRegex(planner.PlannerError, "id/order"):
                planner.run(SUITE_PATH, str(output), str(receipt), "medium", True,
                            api_key="test", opener=bad_plan)
            self.assertFalse(output.exists())
            self.assertFalse(receipt.exists())
            self.assertTrue((root / "invalid-plan-receipt.response.json").is_file())

    def test_http_error_body_and_bad_key_are_not_exposed(self):
        secret = "test-key-should-not-appear"

        def failing(_request, **_kwargs):
            raise HTTPError(planner.API_URL, 429, "busy", {}, None)

        with self.assertRaisesRegex(planner.PlannerError, "HTTP status 429") as caught:
            planner._api_request({}, secret, failing)
        self.assertNotIn(secret, str(caught.exception))
        with self.assertRaisesRegex(planner.PlannerError, "invalid whitespace"):
            planner._api_request({}, "bad\nkey", failing)


if __name__ == "__main__":
    unittest.main()
