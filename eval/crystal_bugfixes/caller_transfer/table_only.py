"""Remove only the public contract from the primary given-table requests."""

import argparse
from pathlib import Path

import probe


def run(primary, out):
    native = probe.native
    fixtures = native.read(primary / "fixtures.json")
    seal = native.read(primary / "protocol.json")
    assert probe.transfer.fingerprint(fixtures) == seal["fixtures"]
    for path, digest in seal["files"].items():
        assert probe.runner.digest(Path(path)) == digest
    adapters = {"parent": probe.runner.ORIGINAL_ADAPTER,
                **{k: probe.transfer.EXPERIMENT / "train" / k / "adapter.bin" for k in ("own", "peer")}}
    assert all(probe.runner.digest(p) == seal["pins"][k] for k, p in adapters.items())
    assert probe.transfer.pins()["model"] == seal["pins"]["model"]
    native.keep(out / "protocol.json", dict(primary_sha256=probe.runner.digest(primary / "protocol.json"),
                results_sha256=probe.runner.digest(primary / "results.json"),
                script_sha256=probe.runner.digest(Path(__file__)), sampling=seal["sampling"],
                intervention="delete only the leading issue and its blank line from table_repair"))
    _, runtime, recovery = probe.runner.load_runtime()
    results = []
    for arm, adapter in adapters.items():
        dest = out / "runs" / arm
        worker = probe.runner.ArmWorker(runtime.ev.evaluator, [str(probe.runner.SAMPLED_NATIVE), "generate",
            str(probe.runner.MODEL), str(adapter), str(probe.transfer.MAX_NEW), "4"],
            dest / "worker", dest / "worker.log", 240)
        cache = {}
        try:
            for row in fixtures:
                key = "/".join((row["family"], row["variant"]))
                prior = native.read(primary / "runs" / arm / key / "table_repair/request.json")
                system, user = prior["messages"]
                prefix = row["issue"] + "\n\n"
                assert user["content"].startswith(prefix)
                messages = [system, dict(role="user", content=user["content"][len(prefix):])]
                request, count = native.checked_request(messages, prior["id"])
                request_hash = probe.transfer.fingerprint(request)
                if request_hash not in cache:
                    cache[request_hash] = native.sample(worker, recovery, request, dest / request_hash)
                response = cache[request_hash]
                loc = dest / key
                score = probe.score_patch(row, probe.normalize(response), loc / "checks")
                result = dict(arm=arm, case=key, request_hash=request_hash, raw=response["answer"],
                              prefix_tokens=count, truth=row["truth"], score=score)
                native.write(loc / "result.json", result)
                results.append(result)
            assert len(cache) == 2
        finally:
            assert worker.close() == 0
    native.write(out / "results.json", dict(protocol_sha256=probe.runner.digest(out / "protocol.json"),
                                            rows=results))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--primary", type=Path, default=probe.OUT)
    cli.add_argument("--output", type=Path, default=probe.OUT / "table-only")
    args = cli.parse_args()
    run(args.primary.resolve(), args.output.resolve())
