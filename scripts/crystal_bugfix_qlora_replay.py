#!/usr/bin/env python3
"""Compile and run one historical Crystal replay in a macOS sandbox."""

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import subprocess
import sys
import tempfile


def profile_for(replay, work, developer, brew):
    roots = [
        replay, work, developer, brew / "Cellar", brew / "opt", brew / "lib",
        brew / "bin", Path("/System"), Path("/usr/lib"), Path("/usr/bin"),
        Path("/bin"), Path("/usr/share"), Path("/private/var/db/dyld"),
    ]
    paths = " ".join(f"(subpath {json.dumps(str(path))})" for path in roots)
    return f"""(version 1)
(deny default)
(allow process-fork)
(allow process-exec {paths})
(allow file-map-executable {paths})
(allow file-read* {paths}
  (literal "/") (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/random")
  (literal "/private/etc/passwd") (literal "/private/etc/localtime"))
(allow file-write* (subpath {json.dumps(str(work))}) (literal "/dev/null"))
(allow file-read-metadata)
(allow ipc-posix-sem (ipc-posix-name-regex #"^/CRYSTAL/[0-9]+$"))
(allow sysctl-read)
(allow signal (target self))
"""


def limit_output():
    limit = 64 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))


def run_stage(name, command, replay, logs, env, profile, timeout):
    with (logs / f"{name}.log").open("wb") as log:
        with subprocess.Popen(
            ["/usr/bin/sandbox-exec", "-f", str(profile), *command], cwd=replay, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, preexec_fn=limit_output,
        ) as process:
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code = 124
    (logs / f"{name}.status").write_text(f"{code if code >= 0 else 128 - code}\n", encoding="ascii")
    return code


def verify_sandbox(replay, logs, env, profile, canary):
    probe = """
import errno
import socket
import sys

with open(sys.argv[1], encoding="ascii") as file:
    assert file.read() == "readable"
try:
    with open(sys.argv[2], encoding="ascii"):
        raise AssertionError("canary outside allowed roots was readable")
except PermissionError:
    pass
try:
    socket.create_connection(("127.0.0.1", int(sys.argv[3])), timeout=1).close()
except OSError as exc:
    if exc.errno not in (errno.EACCES, errno.EPERM):
        raise
else:
    raise AssertionError("outbound network was reachable")
print("Sandbox restricted file access and blocked loopback network.")
"""
    readable = profile.parent / "readable"
    readable.write_text("readable", encoding="ascii")
    canary.write_text("private", encoding="ascii")
    with socket.create_server(("127.0.0.1", 0)) as server:
        command = [str(Path(sys.executable).resolve()), "-I", "-c", probe, str(readable), str(canary),
                   str(server.getsockname()[1])]
        return run_stage("isolation", command, replay, logs, env, profile, 20) == 0


def xcrun(tool):
    return subprocess.check_output(["/usr/bin/xcrun", "--find", tool], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("logs", type=Path)
    parser.add_argument("backend", choices=["cpu", "metal"])
    args = parser.parse_args()
    replay, logs = args.replay.resolve(), args.logs.resolve()
    crystal = shutil.which("crystal")
    if sys.platform != "darwin" or not crystal:
        parser.error("macOS, sandbox-exec, and Crystal are required")
    developer = Path(subprocess.check_output(
        ["/usr/bin/xcode-select", "-p"], text=True).strip()).resolve()
    crystal_path = Path(crystal).resolve()
    brew = Path("/opt/homebrew" if os.uname().machine == "arm64" else "/usr/local")
    if len(crystal_path.parents) > 4 and crystal_path.parents[3].name == "Cellar":
        brew = crystal_path.parents[4]
    clang = xcrun("clang")
    clangxx = xcrun("clang++")
    sdk = subprocess.check_output(["/usr/bin/xcrun", "--show-sdk-path"], text=True).strip()

    with tempfile.TemporaryDirectory(prefix="cogni-qlora-replay-") as temp:
        scratch = Path(temp).resolve()
        work = scratch / "allowed"
        work.mkdir()
        env = {
            "PATH": f"{brew}/opt/llvm/bin:{brew}/bin:/usr/bin:/bin",
            "HOME": str(work),
            "TMPDIR": str(work),
            "CRYSTAL_CACHE_DIR": str(work / "crystal-cache"),
            "CC": clang,
            "SDKROOT": sdk,
            # Crystal tempfiles use Time.local. UTC avoids reading the host's
            # time zone database from inside the sandbox.
            "TZ": "UTC",
        }
        profile = work / "sandbox.sb"
        profile.write_text(profile_for(replay, work, developer, brew), encoding="utf-8")
        if not verify_sandbox(replay, logs, env, profile, scratch / "outside"):
            return 2

        executable = work / "regression"
        command = [str(crystal_path), "build", "--no-color", "spec/pilot_regression_spec.cr",
                   "-o", str(executable)]
        if args.backend == "cpu":
            command.append("-Dcpu_only")
        else:
            bridge = work / "bridge.o"
            bridge_cmd = [clangxx, "-c", "src/ml/metal/bridge.mm", "-o", str(bridge),
                          "-std=c++17", "-fobjc-arc", "-fPIC", "-fno-objc-msgsend-selector-stubs"]
            if run_stage("bridge", bridge_cmd, replay, logs, env, profile, 90) != 0:
                return 2
            command.append(f"--link-flags={bridge} -framework Metal -framework Foundation -lc++")

        if run_stage("compile", command, replay, logs, env, profile, 120) == 0:
            run_stage("test", [str(executable)], replay, logs, env, profile, 30)
    return 0


if __name__ == "__main__":
    sys.exit(main())
