#!/usr/bin/env python3
"""Redirect a diagnostic target's stderr to a NEW live log, then exec in place.

Usage under run_safe.sh: python3 qwen_live_stderr_exec.py /abs/new.log -- /abs/program args...
Wrapper stderr remains separate. No shell, extra child, retry, or guard changes.
The target must flush its trace records; this does not alter its buffering.
"""
import os
from pathlib import Path
import sys


def main(args):
    if (len(args) < 3 or args[1] != '--' or
            not Path(args[0]).is_absolute() or not Path(args[2]).is_absolute()):
        print('expected /absolute/new.log -- /absolute/program [args...]', file=sys.stderr)
        return 64
    try:
        fd = os.open(args[0], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as error:
        print(f'cannot create exclusive diagnostic log: {error.strerror}', file=sys.stderr)
        return 64
    try:
        os.dup2(fd, 2, inheritable=True)
    finally:
        if fd != 2:
            os.close(fd)
    try:
        os.execv(args[2], args[2:])
    except OSError as error:
        print(f'cannot exec diagnostic target: {error.strerror}', file=sys.stderr, flush=True)
        return 126


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
