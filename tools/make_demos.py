#!/usr/bin/env python3
"""Regenerate the README demo GIFs.

    python tools/make_demos.py
    LUA="C:\\Program Files\\Lua\\lua54.exe" python tools/make_demos.py

Needs the exported checkpoint, so it is not part of any offline check:

    python tools/export_mamba.py
    python tools/reference.py 'Mamba is' --output reference.ref

Two demos:

  demo-generate.gif   a real 30-token greedy run
  demo-validate.gif   the parity check against PyTorch

The validation output is 77 lines -- too tall to read as a GIF -- so it is
ELIDED, not re-run or rewritten: the middle layers are dropped and the cut
is marked on screen. The full transcript is committed at
examples/validation.txt, and this reads that file rather than producing its
own numbers, so the GIF cannot show a result the repo does not also ship.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LUA = os.environ.get("LUA", "lua54")
LUA_Q = LUA if " " not in LUA else '"%s"' % LUA

CKPT = os.path.join(ROOT, "mamba-130m.lmb")
TOKENIZER = os.path.join(ROOT, "tokenizer.lmt")
VALIDATION = os.path.join(ROOT, "examples", "validation.txt")

KEEP_HEAD = 8      # commands, blank line, first layers
KEEP_TAIL = 4      # deepest layer, worst, verdict


def trimmed_validation():
    """The committed transcript, middle elided, cut marked."""
    with open(VALIDATION, encoding="utf-8") as fh:
        lines = [l.rstrip("\n") for l in fh]
    if len(lines) <= KEEP_HEAD + KEEP_TAIL:
        return lines
    dropped = len(lines) - KEEP_HEAD - KEEP_TAIL
    return (lines[:KEEP_HEAD]
            + ["    ... %d more layer comparisons ..." % dropped]
            + lines[-KEEP_TAIL:])


def render(cmd, out, hold="3.5"):
    return subprocess.run(
        [sys.executable, os.path.join(HERE, "make_gif.py"),
         "--cmd", cmd, "--out", os.path.join("assets", out), "--hold", hold],
        cwd=ROOT).returncode


def main():
    failed = 0

    if os.path.exists(CKPT) and os.path.exists(TOKENIZER):
        cmd = ('%s main.lua mamba-130m.lmb tokenizer.lmt "Mamba is" 30 0'
               % LUA_Q)
        failed += render(cmd, "demo-generate.gif") != 0
    else:
        print("skip: no mamba-130m.lmb -- run tools/export_mamba.py first")

    if os.path.exists(VALIDATION):
        # Render the already-captured transcript through --text-file rather
        # than shell-escaping 77 lines into a command.
        trimmed = os.path.join(ROOT, "assets", ".validation-trimmed.txt")
        os.makedirs(os.path.dirname(trimmed), exist_ok=True)
        with open(trimmed, "w", encoding="utf-8") as fh:
            fh.write("\n".join(trimmed_validation()) + "\n")
        rc = subprocess.run(
            [sys.executable, os.path.join(HERE, "make_gif.py"),
             "--text-file", trimmed,
             "--out", os.path.join("assets", "demo-validate.gif"),
             "--hold", "3.5"], cwd=ROOT).returncode
        os.remove(trimmed)
        failed += rc != 0
    else:
        print("skip: no examples/validation.txt")

    if failed:
        raise SystemExit("%d demo(s) failed" % failed)


if __name__ == "__main__":
    main()
