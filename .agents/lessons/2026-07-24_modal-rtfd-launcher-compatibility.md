---
date: 2026-07-24
experiment: wan2.1_rtfd_smoke
category: infrastructure
severity: important
---

# Keep Runtime-Cloned Modal Launchers Compatible with Current Tooling

## What Happened

The first RTFD Modal smoke attempts failed before training: a local override
path depended on the launcher's original nested `__file__`, current `uv`
rejected a system install without `--system`, and Hugging Face downloads were
forced through the optional `hf_transfer` package even though it was absent.

## Root Cause

Modal imports the launcher as `/root/modal_rtfd.py` in the remote container,
the launcher upgrades `uv` on every image build, and
`HF_HUB_ENABLE_HF_TRANSFER=1` was stale relative to the installed dependency
set.

## Fix / Workaround

Use a repository-relative local override path, pass `--system` to runtime
`uv pip install` commands, and leave `HF_HUB_ENABLE_HF_TRANSFER` unset so the
installed Hugging Face client selects its supported downloader.

## Prevention

Treat launcher paths as relocatable, make system-package intent explicit for
`uv`, and do not enable optional download backends unless their packages are
installed in the same image. Retain the five-step Modal smoke as the gate
before longer RTFD experiments.
