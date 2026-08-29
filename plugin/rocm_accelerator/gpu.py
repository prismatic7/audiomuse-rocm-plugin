# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""GPU arch detection and ONNX Runtime capability probes."""

import logging
import subprocess
from typing import Optional, Tuple

logger = logging.getLogger("plugin.rocm_accelerator.gpu")

_ROCMINFO_TIMEOUT = 10


def detect_arch() -> Optional[str]:
    """Return this machine's GPU arch (``"gfx1030"``, ...), or None if unknown.

    Parses ``rocminfo`` output rather than asking torch/HIP, because this runs in
    a process that later fork()s the workers doing the actual inference: a HIP
    context created here does not survive the fork, and the children would fail
    their first GPU call with a handle that looks initialized but is not. A
    separate process cannot leak a context into this one.
    """
    try:
        out = subprocess.run(
            ["rocminfo"], capture_output=True, text=True,
            timeout=_ROCMINFO_TIMEOUT, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        logger.debug("rocminfo unavailable - GPU arch unknown", exc_info=True)
        return None
    # rocminfo lists every agent, CPUs included; only GPU agents name a gfx ISA.
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
            if name.startswith("gfx"):
                return name
    return None


def available_providers() -> Tuple[str, ...]:
    """The execution providers this image's ONNX Runtime was built with."""
    try:
        import onnxruntime as ort

        return tuple(ort.get_available_providers())
    except Exception:
        logger.debug("onnxruntime not importable", exc_info=True)
        return ()


def faster_whisper_available() -> bool:
    """Whether the image ships a usable faster-whisper.

    Broad on purpose: a half-built CTranslate2 raises out of its extension
    module rather than an ImportError, and either way it cannot be used.

    gfx1150 (Strix Point APU) note: on ROCm 10.1 / this image line, importing
    faster_whisper (CTranslate2's HIP extension) hard-aborts the whole process
    with `LLVM ERROR: support is already registered for analysis: AnalysisName
    (PointerFlowAnalysisResult)` - duplicate LLVM analysis registration between
    CTranslate2's bundled LLVM and the MIGraphX/ORT one, inside one address
    space. It kills the WORKER, not just the load. Even though the import
    succeeds, refuse this backend so lyrics ASR falls back instead of taking
    analysis down (measured 2026-08-29, AudioMuse-AI v3.5.0, MIGraphX develop
    b373a823).
    """
    try:
        import faster_whisper  # noqa: F401

    except Exception:
        return False
    if _ct2_llvm_conflict_expected():
        logger.warning(
            "faster-whisper on this arch/image aborts the process at import "
            "(duplicate LLVM analysis registration); refusing it so ASR falls "
            "back to whisper_cpp/ONNX. Set ROCM_ALLOW_CTRANSLATE2=1 to override."
        )
        return False
    return True


def _ct2_llvm_conflict_expected() -> bool:
    """Arch/ROCm combos where CT2's bundled LLVM collides with MIGraphX's.

    Currently: gfx1150 on this image line (see the docstring above for the
    measured failure). Other arches keep the old behaviour (available=True).
    """
    import os

    if os.environ.get("ROCM_ALLOW_CTRANSLATE2", "").strip() == "1":
        return False
    try:
        return detect_arch() in ("gfx1150", "gfx1151", "gfx1152", "gfx1153")
    except Exception:
        return False
