# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Polaris / GCN 4. Deviations here are forced by hardware limits specific to
this arch, not by its base image - it builds on the same ROCm 7.14 base
(Schaka/rocm-gfx803) as every other arch.

Full findings behind each one: ``docs/ARCH_NOTES.md``.
"""

from types import MappingProxyType

from .base import ArchProfile


class Gfx803Profile(ArchProfile):
    arches = frozenset({"gfx803", "gfx802", "gfx805"})

    # Same env var, same fix class as gfx1201's - CTranslate2's own
    # cub_caching allocator, not anything fork-specific.
    env = MappingProxyType({"CT2_CUDA_ALLOCATOR": "cub_caching"})

    # GCN 4 has no packed FP16: fp16 math runs at a fraction of the fp32 rate,
    # so enabling it costs precision for no speedup.
    fp16_supported = False
