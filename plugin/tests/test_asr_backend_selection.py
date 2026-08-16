"""asr_backend/asr_backend_variant resolution: the fallback chain a blocked
combo goes through, and register()'s end-to-end behavior per backend.
"""

import pytest

from plugin.rocm_accelerator import _resolve_asr_backend, register
from plugin.rocm_accelerator.arch.base import ArchProfile

pytestmark = pytest.mark.usefixtures("settings", "cache_root")


class TestResolveAsrBackend:
    def test_defaults_to_faster_whisper_vulkan(self):
        profile = ArchProfile()
        assert _resolve_asr_backend(profile) == ("faster_whisper", "vulkan")

    def test_honors_an_unblocked_selection(self, settings):
        settings["asr_backend"] = "whisper_cpp"
        settings["asr_backend_variant"] = "hip"
        profile = ArchProfile()

        assert _resolve_asr_backend(profile) == ("whisper_cpp", "hip")

    def test_falls_back_to_vulkan_when_the_variant_is_blocked(self, settings):
        settings["asr_backend"] = "parakeet_cpp"
        settings["asr_backend_variant"] = "hip"
        profile = ArchProfile()
        profile.blocked_asr_backends = frozenset({("parakeet_cpp", "hip")})

        assert _resolve_asr_backend(profile) == ("parakeet_cpp", "vulkan")

    def test_falls_back_to_faster_whisper_when_even_vulkan_is_blocked(self, settings):
        settings["asr_backend"] = "parakeet_cpp"
        settings["asr_backend_variant"] = "hip"
        profile = ArchProfile()
        profile.blocked_asr_backends = frozenset({
            ("parakeet_cpp", "hip"), ("parakeet_cpp", "vulkan"),
        })

        backend, _variant = _resolve_asr_backend(profile)
        assert backend == "faster_whisper"


class TestRegisterDispatchesByBackend:
    """The worker images ship whisper.cpp/parakeet.cpp (Vulkan + HIP variants)
    plus their model files for every arch, gfx803 included, so a
    non-faster_whisper selection resolves to "available" and register() does
    register the ASR provider. In a bare checkout without those binaries the
    same path resolves to "unavailable" and no provider is registered. Either
    way the behavior under test is that register() reaches that conclusion
    without raising - never that the backend actually runs.
    """

    def test_selecting_whisper_cpp_registers_when_available(self, ctx, gpu, settings):
        from plugin.rocm_accelerator import whisper_cpp_backend

        settings["asr_backend"] = "whisper_cpp"

        register(ctx)  # must not raise, with or without the binaries

        assert ("asr" in ctx.analysis_providers) == whisper_cpp_backend.available("vulkan")

    def test_selecting_parakeet_cpp_registers_when_available(self, ctx, gpu, settings):
        from plugin.rocm_accelerator import parakeet_cpp_backend

        settings["asr_backend"] = "parakeet_cpp"

        register(ctx)

        assert ("asr" in ctx.analysis_providers) == parakeet_cpp_backend.available("vulkan")
