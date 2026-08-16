"""Profile lookup, the defaults, and what gfx803 declares."""

import pytest

from plugin.rocm_accelerator import arch
from plugin.rocm_accelerator.arch.base import DEFAULT_MIGRAPHX_MODELS, ArchProfile
from plugin.rocm_accelerator.arch.gfx803 import Gfx803Profile
from plugin.rocm_accelerator.arch.gfx1201 import Gfx1201Profile

MIGRAPHX = "MIGraphXExecutionProvider"
ROCM_EP = "ROCMExecutionProvider"
CPU = "CPUExecutionProvider"


@pytest.mark.parametrize("arch_name", ["gfx803", "gfx802", "gfx805"])
def test_polaris_arches_get_their_own_profile(arch_name):
    assert isinstance(arch.profile_for(arch_name), Gfx803Profile)


@pytest.mark.parametrize("arch_name", ["gfx900", "gfx906", "gfx1030", None, ""])
def test_arches_without_a_profile_get_the_defaults(arch_name):
    profile = arch.profile_for(arch_name)
    assert type(profile) is ArchProfile


@pytest.mark.parametrize("arch_name", ["gfx1201"])
def test_gfx1201_gets_its_own_profile(arch_name):
    assert isinstance(arch.profile_for(arch_name), Gfx1201Profile)


def test_defaults_change_nothing():
    profile = arch.profile_for("gfx1030")
    providers = (MIGRAPHX, ROCM_EP, CPU)
    assert profile.fp16_supported is True
    assert profile.supports_model_cache_dir is True
    assert profile.blocked_asr_backends == frozenset()
    assert profile.env == {}
    assert profile.migraphx_options() == {}
    assert profile.migraphx_models(providers) == DEFAULT_MIGRAPHX_MODELS
    assert profile.extra_providers(providers) == ()


def test_every_registered_profile_claims_at_least_one_arch():
    for profile in arch.PROFILES:
        assert profile.arches, f"{profile.__name__} claims no arch"


def test_profiles_do_not_claim_the_same_arch_twice():
    claimed = [name for profile in arch.PROFILES for name in profile.arches]
    assert len(claimed) == len(set(claimed))


def test_lookup_returns_a_fresh_profile_each_time():
    # Nothing may reach one arch's profile through another's.
    assert arch.profile_for("gfx1030") is not arch.profile_for("gfx1030")
    assert arch.profile_for("gfx803") is not arch.profile_for("gfx803")


def test_the_inherited_env_cannot_be_mutated():
    # A profile appending to the inherited default would be setting variables
    # for every other arch as well.
    with pytest.raises(TypeError):
        ArchProfile().env["MIOPEN_DEBUG_CONV_FFT"] = "0"


class TestGfx803:
    def test_declares_no_fp16(self):
        profile = Gfx803Profile()
        assert profile.fp16_supported is False
        assert profile.supports_model_cache_dir is True


class TestGfx1201:
    def test_declares_cub_caching(self):
        profile = Gfx1201Profile()
        assert profile.env == {"CT2_CUDA_ALLOCATOR": "cub_caching"}

class TestApplyEnv:
    def test_sets_the_profiles_variables(self, monkeypatch):
        monkeypatch.delenv("MIOPEN_TEST_KNOB", raising=False)
        profile = ArchProfile()
        profile.env = {"MIOPEN_TEST_KNOB": "0"}

        assert arch.apply_env(profile) == {"MIOPEN_TEST_KNOB": "0"}

        import os

        assert os.environ["MIOPEN_TEST_KNOB"] == "0"

    def test_leaves_an_already_set_variable_alone(self, monkeypatch):
        monkeypatch.setenv("MIOPEN_TEST_KNOB", "operator-choice")
        profile = ArchProfile()
        profile.env = {"MIOPEN_TEST_KNOB": "0"}

        assert arch.apply_env(profile) == {}

        import os

        assert os.environ["MIOPEN_TEST_KNOB"] == "operator-choice"

    def test_stringifies_values(self, monkeypatch):
        monkeypatch.delenv("MIOPEN_TEST_KNOB", raising=False)
        profile = ArchProfile()
        profile.env = {"MIOPEN_TEST_KNOB": 1}

        assert arch.apply_env(profile) == {"MIOPEN_TEST_KNOB": "1"}

    def test_no_env_is_not_an_error(self):
        assert arch.apply_env(ArchProfile()) == {}
