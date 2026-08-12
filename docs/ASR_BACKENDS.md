# ASR backend findings

Speech-to-text options evaluated against real hardware, standalone
(`local-test/asr_backends/`), outside the plugin, before implementation. See
that directory for the probe scripts and Dockerfiles behind every result
below.

## gfx803 (Polaris: RX 460-590)

### Vulkan: works cleanly for everything

`whisper.cpp` and `parakeet.cpp`, both built with `GGML_VULKAN`/
`PARAKEET_GGML_VULKAN`, produce correct transcripts on this card - no crashes,
no garbage output, no per-arch tuning. Vulkan runs through the host's
RADV/mesa driver, a completely separate stack from ROCm/HIP with no
dependency on rocBLAS, Tensile, or MIOpen.

### HIP/ROCm

- `whisper.cpp` built with `GGML_HIP`: correct transcript, GPU used (`ROCm0`
  backend confirmed in logs).
- `faster-whisper` (CTranslate2): now the plugin's default on this arch, with
  `CT2_CUDA_ALLOCATOR=cub_caching` fixing CT2's default `hipMallocAsync`
  allocator page-faulting. Confirmed correct on real hardware for all three
  compute types - `float16`, `float32`, and `int8_float32` all transcribe the
  JFK sample identically and correctly on the current ROCm 7.14 base. Full
  write-up: `ARCH_NOTES.md`, "faster-whisper on gfx803".

  This arch moved from a ROCm 6.4.4 base with its own CTranslate2 fork
  (`arlo-phoenix/CTranslate2-rocm`, which added a from-scratch MIOpen Conv1D
  backend the fork needed and upstream didn't have) to
  [Schaka/rocm-gfx803](https://github.com/Schaka/rocm-gfx803)'s ROCm 7.14
  base, the same base every other arch uses. CTranslate2 is now built from
  upstream OpenNMT source for this arch too, same bucket as Vega/CDNA
  (gfx900/906/908/90a/942) - no fork, no arch-specific patch. The Conv1D
  workspace-cap patch the old fork needed is gone with it: it patched code
  that only existed in that fork.
- Parakeet-TDT 0.6B via `parakeet.cpp` built with `PARAKEET_GGML_HIP`
  (~20+ layer Conformer encoder + TDT decoder): confirmed **working** on the
  new ROCm 7.14 base - correct transcript, GPU used. The earlier silent-empty
  result below was against the old ROCm 6.4.4 base's rocBLAS/MIOpen, since
  fixed by that base's own kernel-correctness work (see
  [Schaka/rocm-gfx803](https://github.com/Schaka/rocm-gfx803)); not
  re-tested on the intermediate combination (old CTranslate2 fixes, still on
  ROCm 6.4.4) since the base moved on before that mattered.

  <details><summary>Original ROCm 6.4.4 finding (superseded)</summary>

  Silent **empty** output on GPU, `exit 0`, no error. Same input on CPU (same
  binary, no `--device` passthrough): perfect transcript with word-level
  timestamps.

  </details>

### Verdict for gfx803

Vulkan for `whisper.cpp`/`parakeet.cpp` when HIP isn't confirmed working for
them; on the current ROCm 7.14 base, HIP is confirmed working for both
engines and for faster-whisper, so all three ship their HIP variant here too.

## gfx900+ (Vega and newer)

Not part of the gfx803 investigation above - HIP is expected to actually work
correctly here (no history of the Tensile kernel-selection class of bug on
supported/current architectures), so `parakeet.cpp` with HIP is a real backend
option, not just Vulkan.

Confirmed working, smoke-tested against an RX 9070 XT (gfx1201) on the
plugin's own `rocm-migraphx-ort-torch-builder` base image (ROCm 7.14):

- `parakeet.cpp` built with `PARAKEET_GGML_HIP`: correct transcript,
  `ROCm0` backend, no crash, no garbage - the exact opposite of gfx803's
  result with the identical binary/model/build flags. Confirms the gfx803
  bug is arch-specific, not something inherent to HIP + Parakeet.

One packaging note from getting this running: gfx1201's base image is on
ROCm 7.14, which renamed the hipBLAS/rocBLAS dev packages
(`amdrocm-blas-dev7.14` etc.) - the `hipblas-dev`/`rocblas-dev` package names
that work on gfx803's ROCm 6.4 base don't exist there, but aren't needed
either: ROCm 7.14's base image already carries its own BLAS dev headers.

## Docker image implications

`faster_whisper`, `whisper_cpp` and `parakeet_cpp` are all shipped and
selectable via the `asr_backend` setting (see the
[plugin README](../plugin/rocm_accelerator/README.md#settings)). Remaining
notes from the investigation:

- Every backend's runtime dependencies need to be in the image the plugin
  ships (the plugin itself carries no pip requirements) - same pattern
  `docker/Dockerfile` already uses for CTranslate2/faster-whisper.
- **Models should be shared across backends where the same checkpoint
  works for more than one of them** (e.g. the Parakeet gguf checkpoint used
  by both `parakeet.cpp`'s Vulkan and HIP builds is the same file) - no
  reason to bake in duplicate copies of the same weights just because two
  backends can serve them.
- gfx803 only ever needs the Vulkan variant of each backend; gfx900+ images
  can carry HIP variants instead, keeping gfx803 images smaller and avoiding
  shipping a HIP path on that arch that's known broken for anything but
  whisper.cpp.

## Canary models

`parakeet.cpp` only covers the Parakeet family (CTC/RNNT/TDT/hybrid,
0.6B/1.1B/110M, English + multilingual v3) - no Canary support.
[CrispASR](https://github.com/CrispStrobe/CrispASR), a whisper.cpp fork with a
broader ggml model zoo, can run Canary GGUFs (`crispasr -m
canary-1b-v2.gguf`). Not evaluated against any of our arches yet; worth a look
if Canary support is ever needed.
