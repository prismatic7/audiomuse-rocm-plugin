# Per-arch findings

Why the arch profiles in `plugin/rocm_accelerator/arch/` look the way they do.
Everything here was reproduced on real hardware; nothing is inferred from
documentation alone.

## gfx803 (Polaris: RX 460–590)

Runs on [Schaka/rocm-gfx803](https://github.com/Schaka/rocm-gfx803)'s ROCm
7.14 base (onnxruntime 1.28.0, MIGraphX `release/rocm-rel-7.14`) — a
from-source ROCR-Runtime/CLR rebuild restores Polaris enumeration, which ROCm
7 otherwise drops outright. This build only compiles ORT's MIGraphX EP
(`--use_migraphx`, no `--use_rocm`), so `ROCMExecutionProvider` does not exist
here at all — every finding below that assumed a working ROCM EP was found
against the older, no-longer-built ROCm 6.4.4 base and is kept only as
historical context; current behavior routes everything through MIGraphX.

### No packed FP16 → `fp16_supported = False`

GCN 4 has no packed FP16 ALUs. FP16 math runs at a fraction of the FP32 rate,
so `migraphx_fp16_enable` buys no throughput and only adds precision risk.
Packed FP16 starts at Vega (gfx900).

### `migraphx_model_cache_dir` → `supports_model_cache_dir = True`

Confirmed on real hardware against the current ROCm 7.14/onnxruntime 1.28.0
base: session creation with `migraphx_model_cache_dir` set succeeds, a
`.mxr` file is written to the cache directory, and output matches the CPU
EP's within floating-point noise. gfx803 now uses the same single-directory
cache path as every other arch (`cache.cache_dir_options`), not the
`per_model_options` file-per-model fallback — that fallback existed only for
the older onnxruntime build (1.21.1) that predated this option and is now
unused by any profile, but stays in `cache.py` for any future EP build that
lacks it.

### CLAP runs on MIGraphX

This build has no `ROCMExecutionProvider` at all, so CLAP has one path: the
MIGraphX EP, same as musicnn. CLAP's audio encoder has a Resize node
carrying an explicit `keep_aspect_ratio_policy` attribute (opset-19 exporter
behavior); whether the current MIGraphX release (`release/rocm-rel-7.14`)
parses that attribute is **unconfirmed against the real CLAP checkpoint** —
hardware testing so far only exercised a synthetic Conv+BatchNorm+ReLU graph
(no Resize node) to validate `migraphx_model_cache_dir` itself. If CLAP fails
to compile, ORT's per-session behavior on an unsupported node is what
determines whether it falls back to CPU cleanly or fails session creation
outright — check that against a real CLAP checkpoint before relying on this.

<details><summary>Historical: ROCm 6.4.4 base, routed CLAP through the ROCM EP (superseded)</summary>

The ROCm 6.4.4 base's MIGraphX threw outright on the Resize node's attribute
(`parse_resize.cpp`: `keep_aspect_ratio_policy is not supported!`) until
patched
([parse-resize-fixes.patch](https://github.com/Schaka/rocm-gfx803/blob/main/rocm6.4.4/patches/migraphx/parse-resize-fixes.patch)),
and putting MIGraphX and the ROCM EP in the same session SIGSEGV'd the whole
worker process (`hip_global.cpp: Module not initialized`) — see
[microsoft/onnxruntime#14679](https://github.com/microsoft/onnxruntime/issues/14679).
So CLAP got routed to `[ROCMExecutionProvider, CPUExecutionProvider]` alone,
gated by `ProviderSpec.disable_optimizers` disabling `ConvActivationFusion`
(MIOpen's Fusion Plan path produced wrong output and could still crash on
that base's MIOpen — cos(CPU, ROCM) 0.24–0.84 with the optimizer enabled vs.
0.98+ disabled). musicnn stayed on MIGraphX throughout since it never hit the
Resize node and MIGraphX benchmarked faster for it anyway.

None of this applies to the current build: no ROCM EP exists to route to, no
`ConvActivationFusion` guard has a provider to guard, and `arch/gfx803.py` no
longer overrides `migraphx_models()`/`extra_providers()` at all.

</details>

### faster-whisper on gfx803: all three compute types now correct

Confirmed on real hardware against the current ROCm 7.14 base's upstream
OpenNMT/CTranslate2 source build (JFK sample, `faster-whisper-small`):
`float16`, `float32`, and `int8_float32` all produce the identical correct
transcript. `LYRICS_WHISPER_FASTER_COMPUTE_TYPE` can select any of the three
without a correctness caveat on this arch.

<details><summary>Historical: ROCm 6.4.4 base with the arlo-phoenix CTranslate2 fork (superseded)</summary>

Found against the old ROCm 6.4.4 base's `arlo-phoenix/CTranslate2-rocm` fork,
which this arch no longer builds (upstream OpenNMT/CTranslate2 source, same
bucket as every other arch, replaced it). Three independent failure modes
were separated on that fork that previously masked one another:

1. **Crash (`Memory access fault … Page not present`), any compute type.**
   CTranslate2's default HIP allocator is the stream-ordered `hipMallocAsync`
   mempool, and kernels faulted on its pages. `CT2_CUDA_ALLOCATOR=cub_caching`
   — which the gfx803 profile still sets — eliminated it.
2. **Spurious `CUDA failed with error out of memory` mid-transcribe.** That
   fork's from-scratch MIOpen Conv1D backend (upstream doesn't have one)
   reported a worst-case ~1.44GB workspace for Whisper's first encoder
   Conv1D. Fixed by a workspace-cap patch specific to that fork's Conv1D
   code, which no longer exists in this build.
3. **Silent wrong output** on `float32` and `int8_float32` — root-caused to
   that base's then-unfixed rocBLAS sgemm Tensile logic, not CTranslate2.
   `float16` was the only compute type that transcribed correctly.

</details>

## gfx1201 (RDNA4: RX 9070 / XT)

- **GPU page faults in MIGraphX-compiled kernels** (`mul_add_kernel` /
  `convert_mul_add_kernel`) have been seen during musicnn/CLAP inference with
  fp16 both on *and* off. Not fp16-specific, so turning fp16 off is not a
  workaround — it only gives up throughput. No profile override for this.
- **CTranslate2's default HIP allocator faults mid-transcribe** (page not
  present), taking the host down with it. Fixed by `CT2_CUDA_ALLOCATOR=cub_caching`,
  which the worker image sets for every arch. Root cause is an upstream ROCm
  LLVM codegen bug, not CTranslate2 —
  [OpenNMT/CTranslate2#2021](https://github.com/OpenNMT/CTranslate2/issues/2021).
  That issue also shows a GPU load failing with "out of memory" on a card with
  free VRAM, which is why `whisper_faster` logs actual free/total VRAM on a
  failed load.

## gfx900 / gfx906 (Vega)

No plugin-side override yet. Both need rocBLAS rebuilt from source and
CTranslate2 compiled for the arch, but that is the worker image's job and is
already handled. FP16 throughput on Vega is untested here; if it turns out not
to pay off, that is a two-line `Gfx9Profile` — see
[ARCH_PROFILES.md](ARCH_PROFILES.md).

## Why arch detection shells out to rocminfo

`register()` runs once in the long-lived RQ worker process, which then forks a
child per job. A HIP context does not survive `fork()`: initializing one in the
parent (which `torch.cuda.get_device_properties()` does) leaves every child with
a driver handle that looks initialized but is not, and the child's first real GPU
call fails with a generic error — `hipMemGetInfo` raising `Failed getting
available memory: invalid argument` was the actual symptom. Same root cause as
the "Cannot re-initialize CUDA in forked subprocess" warning seen elsewhere.

`rocminfo` is a separate process with its own address space, so parsing its
output detects the arch without initializing anything here, and GPU init still
happens for the first time in the job's own child process.
