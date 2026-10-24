# GLM-5.3-Flash on the V100 engine: port status

This branch starts the port of Project Maya's GLM-5.3-Flash engine into this fork. The port is not
finished. This page records what works, what is missing, and the order of the remaining work.

## Why a port is needed

This engine has native kernels for Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash. It has no GLM code.
Project Maya (mw00/project-maya, MIT) is a fork of this engine where the engine was rewritten for
GLM-5.3-Flash. Its repository history starts with one commit that holds the rewritten engine, so the
GLM work cannot be cherry-picked. The GLM sources are copied file by file.

## What this branch contains

| Part | State |
| --- | --- |
| GLM model, MTP, prefill, rows, window, multi, fast path, DFlash2 (`src/core/glm_*`) | copied, compiled against this fork's headers |
| GLM kernels (`src/kernels/cuda/glm_*`) | copied |
| GLM headers (`include/strata/core/glm_*`, `include/strata/kernels/glm_*`) | copied |
| `GlmExpertHelper` | extracted into `include/strata/core/glm_expert_helper.hpp` and `src/core/glm_expert_helper.cpp` |
| Split CPU rows (`native_gu_rows_split`, `native_down_rows_split`) | added, ggml's per-row dot |
| `GgufFile::get_full` | added |
| `SamplerParams::host_history` | added |
| `gemm_iq_f16_grouped_supported` | added |
| MMQ helpers (`q8_block_bytes`, `to_f16`, `to_f16_supported`, `glu_supported`, `run_glu`, `pool_slab`) | added |
| `convert.cu` in the MMQ library | added, for the FP16 conversion |

The sources are behind `-DSTRATA_GLM_PORT=ON` (default OFF), so a normal build is unchanged. The
default build passes on this PC (CUDA 12.8, sm_70).

## What is missing

The port compiles to the GLM fast path. These layers remain:

1. **Speculative sampling.** Maya rewrote `src/kernels/cuda/sampler.cu` and added `SpecList`,
   `sample_draft` and `sample_verify`. This fork's sampler has no such path. The GLM fast path calls
   all three.
2. **Fast-path glue.** `native_multi_rows_exact` (the split CPU rows use ggml's per-row dot here)
   and small `std::min` type mismatches.
3. **The program layer.** Maya's `glm_pack_generate` (3225 lines) plus 25 `Options` fields
   (`--glm-pack`, `--spec-follow`, `--dump-routing`, and others).
4. **The server layer.** `--glm-pack` in a config, and the GLM request paths.
5. **Fused MMQ SwiGLU.** This fork's ggml-cuda has no `mmq-fused-swiglu` patch. `glu_supported()`
   reports false and the plain product runs. This costs prompt speed, not correctness.

## What this means for V100 speed

The GLM fast path is where Maya's V100 tuning lives: per-GPU expert pools, the CPU lane, the
pipeline windows and the prompt path. When the port links, it brings that tuning with it. The
fork's own work (pair kernels, rows-per-block grids, the startup VRAM plan, idle prefill) stays in
the same binaries and does not conflict.

## How to try it

```
cmake -S . -B build-glm -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GLM_PORT=ON \
      -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
cmake --build build-glm -j10
```

The build stops in `glm_fast_path.cu` at the sampling layer above.

Until the port completes, Project Maya runs GLM-5.3-Flash on this PC: 14.0 tok/s decode and
381 tok/s prefill on an 8192-token prompt, both V100-16GB cards.
