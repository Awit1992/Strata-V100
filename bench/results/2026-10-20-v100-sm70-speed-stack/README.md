# Dual-V100 IQ3_S: the measured Volta profile

Measured on 2026-10-20 by the Strata-V100 fork. This report is a **community diagnostic of the engine's
internals**, not the standard Strata benchmark protocol. It states what was tested, what the engine's own
timing lines report, and the main limitation: one rig, one model, one layer split.

## Hardware and software

- 2 x Tesla V100 PCIe 16 GB, compute capability 7.0. GPU 0 on a PCIe 3.0 **x4** link, GPU 1 on **x16**.
  The cards reach each other through the host bridge; there is no NVLink.
- AMD Ryzen 5 3600, 6 cores / 12 threads, AVX2, no AVX-512. 94 GiB of RAM. The storage holds the model.
- Linux, NVIDIA driver 580.178.04, CUDA 12.8, the fork's `sm_70` source build (`-DCMAKE_CUDA_ARCHITECTURES=70`).
- The engine runs alone on the machine. No power limit is set; the cards are at their 175 W default.

## Model and configuration

- Qwen3.8-Flash-Next GSQ-RCO **IQ3_S**, the regular pack (`$MODEL_ROOT/packs/iq3_s`), the two GGUF shards of
  `$MODEL_ROOT/models/IQ3_S`, the shared vision projector of the same folder.
- 524,288-token context, **int8** KV, the MTP draft (`--spec 8 --spec-min-p 0.70`), vision on, greedy decoding
  (`temperature=0`).
- Layer split **16**: GPU 0 runs layers 0-15 on the x4 link, GPU 1 runs layers 16-47 on x16.
  `--pipeline-windows 2`, `--expert-cache auto`, `--prefill auto`, `--vram-reserve-mib 750`,
  `--conversation-cache-mib 25600`, `--conversation-cache-slots 32`.

```text
engine/strata --serve --pack $MODEL_ROOT/packs/iq3_s --native $MODEL_ROOT/models/IQ3_S/...-00001-of-00002.gguf
  --ple-gguf $MODEL_ROOT/models/IQ3_S/...-00002-of-00002.gguf --expert-profile $MODEL_ROOT/data/expert-profile.bin
  --expert-cache auto --prefill auto --spec 8 --mtp $MODEL_ROOT/mtp/rt --max-context 524288 --rope-scaling yarn
  --rope-scale 2 --kv int8 --spec-min-p 0.70 --vision --vram-reserve-mib 750 --conversation-cache-mib 25600
  --conversation-cache-slots 32 --pipeline-windows 2 --layer-split 16
```

## Method

Each arm is a fresh engine with the same binary, model and configuration and one change. The script is
[measure.py](measure.py); it speaks the engine's own stdin/stdout serve protocol, so the prompt tokens are
exact and the timings are the engine's own.

- The **prefill** arms read a 32,768-token prompt once, with the conversation cache off, so every token is read
  from scratch. The rate is the engine's `prompt_ms`.
- The **decode** arms read a 512-token prompt and generate 256 tokens, three repeats after one warm-up
  request. The stable number is the engine's **ms per token** (its `ms/window` divided by its
  `tokens/window`), from `STRATA_DECODE_TIMING=1`. A single request's decode rate moves by more than 10 %
  between identical runs, because the speculative drafter's window structure changes with the timing. The two
  baseline arms of the prompt arms differ by 0.006 %, and the two decode baselines differ by more than every
  candidate's difference, so only a change that moves both numbers is called a change.
- The **profile** arm adds `STRATA_VERIFY_PROFILE=1`, which prints the GPU's own stamps per verify window.

## Results

### The prompt path: the 8,192-token chunk is the best one on this rig

| Prompt chunk | Prompt tok/s (32,768 tokens) | Against the default |
| --- | ---: | ---: |
| **8,192 (the default)** | **1,731.2 / 1,731.1** | - |
| 10,240 (what `auto:16384` fell back to) | 1,606.6 | -7.5 % |
| 4,096 | 1,330.6 | -23 % |
| 2,048 | 791.1 | -54 % |

A bigger chunk borrows more of the expert cache for its buffers, so the cache keeps fewer experts and the
PCIe pass grows. The engine says so at start-up: "a 16,384-token chunk borrows 2,748 of them". A smaller chunk
pays the fixed pass over the experts again for every chunk.

The streamed ring's depth is the other half of the same trade-off. A deeper ring prefetches further over
PCIe, but it is VRAM the expert cache does not keep.

| Streamed ring | Prompt tok/s (32,768 tokens) |
| --- | ---: |
| **192 (the service's setting)** | **1,731.2** |
| 384 | 1,731.0 |
| 512 (the engine's cap) | 1,716.3 |

**Do not raise `STRATA_PREFILL_AUTO_MAX` on this rig.** The general 8,192-token ceiling is the measured one
here.

### The decode window: where the 40 ms goes

`STRATA_VERIFY_PROFILE=1` prints the GPU's own stamps per verify window. One window carries 2.4 to 2.5 tokens
and takes about 40 ms. The second stage (32 layers and the head) is the critical path.

| Stamp | What it waits for | Stage 1 (ms per window) |
| --- | --- | ---: |
| the dense projections, the head, the VRAM experts | the real work | about 20 |
| `waitB` | the PCIe share of the missed experts, copied into the staging buffer inside the layer's chain | 6.5 - 8.0 |
| `waitCPU` | the CPU pool's rows for the layer | 4.2 - 5.2 |
| `waitA` | the host's plan for the layer (the doorbell handshake) | 1.4 - 1.6 |

The first stage (16 layers) takes 11.5 ms per window, of which `waitA` is 2.7 ms. A decode rate of about
70 tok/s against a bandwidth floor of about 10 ms per window is not a bandwidth problem. It is these three
waits. The PCIe traffic behind `waitB` is about 82 MiB per window for the missed experts, plus about 45 MiB
per window for the adaptive tier's swaps.

The engine's own summary of the same window: 27.9 experts per layer-window from VRAM, 2.3 from the CPU pool
and 0.9 over PCIe, with 23.6 adaptive swaps per window.

### The decode candidates: none of them moves the needle

| Arm | Decode ms per token | Pooled tok/s |
| --- | ---: | ---: |
| baseline | 15.60 - 15.63 | 66.7 - 68.9 |
| baseline again | 15.60 | 68.9 |
| upstream's six-switch Volta stack (`SIDE_MIXER=2`, `GR_UP75`, `B6_MMVQ_ROWS`, `GU_SWIGLU`, `QSA_ROWS`, `ATTN_DAILY`) | - | 67.0 |
| upstream's thirteen-switch stack (the six plus `GR_NORM_FAST`, `ARGMAX_ROWS_SM70`, `QIL`, `EXPDEC`, `GR_DOWN_TG`, `DEQUANT_WIDE`) | - | 68.1 |
| `--pcie-mode dma` (the PCIe share copied beside the CPU's work) | 19.07 | 61.1 |
| `--pcie-frac 0` (every miss to the CPU pool) | 15.37 | 64.0 |
| `--pool-workers 10` (the SMT siblings of the five pool workers) | 15.20 | 66.9 |
| `--adapt-swaps 24` (a quarter of the adaptive tier's PCIe traffic) | 16.34 | 64.6 |
| `--adapt-every 8` | 14.86 | 70.5 |
| `--adapt-every 0` (the adaptive tier off) | 15.46 | 66.5 |

**Upstream's Volta stack does not help on this rig.** It was measured on a dual V100 32 GB with a Ryzen 9
9950X, a 24/24 layer split and `--spec 4 --spec-min-p 0.5`. On the 16 GB cards with a six-core Ryzen 5 the
same switches are inside the noise. That upstream PR is a draft that asks for a maintainer's scope decision;
this measurement is a reason to measure before pulling it in.

`--pcie-mode dma` is clearly worse here. The DMA's landing signal is a host callback, and the callback's
latency on a six-core host is larger than the copy it waits for.

### The VRAM reserve: the 750 MiB the service keeps is the right one

The reserve is the VRAM the engine leaves free on each card. A smaller reserve gives the expert cache more
slots, and more resident experts mean fewer misses over PCIe. On this rig that does not pay:

| VRAM reserve | Expert cache (CUDA 0) | Decode ms per token |
| --- | ---: | ---: |
| **750 MiB (the service's setting)** | 6,057 slots | 14.08 |
| 300 MiB | 6,308 slots | 14.79 |
| 100 MiB | 6,420 slots | the engine does not start: 20 MiB of VRAM free, and the window graphs do not fit |

The 300 MiB arm is 4 % more cache and no faster. The 100 MiB arm shows why the reserve is there: the window
graphs are captured on first use and need room too.

### The system side, and what it can and cannot do here

The question behind this report was whether the RAM-to-GPU path can be made cheaper without touching the
kernels.

| Item | State on this rig | Changeable here? |
| --- | --- | --- |
| the expert arena in RAM | `cudaHostRegister PORTABLE` (page-locked) and `MADV_HUGEPAGE` requested, so the copies are real DMAs | done by the engine |
| GPU persistence mode | enabled on both cards | already on |
| the cards' PCIe links | GPU 0 negotiated **x4**, GPU 1 **x16**; the H2D probe measures 2.9 GB/s and 13.1 GB/s | hardware; the engine scales its PCIe share to the measured bandwidth |
| the CPU's frequency governor | `powersave` with `amd_pstate=active` | **no**: `cpufreq` and `sysfs` are not writable by the service user |
| the NVIDIA driver's interrupt affinity | both cards' IRQs are on **one** logical CPU, where the driver's host-callback thread also runs | **no**: `/proc/irq/*/smp_affinity` needs root |
| ASPM and the PCIe max-payload size on the x4 slot | `lspci` cannot read the PCI config space as a non-root user | **no** |

The engine's numbers put the decode's PCIe traffic at about 127 MiB per window. That is a bandwidth problem,
not a latency problem: a better governor or a better interrupt affinity takes microseconds off the setup of a
transfer. It cannot move 8 ms per window.

### The Volta master switch on this rig

The engine's Volta master switch (`STRATA_SM70_TABLE=1`) selects the interleaved dense table, the expert
mode 8 layout and the latency-hidden hyper-connection up projection. An A/B of the engine without the table
against the engine with it, both with the switch unset in the environment, four fresh engines, alternating:

| Engine | Decode ms per token (two arms) |
| --- | ---: |
| without the table | 15.17 / 14.84 |
| with the table | 14.92 / 15.42 |

The table is inside the noise of this rig's own run-to-run spread. The change is therefore **neutral in
speed** here: it moves a switch that the service set by hand into the engine's own defaults, so that a
Volta card gets the measured kernels without a hand-set environment variable.

## The fork's V100 bench, A/B

`bench/run_v100_prefill.py` (the fork's own harness: the engine's stdin/stdout protocol, exact prompt token
counts, a fresh engine per arm, the cards cooled to 55 C between the requests). Four prompt lengths, five
repeats each, 128 generated tokens per request, greedy. The arms are the released engine, the PR's engine
with its default settings, and the PR's engine with `--pcie-mode dma`.

**Prefill, prompt tok/s (the median of five):**

| Prompt tokens | before | after | after (DMA path) |
| --- | ---: | ---: | ---: |
| 2,048 | 449.3 | 449.3 | 449.3 |
| 8,192 | 1,149.0 | 1,148.7 | 1,148.8 |
| 16,384 | 1,489.1 | 1,489.6 | 1,484.5 |
| 32,768 | 1,724.6 | 1,725.0 | 1,721.7 |

The prompt path is not touched: the three arms agree to 0.3 %, and every arm's **first generated token is the
same** at every length (the harness's parity signal).

**Decode, tok/s (the median of five per length, and the pooled rate over all twenty rows):**

| Prompt tokens | before | after | after (DMA path) |
| --- | ---: | ---: | ---: |
| 2,048 | 68.5 | 70.5 | 75.2 |
| 8,192 | 70.9 | 77.2 | 76.7 |
| 16,384 | 68.4 | 64.0 | 63.9 |
| 32,768 | 65.0 | 67.2 | 64.1 |
| **pooled (20 rows)** | **66.72** | **68.92** | **69.26** |

**How to read the decode column.** The spread inside one arm is larger than the difference between the arms:
the 2,048-token length alone runs 51 to 83 tok/s across five requests of the *same* engine. The engine's
output text is not reproducible from run to run (the adaptive tier's copies decide which window computes a
swapped-in expert on the GPU and which on the CPU, and the two round differently), so a request's routing, and
with it its speed, moves with the timing. The per-length medians are therefore not decisive; the pooled rate
over twenty rows is the number to compare, and it is **+3.3 %** for the PR's engine and **+3.8 %** with the
DMA path. The dedicated decode measurement (a 512-token prompt, 256 generated tokens, four repeats per arm,
alternating) gave **+8.1 %** for the DMA path; the difference between the two shapes is the workload: a short
answer after a long prompt spends more of its time in the attention over the long KV, where the PCIe share of
the experts matters less.

## The before/after table

The same rig, the same model, the same configuration, one change: the engine's **DMA path with the device-raised
flag** (`--pcie-mode dma`) instead of the copy kernel in the layer's chain.

| Measurement | Before (the released engine, the copy kernel) | After (the device flag, the DMA path) | Change |
| --- | ---: | ---: | ---: |
| **Prefill**, 32,768-token prompt | 1,730.8 tok/s | 1,729.9 tok/s | -0.05 % (the prefill path is not touched) |
| **Decode**, ms per token | 15.39 / 15.79 | 14.58 / 14.53 | **-6.6 %** |
| **Decode**, pooled tok/s | 66.5 / 65.8 | 72.9 / 70.2 | **+8.1 %** |

Two arms per cell, four requests of 256 tokens each per arm, alternating between the arms. The prefill numbers
are a single 32,768-token prompt per arm; the prefill rate repeats to 0.006 % on this rig, so the two cells
are the same measurement.

### Why the DMA path was slower before, and what the device flag changes

The decode's second stage waits 6.5 to 8.0 ms per window for the PCIe share of the missed experts: the copy
of the missed experts into the staging buffer sits **inside the layer's chain**, between the VRAM experts and
the PCIe experts. The engine's `--pcie-mode dma` copies the same blobs on a second stream, beside the layer's
compute, so the wait is gone.

That path was slower than the copy kernel, because the copy's landing signal was a **host callback**
(`cudaLaunchHostFunc`): the driver's callback thread sits between the copy and the waiting graph, and its
latency on a six-core host that also runs the expert pool is larger than the copy itself. Measured: **19.07 ms
per token** with the host callback, against the copy kernel's 15.6.

The device can raise the flag instead. A one-thread kernel on the copy stream runs after the copy (the stream
orders them) and writes the mapped flag with a system fence, so the graph that spins on the flag sees the
value without the host calling into the driver inside a window. The engine already uses the same pattern for
the split hand-off (`handoff_publish`) and for the doorbell. With it, the DMA path measures **14.56 ms per
token**: 24 % faster than the same path with the host callback, and 6.6 % faster than the copy kernel.

The copies are the same bytes in the same order, so the experts the GPU reads are the same. The engine's
output text is not bit-reproducible from run to run (the adaptive tier's copies decide which window computes
a swapped-in expert on the GPU and which on the CPU, and the two round differently), so the check is the
engine's own test suite, not a text comparison.

**The limit to know about:** the DMA path queues a copy to a card whose window spins on a host flag, and that
is the pattern of upstream issue #31 (a copy that blocks in the driver until the window ends). The upstream
therefore keeps the copy kernel as the default. The device-raised flag removes the host callback from the
window, but the host's `cudaMemcpyAsync` remains, so the deadlock pattern is not gone: keep the copy kernel
unless the engine's stall watchdog (300 s) is acceptable as the backstop.

### The next step, with the measurement that says where it is

The copy itself is 1.6 MiB per layer, about 123 us at the x16 link's 13 GB/s, and the layer's compute is
about 450 us. So the copy fits inside the layer's compute and only has to be *moved off* the chain to
disappear. The safe way is a second stream **inside the captured graph**: the copy kernel on the copy stream
after the plan's flag, the layer's PCIe group waiting on an event, and the staging buffer's reuse ordered
with a per-group event (the copy of layer L waits for the PCIe group of layer L-2). That keeps every copy
inside the graph, so no host call can block in the driver, and it would remove the whole 6.5 to 8.0 ms.



1. **`src/core/verify.cpp` and `src/kernels/cuda/verify_kernels.cu`: the device-raised flag.** The DMA path's
   landing signal is a one-thread kernel on the copy stream instead of a host callback. The path measured 19.07
   ms per token with the host callback and 14.56 with the device flag.
2. **`src/core/nv_defaults.cpp`: a Volta defaults table.** The sm_86 and sm_60 tables hold the switches that
   are bitwise the default engine's output and faster on that card. The sm_70 table holds the Volta master
   switch `STRATA_SM70_TABLE=1` (the interleaved dense table, the expert mode 8 layout and the latency-hidden
   hyper-connection up projection), which the fork's service had to set by hand. `STRATA_SM70_PERF=0` turns the
   table off, and a switch the user set is never overridden.
3. **`src/core/nv_defaults_test.cpp`**: the table's test (the architecture dispatch, the opt-out, the user's
   own setting kept, idempotence).
4. **`docs/NVIDIA_V100.md`**: the profile and the opt-outs.
5. **`service/strata-v100.service.d/`**: the measured service profile, as installed files.

The model, the quantisation, the KV format, the context, the L2/L3 conversation tiers and the layer split
are unchanged.

## Correctness and limitations

- The engine's own output is the check: every arm read the same prompt and generated the same text (the
  arms' token ids are in the JSON files of this directory).
- One model (IQ3_S), one quantisation, one context (524,288), int8 KV, one layer split (16) and one pipeline
  depth (2). Another split, a single card or another quantisation can rank the arms differently.
- The decode arms' tok/s is noisy. The ms per window and the ms per token are the numbers to compare.
- No claim is made for cards other than compute capability 7.0.
- The raw arms are in this directory: one JSON file per arm, with the private paths replaced.
