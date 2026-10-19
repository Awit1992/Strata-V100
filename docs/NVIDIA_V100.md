# NVIDIA Tesla V100 / Titan V (Volta, sm_70): the community build

Strata's ready-made engine and the support matrix start at compute capability 7.5 (RTX 20). A V100 (7.0) runs the same
engine compiled for it, through the experimental build that issue #236 added. It is **community-tested, not part of the
supported cards**: no ready-made engine is published for it and the numbers below come from one machine.

## Build

CUDA 13 dropped Volta, so the build needs a **CUDA 12.x** toolkit (12.8 was used) and a host GCC that toolkit accepts.

```sh
cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON
cmake --build build --target strata -j
```

Setup does the same by itself for a model on a V100 (the experimental CUDA 12 engine, in `engine-cuda12/`: ready-made
on Windows, compiled with a CUDA 12.x toolkit on Linux; [OLDER_GPUS.md](OLDER_GPUS.md)). For a PC with a V100 and a newer card, a build for both architectures (`-DCMAKE_CUDA_ARCHITECTURES="70;86"`) is the
natural choice (a community report in issue #509 ran an RTX 3080 beside a V100 that way; it was not repeated here). On the
measuring PC a Quadro RTX 4000 (sm_75) sits beside the V100s and was hidden with `CUDA_VISIBLE_DEVICES` because the sm_70-only
build has no code for it.

## What runs on Volta

The flag only lowers the floor (CMake and the runtime device check) and replaces two intrinsics Volta lacks. It does not
select "old" kernels everywhere: which kernel runs is decided per architecture when the engine is built.

| Part of the prompt path | On sm_70 |
| --- | --- |
| MoE experts | opt-in (`STRATA_PF_WMMA=1`, only when every visible card is compute capability 7+; MMQ otherwise): `gemm_iq_f16_grouped`: a group of experts per launch on the FP16 tensor cores, the weights dequantized in shared memory (+10% prompt speed against MMQ's `dp4a` kernels, [bench](../bench/results/2026-10-07-v100-prompt-experts/README.md); FP16 activations instead of MMQ's q8_1, so the answers differ) |
| Dense projections (dequantized weights) | FP16 tensor-core GEMMs (cuBLAS / CUTLASS `s884`) |
| BF16 projections (hyper-connection, router, indexer, ...) | converted to FP16 and run on the FP16 tensor cores (#655, #540; `STRATA_BF16_TC=0`: cuBLAS BF16, an FP32 SIMT kernel on Volta) |
| QSA attention for decode and verify windows (and prompts with `STRATA_PROMPT_ATTN_OLD=1`) | #540's kernel (fewer shuffles, bit-exact; `STRATA_ATTN_PRE75=0`: the one other cards run) |
| QSA prompt attention, int8 / FP16 / K8V4 KV | `prompt_attn_v70_kernel` on `mma.m8n8k4` (`STRATA_PROMPT_ATTN_OLD=1`: the decode kernel, one query at a time) |
| QSA prompt attention, Q4_0 KV | the decode kernel |
| QSA block scores | the warp kernel (the tensor-core scorer needs sm_80) |

| Part of a decode / verify window | On sm_70 |
| --- | --- |
| Routed experts in VRAM | gfx906's expert mode 8: the codebook grid and the group's activations in shared memory, SwiGLU fused (`STRATA_EXP_MODE=0`: the CUDA layout other cards run) |
| Dense 2-4 column GEMVs and the head | the interleaved `native_mmvq_il` with a Volta rows table (`STRATA_MMVQ_IL=0`: `native_mmvq`) |
| Hyper-connection read, up projection | gfx906's latency-hidden up kernel (`STRATA_GR_FAST=0`: the plain one) |

All three are bitwise the kernels they replace; on a V100-SXM2 they take a verify window's GPU work from 22.8 to 21.2 ms
([bench/results/2026-10-07-v100-decode-kernels](../bench/results/2026-10-07-v100-decode-kernels/README.md)).

## Measured

One V100-PCIE-32GB, i9-7960X, Unsloth `UD-IQ4_XS` (not a setup model), int8 KV, MTP `--spec 2`; prompt read speed of random-word prompts, tokens/s,
with and without the Volta attention kernel (`STRATA_PROMPT_ATTN_OLD=1`):

| Prompt tokens | old attention path | `prompt_attn_v70_kernel` |
| --- | ---: | ---: |
| 7,194 | 1,010 | 1,164 |
| 28,650 | 1,063 | 1,251 |
| 114,338 | 973 | 1,123 |

Decode was 38-51 tok/s in every run and is not affected. The profile, the parity numbers, the needle test and the limits are in
[bench/results/2026-10-03-v100-prompt-attn](../bench/results/2026-10-03-v100-prompt-attn/README.md).

What the kernel does and does not guarantee: against an FP64 reference its error is about 2-3e-6 (output scale 3.6; the FP32 kernel it replaces: 2e-6; 5e-6 before the hi and lo halves got separate accumulator chains); 15 of 15
needles were found; greedy output was identical on the 28,650- and 114,338-token prompts and differed late in the answer on the 7,194-token one (another
summation order).

## Checking and benchmarking the Volta kernels

```sh
cmake -S . -B build -DSTRATA_PARITY_PROMPT_ATTN=ON ... && cmake --build build --target qsa_prompt_attn_parity
CUDA_VISIBLE_DEVICES=0 ./build/qsa_prompt_attn_parity 32768 2048 5     # synthetic, no model; exit 0 = PASS
```

It compares the tensor-core attention with the FP32 kernel and an FP64 host reference and times both. The Q4_0 cases are skipped below
sm_80.

## Limits

- A layer split over two V100 was not measured with this kernel. The cards work in turn on a single request, so a split adds expert-cache room, not prompt speed.
- sm_60 (Pascal) is covered by the same flag, and community runs exist: two Tesla P40 with a layer split (#1028, bench
  [2026-10-05-community-2x-p40](../bench/results/2026-10-05-community-2x-p40/README.md)), a Tesla P100 alone and with an RTX 2070 SUPER as
  the expert-helper card (#1069: 30.5 tok/s alone, 35.8-38.4 with the helper), and a V100 + P100 pair (#1079: +6-11% decode from 0.1.39 to
  0.1.40 with the P100 as helper cache). One reporter's numbers each; this page's own kernel measurements are V100 only.
- Volta's tensor cores take FP16 only: a model's BF16 weights are converted, and a weight outside FP16's range would saturate (none did in
  the check above).

## Selected v0.1.41 kernels

Set `STRATA_SM70_TABLE=1` to select the interleaved dense-matvec table,
native expert mode 8, and latency-hidden HC up projection on Volta.
The interleaved 2-4 column matvec with a Volta rows table and the
latency-hidden norm/up are bitwise the default kernels' output (GPU work
per verify window 22.8 -> 21.2 ms on a V100-SXM2 in the contributor's
measurement); they become the default once the contributor has confirmed
them on a V100 with the 0.1.41 code. The individual settings
`STRATA_MMVQ_IL`, `STRATA_EXP_MODE`, and `STRATA_GR_FAST` take precedence.
Without the master setting, Volta retains the previous decode selection
(what 0.1.40.3 ran).

Set `STRATA_EMB_REUSE_ACCOUNT=1` to let the prompt planner use the embedding
buffer space reused by half outputs. Default accounting remains conservative.
On the measured 8K chunk, this option reduces each stage's prompt loan by
about 80 MiB. It does not establish a general prompt-speed gain.

For the measured dual-V100 IQ3_S arena, `STRATA_PREFILL_CPU_SHARE=auto`
and `STRATA_PREFILL_STREAM_MIN=3072` select CPU-assisted staging below
3,072 tokens. The larger-chunk option can change prompt rounding.
Do not enable it for batch slots or assume the same gain on other models.

**Retained CPU and Volta changes.** These changes keep the model formats, the
int8 KV cache at a 524,288-token context, vision, MTP, the L2/L3 conversation
tiers and the NVMe-backed n-gram table. No 20% end-to-end gain is proved yet,
and no performance guarantee follows.

On sm_70 the prompt path's QSA top-k may use the active-block bound of a
prefill. The bound selects the register kernel when the prompt's active blocks
fit it, even when a long context capacity makes the score-row stride large.
Decode and captured graphs pass no bound and keep the capacity rule.

On a CPU with AVX2, no AVX-512 and no AVX-VNNI, the AVX-2 expert kernel serves
IQ3_XXS, IQ3_S and IQ2_S rows for one token (NT1) by default. NT means the
tokens in one group; NT1 is a group of one token. The IQ3_S and IQ2_S decodes
use 32-bit loads to read packed index bytes, which shortens the grid assembly.
These rows reduce in ggml's own float order and apply the format normalization
factor after the reduction, so a row is ggml's dot product bit for bit,
including subnormal products. IQ4_NL down rows are unchanged. An explicit
`STRATA_IQ_MT_MIN` keeps its rule.

After each non-host physical core has a worker, the pool pins additional
workers to the allowed SMT siblings of the worker cores. SMT (simultaneous
multithreading) means a physical core's second logical processor. The host
core's sibling comes last. The default placement is unchanged: five workers and
the host thread on the measured six-core CPU.

See the [release selection](../improvements/RELEASE-v0.1.41-V100.md)
and [prefill/decode measurements](../benchmarks/v100-iq3_s-v0141-2026-10-08.md).
The options do not change an installed service until its environment and
engine are changed. Do not enable stage-buffer pinning for IQ3_S.

## Shared-root return latency

The dual-V100 protected study measures repeat shared-root HTTP latency
at 1,799.7 ms before the cache changes and 902.6 ms after them.
The engine retains a disk-restored root in the bounded RAM cache.
The unrelated small-chat and live-continuation cases do not improve
in this sample. Do not treat this result as a general throughput gain.
See the [agentic latency measurements](../benchmarks/v100-iq3_s-prefill-wmma.md#agentic-short-turn-latency-2026-10-10).

## Selected v0.1.43 performance changes

The [release review](../improvements/RELEASE-v0.1.43-V100.md) lists the selected
ports and the changes that do not apply to Volta.

- `STRATA_ROUTE_RESIDENT=1` selects resident-biased routing. Its route swap now
  uses one warp per token. The option remains off by default.
- Linux CUDA can overlap adaptive expert copies with the next decode window.
  Layer splits remain excluded. Use `STRATA_ADAPT_OVERLAP=0` to turn it off.
  Resident RAM sources require `STRATA_ADAPT_OVERLAP_RESIDENT=1`.
- The automatic route tail skip stays off at 90% expert-cache residency or
  more. `STRATA_TAIL_SKIP_MAX_CACHED` changes this limit. An explicit
  `STRATA_ROUTE_TAIL_SKIP` takes precedence.
- Linux CUDA uses the measured PCIe share for the canonical Q2_0 pack.
  Native packs already used a measured share.

These controls do not establish a speed gain for every V100 configuration.
See the [verification report](../benchmarks/v100-upstream-v0.1.43-2026-10-11.md).

## Dual-PCIe pipeline service profile

The [recent upstream PR review](../benchmarks/v100-upstream-pr-inventory-2026-10-15.md)
includes the V100 pipeline report in upstream PR #2395. The tested local
profile uses GPU 0 on PCIe x4 and GPU 1 on PCIe x16. Keep the GPU order
`0,1` and the explicit layer split `16`. Do not use an equal-link assumption.

With `STRATA_KV_GROW=1`, use a build with the
[pipeline KV-growth repair](../benchmarks/v100-pipeline-kv-growth-fix-2026-10-16.md).
The original pipelined decode loop did not grow the mapped cells during a
long response. It could fail with a CUDA illegal memory access even when
the configured context had space left. The repair waits for speculative
rollback and adaptive copies before growing both stages, then resumes
pipelining. Short-output speed checks alone do not test this boundary.

The service template uses a separate private configuration,
`strata-iq3_s-pipeline.json`. Prepare it from the existing configuration:

```sh
python3 - <<'PY'
import json
from pathlib import Path

source = Path("strata-iq3_s-arena.json")
target = Path("strata-iq3_s-pipeline.json")
config = json.loads(source.read_text())
args = config["args"]
if "--pipeline-windows" in args:
    args[args.index("--pipeline-windows") + 1] = "2"
else:
    args.extend(["--pipeline-windows", "2"])
target.write_text(json.dumps(config, indent=2) + "\n")
target.chmod(0o600)
PY
```

Keep both configurations private. Do not commit credentials, model paths,
or conversation data. Install the service template only after the new
configuration exists. Keep the previous engine and service settings for
rollback. This profile does not change the model, quantization, context
capacity, vision settings, or cache budgets.

`--pipeline-windows 2` needs two GPU stages and MTP. It is not a batch-slot
profile. It reserves extra memory for a second verifier and stage snapshots.
Do not apply it to a different machine without measurements. See the
[local evaluation report](../benchmarks/v100-upstream-pr-evaluation-2026-10-15.md)
for results, limits, and reproduction commands.

## Prefix-aware conversation storage

L2 keeps one shared K/V backing for matching conversation prefixes.
A short branch has its own token metadata and changed suffix. It does not
remove the original long continuation. The cache charges shared payload
allocations once, including when it admits a branch or evicts another owner.
Writes detach only the ranges that change. The allocation forecast includes
buffer-directory growth and shared suffix writes.

K/V reuse and recurrent-state restore are separate operations. The engine
can share K/V through a branch point without having a recurrent state there.
It must restore a valid earlier state and process the remaining prompt.
An explicit pin can retain an exact boundary. Ordinary history uses bounded
anchors instead of one permanent state per turn. The same anchor selection
applies to every split stage.
A branch inside the saved input freezes its source archive. A later append
can share that source's K/V without rotating away its original resume state.
The active list also admits a predecode input anchor. Reply rewrites can
resume there instead of loading a parked state from the obsolete reply.

Before it shares saved backing after a rewrite, the engine compares the
retained prefix with the active K/V bytes. This includes live cells in a
partial page and completed indexer rows. It excludes unused cells and the
moving spare row. Equal token IDs alone do not permit reuse. If replay
changes boundary cells, the engine can share a shorter verified prefix.

L3 uses prefix references and K/V delta panels. A branch can reference a
shorter extent of a longer chain without truncating that chain. Bounded
`.states` slots hold retained recurrent, PLE, indexer, and draft state.
New panels do not add an inline state copy on every turn. A sidecar write
failure fails the cache write instead of using an unbounded fallback.
The first stored generated end that later input consumes can retain one
ordinary state slot. Dense input checkpoints do not rotate away this original
resume point. Its marker survives restart without increasing the slot limit.
Legacy inline states remain readable. They occupy disk space until their
chain is evicted or replaced. A cut inside a legacy panel still requires
the reader to verify that complete panel's checksum.

The L2 byte limit, slot limit, and free-RAM floor still apply. L2 capacity
victims can move to L3. Disk limits include chain and sidecar bytes.
Pins consume state storage. Finite budgets can still evict conversations.
These changes do not increase the active context limit or remove long-context
decode costs.

The saved input-prompt boundary identifies an obsolete generated tail.
A client can replace that tail without creating a new conversation on each
turn. A branch before that boundary keeps its source continuation.
A nearby checkpoint is not proof of a terminal rewrite. Records without
boundary provenance use the conservative branch path.
If replay changes cells before the input boundary, replacement requires the
remaining original input tokens and image keys to match. An input edit
still creates a separate branch.

### Current native branch check

A native V100 check used a 300,000-token parent and a 20,008-token branch.
No exact boundary pin was set. The first branch reused 16,384 tokens and
processed the remaining prefix. Twenty growing-input turns then reused the
resident input anchor. Their prompt time was 356 to 407 ms.

After those turns, the short branch reused 20,007 tokens. The original long
continuation reused 300,015 tokens and produced the same output as its first
return. Peak L2 storage was 8,508,704,232 bytes across four parked groups.
No capacity eviction occurred. Archive history stayed at six checkpoints.
These figures describe cache allocations, not process RSS or GPU memory.

A 300,000-token L3-only retention check also passed. This check preceded the
final replay-boundary correction. The engine restarted before the 20
growing-input turns. Both long returns
reused 300,015 tokens. The final short return reused 20,007 tokens.
The store held two chains and 6,189,411,036 bytes in total, including
1,298,695,712 bytes of bounded state sidecars. Allocated disk space was
6,189,428,736 bytes.

Disk-only append prompt time was 7.84 to 9.73 seconds in this check.
Disk-only operation still pays for prefix verification and state writes.
The storage and recovery result does not imply RAM-speed disk reuse.

The final writer was also exercised with K/V and running states captured
from that 300,000-token run. The short branch referenced 16,384 source tokens.
Its chain held 55,581,873 bytes, not a second copy of the matching prefix.
The source chain held 4,580,646,417 bytes. Total storage was 5,226,544,706
bytes with the states retained for this replay. The original 300,015-token
state and the 20,023-token branch restored with exact state payloads and K/V.

The final native writer also passed a 40,000-token parent check with a
20,008-token branch, restart, and 20 growing-input turns. It retained two
chains. After repeated parking, the short chain still referenced 16,384
source tokens and held 55,889,222 bytes. Total storage was 1,851,520,352
bytes, including 1,180,632,496 bytes of bounded state sidecars.
The original parent return reused 40,014 tokens and produced the same output.

A separate final check pinned the 19,998-token boundary. The short chain
referenced all 19,998 tokens and held 429,625 bytes. Generated token IDs and
all nine canonical main-state hash fields matched the RAM reference in
four request cases.

All nine focused conversation and session tests passed. All 84 cache-tool
and prefill-tool tests passed. AddressSanitizer and UndefinedBehaviorSanitizer
checks passed 4,394 RAM assertions and 1,863 disk assertions. The disk checks
include branch preservation, bounded state slots,
restart recovery, missing sidecars, and state-publication failures. A dedicated
check replaces 20 generated tails as the input grows. It verifies the latest
image after restart, the original input anchor, and historical-branch refusal.

### Earlier unbounded-archive checks

The following results used the earlier retention policy. They are historical
evidence, not acceptance results for the bounded policy described above.

A deterministic single-GPU engine check used a 12,293-token synthetic
prompt, eight chat continuations with replaced reply tails, and an unrelated
full prompt read between continuations. One archive retained the chain.
After that full read, the archive grew from 28 to 35 checkpoints. The last
three extensions reused 189,330,432 to 189,578,880 KV bytes and took 193.1
to 193.9 ms. No capacity eviction or host-RAM admission failure occurred.
This check does not establish decode speed at a 340,000-token context.

A single-V100 smoke run used a 12,440-token synthetic prompt, eight short
continuations, and edits before the active checkpoint window. The run used
these limits:

```text
--prompt-cache 3 --prompt-cache-every 512
--conversation-cache-mib 8192 --conversation-cache-slots 2
--conversation-cache-min-tokens 0 --conversation-cache-min-free-mib 1024
--conversation-cache-disk <scratch-directory>
--conversation-cache-disk-gib 32
--conversation-cache-disk-min-free-mib 128
```

The L2 archive retained 31 checkpoints after the eight continuations.
An edit at token 6,144 restored the historical L2 checkpoint. The engine
processed the remaining 6,296 tokens. Three unrelated requests then forced
L2 evictions. An edit at token 7,168 restored an L3 checkpoint. The engine
processed the remaining 5,272 tokens. Both requests produced the same output
and main-model state hashes as their cold-reference requests. The logs showed
L2-to-L3 spills and L3-to-L2 promotion. L2 occupancy stayed below 8 GiB and
two slots.

The installed service also passed an HTTP smoke check with its existing
settings: a 524,288-token context, a 25 GiB L2, 32 cache slots, and the
default active checkpoint count. Ten short chat turns completed. A change
to the fifth user message then restored 124 tokens from L2 for a 293-token
prompt. The earlier continuation reference reused 48 tokens. The runtime
log confirmed an L2 restore, not an L3 restore.

The first restart checks failed because of the host memory, not the cache
code. The host uses non-ECC memory and has no error log, so the fault made
no hardware record. A CPU-only pattern test found one bit that always read
back wrong. Two cold engine runs of the same prompt then gave different
model-state fingerprints, so no state comparison was possible.

The operator removed the two 16 GB modules and kept the two 32 GB modules.
A 54 GiB pattern scan of the 62.7 GiB of installed memory found no fault.
Two cold engine runs then gave identical model-state fingerprints, and the
restart and promotion parity checks passed:

- the L3 restore after a full engine restart matched the cold reference;
- the L2 promotion matched the cold reference;
- the model-state fingerprints and the outputs were identical.

The defective module is one of the two removed 16 GB modules. That pair
carries no part number and serial number 00000000. Test each module alone to
confirm which one is defective.

Finite cache budgets still limit history. A checkpoint that both L2 and L3
have evicted is not available for reuse. The engine must process that prefix
again.
## More opt-in switches (CUDA 12 experimental build)

None of these is on by default. Each was measured by its contributor on a V100, not by the maintainers.

| Switch | What it does | Contributor's number |
| --- | --- | --- |
| `STRATA_FUSE_DECODE=1` | the decode GEMV / MoE operator fusions of #1097 (gate/up + SwiGLU in one kernel, SwiGLU + quantizer in one kernel, all activation images of a vector in one pass, two BF16 projections in one kernel); bitwise the unfused kernels | 4-16 us per token saved, not visible end to end |
| `STRATA_PF_WMMA=1` | the FP16 tensor-core prompt experts (above); also takes IQ4_NL down projections (K=640, #1771) | +10% prompt speed (#1401) |
| `STRATA_PF_WMMA2=1` | a second prompt-expert kernel for IQ2_S gate/up and IQ4_NL down (#2356) | +2.5-4.3% prompt speed |
| `STRATA_ARGMAX_ROWS_SM70=1` | the multi-block greedy pick on sm_7x (#2357), bitwise the one-block pick | +0.7-0.8% decode |
| `STRATA_GR_NORM_FAST=1` | the register-split hyper-connection norm on CUDA (#2358) | +0.5-0.6% decode |
| `STRATA_Q8_GATHER_WIDE=0` | turns off the 8-wide int8 KV gather (#1095) that the experimental build uses below compute capability 7.5 | +11% on the gather kernel |
| `STRATA_GDN_REC_WARP=2` | Select the chain-split DeltaNet recurrence explicitly. `1` selects the sync-free control. The fork's normal dispatch already uses `STRATA_GDN_REC_FAST` on Volta. For the original pipelined control, set both `STRATA_GDN_REC_WARP=0` and `STRATA_GDN_REC_FAST=0`. `STRATA_GDN_CHUNK=1` selects the chunked path. | Contributor: +6-9% on the recurrence; not a new gain over this fork |
