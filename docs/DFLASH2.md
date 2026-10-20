# DFlash2 speculative decoding

## Purpose

Use `--dflash2 DIR` to load the
[incoai/Qwen3.8-Flash-Next-DFlash2 draft model](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2).
The draft model is not a separate language model. It uses the target model's
input embeddings, output projection, and intermediate features.

The draft model predicts a block in one pass. A candidate selector chooses a
path through the block. The target model then checks that path. Only the
accepted prefix changes the target session. Rejected draft features do not
enter the draft model's context.

The checkpoint uses the completed target layers 5, 15, 25, 35, and 45.
The engine averages the four hyper-connection streams at each layer. It joins
those five features in checkpoint order. Both prompt processing paths capture
these features. The draft model keeps only the context needed by its
2,048-position sliding attention window.

## Get the checkpoint

Install the Hugging Face CLI. Download the configuration and weights to the
same directory. This revision identifies the checkpoint used for development:

```sh
hf download incoai/Qwen3.8-Flash-Next-DFlash2 \
  config.json model.safetensors \
  --revision a22a1744e85c17364dcc54a6a420af7135d4c7f6 \
  --local-dir /models/dflash2
```

The engine reads the safetensors file directly. No pack conversion or Python
machine-learning library is needed. The loader checks the model configuration
and tensor dimensions before it uses the weights.

## Select the draft model

Start with a working native Qwen3.8-Flash-Next configuration. Keep the target
pack, tokenizer, native shards, and PLE shard together. Do not change the target
model to load a draft model.

1. Remove `--mtp` and its directory from the engine arguments.
2. Add `--dflash2 /models/dflash2`.
3. Set `--spec 4` and `--dflash2-block 4` for a comparison with three draft tokens.
4. Set `--spec-min-p 0` to prevent the confidence gate from shortening that comparison.
5. Set `--suffix-draft 0` and `--lookup-chain 0` to measure only the selected draft model.
6. Use `--pipeline-windows 0` or `--pipeline-windows 1`.

`--dflash2-block N` selects a block from 2 through 8 positions. Position zero
is the anchor token. The other `N - 1` positions are draft tokens. If this
option is absent, the engine uses the checkpoint's eight-position block.
`--spec` limits the verification window. Use the same value for both options
to check the complete block.

The server passes its configuration's `args` array to the engine. No separate
server switch is needed. Use a separate test configuration. Do not put API
keys or private runtime configurations in a commit.

## Sampling

Greedy requests use a greedy selector path. The target's greedy result decides
which draft tokens to keep.

For sampled selector paths and rejection sampling, set both environment
variables in the engine's environment before it starts:

```sh
export STRATA_SPEC_COUPLED=1
export STRATA_SPEC_PROB=1
```

The selector supplies a conditional probability distribution for each proposed
position. The verifier uses this distribution as `q`. It does not use the
unadjusted draft logits as `q`. The target sampler still applies the request's
sampling controls. On rejection, it samples from the residual target
probability distribution.

## Limits and memory

- This path requires the native Flash-Next target and its output projection.
- Use one request at a time. Batch slots are not supported by this draft path.
- Do not combine DFlash2 with MTP or `--batch-mtp`.
- `--pipeline-windows 2` uses an MTP-specific, teacher-forced draft chain.
  DFlash2 does not use that chain. An incompatible combination fails at startup.
- A layer split can capture the required features on both cards. Accepted
  feature rows then move to the card that runs the target output projection.
  Prompt capture uses a separate buffer slot for each in-flight chunk.
- A restored conversation reuses the target session, but starts a fresh draft
  feature history. The missing cached draft features can reduce early
  acceptance. They do not change the target's verification rule.
- The draft weights and scratch space consume additional VRAM. The engine loads
  the draft model before it sizes the automatic expert cache. Prompt feature
  capture also needs bounded buffers on the participating cards.
- On V100, projection weights and matrix-multiplication inputs use FP16.
  Matrix-multiplication outputs, other activations, residuals, and draft K/V use
  FP32. FP16 outputs can overflow with this checkpoint and must not replace them.

Keep the daily service configuration unchanged during comparisons. Restore its
approved model and pipeline settings after the test.

## Acceptance target

The model card reports the following acceptance lengths at concurrency one,
with three draft tokens per verification step:

| Task | MTP | DFlash2 | Relative increase |
| --- | ---: | ---: | ---: |
| GSM8K | 3.08 | 3.26 | 5.8% |
| MATH-500 | 2.93 | 3.17 | 8.2% |
| HumanEval | 2.52 | 2.94 | 16.7% |
| MBPP | 2.65 | 3.06 | 15.5% |
| MT-Bench | 2.50 | 2.83 | 13.2% |

Acceptance length is the completion-token count divided by the number of
verification steps. Take the mean of this value across requests to compare
with the model card. It is not the percentage of offered draft tokens accepted.
Report that percentage separately.

The published run uses an FP8 target on one GB300. Its sampling controls are
temperature 1.0, top-p 0.95, top-k 20, and `xhigh` reasoning effort. It permits
4,096 new tokens per request. Use the same block size, prompt format, and
sampling controls for a matched acceptance comparison. Record the target
quantization, routing mode, request count, and token limit with local results.
The published improvement is a measurement on that system, not a guarantee
for an IQ3_S target on V100.

The card does not give the MTP launch command. The pinned SGLang version
defaults to argmax MTP drafts and target-only verification. Strata can instead
sample MTP drafts from their full distribution and use exact rejection
sampling. Record which MTP mode you use. Do not treat these two baselines
as equivalent.

## Run a comparison

Use [`bench/run_dflash2_bench.py`](../bench/run_dflash2_bench.py) against a
separate test server. Stop other clients before a run. Restart the engine
between arms to clear its conversation cache.

```sh
python bench/run_dflash2_bench.py --suite short \
  --base http://127.0.0.1:8089 --config TEST_SERVICE_JSON \
  --label dflash2-block4 --out short-dflash2-block4.json
```

The short suite sends code, math, and prose requests three times. It uses
greedy decoding, a 256-token limit, and no reasoning. It rejects a request
that reuses cached prompt tokens.

For a canonical target comparison, set `STRATA_ROUTE_RESIDENT=0` and
`STRATA_ROUTE_TAIL_SKIP=0`. These settings disable approximate routing.
Use `--prompt-cache 0 --conversation-cache-mib 0 --conversation-cache-slots 0`
in every arm. Omit `conversation_cache_disk` and `conversation_cache_disk_gib`
from the test configuration. Do not set those two keys to `false`: the server
requires a path and a positive budget when either key is present.
This protocol reads every prompt from the beginning, including repeated
MT-Bench prompts. Record a different cache policy if you use one.

For an acceptance run, set `STRATA_DECODE_TIMING=1` in the engine environment.
Set `answer_room`, `reasoning_close_retry`, `reasoning_loop_recovery`, and
`tool_call_recovery` to `false` in the test server configuration. Set
`repeat_stop_tokens` to `0`. These settings prevent server-side changes to
the measured output.

Prepare the full cohort with
[`bench/prepare_dflash2_prompts.py`](../bench/prepare_dflash2_prompts.py).
It uses the prompt formats and selection order from the
[pinned reference benchmark](https://github.com/z-lab/dflash/blob/265add3a0e4c62d02566dc44ad9a735f350084c2/dflash/benchmark.py).
For each dataset, it shuffles the full corpus with seed 42 and selects
128 requests. A short corpus repeats its shuffled order. The card's server
benchmark uses only the first turn, including MT-Bench. One separate request
with a 64-token limit runs before each dataset as a warmup.

```sh
python bench/prepare_dflash2_prompts.py --out PROMPTS_JSON
python bench/run_dflash2_bench.py --suite acceptance \
  --base http://127.0.0.1:8089 --config TEST_SERVICE_JSON \
  --prompts PROMPTS_JSON --engine-log ENGINE_LOG --limit 128 \
  --label dflash2-block4 --out acceptance-dflash2-block4.json
```

For a custom workload, supply a JSON array. Each record needs a `dataset`
label, a zero-based `row`, and a `turns` array of user prompts. An optional
`warmup: true` record runs without adding a result. Do not commit raw prompts.

The runner includes the assistant reply in the next turn of a multi-turn
item. It does not write reply text or credentials to the results. It reads
the exact verification-step count from each request's engine log span.
It rejects a request that starts more than one engine generation.

Sources: [model card](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2),
[reference model](https://github.com/z-lab/dflash/blob/main/dflash/model.py),
[benchmark protocol](https://github.com/z-lab/dflash/blob/265add3a0e4c62d02566dc44ad9a735f350084c2/dflash/benchmark.py), and
[SGLang speculative defaults](https://github.com/sgl-project/sglang/blob/3c560089e1818a5217a73f2168273485a5350341/python/sglang/srt/arg_groups/fields/spec.py).
