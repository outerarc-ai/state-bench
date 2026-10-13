# Related work

Upstream discussions and implementations that this project builds on or compares against.
Status was checked on 2026-10-12, and on 2026-10-13 for #37898, #40865, #40907,
#41701, #43135, and ATOM #2045. It will change; follow the links for the current state.
Figures quoted below are as reported by their authors and have not been reproduced here.

## Where to place recurrent-state checkpoints

| Project | Item | Status | Summary |
| --- | --- | --- | --- |
| SGLang | [#20415](https://github.com/sgl-project/sglang/issues/20415) | Open roadmap | Unified Hybrid Radix Cache refactor; tracks full/SWA/Mamba components in one tree |
| SGLang | [#39115](https://github.com/sgl-project/sglang/pull/39115) | Merged | Checkpoint depth for prefixes that end off the radix page |
| SGLang | [#40907](https://github.com/sgl-project/sglang/pull/40907) | Merged | Restores non-DCP checkpoint donation for agent workloads with HiCache |
| SGLang | [#41701](https://github.com/sgl-project/sglang/pull/41701) | Open | Keeps chunked-prefill cuts on the Mamba chunk grid so DCP can donate checkpoints |
| SGLang | [#43135](https://github.com/sgl-project/sglang/pull/43135) | Draft | The same alignment without DCP |
| SGLang | [#40865](https://github.com/sgl-project/sglang/issues/40865) | Closed by its author | RFC on budgeted tail replay and retaining recent checkpoints; defines "retreat" as full-KV hit length minus usable prefix |
| vLLM | [#37898](https://github.com/vllm-project/vllm/pull/37898) | Merged | Marconi-style admission: adds a checkpoint at a shared prefix revealed by an attention hit with a state miss |
| vLLM | [#45238](https://github.com/vllm-project/vllm/issues/45238) | Open | Align-mode checkpoints can land in request-unique tokens, dropping hybrid prefix hits to zero |
| vLLM | [#49574](https://github.com/vllm-project/vllm/pull/49574) | Closed (experiment) | Checkpoints at chat-template boundaries (input, reasoning, response end). Preliminary observation: an input-end checkpoint alone added no hits over the prompt-end baseline on canonical multi-turn and tool-call workloads |
| vLLM | [#50551](https://github.com/vllm-project/vllm/pull/50551) | Open | Retains the final decode checkpoint of normally finished requests |
| ATOM | [#2045](https://github.com/ROCm/ATOM/pull/2045) | Merged | Paged state cache with a prompt-end anchor. On the SemiAnalysis cc-traces, the author reports 85.2% read-back for the anchor, 2.8% for demand checkpoints, and 0.0% for fixed-interval checkpoints, measured mainly on Kimi-K3 |

## Papers

| Work | Summary |
| --- | --- |
| [Marconi](https://arxiv.org/abs/2411.19379) (Pan et al., MLSys 2025) | Prefix caching for hybrid models: admits states at shared-prefix branch points and at the last decoded token, and evicts by a mix of recency and FLOP efficiency (compute saved per byte) |

## Capacity, tiering, and retention

| Project | Item | Status | Summary |
| --- | --- | --- | --- |
| SGLang | [#33713](https://github.com/sgl-project/sglang/issues/33713) | Open | After host Mamba states are evicted, host KV for the same prefix can no longer be loaded back; reproduced on current main as of 2026-10-11 |
| SGLang | [#42443](https://github.com/sgl-project/sglang/pull/42443) | Open | `--hicache-mamba-size-gb` to size the host Mamba pool independently, plus per-pool host metrics |
| SGLang | [#43727](https://github.com/sgl-project/sglang/pull/43727) | Open | Exports host Mamba pool occupancy |
| SGLang | [#36208](https://github.com/sgl-project/sglang/issues/36208) | Open | RFC for per-token-range KV retention priority |
| SGLang | [#24656](https://github.com/sgl-project/sglang/issues/24656) | Open | RFC for agent-aware KV cache hints and eviction |
| vLLM | [#60008](https://github.com/vllm-project/vllm/issues/60008) | Open | Hybrid prefix caching in align mode costs throughput even without hits |
| LMCache | [#4025](https://github.com/LMCache/LMCache/issues/4025) | Open | 2026 Q3 roadmap, including an SGLang HiCache integration item |

## How this project relates

The [simulator](SIMULATOR.md) implements several of the rules above so they can be
compared on the same traces and state budgets: SGLang's current behavior (see
[SGLang Mamba checkpoints](SGLANG_MAMBA_CHECKPOINTS.md)) and its DCP donation mask,
ATOM's prompt-end anchor, vLLM's Marconi-style admission, the retention proposed in
#40865, and an offline reference that bounds what any placement can save. Simulated
comparisons are labeled as such; request-level results follow the
[measurement protocol](MEASUREMENT.md).
