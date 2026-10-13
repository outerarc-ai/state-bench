# SGLang Mamba checkpoints

How SGLang's Unified Radix Cache places, inserts, and evicts recurrent-state (Mamba/GDN)
checkpoints. Read from SGLang commit `166b4ae7f`. As of `450837e` (14 commits later), the
only change to `components/mamba.py` is in the HiCache buffer load-back path; the rules
below are unchanged.

**Scope.** `--mamba-radix-cache-strategy extra_buffer` (chosen by the default `auto` when
overlap scheduling is on or `page_size > 1`), without DCP, speculative decoding, or
HiCache. Differences for `no_buffer` and DCP are listed at the end.

**References** are `file:line`, relative to `python/sglang/srt/`. Rule numbers (R1–R16)
are for cross-reference within this document only.

If you find a discrepancy with current SGLang, please open an issue with the commit you
read.

## Granularities

| Name | Definition | Source |
| --- | --- | --- |
| `cache_chunk_size` | max(model `mamba_chunk_size`, else FLA chunk size 64; `page_size`) | `arg_groups/overrides.py:1831` |
| `checkpoint_grid` | lcm(`cache_chunk_size`, radix page size) | `runtime_context.py:1959` |
| `track_grid` | lcm(`checkpoint_grid`, `mamba_track_interval`); the interval defaults to 256 | `runtime_context.py:1967`, `arg_groups/fields/exec_.py:430` |

Example: Qwen3.5 (GDN) with `page_size=1` gives `cache_chunk_size = 64`,
`checkpoint_grid = 64`, and `track_grid = 256`.

## Matching a new request

- **R1 Usable prefix is the deepest node with a state.** Matching follows the full-KV
  prefix, but a request can resume only from the deepest node on that path holding a
  Mamba state on device or host: `mamba_boundary_len = device_prefix_len + host_hit_length`.
  KV matched beyond that point is recomputed. `components/mamba.py:146-168`
- **R2 Branching point.** `branching_seqlen = floor(full_kv_hit / checkpoint_grid) *
  checkpoint_grid`, set only when it exceeds `mamba_boundary_len`. `components/mamba.py:175-180`
- **R3 States live on nodes.** When a node splits, its state stays with the child and the
  new parent has none. `components/mamba.py:308-318`
- **R4 LRU refresh touches only the used state.** A match refreshes the matched node's
  state but not its ancestors; a newly inserted leaf enters at MRU.
  `components/mamba.py:122-144, 222-248`

## Snapshots during prefill

- **R5 At most one snapshot per forward pass.** Each prefill chunk records at most one
  state into the request's two-slot ping-pong buffer. `managers/schedule_batch.py:3008-3101`
- **R6 Default position is aligned relative to the chunk start.** Without DCP:
  `track = prefix_len + floor(extend_len / checkpoint_grid) * checkpoint_grid`, recorded
  only when `extend_len >= checkpoint_grid`; otherwise the previous track position is kept.
  If `prefix_len` is off the absolute grid, so is the snapshot.
  `managers/schedule_batch.py:3044-3051`
- **R7 The branching point replaces the chunk's default snapshot.** When the branching
  point from R2 lies inside the chunk and `(branching_seqlen - prefix_len) %
  cache_chunk_size == 0`, the chunk records the branching point instead of its default
  position. When the alignment condition fails, the branching point is not recorded.
  `managers/schedule_batch.py:3085-3099`
- **R8 Chunk boundaries.** Each chunk takes the remaining `chunked_prefill_size`
  (`managers/schedule_policy.py:1277`), measured from the matched prefix rather than an
  absolute grid; exact chunk fill is disabled when Mamba prefix sharing is on
  (`managers/schedule_policy.py:611-618`). Open pull requests
  [#41701](https://github.com/sgl-project/sglang/pull/41701) and draft
  [#43135](https://github.com/sgl-project/sglang/pull/43135) align chunk boundaries to the grid.

## Insertion into the tree

- **R9 Every prefill chunk is inserted, including the last.** Intermediate chunks go
  through `stash_chunked_request` (`managers/scheduler.py:3631-3635`). After the last
  chunk of an unfinished request, `checkpoint_kv_cache` is called from result processing
  (`managers/scheduler_components/batch_result_processor.py:345-388`). Both call
  `tree_cache.checkpoint(up_to=extend_end)` (`mem_cache/common.py:252-262`). The state at
  the end of the prompt, aligned as in R6, is therefore in the tree once prefill finishes.
- **R10 Inserted KV stops at the snapshot.** The inserted length is the minimum over
  components; the Mamba component reports `mamba_last_track_seqlen`, then the key is
  page-aligned. KV computed past the latest snapshot is not inserted for an unfinished
  request. `mem_cache/unified_radix_cache.py:1084-1128`
- **R11 Snapshots are donated, not copied.** An unfinished request hands its ping-pong
  slot to the tree and allocates a new one. `components/mamba.py:606-641`

## Decode and request completion

- **R12 Decode snapshots use absolute positions.** A state is recorded whenever
  `seq_len % track_grid == 0`; the previous one becomes `prev`. Nothing is inserted during
  decode. `managers/schedule_batch.py:3620-3640`,
  `managers/scheduler_components/batch_result_processor.py:1373-1378`
- **R13 Completion inserts the latest snapshot, not the sequence end.** The position is
  `mamba_last_track_seqlen`: the last `track_grid` multiple reached in decode, or else the
  prefill snapshot already inserted by R9. If it exceeds the key length, the `prev` slot is
  used; if that also exceeds it, nothing is inserted for this request (the R9 insertion
  remains). `components/mamba.py:541-605`

For an append-style next turn (this input + the reply + a tool result), the usable prefix
is the deeper of the R13 decode snapshot (valid only when the reply is replayed token for
token) and the R9 end-of-prompt snapshot.

## Eviction

- **R14 State pool LRU.** Failing to allocate a state slot triggers eviction
  (`components/mamba.py:505-512`), which walks the Mamba LRU from the cold end
  (`components/mamba.py:358-435`). Leaves are evicted with their KV; internal nodes lose
  only the state and keep KV as a tombstone.
- **R15 Per-path cap.** `mamba_max_states_per_path` defaults to −1 (unlimited). When
  positive, each insertion removes the shallowest states beyond the cap, skipping the
  tail, fork points (nodes with more than one child), locked nodes, and device leaves, so
  the cap is best effort. `components/mamba.py:250-306`, `arg_groups/fields/exec_.py:397-403`
- **R16 Session-aware cache.** `enable_session_radix_cache` changes the eviction cursor
  order. Not covered here.

## Other configurations

- `no_buffer` requires `page_size == 1`. Each chunk and completion snapshot is taken at
  the sequence end (less the unflushed ReplaySSM ring depth at completion); there is no
  branching-point snapshot. `components/mamba.py:570-585`
- DCP aligns R6 to the absolute grid and additionally requires the offset from the chunk
  start to be a multiple of `cache_chunk_size`. `managers/schedule_batch.py:3036-3043`
- int8 checkpoints and HiCache write-back change storage cost and tiering, not positions.

## Open questions for measurement

- How often R7 drops a branching point on agent traces.
- How often replies fail to round-trip through chat templates, which decides whether
  R13 or R9 supplies the usable prefix.
- Whether a chunk with `extend_len < checkpoint_grid` can carry a track position over from
  an earlier request; new requests initialize it to `None`
  (`managers/schedule_batch.py:1978`).
