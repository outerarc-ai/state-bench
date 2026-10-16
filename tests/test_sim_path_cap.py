"""Hand-checked cases for the per-path snapshot cap (SGLang R15)."""

import json

import pytest
from sim_helpers import make_trace, toks

from state_bench import cli
from state_bench.sim import Policy, SimConfig, Simulation
from state_bench.traces import schema, synth


class At(Policy):
    """Places a snapshot at every listed position past the resume point."""

    name = "at"

    def __init__(self, *positions: int) -> None:
        self.positions = positions

    def plan_prefill(self, req, start, kv_hit, cfg):
        return [p for p in self.positions if start < p <= req.prompt_tokens]


def run_sim(trace, policy, **cfg) -> Simulation:
    sim = Simulation(trace, policy, SimConfig(**cfg))
    sim.run()
    return sim


def held(sim: Simulation) -> list[tuple[str, int]]:
    return sorted((s.owner, s.pos) for s in sim.snapshots())


def test_cap_evicts_shallowest_and_keeps_new_snapshot():
    trace = make_trace(("A", toks(1, 300), 1, None, None))
    sim = run_sim(trace, At(64, 128, 192, 256), max_states_per_path=2)
    assert held(sim) == [("A", 192), ("A", 256)]
    assert sim.result.evictions == 2
    assert sim.result.snapshots_written == 4


def test_unlimited_by_default():
    trace = make_trace(("A", toks(1, 300), 1, None, None))
    sim = run_sim(trace, At(64, 128, 192, 256))
    assert len(held(sim)) == 4 and sim.result.evictions == 0


def test_fork_point_is_kept():
    shared = toks(1, 192)
    trace = make_trace(
        ("A", shared + toks(2, 108), 1, None, None),
        ("B", shared + toks(3, 108), 1, "A", "fork"),
    )
    sim = run_sim(trace, At(64, 128, 192, 256), max_states_per_path=1)
    # A keeps only its deepest snapshot. B re-creates the shared ones; once B's own
    # block follows 192, that position has two continuations and survives the cap.
    assert held(sim) == [("A", 256), ("B", 192), ("B", 256)]


def test_snapshot_in_use_is_kept():
    base = toks(1, 300)
    trace = make_trace(
        ("A", base, 1, None, None),
        ("A2", base + toks(2, 200), 1, "A", "append"),
    )
    sim = run_sim(trace, At(64, 128, 192, 256, 320, 384, 448), max_states_per_path=1)
    # A2 resumes from 256; it stays while A2 adds 320, 384, 448 (each replaced by the next).
    assert held(sim) == [("A", 256), ("A2", 448)]
    rec = sim.result.records[1]
    assert rec.usable_prefix == 256


def test_invalid_cap():
    with pytest.raises(ValueError):
        SimConfig(max_states_per_path=0)


def test_cli_flag(tmp_path, capsys):
    path = tmp_path / "t.jsonl"
    trace = synth.generate_trace(synth.preset_params("mixed", sessions=2, seed=0))
    schema.dump_trace(trace, path)
    args = ["sim", str(path), "--state-slots", "16"]
    assert cli.main([*args, "--max-states-per-path", "2"]) == 0
    capped = json.loads(capsys.readouterr().out)[0]
    assert capped["config"]["max_states_per_path"] == 2
    assert cli.main(args) == 0
    plain = json.loads(capsys.readouterr().out)[0]
    assert plain["config"]["max_states_per_path"] is None


def test_cap_applies_only_under_pressure():
    trace = make_trace(("A", toks(1, 300), 1, None, None))
    pos = At(64, 128, 192, 256)
    # Pool of 8: usage reaches 0.5 only at the fourth snapshot, which then evicts the
    # two shallowest down to the cap of 2.
    sim = run_sim(trace, pos, state_slots=8, max_states_per_path=2, path_cap_min_usage=0.5)
    assert held(sim) == [("A", 192), ("A", 256)]
    # Below the threshold nothing is evicted.
    sim = run_sim(trace, pos, state_slots=10, max_states_per_path=2, path_cap_min_usage=0.5)
    assert len(held(sim)) == 4
    # An unlimited pool never counts as under pressure.
    sim = run_sim(trace, pos, max_states_per_path=2, path_cap_min_usage=0.5)
    assert len(held(sim)) == 4


def test_invalid_min_usage():
    with pytest.raises(ValueError):
        SimConfig(path_cap_min_usage=1.5)
