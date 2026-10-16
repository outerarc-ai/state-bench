import json
import sys
import textwrap

import pytest

from state_bench import bench, cli
from state_bench.bench import RequestResult, Slo
from state_bench.bench.records import drop_warmup, matched_ttft_diff

PY = sys.executable


# -- statistics ----------------------------------------------------------------------


def test_percentile_interpolates():
    assert bench.percentile([1, 2, 3, 4], 50) == 2.5
    assert bench.percentile([5], 90) == 5
    with pytest.raises(ValueError):
        bench.percentile([], 50)


def test_bootstrap_is_deterministic():
    diffs = [-0.1, -0.2, -0.15, -0.12, -0.18]
    a = bench.bootstrap_median_ci(diffs, seed=3)
    assert a == bench.bootstrap_median_ci(diffs, seed=3)
    assert a[0] <= -0.12 and a[1] <= -0.1
    with pytest.raises(ValueError):
        bench.bootstrap_median_ci([])


@pytest.mark.parametrize(
    "pairs, verdict",
    [
        ([(1.0, 0.8)] * 5, "candidate lower"),
        ([(1.0, 1.2)] * 5, "candidate higher"),
        (
            [(1.0, 0.9), (1.0, 1.1), (1.0, 0.95), (1.0, 1.05), (1.0, 1.0)],
            "no significant difference",
        ),
        ([(1.0, 0.8)] * 4, "too few pairs"),
    ],
)
def test_compare_pairs_verdicts(pairs, verdict):
    r = bench.compare_pairs("m", pairs)
    assert r.verdict == verdict
    assert r.pairs == len(pairs)


def test_compare_pairs_relative_and_errors():
    r = bench.compare_pairs("m", [(2.0, 1.0)] * 5)
    assert r.diff_median == -1.0 and r.relative_diff_median == -0.5
    assert bench.compare_pairs("m", [(0.0, 1.0)] * 5).relative_diff_median is None
    with pytest.raises(ValueError):
        bench.compare_pairs("m", [])


# -- records ---------------------------------------------------------------------------


def rec(i, **kw):
    return RequestResult(req_id=f"r{i}", t_send_s=float(i), **kw)


def test_denominators_and_slo():
    rs = [
        rec(0, ttft_s=0.5, e2e_s=1.0, valid=True),
        rec(1, ttft_s=3.0, e2e_s=4.0, valid=True),  # misses the TTFT SLO
        rec(2, ttft_s=0.5, e2e_s=1.0, valid=False),  # wrong output
        rec(3, admitted=False, completed=False),
        rec(4, completed=False, ttft_s=0.4),
    ]
    d = bench.denominators(rs, Slo(ttft_s=2.0))
    assert d == {
        "offered": 5,
        "admitted": 4,
        "completed": 3,
        "on_time": 2,
        "on_time_valid": 1,
        "validity_checked": True,
    }
    unchecked = bench.denominators([rec(0, ttft_s=0.1)], Slo(e2e_s=1.0))
    assert unchecked["on_time"] == 0 and unchecked["on_time_valid"] is None


def test_run_metrics_and_matching():
    base = [
        rec(i, ttft_s=1.0 + i, e2e_s=2.0, prompt_tokens=100, cached_tokens=50) for i in range(3)
    ]
    m = bench.run_metrics(base)
    assert m["ttft_p50_s"] == 2.0 and m["cached_token_share"] == 0.5
    cand = [rec(i, ttft_s=0.5 + i) for i in range(2)]
    assert matched_ttft_diff(base, cand) == -0.5
    assert matched_ttft_diff(base, []) is None
    assert bench.run_metrics([rec(0, completed=False)])["ttft_p50_s"] is None
    assert [r.req_id for r in drop_warmup(list(reversed(base)), 1)] == ["r1", "r2"]


def test_load_records(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps({"req_id": "a", "t_send_s": 0, "ttft_s": 0.2}) + "\n\n")
    assert bench.load_records(p)[0].ttft_s == 0.2
    p.write_text(json.dumps({"req_id": "a", "t_send_s": 0, "bogus": 1}) + "\n")
    with pytest.raises(ValueError, match="unknown"):
        bench.load_records(p)
    p.write_text(json.dumps({"req_id": "a"}) + "\n")
    with pytest.raises(ValueError, match="required"):
        bench.load_records(p)


# -- plan, run, report ---------------------------------------------------------------------


def test_plan_orders():
    ab = [(r.pair, r.arm) for r in bench.plan(2)]
    assert ab == [(0, "baseline"), (0, "candidate"), (1, "baseline"), (1, "candidate")]
    abba = [r.arm for r in bench.plan(2, "abba")]
    assert abba == ["baseline", "candidate", "candidate", "baseline"]
    with pytest.raises(ValueError):
        bench.plan(0)
    with pytest.raises(ValueError):
        bench.plan(1, "ba")


FAKE_LOAD = textwrap.dedent(
    """
    import json, sys
    out, arm, run = sys.argv[1], sys.argv[2], int(sys.argv[3])
    scale = 0.8 if arm == "candidate" else 1.0
    with open(out, "w") as f:
        for i in range(10):
            ttft = scale * (1.0 + 0.01 * ((run * 7 + i) % 5))
            f.write(json.dumps({"req_id": f"q{i}", "t_send_s": i, "ttft_s": ttft,
                                "e2e_s": ttft + 1, "prompt_tokens": 100,
                                "cached_tokens": 60 if arm == "candidate" else 40,
                                "valid": True}) + "\\n")
    """
)


def config(tmp_path, ready_ok=True):
    load = tmp_path / "load.py"
    load.write_text(FAKE_LOAD)
    server = [PY, "-c", "import time; time.sleep(60)"]
    return {
        "name": "fake",
        "pairs": 5,
        "order": "abba",
        "warmup_requests": 2,
        "slo": {"ttft_s": 2.0},
        "device_check": [PY, "-c", "print('idle')"],
        "device_idle_pattern": "idle",
        "ready": {
            "command": [PY, "-c", f"import sys; sys.exit({0 if ready_ok else 1})"],
            "timeout_s": 0.3 if not ready_ok else 30,
            "interval_s": 0.1,
        },
        "load": [PY, str(load), "{records}", "{arm}", "{run}"],
        "arms": {"baseline": {"start": server}, "candidate": {"start": server, "env": {"X": 1}}},
    }


def test_run_and_report(tmp_path):
    out = tmp_path / "run"
    bench.run(config(tmp_path), out, log=lambda _m: None)
    manifest = json.loads((out / "raw" / "runs.json").read_text())
    assert [e["arm"] for e in manifest][:4] == ["baseline", "candidate", "candidate", "baseline"]
    assert all(e["status"] == "ok" and e["device_before"] == "idle" for e in manifest)
    rep = bench.report(out)
    assert rep["complete_pairs"] == 5 and rep["warnings"] == []
    assert rep["runs"][0]["denominators"]["offered"] == 8  # 10 minus 2 warm-up
    by = {c["metric"]: c for c in rep["comparisons"]}
    assert by["ttft_p50_s"]["verdict"] == "candidate lower"
    assert by["cached_token_share"]["verdict"] == "candidate higher"
    assert by["matched_ttft_diff_s"]["diff_median"] < 0


def test_server_not_ready_is_reported(tmp_path):
    cfg = config(tmp_path, ready_ok=False)
    cfg["pairs"] = 1
    cfg["device_idle_pattern"] = "busy-free"
    out = tmp_path / "run"
    bench.run(cfg, out, log=lambda _m: None)
    rep = bench.report(out)
    assert rep["complete_pairs"] == 0
    assert any("server not ready" in w for w in rep["warnings"])
    assert any("device not idle" in w for w in rep["warnings"])
    assert any("at least 5" in w for w in rep["warnings"])


def test_cli_paired(tmp_path, capsys):
    assert cli.main(["paired", "plan", "--pairs", "1"]) == 0
    assert "baseline" in capsys.readouterr().out
    cfg = config(tmp_path)
    cfg["pairs"] = 1
    stop = [PY, "-c", "pass"]
    cfg["arms"]["baseline"]["stop"] = stop
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    out = tmp_path / "run"
    assert cli.main(["paired", "run", str(path), "-o", str(out)]) == 0
    capsys.readouterr()
    assert cli.main(["paired", "report", str(out)]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["complete_pairs"] == 1
    assert rep["comparisons"][0]["verdict"] == "too few pairs"


# -- SGLang load ---------------------------------------------------------------------------


def fake_sglang():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen: set = set()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            ids = body["input_ids"]
            if len(ids) > 2500:
                self.send_response(400)
                self.end_headers()
                return
            key = tuple(ids[:64])
            cached = 64 if key in seen else 0
            seen.add(key)
            self.send_response(200)
            self.end_headers()
            for i in range(2):
                meta = {"completion_tokens": i + 1, "cached_tokens": cached}
                self.wfile.write(f"data: {json.dumps({'meta_info': meta})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_build_prompts_is_deterministic_append_chains():
    from state_bench.bench.sglang_load import LoadShape, build_prompts

    shape = LoadShape(sessions=2, turns=3, shared_prefix=10, turn_min=5, turn_max=7)
    a, b = build_prompts(shape), build_prompts(shape)
    assert a == b
    assert [r for r, _ in a] == ["s0-t0", "s1-t0", "s0-t1", "s1-t1", "s0-t2", "s1-t2"]
    assert a[0][1][:10] == a[1][1][:10]  # shared prefix
    assert a[2][1][: len(a[0][1])] == a[0][1]  # append chain
    with pytest.raises(ValueError):
        LoadShape(turn_min=0)
    with pytest.raises(ValueError):
        LoadShape(sessions=0)
    with pytest.raises(ValueError):
        LoadShape(vocab_low=5, vocab_high=5)


def test_run_load_against_fake_server(tmp_path, capsys):
    srv, url = fake_sglang()
    try:
        out = tmp_path / "r.jsonl"
        args = ["paired", "load", url, "-o", str(out), "--sessions", "2", "--turns", "3"]
        args += ["--shared-prefix", "100", "--turn-min", "900", "--turn-max", "1000"]
        assert cli.main(args) == 0
        assert "failed requests" in capsys.readouterr().err
        recs = bench.load_records(out)
        assert len(recs) == 6
        assert recs[0].completed and recs[0].ttft_s is not None and recs[0].output_tokens == 2
        assert recs[1].cached_tokens == 64  # second session shares the prefix
        assert any(not r.admitted and r.error for r in recs)  # prompts over 2500 tokens
    finally:
        srv.shutdown()
