# Adapted from SGLang 9f5d8712514c2d9dd29c76c6080774db3fd50aa2
# test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_mamba.py
# (Apache-2.0). Changes for a single gfx1151 GPU are marked inline; only the first
# test class is kept.
import os
import shutil
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache_kit import (
    AccuracyTwoPassMixin,
    UnifiedRadixTreeTestMixin,
)
from sglang.test.kl_multiturn_utils import (
    get_input_ids,
    make_mamba_decode_assert,
    make_mamba_prefill_assert,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=742, stage="extra-b", runner_config="4-gpu-h100")

MAMBA_MODEL = "/models/Qwen/Qwen3.5-4B"  # local functional check (upstream: Qwen3-Next-80B FP8, TP4)
MAMBA_CHUNK_SIZE = 64
MAMBA_TRACK_INTERVAL = 128
MAMBA_CHUNKED_PREFILL_SIZE = 2048


class TestUnifiedMambaRadixCache(UnifiedRadixTreeTestMixin, CustomTestCase):
    """Mamba hybrid + UnifiedRadixCache."""

    # fp8 cache-hit KL is noisy and flaked at 0.003, match the HiCache variant
    kl_threshold = 0.005
    prefill_cache_assert = staticmethod(
        make_mamba_prefill_assert(chunk_size=MAMBA_CHUNK_SIZE)
    )
    decode_cache_assert = staticmethod(
        make_mamba_decode_assert(track_interval=MAMBA_TRACK_INTERVAL)
    )

    @classmethod
    def setUpClass(cls):
        cls.model = MAMBA_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "1",
                "--chunked-prefill-size",
                str(MAMBA_CHUNKED_PREFILL_SIZE),
                "--mem-fraction-static",
                "0.35",
                "--mamba-radix-cache-strategy",
                "extra_buffer",
                "--mamba-track-interval",
                str(MAMBA_TRACK_INTERVAL),
                "--mamba-max-states-per-path",
                "3",
                # gfx1151: the default aiter attention backend aborts at init; use Triton.
                "--attention-backend",
                "triton",
                "--linear-attn-backend",
                "triton",
                "--mamba-backend",
                "triton",
                # Without this, auto-sizing gave 22 slots under memory pressure and
                # cached states were evicted (cached_tokens=0); see summary.md.
                "--max-mamba-cache-size",
                "64",
            ],
            env={
                "SGLANG_ENABLE_RANK_CONSENSUS_CHECKER": "1",
            },
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process, wait_timeout=60)



if __name__ == "__main__":
    unittest.main()
