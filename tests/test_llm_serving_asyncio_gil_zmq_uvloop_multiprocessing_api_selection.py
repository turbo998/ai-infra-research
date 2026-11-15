import unittest
from experiments.llm_serving_asyncio_gil_zmq_uvloop_multiprocessing_api_selection.run import (
    evaluate_serving_gil_zmq_async_schedule_tiers,
)


class TestServingGilZmqAsyncScheduleSelection(unittest.TestCase):
    def test_serving_gil_tiers(self) -> None:
        data = evaluate_serving_gil_zmq_async_schedule_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_monolithic_single_process_fastapi_detokenizer_scheduler_gil"]
        t4 = tiers["tier4_zmq_multiprocess_plus_async_pre_scheduling_and_gpu_d2d_token_forward"]
        self.assertGreater(t1["exposed_cpu_overhead_ms"], 14.0)
        self.assertLess(t4["exposed_cpu_overhead_ms"], 0.5)
        self.assertGreater(t4["gpu_compute_duty_cycle_pct"], 95.0)


if __name__ == "__main__":
    unittest.main()
