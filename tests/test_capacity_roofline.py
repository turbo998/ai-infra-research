import io
import json
from contextlib import redirect_stderr, redirect_stdout
import unittest
from unittest.mock import patch

from experiments.capacity_roofline import budget, run
from tests.fixture_support import FixtureCase


class CapacityTests(unittest.TestCase):
    def test_independent_kv_reference(self) -> None:
        value = budget.kv_cache_bytes(24, 4, 128, 8192, 1, 2)
        self.assertEqual(value, 402_653_184)
        self.assertEqual(budget.bytes_to_gib(value), 0.375)
        batch_two = budget.kv_cache_bytes(24, 4, 128, 8192, 2, 2)
        self.assertEqual(batch_two, 805_306_368)
        self.assertEqual(budget.bytes_to_gib(batch_two), 0.75)

    def test_mha_and_gqa_use_kv_heads(self) -> None:
        self.assertEqual(budget.kv_cache_bytes(1, 1, 1, 1, 1, 2, architecture="mha"), 4)
        self.assertEqual(budget.kv_cache_bytes(1, 4, 1, 1, 1, 2, architecture="gqa"), 16)

    def test_weight_packing_ceiling(self) -> None:
        self.assertEqual(budget.weight_bytes(1_000_000_000, 16), 2_000_000_000)
        self.assertEqual(budget.weight_bytes(3, 4), 2)
        self.assertEqual(budget.weight_bytes(1, 1), 1)
        self.assertEqual(budget.weight_bytes(8, 1), 1)

    def test_decimal_and_binary_units(self) -> None:
        self.assertEqual(budget.bytes_to_gb(1_000_000_000), 1)
        self.assertEqual(budget.bytes_to_gib(1_073_741_824), 1)
        self.assertEqual(budget.bytes_to_gb(1_073_741_824), 1.073741824)
        self.assertEqual(budget.bytes_to_gib(0), 0)

    def test_block_rounding_and_accounted_capacity(self) -> None:
        self.assertEqual(budget.allocated_tokens(17, 16), 32)
        self.assertEqual(budget.allocated_tokens(16, 16), 16)
        self.assertEqual(budget.allocated_tokens(1, 16), 16)
        kwargs = dict(
            parameters=3, bits_per_parameter=4, layers=1, kv_heads=1, head_dim=1,
            sequence=17, batch=2, bytes_per_element=2, token_block_size=16,
            workspace_bytes=7, allocator_reserve_bytes=9,
        )
        result = budget.memory_budget(**kwargs)
        self.assertEqual(result["logical_kv_bytes"], 136)
        self.assertEqual(result["allocated_kv_bytes"], 256)
        self.assertEqual(result["kv_allocation_padding_bytes"], 120)
        self.assertEqual(result["accounted_total_bytes"], 274)
        self.assertIsNone(result["accounted_headroom_bytes"])
        self.assertIsNone(result["fits_accounted_capacity"])
        self.assertTrue(budget.memory_budget(**kwargs, capacity_bytes=274)["fits_accounted_capacity"])
        self.assertFalse(budget.memory_budget(**kwargs, capacity_bytes=273)["fits_accounted_capacity"])
        self.assertEqual(budget.memory_budget(**kwargs, capacity_bytes=273)["accounted_headroom_bytes"], -1)

    def test_independent_reference_overhead_capacity_headroom(self) -> None:
        result = budget.memory_budget(
            parameters=1_000_000_000, bits_per_parameter=16, layers=24, kv_heads=4,
            head_dim=128, sequence=8192, batch=2, bytes_per_element=2,
            token_block_size=16, workspace_bytes=268_435_456,
            allocator_reserve_bytes=268_435_456, capacity_bytes=4_294_967_296,
        )
        self.assertEqual(result["accounted_total_bytes"], 3_342_177_280)
        self.assertEqual(result["accounted_headroom_bytes"], 952_790_016)
        self.assertTrue(result["fits_accounted_capacity"])

    def test_invalid_dimensions_and_fractional_element_bytes(self) -> None:
        dimensions = [24, 4, 128, 8192, 2, 2]
        for index in range(len(dimensions)):
            for invalid in (0, -1, 1.5, True, "2", None):
                values = list(dimensions)
                values[index] = invalid
                with self.subTest(index=index, invalid=invalid), self.assertRaises(ValueError):
                    budget.kv_cache_bytes(*values)
        with self.assertRaises(ValueError):
            budget.kv_cache_bytes(1, 1, 1, 1, 1, 2.0)

    def test_invalid_packing_units_and_blocks(self) -> None:
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                budget.weight_bytes(1, invalid)
            with self.assertRaises(ValueError):
                budget.allocated_tokens(1, invalid)
        for invalid in (-1, True, 0.5):
            with self.assertRaises(ValueError):
                budget.bytes_to_gb(invalid)
            with self.assertRaises(ValueError):
                budget.bytes_to_gib(invalid)

    def test_invalid_overhead_and_capacity(self) -> None:
        valid = dict(
            parameters=1, bits_per_parameter=16, layers=1, kv_heads=1,
            head_dim=1, sequence=1, batch=1, bytes_per_element=2,
        )
        for field in ("workspace_bytes", "allocator_reserve_bytes"):
            for invalid in (-1, True, 0.5):
                with self.assertRaises(ValueError):
                    budget.memory_budget(**valid, **{field: invalid})
        for invalid in (0, -1, True, 0.5):
            with self.assertRaises(ValueError):
                budget.memory_budget(**valid, capacity_bytes=invalid)

    def test_explicitly_unsupported_architectures(self) -> None:
        for architecture in ("mla", "hybrid", "mamba", "", None):
            with self.subTest(architecture=architecture), self.assertRaisesRegex(ValueError, "unsupported"):
                budget.kv_cache_bytes(1, 1, 1, 1, 1, 2, architecture=architecture)

    def test_roofline_independent_reference(self) -> None:
        result = budget.roofline(
            flops=1e12, bytes_moved=1e10, peak_flops_per_s=1e14, memory_bytes_per_s=1e12,
        )
        self.assertEqual(result, {
            "arithmetic_intensity_flops_per_byte": 100.0,
            "ridge_flops_per_byte": 100.0,
            "performance_upper_bound_flops_per_s": 1e14,
            "elapsed_lower_bound_seconds": 0.01,
        })

    def test_roofline_memory_and_compute_bounds(self) -> None:
        memory = budget.roofline(flops=1000, bytes_moved=1000, peak_flops_per_s=1000, memory_bytes_per_s=10)
        self.assertEqual(memory["performance_upper_bound_flops_per_s"], 10)
        self.assertEqual(memory["elapsed_lower_bound_seconds"], 100)
        compute = budget.roofline(flops=1000, bytes_moved=1, peak_flops_per_s=1000, memory_bytes_per_s=10)
        self.assertEqual(compute["performance_upper_bound_flops_per_s"], 1000)
        self.assertEqual(compute["elapsed_lower_bound_seconds"], 1)
        for result in (memory, compute):
            self.assertEqual(1000 / result["performance_upper_bound_flops_per_s"], result["elapsed_lower_bound_seconds"])

    def test_roofline_nonfinite_and_invalid_inputs(self) -> None:
        valid = dict(flops=1e12, bytes_moved=1e10, peak_flops_per_s=1e14, memory_bytes_per_s=1e12)
        for field in valid:
            for invalid in (0, -1, float("nan"), float("inf"), True, "1", 10**400):
                with self.subTest(field=field, invalid=type(invalid).__name__), self.assertRaises(ValueError):
                    budget.roofline(**{**valid, field: invalid})
        with self.assertRaises(ValueError):
            budget.roofline(flops=1e308, bytes_moved=1e-308, peak_flops_per_s=1, memory_bytes_per_s=1)


class ExperimentCliTests(FixtureCase):
    def test_fixed_outputs_reproduce_and_check_without_write(self) -> None:
        fixture = json.loads((run.HERE / "input.json").read_text(encoding="utf-8"))
        result = run.calculate(fixture)
        self.assertEqual(result["memory"]["logical_kv_bytes"], 805_306_368)
        self.assertEqual(result["memory"]["accounted_total_bytes"], 3_342_177_280)
        self.assertEqual(result["memory"]["accounted_headroom_bytes"], 952_790_016)
        with patch.object(run, "HERE", self.root), redirect_stdout(io.StringIO()):
            input_path = self.write_json("input.json", fixture)
            self.assertEqual(run.main(["--input", str(input_path)]), 0)
            contents = {name: (self.root / name).read_bytes() for name in ("results.json", "results.csv")}
            before = {name: (self.root / name).stat().st_mtime_ns for name in contents}
            self.assertEqual(run.main(["--input", str(input_path), "--check"]), 0)
            after = {name: (self.root / name).stat().st_mtime_ns for name in contents}
            self.assertEqual(before, after)
            for name, content in contents.items():
                self.assertEqual((self.root / name).read_bytes(), content)
            self.write("results.csv", "different")
            with redirect_stderr(io.StringIO()):
                self.assertEqual(run.main(["--input", str(input_path), "--check"]), 1)
            self.assertEqual((self.root / "results.csv").read_text(), "different")

    def test_invalid_fixture_explicit_failure(self) -> None:
        path = self.write_json("input.json", {"schema_version": 1, "synthetic": False})
        error = io.StringIO()
        with redirect_stderr(error):
            self.assertEqual(run.main(["--input", str(path)]), 1)
        self.assertIn("Experiment failed", error.getvalue())
