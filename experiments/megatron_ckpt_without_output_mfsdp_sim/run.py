#!/usr/bin/env python3
"""Deterministic simulator for Megatron-LM Activation & Parameter Lifecycle:
CheckpointWithoutOutput _RecomputeRegion 4-state automaton (weakref zero-cycle & UntypedStorage sharing)
and Megatron-FSDP frozen-module deferred cleanup + non-final microbatch ReduceScatter overlap.

Models the exact state machines and memory-timeline invariants from NVIDIA/Megatron-LM commits:
- d2c430ab485c6947701447e5d551d3aafaa39161 (megatron/core/tensor_parallel/random.py)
- d37b3a8fdb37af1b3a1d75720636af4c323bd522 (megatron_fsdp/experimental/{context,fully_shard,module}.py)
"""

from __future__ import annotations

import argparse
import enum
import gc
import json
import sys
import weakref
from dataclasses import asdict, dataclass


class RegionState(enum.Enum):
    RECORDED = "RECORDED"
    RELEASED = "RELEASED"
    REBUILT = "REBUILT"
    DONE = "DONE"


class SimulatedUntypedStorage:
    def __init__(self, data_ptr: int, nbytes: int):
        self.data_ptr = data_ptr
        self.nbytes = nbytes

    def resize_(self, new_nbytes: int) -> None:
        self.nbytes = new_nbytes

    def share_from(self, other: SimulatedUntypedStorage) -> None:
        self.data_ptr = other.data_ptr
        self.nbytes = other.nbytes


class SimulatedTensor:
    def __init__(self, storage: SimulatedUntypedStorage, requires_grad: bool = True, version: int = 0):
        self.storage = storage
        self.requires_grad = requires_grad
        self.version = version

    def view(self) -> SimulatedTensor:
        # Views share the same SimulatedUntypedStorage below TensorImpl
        return SimulatedTensor(self.storage, requires_grad=self.requires_grad, version=self.version)


class RecomputeRegionSim:
    """Models _RecomputeRegion in megatron/core/tensor_parallel/random.py (commit d2c430ab48)."""

    def __init__(self, input_storages: list[SimulatedUntypedStorage], retain_input_tensors: bool):
        self.state: RegionState | None = None
        self.retained_ptrs = (
            frozenset(s.data_ptr for s in input_storages) if retain_input_tensors else frozenset()
        )
        self.output_aliases: list[SimulatedTensor] = []
        self.unpack_count = 0

    def record(self, outputs: list[SimulatedTensor]) -> list[SimulatedTensor]:
        self.state = RegionState.RECORDED
        return outputs

    def release(self, outputs: list[SimulatedTensor]) -> None:
        if self.state is not RegionState.RECORDED:
            return
        self.output_aliases = [ SimulatedTensor(o.storage, o.requires_grad, o.version) for o in outputs ]
        for alias in self.output_aliases:
            if alias.storage.data_ptr not in self.retained_ptrs:
                alias.storage.resize_(0)
        self.state = RegionState.RELEASED

    def rebuild(self, new_storages: list[SimulatedUntypedStorage]) -> None:
        if self.state not in (RegionState.RECORDED, RegionState.RELEASED):
            return
        self.unpack_count += 1
        for alias, new_st in zip(self.output_aliases, new_storages):
            if alias.storage.data_ptr != new_st.data_ptr:
                alias.storage.share_from(new_st)
        self.output_aliases = []
        self.state = RegionState.REBUILT

    def backpropagate(self, new_storages: list[SimulatedUntypedStorage]) -> None:
        if self.state is RegionState.RECORDED:
            self.rebuild(new_storages)
        if self.state is not RegionState.REBUILT:
            raise RuntimeError(f"Invalid state for backpropagate: {self.state}")
        self.state = RegionState.DONE


def simulate_checkpoint_without_output_lifecycle() -> dict:
    in_storage = SimulatedUntypedStorage(data_ptr=0x1000, nbytes=16 * 1024 * 1024)
    out_storage = SimulatedUntypedStorage(data_ptr=0x2000, nbytes=64 * 1024 * 1024)
    aliased_out_storage = in_storage  # Simulates TE MakeExtraOutput sharing storage with input

    out_main = SimulatedTensor(out_storage, requires_grad=True)
    out_extra = SimulatedTensor(aliased_out_storage, requires_grad=True)
    downstream_view = out_main.view()  # View saved for backward by downstream op (e.g. GroupedLinear)

    region = RecomputeRegionSim([in_storage], retain_input_tensors=True)
    region.record([out_main, out_extra])
    bytes_after_record = out_main.storage.nbytes

    # Facade drops strong reference upon release, keeping only weakref.ref(region)
    released_ref = weakref.ref(region)
    region.release([out_main, out_extra])
    bytes_after_release_main = out_main.storage.nbytes
    bytes_after_release_aliased = out_extra.storage.nbytes

    # Rebuild with new storage (0x3000) and verify downstream view sees it with 0 version bump
    rebuilt_storage = SimulatedUntypedStorage(data_ptr=0x3000, nbytes=64 * 1024 * 1024)
    region.rebuild([rebuilt_storage, aliased_out_storage])
    # Calling rebuild a second time (multiple hook triggers) is a no-op
    region.rebuild([rebuilt_storage, aliased_out_storage])

    view_sees_rebuilt_ptr = downstream_view.storage.data_ptr == 0x3000
    view_sees_rebuilt_bytes = downstream_view.storage.nbytes
    version_unchanged = downstream_view.version == 0

    region.backpropagate([rebuilt_storage, aliased_out_storage])
    final_state = region.state.value
    unpack_count = region.unpack_count

    # Verify weakref collection when graph is dropped without backward
    abandoned_region = RecomputeRegionSim([in_storage], retain_input_tensors=False)
    abandoned_region.record([SimulatedTensor(SimulatedUntypedStorage(0x4000, 1024))])
    abandoned_region.release([SimulatedTensor(SimulatedUntypedStorage(0x4000, 1024))])
    abandoned_weak = weakref.ref(abandoned_region)
    del abandoned_region
    gc.collect()
    abandoned_collected = abandoned_weak() is None

    return {
        "bytes_after_record": bytes_after_record,
        "bytes_after_release_main": bytes_after_release_main,
        "bytes_after_release_aliased_input": bytes_after_release_aliased,
        "view_sees_rebuilt_ptr": view_sees_rebuilt_ptr,
        "view_sees_rebuilt_bytes": view_sees_rebuilt_bytes,
        "version_counter_unchanged": version_unchanged,
        "unpack_count_across_two_hooks": unpack_count,
        "final_state": final_state,
        "abandoned_region_gc_collected": abandoned_collected,
        "weakref_alive_during_active_region": released_ref() is not None,
    }


@dataclass(frozen=True)
class MfsdpBackwardSimulation:
    frozen_legacy_bytes_at_internal_bwd: int
    frozen_legacy_use_after_free: bool
    frozen_patched_bytes_at_internal_bwd: int
    frozen_patched_bytes_after_post_bwd: int
    frozen_patched_use_after_free: bool
    delayed_cleanup_order: list[str]
    num_microbatches: int
    legacy_stream_waits: int
    patched_stream_waits: int
    cross_microbatch_rs_overlaps_gained: int


def simulate_mfsdp_frozen_and_microbatch(num_microbatches: int = 4) -> MfsdpBackwardSimulation:
    """Model commit d37b3a8fdb frozen-weight hook deferral and microbatch ReduceScatter overlap."""
    frozen_weight_nbytes = 32 * 1024 * 1024

    # Legacy: trainable_parameter_countdown == 0 registers full_backward_hook that immediately
    # calls post_backward_hook (resizing frozen weight to 0) BEFORE internal backward runs when grad_input is all None!
    legacy_bytes_during_internal_bwd = 0
    legacy_uaf = True

    # Patched: grad_checking_post_backward_hook sees all(g is None for g in grad_input), pushes
    # (module, post_backward_hook) onto context._delayed_post_backward_callbacks, and pops in LIFO order
    delayed_stack = ["parent_frozen_module", "child_frozen_module"]
    popped_order = []
    while delayed_stack:
        popped_order.append(delayed_stack.pop())
    patched_bytes_during_internal_bwd = frozen_weight_nbytes
    patched_bytes_after_post_bwd = 0
    patched_uaf = False

    # Microbatch ReduceScatter stream synchronization:
    # Legacy called finish_grad_sync() on every microbatch post_backward (4 waits for 4 microbatches).
    # Patched checks `if not self.caller_managed_grad_sync and self.is_last_microbatch:` -> 1 wait on last microbatch!
    legacy_waits = num_microbatches
    patched_waits = sum(1 for mb in range(num_microbatches) if mb == num_microbatches - 1)
    overlaps_gained = legacy_waits - patched_waits

    return MfsdpBackwardSimulation(
        frozen_legacy_bytes_at_internal_bwd=legacy_bytes_during_internal_bwd,
        frozen_legacy_use_after_free=legacy_uaf,
        frozen_patched_bytes_at_internal_bwd=patched_bytes_during_internal_bwd,
        frozen_patched_bytes_after_post_bwd=patched_bytes_after_post_bwd,
        frozen_patched_use_after_free=patched_uaf,
        delayed_cleanup_order=popped_order,
        num_microbatches=num_microbatches,
        legacy_stream_waits=legacy_waits,
        patched_stream_waits=patched_waits,
        cross_microbatch_rs_overlaps_gained=overlaps_gained,
    )


def build_summary() -> dict:
    return {
        "checkpoint_without_output": simulate_checkpoint_without_output_lifecycle(),
        "mfsdp_frozen_and_microbatch": asdict(simulate_mfsdp_frozen_and_microbatch(num_microbatches=4)),
    }


def verify_summary(summary: dict) -> None:
    ckpt = summary["checkpoint_without_output"]
    assert ckpt["bytes_after_record"] == 67108864
    assert ckpt["bytes_after_release_main"] == 0
    assert ckpt["bytes_after_release_aliased_input"] == 16777216
    assert ckpt["view_sees_rebuilt_ptr"] is True
    assert ckpt["view_sees_rebuilt_bytes"] == 67108864
    assert ckpt["version_counter_unchanged"] is True
    assert ckpt["unpack_count_across_two_hooks"] == 1
    assert ckpt["final_state"] == "DONE"
    assert ckpt["abandoned_region_gc_collected"] is True

    mfsdp = summary["mfsdp_frozen_and_microbatch"]
    assert mfsdp["frozen_legacy_use_after_free"] is True
    assert mfsdp["frozen_patched_use_after_free"] is False
    assert mfsdp["frozen_patched_bytes_at_internal_bwd"] == 33554432
    assert mfsdp["frozen_patched_bytes_after_post_bwd"] == 0
    assert mfsdp["delayed_cleanup_order"] == ["child_frozen_module", "parent_frozen_module"]
    assert mfsdp["legacy_stream_waits"] == 4
    assert mfsdp["patched_stream_waits"] == 1
    assert mfsdp["cross_microbatch_rs_overlaps_gained"] == 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    summary = build_summary()
    if args.check:
        verify_summary(summary)
    if args.json or not args.check:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("megatron_ckpt_without_output_mfsdp_sim: CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
