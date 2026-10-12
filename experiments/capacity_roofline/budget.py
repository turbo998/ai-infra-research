"""Analytical payload accounting and Roofline bounds, never measured timings."""

import math


def integer(value: object, field: str, *, allow_zero: bool = False) -> int:
    """Require an actual integer, excluding booleans and fractional element widths."""
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{field}: expected {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def positive_number(value: object, field: str) -> float:
    """Require a finite positive numeric scalar, excluding booleans."""
    if type(value) not in {int, float}:
        raise ValueError(f"{field}: expected finite positive number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field}: expected finite positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field}: expected finite positive number")
    return number


def bytes_to_gb(byte_count: int) -> float:
    """Convert bytes to decimal GB (1 GB = 10**9 bytes)."""
    return integer(byte_count, "byte_count", allow_zero=True) / 10**9


def bytes_to_gib(byte_count: int) -> float:
    """Convert bytes to binary GiB (1 GiB = 2**30 bytes)."""
    return integer(byte_count, "byte_count", allow_zero=True) / 2**30


def weight_bytes(parameters: int, bits_per_parameter: int) -> int:
    """Return ceiling-rounded packed weight payload; exclude quantization metadata."""
    bits = integer(parameters, "parameters") * integer(bits_per_parameter, "bits_per_parameter")
    return (bits + 7) // 8


def allocated_tokens(sequence: int, token_block_size: int) -> int:
    """Round a positive per-sequence length up to a whole allocation block."""
    sequence = integer(sequence, "sequence")
    block = integer(token_block_size, "token_block_size")
    return ((sequence + block - 1) // block) * block


def kv_cache_bytes(
    layers: int, kv_heads: int, head_dim: int, sequence: int, batch: int,
    bytes_per_element: int, *, architecture: str = "gqa",
) -> int:
    """Compute 2*layers*kv_heads*head_dim*sequence*batch*element_bytes for MHA/GQA."""
    if not isinstance(architecture, str) or architecture.lower() not in {"mha", "gqa"}:
        raise ValueError("architecture: only conventional MHA/GQA supported; MLA/hybrid unsupported")
    dimensions = (
        (layers, "layers"), (kv_heads, "kv_heads"), (head_dim, "head_dim"),
        (sequence, "sequence"), (batch, "batch"), (bytes_per_element, "bytes_per_element"),
    )
    return 2 * math.prod(integer(value, name) for value, name in dimensions)


def memory_budget(
    *, parameters: int, bits_per_parameter: int, layers: int, kv_heads: int,
    head_dim: int, sequence: int, batch: int, bytes_per_element: int,
    architecture: str = "gqa", token_block_size: int = 1,
    workspace_bytes: int = 0, allocator_reserve_bytes: int = 0,
    capacity_bytes: int | None = None,
) -> dict:
    """Account separately for payload and explicit overhead using allocated KV."""
    weights = weight_bytes(parameters, bits_per_parameter)
    workspace = integer(workspace_bytes, "workspace_bytes", allow_zero=True)
    reserve = integer(allocator_reserve_bytes, "allocator_reserve_bytes", allow_zero=True)
    allocated_sequence = allocated_tokens(sequence, token_block_size)
    logical_kv = kv_cache_bytes(
        layers, kv_heads, head_dim, sequence, batch, bytes_per_element, architecture=architecture,
    )
    allocated_kv = kv_cache_bytes(
        layers, kv_heads, head_dim, allocated_sequence, batch, bytes_per_element, architecture=architecture,
    )
    if capacity_bytes is not None:
        integer(capacity_bytes, "capacity_bytes")
    total = weights + allocated_kv + workspace + reserve
    return {
        "weight_payload_bytes": weights,
        "logical_kv_bytes": logical_kv,
        "allocated_sequence_tokens": allocated_sequence,
        "allocated_kv_bytes": allocated_kv,
        "kv_allocation_padding_bytes": allocated_kv - logical_kv,
        "workspace_bytes": workspace,
        "allocator_reserve_bytes": reserve,
        "accounted_total_bytes": total,
        "accounted_total_gb": bytes_to_gb(total),
        "accounted_total_gib": bytes_to_gib(total),
        "capacity_bytes": capacity_bytes,
        "accounted_headroom_bytes": None if capacity_bytes is None else capacity_bytes - total,
        "fits_accounted_capacity": None if capacity_bytes is None else total <= capacity_bytes,
    }


def roofline(
    *, flops: float, bytes_moved: float, peak_flops_per_s: float,
    memory_bytes_per_s: float,
) -> dict[str, float]:
    """Compute ideal upper performance and lower elapsed-time bounds, not timings."""
    work = positive_number(flops, "flops")
    traffic = positive_number(bytes_moved, "bytes_moved")
    peak = positive_number(peak_flops_per_s, "peak_flops_per_s")
    bandwidth = positive_number(memory_bytes_per_s, "memory_bytes_per_s")
    intensity = work / traffic
    ridge = peak / bandwidth
    performance = min(peak, bandwidth * intensity)
    lower_bound = max(work / peak, traffic / bandwidth)
    values = {
        "arithmetic_intensity_flops_per_byte": intensity,
        "ridge_flops_per_byte": ridge,
        "performance_upper_bound_flops_per_s": performance,
        "elapsed_lower_bound_seconds": lower_bound,
    }
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError("roofline: derived bounds are not finite positive numbers")
    return values
