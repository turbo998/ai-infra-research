"""Regenerate deterministic synthetic JSON/CSV, or check them without modification."""

import argparse
import csv
import io
import json
from pathlib import Path
import sys

if __package__:
    from .budget import memory_budget, roofline
else:
    from budget import memory_budget, roofline


HERE = Path(__file__).resolve().parent
LIMITATIONS = [
    "own-offline synthetic arithmetic only; no measured hardware performance or timings",
    "KV formula supports conventional MHA/GQA only, not MLA or hybrid architectures",
    "packed weights omit quantization scales, zero points, metadata, and alignment",
    "workspace and allocator reserve are user inputs; omitted runtime overhead is not zero in reality",
    "capacity comparison covers accounted bytes only; no capacity supplied means fit is not assessed",
    "Roofline omits launches, latency, communication, cache effects, utilization, and overlap constraints",
    "no model-quality, accuracy, serving-throughput, or network-bandwidth claims",
]


def calculate(fixture: dict) -> dict:
    """Calculate versioned own-offline results from an explicit synthetic fixture."""
    if (
        not isinstance(fixture, dict) or type(fixture.get("schema_version")) is not int
        or fixture["schema_version"] != 1 or fixture.get("synthetic") is not True
        or not isinstance(fixture.get("memory"), dict) or not isinstance(fixture.get("roofline"), dict)
    ):
        raise ValueError("fixture: expected version-one synthetic memory and roofline inputs")
    return {
        "schema_version": 1, "evidence_status": "own-offline", "synthetic": True,
        "memory": memory_budget(**fixture["memory"]),
        "roofline": roofline(**fixture["roofline"]),
        "limitations": LIMITATIONS,
    }


def render(fixture: dict) -> dict[str, str]:
    """Return deterministic JSON and flat CSV representations of the same results."""
    result = calculate(fixture)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("evidence_status", "section", "metric", "value"))
    for section in ("memory", "roofline"):
        for metric, value in result[section].items():
            formatted = "not-assessed" if value is None else json.dumps(value, allow_nan=False)
            writer.writerow(("own-offline", section, metric, formatted))
    return {
        "results.json": json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        "results.csv": stream.getvalue(),
    }


def main(argv: list[str] | None = None) -> int:
    """Regenerate the fixed outputs, or verify their exact bytes using --check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "input.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        fixture = json.loads(args.input.read_text(encoding="utf-8"))
        outputs = render(fixture)
        if args.check:
            for name, content in outputs.items():
                path = HERE / name
                if not path.is_file() or path.read_bytes() != content.encode("utf-8"):
                    print(f"Experiment check failed: missing or different {name}", file=sys.stderr)
                    return 1
        else:
            for name, content in outputs.items():
                with (HERE / name).open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write(content)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        print("Experiment failed: unreadable/invalid fixture or output", file=sys.stderr)
        return 1
    print("own-offline outputs verified" if args.check else "own-offline outputs regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
