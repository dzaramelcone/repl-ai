#!/usr/bin/env python3
"""Pre-commit hook to enforce Halstead complexity thresholds."""

import json
import subprocess
import sys

THRESHOLDS = {
    "h1": 14,
    "h2": 43,
    "N1": 27,
    "N2": 51,
    "vocabulary": 56,
    "length": 78,
    "calculated_length": 280,
    "volume": 453,
    "difficulty": 8.5,
    "effort": 3851,
    "time": 214,
    "bugs": 0.16,
}


def check_halstead(files: list[str]) -> int:
    """Run radon hal and check against thresholds. Returns exit code."""
    # Exclude test files - they naturally have high h2 (unique operands) due to
    # unique test names, string literals, and variable names in each test
    files = [f for f in files if not f.startswith("tests/")]
    if not files:
        return 0

    result = subprocess.run(
        ["radon", "hal", "--json", *files],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"radon hal failed: {result.stderr}", file=sys.stderr)
        return 1

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"Failed to parse radon output: {e}", file=sys.stderr)
        return 1

    violations = []

    for filepath, metrics in data.items():
        total = metrics.get("total")
        if not total:
            continue

        for metric, threshold in THRESHOLDS.items():
            value = total.get(metric, 0)
            if value > threshold:
                violations.append(f"  {filepath}: {metric}={value:.2f} (max {threshold})")

    if violations:
        print("Halstead complexity violations:")
        for v in violations:
            print(v)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(check_halstead(sys.argv[1:]))
