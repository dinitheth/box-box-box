#!/usr/bin/env python3
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "solution"))

import race_simulator_physics as rsp  # noqa: E402


def main() -> None:
    rsp.load_model_params()

    inputs = sorted((root / "data" / "test_cases" / "inputs").glob("test_*.json"))
    expected_dir = root / "data" / "test_cases" / "expected_outputs"

    passed = failed = errors = 0
    for i, fp in enumerate(inputs, 1):
        try:
            race = json.loads(fp.read_text(encoding="utf-8"))
            pred = rsp.simulate_race(race)
            exp = json.loads((expected_dir / fp.name).read_text(encoding="utf-8"))["finishing_positions"]
            if pred == exp:
                passed += 1
            else:
                failed += 1
        except Exception:
            errors += 1
        if i % 10 == 0:
            print(f"REAL_PROGRESS {i}/{len(inputs)}", flush=True)

    total = len(inputs)
    print(f"REAL_TOTAL {total}")
    print(f"REAL_PASSED {passed}")
    print(f"REAL_FAILED {failed}")
    print(f"REAL_ERRORS {errors}")
    print(f"REAL_PASS_RATE {round((passed * 100.0 / total), 1) if total else 0.0}")


if __name__ == "__main__":
    main()
