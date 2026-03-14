#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run_command = (root / "solution" / "run_command.txt").read_text(encoding="utf-8").strip()

    input_files = sorted((root / "data" / "test_cases" / "inputs").glob("test_*.json"))
    expected_dir = root / "data" / "test_cases" / "expected_outputs"

    passed = 0
    failed = 0
    errors = 0

    for index, input_file in enumerate(input_files, start=1):
        test_name = input_file.stem
        expected_file = expected_dir / f"{test_name}.json"

        proc = subprocess.run(
            run_command,
            input=input_file.read_text(encoding="utf-8"),
            text=True,
            capture_output=True,
            shell=True,
            cwd=root,
        )

        if proc.returncode != 0:
            errors += 1
            continue

        try:
            predicted_output = json.loads(proc.stdout)
            predicted_positions = predicted_output["finishing_positions"]
        except Exception:
            failed += 1
            continue

        if expected_file.exists():
            expected_positions = json.loads(expected_file.read_text(encoding="utf-8"))["finishing_positions"]
            if predicted_positions == expected_positions:
                passed += 1
            else:
                failed += 1
        else:
            passed += 1

        if index % 10 == 0:
            print(f"PROGRESS {index}/{len(input_files)}", flush=True)

    total = len(input_files)
    pass_rate = round((passed * 100.0 / total), 1) if total else 0.0

    print(f"TOTAL {total}")
    print(f"PASSED {passed}")
    print(f"FAILED {failed}")
    print(f"ERRORS {errors}")
    print(f"PASS_RATE {pass_rate}")


if __name__ == "__main__":
    main()
