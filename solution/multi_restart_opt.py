#!/usr/bin/env python3
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "solution"
CHECKPOINTS = SOLUTION / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True)

SEEDS = [int(x) for x in os.environ.get("MR_SEEDS", "41,42,123,777,2026,9001").split(",") if x.strip()]
ITERS = int(os.environ.get("MR_ITERS", "8000"))


def best_checkpoint_path():
    best_passed = -1
    best_path = None
    for p in CHECKPOINTS.glob("best_passed_*.json"):
        m = re.search(r"best_passed_(\d+)", p.name)
        if not m:
            continue
        passed = int(m.group(1))
        if passed > best_passed:
            best_passed = passed
            best_path = p
    return best_path, best_passed


def run_cmd(cmd: str, env=None):
    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_cmd_stream(cmd: str, env=None):
    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    return proc.returncode


def parse_real_pass_rate(output: str):
    m_passed = re.search(r"REAL_PASSED\s+(\d+)", output)
    m_rate = re.search(r"REAL_PASS_RATE\s+([0-9.]+)", output)
    if not m_passed or not m_rate:
        return None, None
    return int(m_passed.group(1)), float(m_rate.group(1))


def evaluate_real():
    code, out, err = run_cmd("python solution/evaluate_real.py")
    if code != 0:
        raise RuntimeError(f"evaluate_real failed:\n{out}\n{err}")
    passed, rate = parse_real_pass_rate(out)
    if passed is None:
        raise RuntimeError(f"Could not parse evaluate_real output:\n{out}")
    return passed, rate, out


def main():
    print("Starting multi-restart optimization")
    base_passed, base_rate, _ = evaluate_real()
    if base_passed <= 0:
        ckpt, ckpt_passed = best_checkpoint_path()
        if ckpt is not None and ckpt_passed > 0:
            print(f"Baseline is {base_passed}; restoring checkpoint {ckpt}")
            shutil.copyfile(ckpt, SOLUTION / "model_params.json")
            base_passed, base_rate, _ = evaluate_real()
    print(f"Baseline: passed={base_passed} rate={base_rate}")

    best_passed = base_passed
    best_rate = base_rate

    best_path = CHECKPOINTS / f"best_passed_{best_passed}.json"
    shutil.copyfile(SOLUTION / "model_params.json", best_path)

    for seed in SEEDS:
        print(f"\n=== Restart seed={seed} iters={ITERS} ===")
        env = os.environ.copy()
        env["OPT_SEED"] = str(seed)
        env["OPT_ITERS"] = str(ITERS)

        code = run_cmd_stream("python solution/optimize_physics.py", env=env)
        if code != 0:
            print("Optimizer failed for seed", seed)
            continue

        passed, rate, _ = evaluate_real()
        print(f"seed={seed} -> passed={passed} rate={rate}")

        ckpt = CHECKPOINTS / f"seed_{seed}_passed_{passed}.json"
        shutil.copyfile(SOLUTION / "model_params.json", ckpt)

        if passed > best_passed or (passed == best_passed and rate > best_rate):
            best_passed = passed
            best_rate = rate
            best_path = CHECKPOINTS / f"best_passed_{best_passed}_seed_{seed}.json"
            shutil.copyfile(SOLUTION / "model_params.json", best_path)
            print(f"NEW BEST: passed={best_passed} rate={best_rate}")

    shutil.copyfile(best_path, SOLUTION / "model_params.json")
    final_passed, final_rate, _ = evaluate_real()
    print("\n=== FINAL ===")
    print(f"Best checkpoint: {best_path}")
    print(f"Final restored score: passed={final_passed} rate={final_rate}")


if __name__ == "__main__":
    main()
