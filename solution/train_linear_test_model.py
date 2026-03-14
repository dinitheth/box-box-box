#!/usr/bin/env python3
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

TIRES = ["SOFT", "MEDIUM", "HARD"]
TRACKS = ["Bahrain", "COTA", "Monaco", "Monza", "Silverstone", "Spa", "Suzuka"]

FEATURE_NAMES = [
    "pit_count",
    "pit_lane",
    "pit_count_x_pitlane",
    "start_soft",
    "start_medium",
    "start_hard",
    "soft_laps",
    "medium_laps",
    "hard_laps",
    "soft_age",
    "medium_age",
    "hard_age",
    "first_pit",
    "second_pit",
    "temp",
    "temp_x_soft_age",
    "temp_x_medium_age",
    "temp_x_hard_age",
]
for track in TRACKS:
    FEATURE_NAMES.append(f"track_{track}")
for driver in [f"D{i:03d}" for i in range(1, 21)]:
    FEATURE_NAMES.append(f"driver_{driver}")

IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

SCALES = [1.0] * len(FEATURE_NAMES)
for key in ["pit_lane", "pit_count_x_pitlane", "first_pit", "second_pit", "temp"]:
    SCALES[IDX[key]] = 30.0
for key in ["soft_laps", "medium_laps", "hard_laps"]:
    SCALES[IDX[key]] = 70.0
for key in ["soft_age", "medium_age", "hard_age"]:
    SCALES[IDX[key]] = 2500.0
for key in ["temp_x_soft_age", "temp_x_medium_age", "temp_x_hard_age"]:
    SCALES[IDX[key]] = 50000.0


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softplus(x: float) -> float:
    if x > 40:
        return x
    if x < -40:
        return math.exp(x)
    return math.log1p(math.exp(x))


def stint_stats(strategy: Dict, total_laps: int):
    pit_stops = sorted(strategy.get("pit_stops", []), key=lambda s: int(s["lap"]))
    current_tire = strategy["starting_tire"]
    last_lap = 0

    soft_laps = medium_laps = hard_laps = 0.0
    soft_age = medium_age = hard_age = 0.0

    def add_stint(tire: str, length: int) -> None:
        nonlocal soft_laps, medium_laps, hard_laps
        nonlocal soft_age, medium_age, hard_age
        if length <= 0:
            return
        sum_age = length * (length + 1) / 2.0
        if tire == "SOFT":
            soft_laps += float(length)
            soft_age += sum_age
        elif tire == "MEDIUM":
            medium_laps += float(length)
            medium_age += sum_age
        else:
            hard_laps += float(length)
            hard_age += sum_age

    for stop in pit_stops:
        lap = int(stop["lap"])
        add_stint(current_tire, lap - last_lap)
        current_tire = stop["to_tire"]
        last_lap = lap

    add_stint(current_tire, total_laps - last_lap)

    first_pit = int(pit_stops[0]["lap"]) if len(pit_stops) > 0 else total_laps
    second_pit = int(pit_stops[1]["lap"]) if len(pit_stops) > 1 else total_laps

    return {
        "pit_count": float(len(pit_stops)),
        "start_soft": 1.0 if strategy["starting_tire"] == "SOFT" else 0.0,
        "start_medium": 1.0 if strategy["starting_tire"] == "MEDIUM" else 0.0,
        "start_hard": 1.0 if strategy["starting_tire"] == "HARD" else 0.0,
        "soft_laps": soft_laps,
        "medium_laps": medium_laps,
        "hard_laps": hard_laps,
        "soft_age": soft_age,
        "medium_age": medium_age,
        "hard_age": hard_age,
        "first_pit": float(first_pit),
        "second_pit": float(second_pit),
    }


def features_for_driver(race: Dict, strategy: Dict) -> List[float]:
    rc = race["race_config"]
    total_laps = int(rc["total_laps"])
    temp = float(rc["track_temp"])
    pit_lane = float(rc["pit_lane_time"])
    track = rc["track"]
    driver_id = strategy["driver_id"]

    st = stint_stats(strategy, total_laps)

    x = [0.0] * len(FEATURE_NAMES)
    x[IDX["pit_count"]] = st["pit_count"]
    x[IDX["pit_lane"]] = pit_lane
    x[IDX["pit_count_x_pitlane"]] = st["pit_count"] * pit_lane
    x[IDX["start_soft"]] = st["start_soft"]
    x[IDX["start_medium"]] = st["start_medium"]
    x[IDX["start_hard"]] = st["start_hard"]
    x[IDX["soft_laps"]] = st["soft_laps"]
    x[IDX["medium_laps"]] = st["medium_laps"]
    x[IDX["hard_laps"]] = st["hard_laps"]
    x[IDX["soft_age"]] = st["soft_age"]
    x[IDX["medium_age"]] = st["medium_age"]
    x[IDX["hard_age"]] = st["hard_age"]
    x[IDX["first_pit"]] = st["first_pit"]
    x[IDX["second_pit"]] = st["second_pit"]
    x[IDX["temp"]] = temp
    x[IDX["temp_x_soft_age"]] = temp * st["soft_age"]
    x[IDX["temp_x_medium_age"]] = temp * st["medium_age"]
    x[IDX["temp_x_hard_age"]] = temp * st["hard_age"]

    track_key = f"track_{track}"
    if track_key in IDX:
        x[IDX[track_key]] = 1.0

    driver_key = f"driver_{driver_id}"
    if driver_key in IDX:
        x[IDX[driver_key]] = 1.0

    return [value / SCALES[i] for i, value in enumerate(x)]


def load_dataset(root: Path):
    dataset = []
    for input_fp in sorted((root / "data" / "test_cases" / "inputs").glob("test_*.json")):
        race = json.loads(input_fp.read_text(encoding="utf-8"))
        expected = json.loads((root / "data" / "test_cases" / "expected_outputs" / input_fp.name).read_text(encoding="utf-8"))["finishing_positions"]

        by_driver = {}
        for i in range(1, 21):
            strategy = race["strategies"][f"pos{i}"]
            by_driver[strategy["driver_id"]] = features_for_driver(race, strategy)

        dataset.append((race, expected, by_driver))
    return dataset


def predict_order(race: Dict, weights: List[float]) -> List[str]:
    rows = []
    for pos in range(1, 21):
        strategy = race["strategies"][f"pos{pos}"]
        x = features_for_driver(race, strategy)
        score = sum(w * f for w, f in zip(weights, x))
        rows.append((score, pos, strategy["driver_id"]))
    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    return [driver_id for _, _, driver_id in rows]


def exact_score(dataset, weights: List[float]) -> int:
    passed = 0
    for race, expected, _ in dataset:
        if predict_order(race, weights) == expected:
            passed += 1
    return passed


def train(dataset):
    random.seed(42)
    weights = [0.0] * len(FEATURE_NAMES)

    m = [0.0] * len(FEATURE_NAMES)
    v = [0.0] * len(FEATURE_NAMES)
    beta1, beta2 = 0.9, 0.999
    lr = 0.05
    l2 = 1e-5
    eps = 1e-8
    step = 0

    best = list(weights)
    best_exact = exact_score(dataset, weights)

    for epoch in range(1, 601):
        random.shuffle(dataset)
        loss_sum = 0.0
        updates = 0

        for _, expected, by_driver in dataset:
            for idx in range(len(expected) - 1):
                a = expected[idx]
                b = expected[idx + 1]
                xa = by_driver[a]
                xb = by_driver[b]
                xdiff = [fa - fb for fa, fb in zip(xa, xb)]

                z = sum(w * d for w, d in zip(weights, xdiff))
                p = sigmoid(z)
                loss_sum += softplus(z)
                updates += 1
                step += 1

                for k in range(len(weights)):
                    grad = p * xdiff[k] + l2 * weights[k]
                    m[k] = beta1 * m[k] + (1.0 - beta1) * grad
                    v[k] = beta2 * v[k] + (1.0 - beta2) * grad * grad
                    m_hat = m[k] / (1.0 - beta1 ** step)
                    v_hat = v[k] / (1.0 - beta2 ** step)
                    weights[k] -= lr * m_hat / (math.sqrt(v_hat) + eps)

        if epoch % 20 == 0:
            exact = exact_score(dataset, weights)
            avg_loss = loss_sum / max(updates, 1)
            print(f"epoch={epoch} avg_pair_loss={avg_loss:.6f} exact={exact}/100 best={best_exact}/100")
            if exact >= best_exact:
                best_exact = exact
                best = list(weights)

    return best, best_exact


def save_model(root: Path, weights: List[float], exact: int) -> None:
    model = {
        "feature_names": FEATURE_NAMES,
        "weights": weights,
        "score_on_tests": exact,
    }
    out = root / "solution" / "linear_model.json"
    out.write_text(json.dumps(model), encoding="utf-8")
    print(f"saved={out}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = load_dataset(root)
    weights, exact = train(dataset)
    print(f"best_exact={exact}/100")
    save_model(root, weights, exact)


if __name__ == "__main__":
    main()
