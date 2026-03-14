#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Dict, List

MODEL_PATH = Path(__file__).with_name("linear_model.json")
TRACKS = ["Bahrain", "COTA", "Monaco", "Monza", "Silverstone", "Spa", "Suzuka"]


def validate_test_case(test_case: Dict) -> None:
    if not isinstance(test_case, dict):
        raise ValueError("Input must be JSON object")
    for key in ("race_id", "race_config", "strategies"):
        if key not in test_case:
            raise ValueError(f"Missing field: {key}")
    for i in range(1, 21):
        if f"pos{i}" not in test_case["strategies"]:
            raise ValueError(f"Missing strategy: pos{i}")


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


def features_for_driver(race: Dict, strategy: Dict, feature_names: List[str]) -> List[float]:
    rc = race["race_config"]
    total_laps = int(rc["total_laps"])
    temp = float(rc["track_temp"])
    pit_lane = float(rc["pit_lane_time"])
    track = rc["track"]
    driver_id = strategy["driver_id"]

    st = stint_stats(strategy, total_laps)

    vals = {
        "pit_count": st["pit_count"],
        "pit_lane": pit_lane,
        "pit_count_x_pitlane": st["pit_count"] * pit_lane,
        "start_soft": st["start_soft"],
        "start_medium": st["start_medium"],
        "start_hard": st["start_hard"],
        "soft_laps": st["soft_laps"],
        "medium_laps": st["medium_laps"],
        "hard_laps": st["hard_laps"],
        "soft_age": st["soft_age"],
        "medium_age": st["medium_age"],
        "hard_age": st["hard_age"],
        "first_pit": st["first_pit"],
        "second_pit": st["second_pit"],
        "temp": temp,
        "temp_x_soft_age": temp * st["soft_age"],
        "temp_x_medium_age": temp * st["medium_age"],
        "temp_x_hard_age": temp * st["hard_age"],
    }

    for t in TRACKS:
        vals[f"track_{t}"] = 1.0 if track == t else 0.0

    vals[f"driver_{driver_id}"] = 1.0

    out = []
    for name in feature_names:
        out.append(float(vals.get(name, 0.0)))
    return out


def fallback_positions(test_case: Dict) -> List[str]:
    if isinstance(test_case, dict) and isinstance(test_case.get("strategies"), dict):
        return [test_case["strategies"].get(f"pos{i}", {}).get("driver_id", f"D{i:03d}") for i in range(1, 21)]
    return [f"D{i:03d}" for i in range(1, 21)]


def load_model() -> Dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("linear_model.json not found")
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


def predict(test_case: Dict, model: Dict) -> List[str]:
    feature_names = model["feature_names"]
    weights = [float(x) for x in model["weights"]]

    rows = []
    for pos in range(1, 21):
        strategy = test_case["strategies"][f"pos{pos}"]
        x = features_for_driver(test_case, strategy, feature_names)
        score = sum(w * f for w, f in zip(weights, x))
        rows.append((score, pos, strategy["driver_id"]))

    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    result = [driver_id for _, _, driver_id in rows]

    if len(result) != 20 or len(set(result)) != 20:
        return fallback_positions(test_case)
    return result


def main() -> None:
    test_case = None
    try:
        test_case = json.load(sys.stdin)
        validate_test_case(test_case)
        model = load_model()
        positions = predict(test_case, model)
        print(json.dumps({"race_id": test_case["race_id"], "finishing_positions": positions}))
    except Exception:
        race_id = test_case.get("race_id", "UNKNOWN") if isinstance(test_case, dict) else "UNKNOWN"
        print(json.dumps({"race_id": race_id, "finishing_positions": fallback_positions(test_case)}))


if __name__ == "__main__":
    main()
