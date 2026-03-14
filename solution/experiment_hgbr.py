import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

TIRES = ["SOFT", "MEDIUM", "HARD"]
TRACKS = ["Bahrain", "COTA", "Monaco", "Monza", "Silverstone", "Spa", "Suzuka"]


def stint_stats(strategy, total_laps):
    pit_stops = sorted(strategy.get("pit_stops", []), key=lambda s: int(s["lap"]))
    current_tire = strategy["starting_tire"]
    last_lap = 0

    laps = {t: 0.0 for t in TIRES}
    sum_age = {t: 0.0 for t in TIRES}
    sum_age2 = {t: 0.0 for t in TIRES}

    for stop in pit_stops:
        lap = int(stop["lap"])
        length = lap - last_lap
        if length > 0 and current_tire in laps:
            laps[current_tire] += float(length)
            sum_age[current_tire] += length * (length + 1) / 2.0
            sum_age2[current_tire] += length * (length + 1) * (2 * length + 1) / 6.0
        current_tire = stop["to_tire"]
        last_lap = lap

    tail = total_laps - last_lap
    if tail > 0 and current_tire in laps:
        laps[current_tire] += float(tail)
        sum_age[current_tire] += tail * (tail + 1) / 2.0
        sum_age2[current_tire] += tail * (tail + 1) * (2 * tail + 1) / 6.0

    first_pit = int(pit_stops[0]["lap"]) if len(pit_stops) > 0 else total_laps
    second_pit = int(pit_stops[1]["lap"]) if len(pit_stops) > 1 else total_laps
    return laps, sum_age, sum_age2, len(pit_stops), first_pit, second_pit


def row_features(race, pos):
    rc = race["race_config"]
    total_laps = int(rc["total_laps"])
    strategy = race["strategies"][f"pos{pos}"]

    laps, sum_age, sum_age2, pit_count, first_pit, second_pit = stint_stats(strategy, total_laps)

    track_oh = [1.0 if rc["track"] == t else 0.0 for t in TRACKS]
    tire_oh = [1.0 if strategy["starting_tire"] == t else 0.0 for t in TIRES]
    driver_num = float(int(strategy["driver_id"][1:]))

    return [
        float(total_laps),
        float(rc["base_lap_time"]),
        float(rc["pit_lane_time"]),
        float(rc["track_temp"]),
        float(pos),
        driver_num,
        float(pit_count),
        float(first_pit),
        float(second_pit),
        laps["SOFT"], laps["MEDIUM"], laps["HARD"],
        sum_age["SOFT"], sum_age["MEDIUM"], sum_age["HARD"],
        sum_age2["SOFT"], sum_age2["MEDIUM"], sum_age2["HARD"],
        *track_oh,
        *tire_oh,
    ]


def build_train(root):
    X = []
    y = []
    for fp in sorted((root / "data" / "historical_races").glob("*.json")):
        races = json.loads(fp.read_text(encoding="utf-8"))
        for race in races:
            finish_rank = {d: i for i, d in enumerate(race["finishing_positions"], 1)}
            for pos in range(1, 21):
                strategy = race["strategies"][f"pos{pos}"]
                X.append(row_features(race, pos))
                y.append(float(finish_rank[strategy["driver_id"]]))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def evaluate_tests(root, model):
    passed = 0
    total = 0
    for fp in sorted((root / "data" / "test_cases" / "inputs").glob("test_*.json")):
        race = json.loads(fp.read_text(encoding="utf-8"))
        exp = json.loads((root / "data" / "test_cases" / "expected_outputs" / fp.name).read_text(encoding="utf-8"))["finishing_positions"]

        rows = []
        for pos in range(1, 21):
            strategy = race["strategies"][f"pos{pos}"]
            x = np.asarray(row_features(race, pos), dtype=np.float32).reshape(1, -1)
            pred_rank = float(model.predict(x)[0])
            rows.append((pred_rank, pos, strategy["driver_id"]))

        rows.sort(key=lambda r: (r[0], r[1], r[2]))
        pred = [d for _, _, d in rows]
        passed += int(pred == exp)
        total += 1

    print("HGBR_PASS", passed)
    print("HGBR_TOTAL", total)
    print("HGBR_RATE", round(passed * 100.0 / total, 1))


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    X, y = build_train(root)
    print("train_shape", X.shape)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=500,
        max_leaf_nodes=63,
        min_samples_leaf=30,
        l2_regularization=1e-3,
        random_state=42,
        early_stopping=False,
    )
    model.fit(X, y)
    evaluate_tests(root, model)
