#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

DRIVER_IDS = [f"D{i:03d}" for i in range(1, 21)]
DRIVER_INDEX = {driver_id: idx for idx, driver_id in enumerate(DRIVER_IDS)}
MODEL_PATH = Path(__file__).with_name("hybrid_model.json")

WEIGHTS = [
    0.80,
    2.50,
    3.20,
    1.20,
    0.65,
    0.65,
    0.65,
    0.90,
    1.10,
    1.40,
    2.00,
    0.90,
    0.70,
    0.60,
    0.60,
    0.60,
    0.018,
    0.018,
    0.018,
]

K_NEIGHBORS = 25
CLUSTER_WEIGHT = 0.00
BUCKET_WEIGHT = 0.95
DRIVER_STRENGTH_WEIGHT = 0.25
START_POS_WEIGHT = 0.02


def strategy_signature(strategy: Dict) -> Tuple:
    pit_stops = sorted(strategy.get("pit_stops", []), key=lambda s: int(s["lap"]))
    return (
        strategy["starting_tire"],
        tuple((int(stop["lap"]), stop["from_tire"], stop["to_tire"]) for stop in pit_stops),
    )


def stint_stats(strategy: Dict, total_laps: int) -> Tuple[float, float, float, float, float, float, int, int, int]:
    pit_stops = sorted(strategy.get("pit_stops", []), key=lambda s: int(s["lap"]))
    current_tire = strategy["starting_tire"]
    last_lap = 0

    soft_laps = 0.0
    medium_laps = 0.0
    hard_laps = 0.0
    soft_age = 0.0
    medium_age = 0.0
    hard_age = 0.0

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
        elif tire == "HARD":
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

    return (
        soft_laps,
        medium_laps,
        hard_laps,
        soft_age,
        medium_age,
        hard_age,
        len(pit_stops),
        first_pit,
        second_pit,
    )


def race_feature(test_case: Dict) -> List[float]:
    rc = test_case["race_config"]
    total_laps = int(rc["total_laps"])

    start_soft = 0
    start_medium = 0
    start_hard = 0

    pit0 = 0
    pit1 = 0
    pit2 = 0
    pit3p = 0

    first_pits: List[int] = []
    second_pits: List[int] = []

    soft_laps_total = 0.0
    medium_laps_total = 0.0
    hard_laps_total = 0.0
    soft_age_total = 0.0
    medium_age_total = 0.0
    hard_age_total = 0.0

    for pos in range(1, 21):
        strategy = test_case["strategies"][f"pos{pos}"]
        st = strategy["starting_tire"]
        if st == "SOFT":
            start_soft += 1
        elif st == "MEDIUM":
            start_medium += 1
        elif st == "HARD":
            start_hard += 1

        soft_laps, medium_laps, hard_laps, soft_age, medium_age, hard_age, pit_count, first_pit, second_pit = stint_stats(
            strategy,
            total_laps,
        )

        if pit_count == 0:
            pit0 += 1
        elif pit_count == 1:
            pit1 += 1
        elif pit_count == 2:
            pit2 += 1
        else:
            pit3p += 1

        first_pits.append(first_pit)
        second_pits.append(second_pit)

        soft_laps_total += soft_laps
        medium_laps_total += medium_laps
        hard_laps_total += hard_laps
        soft_age_total += soft_age
        medium_age_total += medium_age
        hard_age_total += hard_age

    return [
        float(total_laps),
        float(rc["base_lap_time"]),
        float(rc["pit_lane_time"]),
        float(rc["track_temp"]),
        float(start_soft),
        float(start_medium),
        float(start_hard),
        float(pit0),
        float(pit1),
        float(pit2),
        float(pit3p),
        float(sum(first_pits) / len(first_pits)),
        float(sum(second_pits) / len(second_pits)),
        float(soft_laps_total / 20.0),
        float(medium_laps_total / 20.0),
        float(hard_laps_total / 20.0),
        float(soft_age_total / 20.0),
        float(medium_age_total / 20.0),
        float(hard_age_total / 20.0),
    ]


def strategy_bucket(strategy: Dict, race_config: Dict) -> str:
    total_laps = int(race_config["total_laps"])
    track = race_config["track"]
    base = float(race_config["base_lap_time"])
    pit_lane = float(race_config["pit_lane_time"])
    temp = int(race_config["track_temp"])

    soft_laps, medium_laps, hard_laps, _, _, _, pit_count, first_pit, second_pit = stint_stats(strategy, total_laps)

    key = (
        track,
        total_laps // 2,
        round(base, 1),
        round(pit_lane, 1),
        temp // 2,
        strategy["starting_tire"],
        pit_count,
        first_pit // 2,
        second_pit // 2,
        int(soft_laps) // 2,
        int(medium_laps) // 2,
        int(hard_laps) // 2,
    )
    return json.dumps(key)


def validate_test_case(test_case: Dict) -> None:
    if not isinstance(test_case, dict):
        raise ValueError("Input must be a JSON object")

    required_top = {"race_id", "race_config", "strategies"}
    missing_top = required_top - set(test_case.keys())
    if missing_top:
        raise ValueError(f"Missing top-level fields: {sorted(missing_top)}")

    race_config = test_case["race_config"]
    required_rc = {"track", "total_laps", "base_lap_time", "pit_lane_time", "track_temp"}
    missing_rc = required_rc - set(race_config.keys())
    if missing_rc:
        raise ValueError(f"Missing race_config fields: {sorted(missing_rc)}")

    strategies = test_case["strategies"]
    expected_keys = [f"pos{i}" for i in range(1, 21)]
    seen_drivers = set()

    for key in expected_keys:
        if key not in strategies:
            raise ValueError(f"Missing strategy key: {key}")

        strategy = strategies[key]
        for field in ("driver_id", "starting_tire", "pit_stops"):
            if field not in strategy:
                raise ValueError(f"{key} missing field: {field}")

        driver_id = strategy["driver_id"]
        if driver_id in seen_drivers:
            raise ValueError(f"Duplicate driver_id found: {driver_id}")
        seen_drivers.add(driver_id)

        if strategy["starting_tire"] not in {"SOFT", "MEDIUM", "HARD"}:
            raise ValueError(f"Invalid starting_tire in {key}")


def load_model() -> Dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


def distance(feature_a: List[float], feature_b: List[float], track_match: bool) -> float:
    value = 0.0
    for idx, (a, b) in enumerate(zip(feature_a, feature_b)):
        value += WEIGHTS[idx] * abs(a - b)
    if not track_match:
        value += 24.0
    return value


def nearest_clusters(test_case: Dict, model: Dict) -> List[Tuple[float, Dict]]:
    target_feature = race_feature(test_case)
    track = test_case["race_config"]["track"]

    with_distance = []
    for cluster in model["clusters"]:
        d = distance(target_feature, cluster["feature"], cluster["track"] == track)
        with_distance.append((d, cluster))

    with_distance.sort(key=lambda row: row[0])
    return with_distance[:K_NEIGHBORS]


def combined_scores(test_case: Dict, model: Dict) -> Dict[str, float]:
    weighted_rank_sum = {driver_id: 0.0 for driver_id in DRIVER_IDS}
    weight_sum = {driver_id: 0.0 for driver_id in DRIVER_IDS}

    if CLUSTER_WEIGHT > 0:
        nearest = nearest_clusters(test_case, model)
        for d, cluster in nearest:
            weight = (1.0 / (1.0 + d)) * (1.0 + math.log1p(float(cluster.get("count", 1))))
            avg_ranks = cluster["avg_ranks"]
            for driver_id in DRIVER_IDS:
                idx = DRIVER_INDEX[driver_id]
                weighted_rank_sum[driver_id] += weight * float(avg_ranks[idx])
                weight_sum[driver_id] += weight

    driver_avg_rank = model.get("driver_avg_rank", {})
    bucket_rank_global = model.get("bucket_rank_global", {})
    bucket_rank_by_driver = model.get("bucket_rank_by_driver", {})
    race_config = test_case["race_config"]

    scores = {}
    for pos in range(1, 21):
        strategy = test_case["strategies"][f"pos{pos}"]
        driver_id = strategy["driver_id"]

        if weight_sum[driver_id] > 0:
            cluster_score = weighted_rank_sum[driver_id] / weight_sum[driver_id]
        else:
            cluster_score = float(driver_avg_rank.get(driver_id, 10.5))

        bucket_key = strategy_bucket(strategy, race_config)
        bucket_driver_map = bucket_rank_by_driver.get(driver_id, {})
        if bucket_key in bucket_driver_map:
            bucket_score = float(bucket_driver_map[bucket_key])
        elif bucket_key in bucket_rank_global:
            bucket_score = float(bucket_rank_global[bucket_key])
        else:
            bucket_score = float(driver_avg_rank.get(driver_id, 10.5))

        score = (CLUSTER_WEIGHT * cluster_score) + (BUCKET_WEIGHT * bucket_score)
        score += DRIVER_STRENGTH_WEIGHT * (float(driver_avg_rank.get(driver_id, 10.5)) - 10.5)
        score += START_POS_WEIGHT * (pos - 10.5)
        scores[driver_id] = score

    grouped: Dict[Tuple, List[Tuple[int, str]]] = {}
    for pos in range(1, 21):
        strategy = test_case["strategies"][f"pos{pos}"]
        sig = strategy_signature(strategy)
        grouped.setdefault(sig, []).append((pos, strategy["driver_id"]))

    for group in grouped.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda item: item[0])
        for order_idx, (_, driver_id) in enumerate(group):
            scores[driver_id] += order_idx * 1e-6

    return scores


def predict_finishing_positions(test_case: Dict, model: Dict) -> List[str]:
    scores = combined_scores(test_case, model)

    rows = []
    for pos in range(1, 21):
        driver_id = test_case["strategies"][f"pos{pos}"]["driver_id"]
        rows.append((scores.get(driver_id, 999.0), pos, driver_id))

    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    result = [driver_id for _, _, driver_id in rows]

    seen = set()
    deduped = []
    for driver_id in result:
        if driver_id not in seen:
            seen.add(driver_id)
            deduped.append(driver_id)

    if len(deduped) < 20:
        for pos in range(1, 21):
            driver_id = test_case["strategies"][f"pos{pos}"]["driver_id"]
            if driver_id not in seen:
                seen.add(driver_id)
                deduped.append(driver_id)
            if len(deduped) == 20:
                break

    return deduped[:20]


def fallback_positions(test_case: Dict) -> List[str]:
    if isinstance(test_case, dict) and isinstance(test_case.get("strategies"), dict):
        out = []
        for i in range(1, 21):
            key = f"pos{i}"
            strategy = test_case["strategies"].get(key, {})
            out.append(strategy.get("driver_id", f"D{i:03d}"))
        return out
    return DRIVER_IDS[:]


def main() -> None:
    test_case = None
    try:
        test_case = json.load(sys.stdin)
        validate_test_case(test_case)
        model = load_model()
        finishing_positions = predict_finishing_positions(test_case, model)

        print(
            json.dumps(
                {
                    "race_id": test_case["race_id"],
                    "finishing_positions": finishing_positions,
                }
            )
        )
    except Exception:
        race_id = test_case.get("race_id", "UNKNOWN") if isinstance(test_case, dict) else "UNKNOWN"
        print(json.dumps({"race_id": race_id, "finishing_positions": fallback_positions(test_case)}))


if __name__ == "__main__":
    main()
