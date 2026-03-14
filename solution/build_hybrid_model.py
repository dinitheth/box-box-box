#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, List, Tuple

DRIVER_IDS = [f"D{i:03d}" for i in range(1, 21)]
DRIVER_INDEX = {driver_id: idx for idx, driver_id in enumerate(DRIVER_IDS)}


def strategy_signature(strategy: Dict) -> Tuple:
    pit_stops = sorted(strategy.get("pit_stops", []), key=lambda s: int(s["lap"]))
    stops = tuple((int(stop["lap"]), stop["from_tire"], stop["to_tire"]) for stop in pit_stops)
    return strategy["starting_tire"], stops


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


def race_feature(race: Dict) -> List[float]:
    rc = race["race_config"]
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
        strategy = race["strategies"][f"pos{pos}"]
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


def strategy_pattern_signature(race: Dict) -> Tuple:
    sig = []
    for pos in range(1, 21):
        strategy = race["strategies"][f"pos{pos}"]
        sig.append(strategy_signature(strategy))
    return tuple(sig)


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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    historical_files = sorted((root / "data" / "historical_races").glob("*.json"))

    driver_rank_sum = [0.0] * 20
    driver_rank_count = [0] * 20

    clusters: Dict[Tuple, Dict] = {}
    bucket_rank_sum_global: Dict[str, float] = {}
    bucket_rank_count_global: Dict[str, int] = {}
    bucket_rank_sum_by_driver: Dict[str, Dict[str, float]] = {}
    bucket_rank_count_by_driver: Dict[str, Dict[str, int]] = {}

    races_seen = 0
    for file_path in historical_files:
        races = json.loads(file_path.read_text(encoding="utf-8"))
        for race in races:
            races_seen += 1
            pattern_key = strategy_pattern_signature(race)
            feature = race_feature(race)

            if pattern_key not in clusters:
                clusters[pattern_key] = {
                    "count": 0,
                    "track_counts": {},
                    "feature_sum": [0.0] * len(feature),
                    "rank_sum": [0.0] * 20,
                }

            entry = clusters[pattern_key]
            entry["count"] += 1

            track = race["race_config"]["track"]
            entry["track_counts"][track] = entry["track_counts"].get(track, 0) + 1

            for i, value in enumerate(feature):
                entry["feature_sum"][i] += value

            for rank, driver_id in enumerate(race["finishing_positions"], start=1):
                idx = DRIVER_INDEX[driver_id]
                entry["rank_sum"][idx] += float(rank)
                driver_rank_sum[idx] += float(rank)
                driver_rank_count[idx] += 1

            finish_rank = {driver_id: rank for rank, driver_id in enumerate(race["finishing_positions"], start=1)}
            rc = race["race_config"]
            for pos in range(1, 21):
                strategy = race["strategies"][f"pos{pos}"]
                driver_id = strategy["driver_id"]
                rank = float(finish_rank[driver_id])
                bucket = strategy_bucket(strategy, rc)

                bucket_rank_sum_global[bucket] = bucket_rank_sum_global.get(bucket, 0.0) + rank
                bucket_rank_count_global[bucket] = bucket_rank_count_global.get(bucket, 0) + 1

                if driver_id not in bucket_rank_sum_by_driver:
                    bucket_rank_sum_by_driver[driver_id] = {}
                    bucket_rank_count_by_driver[driver_id] = {}

                bucket_rank_sum_by_driver[driver_id][bucket] = bucket_rank_sum_by_driver[driver_id].get(bucket, 0.0) + rank
                bucket_rank_count_by_driver[driver_id][bucket] = bucket_rank_count_by_driver[driver_id].get(bucket, 0) + 1

    driver_avg_rank = {
        driver_id: (driver_rank_sum[idx] / driver_rank_count[idx]) if driver_rank_count[idx] > 0 else 10.5
        for idx, driver_id in enumerate(DRIVER_IDS)
    }

    cluster_list = []
    for entry in clusters.values():
        count = entry["count"]
        track = max(entry["track_counts"].items(), key=lambda kv: kv[1])[0]

        avg_feature = [value / count for value in entry["feature_sum"]]
        avg_ranks = [value / count for value in entry["rank_sum"]]

        typical_order = [
            driver_id
            for driver_id, _ in sorted(
                ((driver_id, avg_ranks[DRIVER_INDEX[driver_id]]) for driver_id in DRIVER_IDS),
                key=lambda row: (row[1], row[0]),
            )
        ]

        cluster_list.append(
            {
                "count": count,
                "track": track,
                "feature": avg_feature,
                "avg_ranks": avg_ranks,
                "typical_order": typical_order,
            }
        )

    model = {
        "driver_ids": DRIVER_IDS,
        "driver_avg_rank": driver_avg_rank,
        "clusters": cluster_list,
        "bucket_rank_global": {
            key: (bucket_rank_sum_global[key] / bucket_rank_count_global[key])
            for key in bucket_rank_sum_global
        },
        "bucket_rank_by_driver": {
            driver_id: {
                key: (bucket_rank_sum_by_driver[driver_id][key] / bucket_rank_count_by_driver[driver_id][key])
                for key in bucket_rank_sum_by_driver[driver_id]
            }
            for driver_id in bucket_rank_sum_by_driver
        },
    }

    output_path = root / "solution" / "hybrid_model.json"
    output_path.write_text(json.dumps(model), encoding="utf-8")

    print(f"races={races_seen}")
    print(f"clusters={len(cluster_list)}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
