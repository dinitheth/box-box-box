#!/usr/bin/env python3
import json
import math
import random
from pathlib import Path

TIRES = ["SOFT", "MEDIUM", "HARD"]
TRACKS = ["Bahrain", "COTA", "Monaco", "Monza", "Silverstone", "Spa", "Suzuka"]
DRIVERS = [f"D{i:03d}" for i in range(1, 21)]

TIRE_IDX = {t: i for i, t in enumerate(TIRES)}
TRACK_IDX = {t: i for i, t in enumerate(TRACKS)}
DRIVER_IDX = {d: i for i, d in enumerate(DRIVERS)}

# Parameter layout
PIT_IDX = 0
TIRE_BASE_START = 1                 # +3
TIRE_LIN_START = TIRE_BASE_START + 3  # +3
TIRE_QUAD_START = TIRE_LIN_START + 3  # +3
TIRE_TEMP_BASE_START = TIRE_QUAD_START + 3  # +3
TIRE_TEMP_AGE_START = TIRE_TEMP_BASE_START + 3  # +3
TRACK_TIRE_START = TIRE_TEMP_AGE_START + 3  # +21
DRIVER_BIAS_START = TRACK_TIRE_START + (len(TRACKS) * len(TIRES))  # +20
DRIVER_TRACK_START = DRIVER_BIAS_START + len(DRIVERS)  # +140
NUM_PARAMS = DRIVER_TRACK_START + (len(DRIVERS) * len(TRACKS))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softplus(x: float) -> float:
    if x > 50:
        return x
    if x < -50:
        return math.exp(x)
    return math.log1p(math.exp(x))


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

    return laps, sum_age, sum_age2, len(pit_stops)


def sparse_features(race, pos):
    rc = race["race_config"]
    total_laps = int(rc["total_laps"])
    track = rc["track"]
    temp_delta = float(rc["track_temp"]) - 30.0
    pit_lane = float(rc["pit_lane_time"])

    strategy = race["strategies"][f"pos{pos}"]
    driver_id = strategy["driver_id"]

    laps, sum_age, sum_age2, pit_count = stint_stats(strategy, total_laps)

    feats = []
    feats.append((PIT_IDX, float(pit_count) * pit_lane))

    for tire in TIRES:
        ti = TIRE_IDX[tire]
        lp = laps[tire]
        sa = sum_age[tire]
        sa2 = sum_age2[tire]

        if lp != 0.0:
            feats.append((TIRE_BASE_START + ti, lp))
            feats.append((TIRE_TEMP_BASE_START + ti, temp_delta * lp))
            tr_idx = TRACK_TIRE_START + (TRACK_IDX[track] * len(TIRES)) + ti
            feats.append((tr_idx, lp))
        if sa != 0.0:
            feats.append((TIRE_LIN_START + ti, sa))
            feats.append((TIRE_TEMP_AGE_START + ti, temp_delta * sa))
        if sa2 != 0.0:
            feats.append((TIRE_QUAD_START + ti, sa2))

    didx = DRIVER_IDX[driver_id]
    feats.append((DRIVER_BIAS_START + didx, float(total_laps)))
    dt_idx = DRIVER_TRACK_START + (didx * len(TRACKS)) + TRACK_IDX[track]
    feats.append((dt_idx, float(total_laps)))
    return feats


def sparse_dot(w, x):
    return sum(w[idx] * val for idx, val in x)


def sparse_diff(a, b):
    out = {}
    for idx, val in a:
        out[idx] = out.get(idx, 0.0) + val
    for idx, val in b:
        out[idx] = out.get(idx, 0.0) - val
    return [(idx, val) for idx, val in out.items() if val != 0.0]


def load_races(root):
    races = []
    for fp in sorted((root / "data" / "historical_races").glob("*.json")):
        races.extend(json.loads(fp.read_text(encoding="utf-8")))
    return races


def build_cache(races):
    cached = []
    for race in races:
        by_driver = {}
        for pos in range(1, 21):
            strategy = race["strategies"][f"pos{pos}"]
            by_driver[strategy["driver_id"]] = sparse_features(race, pos)
        cached.append((race["finishing_positions"], by_driver))
    return cached


def train(cached):
    rng = random.Random(42)

    w = [0.0] * NUM_PARAMS
    m = [0.0] * NUM_PARAMS
    v = [0.0] * NUM_PARAMS

    # physics-ish init
    w[PIT_IDX] = 1.0
    w[TIRE_BASE_START + TIRE_IDX["SOFT"]] = -0.45
    w[TIRE_BASE_START + TIRE_IDX["MEDIUM"]] = 0.00
    w[TIRE_BASE_START + TIRE_IDX["HARD"]] = 0.40
    w[TIRE_LIN_START + TIRE_IDX["SOFT"]] = 0.055
    w[TIRE_LIN_START + TIRE_IDX["MEDIUM"]] = 0.035
    w[TIRE_LIN_START + TIRE_IDX["HARD"]] = 0.022
    w[TIRE_QUAD_START + TIRE_IDX["SOFT"]] = 0.0017
    w[TIRE_QUAD_START + TIRE_IDX["MEDIUM"]] = 0.0010
    w[TIRE_QUAD_START + TIRE_IDX["HARD"]] = 0.0006
    w[TIRE_TEMP_BASE_START + TIRE_IDX["SOFT"]] = -0.002
    w[TIRE_TEMP_BASE_START + TIRE_IDX["MEDIUM"]] = 0.000
    w[TIRE_TEMP_BASE_START + TIRE_IDX["HARD"]] = 0.002
    w[TIRE_TEMP_AGE_START + TIRE_IDX["SOFT"]] = 0.0045
    w[TIRE_TEMP_AGE_START + TIRE_IDX["MEDIUM"]] = 0.0032
    w[TIRE_TEMP_AGE_START + TIRE_IDX["HARD"]] = 0.0020

    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    lr = 0.03
    l2 = 3e-8
    step = 0

    epochs = 4
    pairs_per_race = 45

    for epoch in range(1, epochs + 1):
        rng.shuffle(cached)
        loss_sum = 0.0
        updates = 0

        for finish_order, by_driver in cached:
            n = len(finish_order)
            for _ in range(pairs_per_race):
                i = rng.randrange(0, n - 1)
                j = rng.randrange(i + 1, n)
                ahead = finish_order[i]
                behind = finish_order[j]

                xdiff = sparse_diff(by_driver[ahead], by_driver[behind])
                z = sparse_dot(w, xdiff)
                p = sigmoid(z)
                loss_sum += softplus(z)
                step += 1
                updates += 1

                for idx, val in xdiff:
                    grad = p * val + (l2 * w[idx])
                    m[idx] = beta1 * m[idx] + (1.0 - beta1) * grad
                    v[idx] = beta2 * v[idx] + (1.0 - beta2) * (grad * grad)
                    m_hat = m[idx] / (1.0 - (beta1 ** step))
                    v_hat = v[idx] / (1.0 - (beta2 ** step))
                    w[idx] -= lr * m_hat / (math.sqrt(v_hat) + eps)

        avg_loss = loss_sum / max(updates, 1)
        print(f"epoch={epoch} updates={updates} avg_pair_loss={avg_loss:.6f}")

    return w


def write_params(root, w):
    def tire_dict(tire):
        ti = TIRE_IDX[tire]
        return {
            "base_delta": w[TIRE_BASE_START + ti],
            "deg_linear": w[TIRE_LIN_START + ti],
            "deg_quad": w[TIRE_QUAD_START + ti],
            "temp_base": w[TIRE_TEMP_BASE_START + ti],
            "temp_age": w[TIRE_TEMP_AGE_START + ti],
        }

    params = {
        "temp_reference": 30.0,
        "pit_lane_weight": w[PIT_IDX],
        "tires": {
            "SOFT": tire_dict("SOFT"),
            "MEDIUM": tire_dict("MEDIUM"),
            "HARD": tire_dict("HARD"),
        },
        "track_tire_delta": {
            track: {
                tire: w[TRACK_TIRE_START + (TRACK_IDX[track] * len(TIRES)) + TIRE_IDX[tire]]
                for tire in TIRES
            }
            for track in TRACKS
        },
        "driver_lap_bias": {
            driver_id: w[DRIVER_BIAS_START + DRIVER_IDX[driver_id]]
            for driver_id in DRIVERS
        },
        "driver_track_bias": {
            driver_id: {
                track: w[DRIVER_TRACK_START + (DRIVER_IDX[driver_id] * len(TRACKS)) + TRACK_IDX[track]]
                for track in TRACKS
            }
            for driver_id in DRIVERS
        },
    }

    out = root / "solution" / "model_params.json"
    out.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"saved={out}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    races = load_races(root)
    print(f"loaded_races={len(races)}")
    cached = build_cache(races)
    w = train(cached)
    write_params(root, w)
