#!/usr/bin/env python3
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

TIRES = ["SOFT", "MEDIUM", "HARD"]
TIRE_INDEX = {tire: idx for idx, tire in enumerate(TIRES)}
TRACKS = ["Bahrain", "COTA", "Monaco", "Monza", "Silverstone", "Spa", "Suzuka"]
TRACK_INDEX = {track: idx for idx, track in enumerate(TRACKS)}
FEATURES_PER_TIRE = 5
PIT_FEATURE_INDEX = len(TIRES) * FEATURES_PER_TIRE
BASE_FEATURES = PIT_FEATURE_INDEX + 1
TRACK_TIRE_START = BASE_FEATURES
TRACK_TIRE_FEATURES = len(TRACKS) * len(TIRES)
NUM_FEATURES = BASE_FEATURES + TRACK_TIRE_FEATURES

SCALES = [
    70.0, 2500.0, 120000.0, 900.0, 35000.0,
    70.0, 2500.0, 120000.0, 900.0, 35000.0,
    70.0, 2500.0, 120000.0, 900.0, 35000.0,
    60.0,
    *([70.0] * TRACK_TIRE_FEATURES),
]

LR = 0.03
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
L2 = 1e-6
EPOCHS = 3
SEED = 42


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


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def scale_vec(vec: List[float]) -> List[float]:
    return [v / s for v, s in zip(vec, SCALES)]


def stint_lengths(strategy: Dict, total_laps: int) -> List[Tuple[str, int]]:
    pit_stops = sorted(strategy.get("pit_stops", []), key=lambda s: int(s["lap"]))
    current_tire = strategy["starting_tire"]
    last_lap = 0
    stints: List[Tuple[str, int]] = []

    for stop in pit_stops:
        lap = int(stop["lap"])
        length = lap - last_lap
        if length > 0:
            stints.append((current_tire, length))
        current_tire = stop["to_tire"]
        last_lap = lap

    tail = total_laps - last_lap
    if tail > 0:
        stints.append((current_tire, tail))
    return stints


def driver_feature_vector(strategy: Dict, race_config: Dict) -> List[float]:
    total_laps = int(race_config["total_laps"])
    track_temp = float(race_config["track_temp"])
    pit_lane_time = float(race_config["pit_lane_time"])
    temp_delta = track_temp - 30.0
    track = race_config["track"]

    x = [0.0] * NUM_FEATURES

    for tire, length in stint_lengths(strategy, total_laps):
        if tire not in TIRE_INDEX:
            continue
        idx = TIRE_INDEX[tire] * FEATURES_PER_TIRE
        sum_age = length * (length + 1) / 2.0
        sum_age2 = length * (length + 1) * (2 * length + 1) / 6.0

        x[idx + 0] += float(length)
        x[idx + 1] += sum_age
        x[idx + 2] += sum_age2
        x[idx + 3] += temp_delta * float(length)
        x[idx + 4] += temp_delta * sum_age

        if track in TRACK_INDEX:
            tidx = TRACK_TIRE_START + (TRACK_INDEX[track] * len(TIRES)) + TIRE_INDEX[tire]
            x[tidx] += float(length)

    x[PIT_FEATURE_INDEX] = float(len(strategy.get("pit_stops", []))) * pit_lane_time
    return x


def race_driver_maps(race: Dict):
    race_config = race["race_config"]
    by_driver_raw = {}
    by_driver_scaled = {}
    for pos in range(1, 21):
        strategy = race["strategies"][f"pos{pos}"]
        driver_id = strategy["driver_id"]
        raw = driver_feature_vector(strategy, race_config)
        by_driver_raw[driver_id] = raw
        by_driver_scaled[driver_id] = scale_vec(raw)
    return by_driver_raw, by_driver_scaled


def infer_order(race: Dict, w_scaled: List[float]) -> List[str]:
    race_config = race["race_config"]
    scored = []
    for pos in range(1, 21):
        strategy = race["strategies"][f"pos{pos}"]
        raw = driver_feature_vector(strategy, race_config)
        scaled = scale_vec(raw)
        score = dot(w_scaled, scaled)
        scored.append((score, pos, strategy["driver_id"]))
    scored.sort(key=lambda row: (row[0], row[1]))
    return [driver_id for _, _, driver_id in scored]


def exact_accuracy(races: List[Dict], w_scaled: List[float]) -> float:
    if not races:
        return 0.0
    correct = 0
    for race in races:
        if infer_order(race, w_scaled) == race["finishing_positions"]:
            correct += 1
    return correct / len(races)


def load_all_races(root: Path) -> List[Dict]:
    races: List[Dict] = []
    for fp in sorted((root / "data" / "historical_races").glob("*.json")):
        with fp.open("r", encoding="utf-8") as f:
            races.extend(json.load(f))
    return races


def initial_physical_weights() -> List[float]:
    physical = [0.0] * NUM_FEATURES

    def set_tire(tire: str, base: float, lin: float, quad: float, temp_base: float, temp_age: float):
        idx = TIRE_INDEX[tire] * FEATURES_PER_TIRE
        physical[idx + 0] = base
        physical[idx + 1] = lin
        physical[idx + 2] = quad
        physical[idx + 3] = temp_base
        physical[idx + 4] = temp_age

    set_tire("SOFT", -0.55, 0.055, 0.0017, -0.002, 0.0042)
    set_tire("MEDIUM", 0.00, 0.034, 0.0010, 0.000, 0.0030)
    set_tire("HARD", 0.42, 0.021, 0.0006, 0.002, 0.0021)
    physical[PIT_FEATURE_INDEX] = 1.0
    return physical


def to_scaled_weights(physical: List[float]) -> List[float]:
    return [p * s for p, s in zip(physical, SCALES)]


def to_physical_weights(scaled: List[float]) -> List[float]:
    return [w / s for w, s in zip(scaled, SCALES)]


def train(races: List[Dict]) -> List[float]:
    random.seed(SEED)

    split = int(len(races) * 0.9)
    train_races = races[:split]
    val_races = races[split:]

    w_scaled = to_scaled_weights(initial_physical_weights())
    m = [0.0] * NUM_FEATURES
    v = [0.0] * NUM_FEATURES

    best_w = list(w_scaled)
    best_val = exact_accuracy(val_races[:600], w_scaled)
    step = 0

    for epoch in range(1, EPOCHS + 1):
        random.shuffle(train_races)
        running_loss = 0.0
        updates = 0

        for race_index, race in enumerate(train_races, start=1):
            _, by_driver_scaled = race_driver_maps(race)
            finishing = race["finishing_positions"]

            for idx in range(len(finishing) - 1):
                ahead = finishing[idx]
                behind = finishing[idx + 1]
                xdiff = [a - b for a, b in zip(by_driver_scaled[ahead], by_driver_scaled[behind])]

                z = dot(w_scaled, xdiff)
                p = sigmoid(z)
                running_loss += softplus(z)

                step += 1
                updates += 1

                for k in range(NUM_FEATURES):
                    grad = p * xdiff[k] + (L2 * w_scaled[k])
                    m[k] = BETA1 * m[k] + (1.0 - BETA1) * grad
                    v[k] = BETA2 * v[k] + (1.0 - BETA2) * (grad * grad)
                    m_hat = m[k] / (1.0 - (BETA1 ** step))
                    v_hat = v[k] / (1.0 - (BETA2 ** step))
                    w_scaled[k] -= LR * m_hat / (math.sqrt(v_hat) + EPS)

            if race_index % 3000 == 0:
                print(f"epoch={epoch} progress={race_index}/{len(train_races)}")

        val_acc = exact_accuracy(val_races[:1000], w_scaled)
        if val_acc >= best_val:
            best_val = val_acc
            best_w = list(w_scaled)

        avg_loss = running_loss / max(updates, 1)
        print(f"epoch={epoch} updates={updates} avg_pair_loss={avg_loss:.6f} val_exact={val_acc:.4f} best={best_val:.4f}")

    return to_physical_weights(best_w)


def weights_to_params(w: List[float]) -> Dict:
    def tire_params(tire: str):
        idx = TIRE_INDEX[tire] * FEATURES_PER_TIRE
        return {
            "base_delta": w[idx + 0],
            "deg_linear": w[idx + 1],
            "deg_quad": w[idx + 2],
            "temp_base": w[idx + 3],
            "temp_age": w[idx + 4],
        }

    return {
        "temp_reference": 30.0,
        "pit_lane_weight": w[PIT_FEATURE_INDEX],
        "tires": {
            "SOFT": tire_params("SOFT"),
            "MEDIUM": tire_params("MEDIUM"),
            "HARD": tire_params("HARD"),
        },
        "track_tire_delta": {
            track: {
                tire: w[TRACK_TIRE_START + (TRACK_INDEX[track] * len(TIRES)) + TIRE_INDEX[tire]]
                for tire in TIRES
            }
            for track in TRACKS
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    races = load_all_races(root)
    print(f"loaded_races={len(races)}")

    learned_physical = train(races)
    params = weights_to_params(learned_physical)

    output_path = root / "solution" / "model_params.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
