#!/usr/bin/env python3
import heapq
import json
from collections import defaultdict
from pathlib import Path

K = 35


def stint_stats(strategy, total_laps):
    pits = sorted(strategy.get('pit_stops', []), key=lambda x: x['lap'])
    current = strategy['starting_tire']
    last = 0

    soft = med = hard = 0.0
    soft_age = med_age = hard_age = 0.0

    def add_stint(tire, length):
        nonlocal soft, med, hard, soft_age, med_age, hard_age
        if length <= 0:
            return
        s1 = length * (length + 1) / 2.0
        if tire == 'SOFT':
            soft += length
            soft_age += s1
        elif tire == 'MEDIUM':
            med += length
            med_age += s1
        else:
            hard += length
            hard_age += s1

    for p in pits:
        lap = int(p['lap'])
        add_stint(current, lap - last)
        current = p['to_tire']
        last = lap

    add_stint(current, total_laps - last)

    first_pit = int(pits[0]['lap']) if len(pits) > 0 else total_laps
    second_pit = int(pits[1]['lap']) if len(pits) > 1 else total_laps

    return soft, med, hard, soft_age, med_age, hard_age, len(pits), first_pit, second_pit


def feature_for_driver(race_config, strategy):
    total_laps = int(race_config['total_laps'])
    soft, med, hard, soft_age, med_age, hard_age, pit_count, first_pit, second_pit = stint_stats(strategy, total_laps)
    st = strategy['starting_tire']
    return {
        'track': race_config['track'],
        'pit_count': pit_count,
        'start_tire': st,
        'vec': (
            float(total_laps),
            float(race_config['base_lap_time']),
            float(race_config['pit_lane_time']),
            float(race_config['track_temp']),
            1.0 if st == 'SOFT' else 0.0,
            1.0 if st == 'MEDIUM' else 0.0,
            1.0 if st == 'HARD' else 0.0,
            float(first_pit),
            float(second_pit),
            float(soft),
            float(med),
            float(hard),
            float(soft_age),
            float(med_age),
            float(hard_age),
        )
    }


def dist(v1, v2):
    weights = [2.0, 3.0, 4.0, 1.5, 5.0, 5.0, 5.0, 2.0, 2.0, 1.5, 1.5, 1.5, 0.03, 0.03, 0.03]
    d = 0.0
    for a, b, w in zip(v1, v2, weights):
        d += w * abs(a - b)
    return d


def build_index(root):
    buckets = defaultdict(list)
    for fp in sorted((root / 'data' / 'historical_races').glob('*.json')):
        races = json.loads(fp.read_text(encoding='utf-8'))
        for race in races:
            fin_rank = {d: i + 1 for i, d in enumerate(race['finishing_positions'])}
            rc = race['race_config']
            for pos in range(1, 21):
                s = race['strategies'][f'pos{pos}']
                f = feature_for_driver(rc, s)
                key = (f['track'], f['pit_count'], f['start_tire'])
                buckets[key].append((f['vec'], float(fin_rank[s['driver_id']])))
    return buckets


def predict_rank(index, feat):
    key = (feat['track'], feat['pit_count'], feat['start_tire'])
    candidates = index.get(key)
    if not candidates:
        key2 = (feat['track'], feat['pit_count'], 'SOFT')
        key3 = (feat['track'], feat['pit_count'], 'MEDIUM')
        key4 = (feat['track'], feat['pit_count'], 'HARD')
        candidates = index.get(key2, []) + index.get(key3, []) + index.get(key4, [])
        if not candidates:
            return 10.5

    vec = feat['vec']
    best = []
    for cvec, rank in candidates:
        d = dist(vec, cvec)
        if len(best) < K:
            heapq.heappush(best, (-d, rank))
        else:
            if d < -best[0][0]:
                heapq.heapreplace(best, (-d, rank))

    if not best:
        return 10.5

    num = 0.0
    den = 0.0
    for neg_d, rank in best:
        d = -neg_d
        w = 1.0 / (1.0 + d)
        num += w * rank
        den += w
    return num / den if den > 0 else 10.5


def main():
    root = Path(__file__).resolve().parents[1]
    index = build_index(root)

    passed = 0
    total = 0

    for input_fp in sorted((root / 'data' / 'test_cases' / 'inputs').glob('test_*.json')):
        total += 1
        race = json.loads(input_fp.read_text(encoding='utf-8'))
        rc = race['race_config']

        scored = []
        for pos in range(1, 21):
            s = race['strategies'][f'pos{pos}']
            feat = feature_for_driver(rc, s)
            pr = predict_rank(index, feat)
            scored.append((pr, pos, s['driver_id']))

        scored.sort(key=lambda x: (x[0], x[1]))
        pred = [x[2] for x in scored]
        exp = json.loads((root / 'data' / 'test_cases' / 'expected_outputs' / input_fp.name).read_text(encoding='utf-8'))['finishing_positions']
        if pred == exp:
            passed += 1

    print('TOTAL', total)
    print('PASSED', passed)
    print('PASS_RATE', round(100.0 * passed / total, 1) if total else 0.0)


if __name__ == '__main__':
    main()
