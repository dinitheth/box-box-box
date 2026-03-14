#!/usr/bin/env python3
import json
from pathlib import Path


def driver_vec(strategy, total_laps, track_temp):
    pits = sorted(strategy.get('pit_stops', []), key=lambda x: x['lap'])
    current = strategy['starting_tire']
    last = 0
    soft = med = hard = 0
    pit_count = len(pits)
    first_pit = pits[0]['lap'] if pits else total_laps
    second_pit = pits[1]['lap'] if len(pits) > 1 else total_laps

    for p in pits:
        length = p['lap'] - last
        if current == 'SOFT': soft += length
        elif current == 'MEDIUM': med += length
        else: hard += length
        current = p['to_tire']
        last = p['lap']

    tail = total_laps - last
    if current == 'SOFT': soft += tail
    elif current == 'MEDIUM': med += tail
    else: hard += tail

    st = strategy['starting_tire']
    return (
        1 if st == 'SOFT' else 0,
        1 if st == 'MEDIUM' else 0,
        1 if st == 'HARD' else 0,
        pit_count,
        first_pit,
        second_pit,
        soft,
        med,
        hard,
        track_temp,
    )


def race_signature(race):
    rc = race['race_config']
    total_laps = rc['total_laps']
    temp = rc['track_temp']
    drivers = []
    for i in range(1, 21):
        s = race['strategies'][f'pos{i}']
        drivers.append(driver_vec(s, total_laps, temp))
    return (rc['track'], rc['total_laps'], rc['base_lap_time'], rc['pit_lane_time'], rc['track_temp'], tuple(drivers))


def race_distance(test_race, hist_race):
    trc = test_race['race_config']
    hrc = hist_race['race_config']
    d = 0.0
    if trc['track'] != hrc['track']:
        d += 200.0
    d += abs(trc['total_laps'] - hrc['total_laps']) * 4.0
    d += abs(trc['base_lap_time'] - hrc['base_lap_time']) * 5.0
    d += abs(trc['pit_lane_time'] - hrc['pit_lane_time']) * 8.0
    d += abs(trc['track_temp'] - hrc['track_temp']) * 2.0

    for i in range(1, 21):
        ts = test_race['strategies'][f'pos{i}']
        hs = hist_race['strategies'][f'pos{i}']
        tv = driver_vec(ts, trc['total_laps'], trc['track_temp'])
        hv = driver_vec(hs, hrc['total_laps'], hrc['track_temp'])
        d += sum(abs(a - b) for a, b in zip(tv, hv))

    return d


def main():
    root = Path(__file__).resolve().parents[1]
    hist = []
    for fp in sorted((root / 'data' / 'historical_races').glob('*.json')):
        hist.extend(json.loads(fp.read_text(encoding='utf-8')))

    tests = sorted((root / 'data' / 'test_cases' / 'inputs').glob('test_*.json'))

    passed = 0
    for tf in tests:
        t = json.loads(tf.read_text(encoding='utf-8'))
        best = None
        best_d = float('inf')
        for hr in hist:
            dist = race_distance(t, hr)
            if dist < best_d:
                best_d = dist
                best = hr
        pred = best['finishing_positions']
        exp = json.loads((root / 'data' / 'test_cases' / 'expected_outputs' / tf.name).read_text(encoding='utf-8'))['finishing_positions']
        if pred == exp:
            passed += 1
    total = len(tests)
    print('TOTAL', total)
    print('PASSED', passed)
    print('PASS_RATE', round(100.0 * passed / total, 1) if total else 0.0)


if __name__ == '__main__':
    main()
