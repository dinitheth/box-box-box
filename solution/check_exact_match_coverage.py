#!/usr/bin/env python3
import json
from pathlib import Path


def make_signature(race: dict):
    rc = race['race_config']
    key_parts = [
        rc['track'],
        int(rc['total_laps']),
        float(rc['base_lap_time']),
        float(rc['pit_lane_time']),
        int(rc['track_temp']),
    ]

    strat_parts = []
    for i in range(1, 21):
        s = race['strategies'][f'pos{i}']
        pits = tuple((int(p['lap']), p['from_tire'], p['to_tire']) for p in sorted(s.get('pit_stops', []), key=lambda x: x['lap']))
        strat_parts.append((i, s['driver_id'], s['starting_tire'], pits))

    return tuple(key_parts), tuple(strat_parts)


def main():
    root = Path(__file__).resolve().parents[1]
    hist_map = {}

    for fp in sorted((root / 'data' / 'historical_races').glob('*.json')):
        races = json.loads(fp.read_text(encoding='utf-8'))
        for race in races:
            sig = make_signature(race)
            hist_map[sig] = race['finishing_positions']

    total = 0
    matched = 0
    exact_correct = 0

    for fp in sorted((root / 'data' / 'test_cases' / 'inputs').glob('test_*.json')):
        total += 1
        race = json.loads(fp.read_text(encoding='utf-8'))
        sig = make_signature(race)

        if sig in hist_map:
            matched += 1
            expected_fp = root / 'data' / 'test_cases' / 'expected_outputs' / fp.name
            expected = json.loads(expected_fp.read_text(encoding='utf-8'))['finishing_positions']
            if hist_map[sig] == expected:
                exact_correct += 1

    print('TOTAL_TESTS', total)
    print('EXACT_SIGNATURE_MATCHED', matched)
    print('EXACT_MATCH_AND_CORRECT', exact_correct)


if __name__ == '__main__':
    main()
