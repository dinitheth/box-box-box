#!/usr/bin/env python3
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_all_races(root: Path):
    races = []
    for fp in sorted((root / 'data' / 'historical_races').glob('*.json')):
        with fp.open('r', encoding='utf-8') as f:
            races.extend(json.load(f))
    return races


def strategy_signature(strategy: dict, total_laps: int):
    pits = sorted(strategy.get('pit_stops', []), key=lambda x: x['lap'])
    seq = [strategy['starting_tire']]
    laps = [0]
    for p in pits:
        seq.append(p['to_tire'])
        laps.append(p['lap'])
    laps.append(total_laps)
    stints = []
    for idx, tire in enumerate(seq):
        stints.append((tire, laps[idx + 1] - laps[idx]))
    return tuple(stints), len(pits)


def main():
    root = Path(__file__).resolve().parents[1]
    races = load_all_races(root)

    n_races = len(races)
    pit_count_dist = Counter()
    start_tire_dist = Counter()
    track_dist = Counter()
    temp_values = []
    lap_values = []
    duplicate_strategy_races = 0

    winner_start_tire = Counter()
    winner_pit_count = Counter()
    winner_stint_patterns = Counter()

    for race in races:
        rc = race['race_config']
        total_laps = rc['total_laps']
        track_dist[rc['track']] += 1
        temp_values.append(rc['track_temp'])
        lap_values.append(total_laps)

        strat_counts = Counter()
        by_driver = {}

        for i in range(1, 21):
            s = race['strategies'][f'pos{i}']
            sig, pit_count = strategy_signature(s, total_laps)
            strat_counts[sig] += 1
            by_driver[s['driver_id']] = (s, sig, pit_count)
            pit_count_dist[pit_count] += 1
            start_tire_dist[s['starting_tire']] += 1

        if any(v > 1 for v in strat_counts.values()):
            duplicate_strategy_races += 1

        winner_id = race['finishing_positions'][0]
        winner_s, winner_sig, winner_pc = by_driver[winner_id]
        winner_start_tire[winner_s['starting_tire']] += 1
        winner_pit_count[winner_pc] += 1
        winner_stint_patterns[winner_sig] += 1

    print('RACES', n_races)
    print('TRACKS', len(track_dist), 'TOP', track_dist.most_common(10))
    print('TOTAL_LAPS_RANGE', min(lap_values), max(lap_values))
    print('TEMP_RANGE', min(temp_values), max(temp_values))
    print('START_TIRE_DIST', dict(start_tire_dist))
    print('PIT_COUNT_DIST', dict(sorted(pit_count_dist.items())))
    print('RACES_WITH_DUPLICATE_STRATEGY', duplicate_strategy_races)
    print('WINNER_START_TIRE', dict(winner_start_tire))
    print('WINNER_PIT_COUNT', dict(sorted(winner_pit_count.items())))
    print('TOP_WINNER_STINT_PATTERNS', winner_stint_patterns.most_common(12))


if __name__ == '__main__':
    main()
