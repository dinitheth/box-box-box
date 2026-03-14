#!/usr/bin/env python3
import glob
import json
from collections import Counter


def strategy_bucket(strategy, total_laps):
    pit_stops = sorted(strategy.get('pit_stops', []), key=lambda x: x['lap'])
    current = strategy['starting_tire']
    last = 0
    stints = []

    for stop in pit_stops:
        lap = int(stop['lap'])
        stints.append((current, lap - last))
        current = stop['to_tire']
        last = lap
    stints.append((current, total_laps - last))

    soft = sum(length for tire, length in stints if tire == 'SOFT')
    medium = sum(length for tire, length in stints if tire == 'MEDIUM')
    hard = sum(length for tire, length in stints if tire == 'HARD')

    first_pit = int(pit_stops[0]['lap']) if pit_stops else total_laps
    second_pit = int(pit_stops[1]['lap']) if len(pit_stops) > 1 else total_laps

    return (
        strategy['starting_tire'],
        len(pit_stops),
        first_pit // 4,
        second_pit // 4,
        soft // 4,
        medium // 4,
        hard // 4,
    )


def main():
    counter = Counter()
    races = 0
    for fp in glob.glob('data/historical_races/*.json'):
        data = json.load(open(fp, 'r', encoding='utf-8'))
        for race in data:
            races += 1
            total_laps = race['race_config']['total_laps']
            rc = race['race_config']
            key = [
                rc['track'],
                total_laps // 3,
                round(float(rc['base_lap_time']), 1),
                round(float(rc['pit_lane_time']), 1),
                int(rc['track_temp']) // 2,
            ]
            bucket_list = []
            for i in range(1, 21):
                bucket_list.append(strategy_bucket(race['strategies'][f'pos{i}'], total_laps))
            bucket_list.sort()
            key.append(tuple(bucket_list))
            counter[tuple(key)] += 1

    print('races', races)
    print('clusters', len(counter))
    values = list(counter.values())
    print('mean_cluster_size', round(sum(values) / len(values), 4) if values else 0.0)
    print('max_cluster_size', max(values) if values else 0)
    print('clusters_with_gt1', sum(1 for v in values if v > 1))
    print('clusters_with_ge5', sum(1 for v in values if v >= 5))
    print('top', counter.most_common(10))


if __name__ == '__main__':
    main()
