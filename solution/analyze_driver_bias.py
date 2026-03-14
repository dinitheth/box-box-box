#!/usr/bin/env python3
import glob
import json

stats = {f"D{i:03d}": [0, 0] for i in range(1, 21)}

for fp in glob.glob('data/historical_races/*.json'):
    races = json.load(open(fp, 'r', encoding='utf-8'))
    for race in races:
        for rank, driver_id in enumerate(race['finishing_positions'], start=1):
            stats[driver_id][0] += rank
            stats[driver_id][1] += 1

rows = []
for driver_id, (sum_rank, cnt) in stats.items():
    rows.append((driver_id, sum_rank / cnt if cnt else 0.0))
rows.sort(key=lambda x: x[1])

for driver_id, avg_rank in rows:
    print(driver_id, round(avg_rank, 4))
