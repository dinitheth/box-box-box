#!/usr/bin/env python3
import glob
import json

same = 0
total = 0
for fp in glob.glob('data/historical_races/*.json'):
    races = json.load(open(fp, 'r', encoding='utf-8'))
    for race in races:
        for i in range(1, 21):
            total += 1
            if race['strategies'][f'pos{i}']['driver_id'] == f'D{i:03d}':
                same += 1

print('same', same)
print('total', total)
print('ratio', same / total if total else 0.0)
