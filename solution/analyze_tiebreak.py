#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path


def sig(strategy: dict, total_laps: int):
    pits = sorted(strategy.get('pit_stops', []), key=lambda x: x['lap'])
    seq = [strategy['starting_tire']]
    laps = [0]
    for p in pits:
        seq.append(p['to_tire'])
        laps.append(p['lap'])
    laps.append(total_laps)
    stints = []
    for i, tire in enumerate(seq):
        stints.append((tire, laps[i + 1] - laps[i]))
    return tuple(stints)


def main():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / 'data' / 'historical_races').glob('*.json'))

    pair_orders_pos = Counter()   # counts whether lower pos finishes ahead
    pair_orders_id = Counter()    # counts whether lower driver id finishes ahead

    total_pairs = 0
    races_with_groups = 0

    for fp in files:
        races = json.loads(fp.read_text(encoding='utf-8'))
        for race in races:
            total_laps = race['race_config']['total_laps']

            by_sig = {}
            fin_rank = {d: i for i, d in enumerate(race['finishing_positions'])}

            for pos in range(1, 21):
                ps = f'pos{pos}'
                st = race['strategies'][ps]
                s = sig(st, total_laps)
                by_sig.setdefault(s, []).append((pos, st['driver_id']))

            has_group = False
            for members in by_sig.values():
                if len(members) < 2:
                    continue
                has_group = True
                members_sorted = sorted(members, key=lambda x: fin_rank[x[1]])
                # pairwise relation inside identical-strategy group
                for i in range(len(members_sorted)):
                    for j in range(i + 1, len(members_sorted)):
                        pos_i, id_i = members_sorted[i]
                        pos_j, id_j = members_sorted[j]
                        total_pairs += 1
                        if pos_i < pos_j:
                            pair_orders_pos['lower_pos_ahead'] += 1
                        else:
                            pair_orders_pos['higher_pos_ahead'] += 1

                        if id_i < id_j:
                            pair_orders_id['lower_id_ahead'] += 1
                        else:
                            pair_orders_id['higher_id_ahead'] += 1
            if has_group:
                races_with_groups += 1

    print('RACES_WITH_IDENTICAL_GROUPS', races_with_groups)
    print('TOTAL_IDENTICAL_PAIRS', total_pairs)
    print('POS_ORDER', dict(pair_orders_pos))
    print('ID_ORDER', dict(pair_orders_id))


if __name__ == '__main__':
    main()
