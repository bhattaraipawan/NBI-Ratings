"""Generate the national few-cluster Holm sensitivity results from exact bridge records."""
from __future__ import annotations

import csv
import gzip
from pathlib import Path

from analysis_utils import BASE_CATS, BASE_CONTROLS, fit_rd, holm_adjust, wild_cluster_bootstrap_t

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/generated'
EXACT = ROOT / 'data/processed/AllEligible_Exact_Border_Bridge_Data_25mi.csv.gz'
SOURCE = RESULTS / 'all_exact_outputs/rd_results.csv'
OUTPUT = RESULTS / 'National_2025_FewCluster_Holm_Sensitivity.csv'


def main():
    with SOURCE.open(newline='', encoding='utf-8-sig') as handle:
        rows = [r for r in csv.DictReader(handle)
                if r['bandwidth'] == '10' and r['outcome'] == 'lowest_rating']
    if len(rows) != 13 or len({r['border_pair'] for r in rows}) != 13:
        raise ValueError('Expected exactly 13 distinct eligible border models')

    small = {r['border_pair'] for r in rows if int(float(r['recency_adjusted_clusters'])) < 10}
    if small != {'CT-MA', 'DE-MD', 'MA-RI'}:
        raise ValueError(f'Unexpected few-cluster border set: {sorted(small)}')
    bridge_rows = {pair: [] for pair in small}
    numeric = set(BASE_CONTROLS + ['inspection_age_months', 'lowest_rating',
                                   'distance_miles_exact', 'signed_distance_miles_exact'])
    with gzip.open(EXACT, 'rt', encoding='utf-8-sig', newline='') as handle:
        for item in csv.DictReader(handle):
            pair = item['border_pair']
            if pair not in small or float(item['distance_miles_exact']) > 10:
                continue
            row = dict(item)
            for field in numeric:
                try:
                    row[field] = float(row[field])
                except (ValueError, TypeError, KeyError):
                    row[field] = float('nan')
            bridge_rows[pair].append(row)

    for row in rows:
        pair = row['border_pair']
        row['p_fewcluster'] = row['recency_adjusted_p_value']
        if pair not in small:
            continue
        fitted = fit_rd(bridge_rows[pair], 'lowest_rating',
                        cont_controls=BASE_CONTROLS + ['inspection_age_months'],
                        cat_controls=BASE_CATS)
        n = int(float(row['recency_adjusted_n']))
        clusters = int(float(row['recency_adjusted_clusters']))
        estimate = float(row['recency_adjusted_estimate_a_minus_b'])
        if fitted.get('n') != n or fitted.get('clusters') != clusters:
            raise ValueError(f'{pair}: reconstructed sample or segment count differs')
        if abs(fitted['estimate_a_minus_b'] - estimate) > 1e-6:
            raise ValueError(f'{pair}: reconstructed estimate differs from exact model')
        result = wild_cluster_bootstrap_t(fitted, seed=42)
        if result.get('method') != 'exhaustive_rademacher_bootstrap_t':
            raise ValueError(f'{pair}: expected exhaustive bootstrap')
        if result['patterns'] != 2 ** clusters:
            raise ValueError(f'{pair}: incorrect number of sign patterns')
        row['p_fewcluster'] = result['p_value']

    pvals = [float(r['p_fewcluster']) for r in rows]
    for row, p in zip(rows, holm_adjust(pvals)):
        row['p_holm_fewcluster'] = p
    columns = list(rows[0].keys())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} national bootstrap/Holm results to {OUTPUT}')


if __name__ == '__main__':
    main()
