"""Check the frozen national inference family against its archived values."""
from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from analysis_utils import holm_adjust


def test_archived_few_cluster_holm():
    path = ROOT / 'results/frozen/National_2025_FewCluster_Holm_Sensitivity.csv'
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 13
    assert len({r['border_pair'] for r in rows}) == 13
    expected = holm_adjust([float(r['p_fewcluster']) for r in rows])
    for row, p in zip(rows, expected):
        assert abs(float(row['p_holm_fewcluster']) - p) < 1e-10
    passing = sorted(r['border_pair'] for r in rows if float(r['p_holm_fewcluster']) < 0.05)
    assert passing == ['IA-NE', 'IN-KY', 'KS-MO', 'NE-SD']
