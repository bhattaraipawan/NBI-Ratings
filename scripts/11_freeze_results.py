"""Copy reviewed revised-analysis outputs into results/frozen and hash them."""
from pathlib import Path
import hashlib,shutil,json

ROOT=Path(__file__).resolve().parents[1]
GEN=ROOT/'results/generated'
FROZEN=ROOT/'results/frozen'
REPORT=GEN/'reproducibility_report.json'

if not (GEN/'National_2025_FewCluster_Holm_Sensitivity.csv').exists():
    raise SystemExit('Generate the national few-cluster Holm results before freezing.')
if not REPORT.exists():
    raise SystemExit('Run scripts/10_validate_outputs.py first.')
rep=json.loads(REPORT.read_text())
if not rep.get('passed'):
    raise SystemExit('Validation did not pass; refusing to freeze.')
if not rep.get('manuscript_ready',True):
    raise SystemExit('Core validation passed, but manuscript_ready=false (for example, county climate fallback still in use). Resolve advisory items before freezing final manuscript results.')

files=[
    'input_manifest.json','reproducibility_report.json',
    'National_2025_NBI_High_Priority_Borders.csv','National_2025_NBI_Border_Screening.csv','National_2025_FewCluster_Holm_Sensitivity.csv','screening_method.json',
    'AllEligible_Exact_RD_Results.json',
    'KS_MO_Confounder_Matching_Results.json','KS_MO_Strict_Matched_Pairs.csv','KS_MO_Model_Results.csv',
    'KS_MO_Final_Robustness_Results.json','KS_MO_Robustness_Models.csv',
    'KS_MO_Climate_Sensitivity_Results.json','KS_MO_Climate_Model_Results.csv',
    'KS_MO_Temporal_Replication_Results.json','KS_MO_Temporal_Replication_Results.csv',
    'KS_MO_Subregion_Sensitivity_Results.json','KS_MO_Subregion_Sensitivity_Results.csv',
]
summary_dir=GEN/'all_exact_outputs'
summary_files=[
    'sample_counts.csv','rd_results.csv','balance_continuous.csv','balance_categorical.csv',
    'covariate_continuity.csv','placebo_results.csv','binned_means.csv','rd_fitted_lines.csv',
    'border_metadata.csv','overlap_results.csv'
]

FROZEN.mkdir(parents=True,exist_ok=True)
for child in list(FROZEN.iterdir()):
    if child.name=='README.md':
        continue
    if child.is_dir():shutil.rmtree(child)
    else:child.unlink()

for name in files:
    src=GEN/name
    if src.exists():shutil.copy2(src,FROZEN/name)

out_summary=FROZEN/'all_exact_outputs'
out_summary.mkdir(exist_ok=True)
for name in summary_files:
    src=summary_dir/name
    if src.exists():shutil.copy2(src,out_summary/name)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

with (FROZEN/'SHA256SUMS.txt').open('w',encoding='utf-8') as f:
    for q in sorted(x for x in FROZEN.rglob('*') if x.is_file() and x.name not in {'SHA256SUMS.txt','README.md'}):
        f.write(f'{sha(q)}  {q.relative_to(FROZEN).as_posix()}\n')
print(f'Frozen revised audit files to {FROZEN}')
