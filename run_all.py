"""Clean one-command NBI border-analysis pipeline.

Core run:       python run_all.py
Climate + metro: python run_all.py --prepare-external
Optional river proximity: python run_all.py --prepare-external --river-lines path/to/river_centerlines.geojson
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(script, *args):
    cmd = [sys.executable, str(ROOT/'scripts'/script), *map(str,args)]
    print('\n' + '='*64 + '\nRUN ' + ' '.join(cmd[1:]) + '\n' + '='*64, flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare-external',action='store_true',help='Download PRISM 4-km climate normals and Census KC metropolitan area geometry')
    ap.add_argument('--river-lines',help='Optional verified river centerlines for a proximity-based boundary sensitivity')
    ap.add_argument('--river-buffer-m',type=int,default=350)
    args = ap.parse_args()
    if args.river_lines and not args.prepare_external:
        ap.error('--river-lines also requires --prepare-external')
    required = [ROOT/f'data/raw/{y}HwyBridgesDelimitedAllStates.txt' for y in (2023,2024,2025)] + [ROOT/'data/raw/tl_2025_us_state.zip']
    missing = [p for p in required if not p.is_file()]
    if missing:
        print('Missing required raw inputs:')
        for p in missing: print('  -', p.relative_to(ROOT))
        print('Download once with: python scripts/download_public_inputs.py')
        raise SystemExit(1)
    generated = ROOT/'results/generated'
    if generated.exists(): shutil.rmtree(generated)
    generated.mkdir(parents=True,exist_ok=True)
    (ROOT/'data/processed').mkdir(parents=True,exist_ok=True)
    for s in ('01_screen_borders.py','02_exact_all_rd.py','02b_national_fewcluster.py','03_overlap_weighting.py','04_deep_dive.py'):
        run(s)
    if args.prepare_external:
        run('prepare_prism_climate.py','--download')
        opts=['--download']
        if args.river_lines:opts += ['--river-lines',args.river_lines,'--river-buffer-m',args.river_buffer_m]
        run('prepare_ksmo_subregions.py',*opts)
    run('00_record_input_manifest.py')
    for s in ('05_final_robustness.py',
              '06_climate_sensitivity.py','07_temporal_replication.py',
              '08_subregion_sensitivity.py','10_validate_outputs.py'):
        run(s)
    print('\nCORE ANALYSIS FINISHED. Check results/generated/reproducibility_report.json.')
    print('If manuscript_ready=true, generate figures and freeze with:')
    print('  python scripts/12_make_manuscript_figures.py')
    print('  python scripts/11_freeze_results.py')


if __name__=='__main__': main()
