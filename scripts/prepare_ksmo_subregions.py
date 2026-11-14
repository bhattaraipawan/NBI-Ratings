"""Build documented Kansas City metro/nonmetro border partitions from Census CBSA.

Optional: if a vetted river-centerline GeoJSON is provided, also produce a
RIVER-PROXIMITY sensitivity using an explicit 350-m buffer, which must NOT be
interpreted as a definitive river/land boundary classification.
"""
from __future__ import annotations
import argparse
import json
import shutil
import urllib.request
from pathlib import Path
import geopandas as gpd
from shapely.ops import unary_union
from analysis_utils import merged_border_parts

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'data/raw/tl_2025_us_state.zip'
CBSA = ROOT/'data/external/tl_2025_us_cbsa.zip'
OUT = ROOT/'data/external/KS_MO_Border_Subregions.geojson'
AUDIT = ROOT/'data/external/KS_MO_Subregion_Provenance.json'
CBSA_URL = 'https://www2.census.gov/geo/tiger/TIGER2025/CBSA/tl_2025_us_cbsa.zip'
KC_CBSA = '28140'  # Kansas City, MO-KS metropolitan statistical area


def add_lines(records, geom, group, label):
    for line in merged_border_parts(geom):
        records.append({'group':group,'label':label,'geometry':line})


def build(states, metro, rivers=None, buffer_m=350):
    s = states.to_crs('EPSG:5070')
    sg = {r.STUSPS:r.geometry for _,r in s.iterrows()}
    boundary = sg['KS'].boundary.intersection(sg['MO'].boundary)
    if boundary.is_empty:
        raise ValueError('Census KS/MO shared boundary was empty')
    k = metro.to_crs('EPSG:5070')
    kc = k[k['GEOID'].astype(str)==KC_CBSA]
    if len(kc) != 1:
        raise ValueError(f'Expected one CBSA {KC_CBSA}, found {len(kc)}')
    urban = kc.geometry.iloc[0]
    inside = boundary.intersection(urban)
    outside = boundary.difference(urban)
    records = []
    add_lines(records, inside, 'metro', 'KC_metro')
    add_lines(records, outside, 'metro', 'nonmetro')
    if not inside.length or not outside.length:
        raise ValueError('KS/MO border does not intersect both KC metro and nonmetro regions')
    river_status = 'not available (requires externally verified hydrography)'
    if rivers is not None:
        rr = rivers.to_crs('EPSG:5070')
        corridor = unary_union(list(rr.geometry)).buffer(buffer_m)
        near = boundary.intersection(corridor)
        far = boundary.difference(corridor)
        if not near.is_empty and not far.is_empty:
            add_lines(records,near,'hydrography_proximity',f'within_{buffer_m}m_river')
            add_lines(records,far,'hydrography_proximity',f'outside_{buffer_m}m_river')
            river_status=f'proximity-based sensitivity, {buffer_m} m buffer; not definitive river classification'
    out = gpd.GeoDataFrame(records,geometry='geometry',crs='EPSG:5070').to_crs('EPSG:4326')
    return out, {'kc_cbsa_code':KC_CBSA,'metro_boundary_length_km':round(inside.length/1000,3),
                 'nonmetro_boundary_length_km':round(outside.length/1000,3),
                 'river_classification_status':river_status,
                 'geometry_source':'U.S. Census 2025 TIGER/Line state boundaries + CBSA',
                 'assignment_rule':'nearest boundary partition feature in EPSG:5070'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--download',action='store_true',help='Download Census 2025 CBSA archive if absent')
    ap.add_argument('--river-lines',type=Path,help='Optional verified river-centerline GIS source (GeoJSON etc)')
    ap.add_argument('--river-buffer-m',type=int,default=350)
    args=ap.parse_args()
    if not STATE.exists():raise FileNotFoundError('Download tl_2025_us_state.zip first')
    if not CBSA.exists():
        if not args.download:raise FileNotFoundError('Missing 2025 CBSA archive, rerun --download')
        print('Downloading official Census 2025 CBSA geometry...',flush=True)
        CBSA.parent.mkdir(parents=True,exist_ok=True)
        tmp=CBSA.with_suffix('.partial')
        try:
            urllib.request.urlretrieve(CBSA_URL,tmp)
            import zipfile
            if not zipfile.is_zipfile(tmp):raise ValueError('Invalid Census CBSA download (not ZIP)')
            tmp.replace(CBSA)
        finally:tmp.unlink(missing_ok=True)
    states=gpd.read_file('zip://'+str(STATE))
    metro=gpd.read_file('zip://'+str(CBSA))
    river=gpd.read_file(args.river_lines) if args.river_lines else None
    out,meta=build(states,metro,river,args.river_buffer_m)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    out.to_file(OUT,driver='GeoJSON')
    meta.update({'cbsa_url':CBSA_URL,'subregion_geojson':OUT.name,'river_lines_source':str(args.river_lines) if args.river_lines else None})
    AUDIT.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(f'Wrote {OUT.relative_to(ROOT)} with {len(out)} partition features',flush=True)


if __name__=='__main__':main()
