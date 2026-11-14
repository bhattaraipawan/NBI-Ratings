"""Frozen-protocol Kansas-Missouri temporal replication in 2023 and 2024.

The 2025 analysis is the discovery vintage. This script applies the same exact
border geometry, 25-km segment definition, outcome, covariates, bandwidths, and
inspection-recency construction to untouched 2023 and 2024 legacy-NBI files.
It also reports a pooled 2023-2025 model with year fixed effects and spatial
segment-clustered inference.
"""
from __future__ import annotations
import csv,gzip,json,math,time
from pathlib import Path
import geopandas as gpd
import numpy as np
import shapely
from pyproj import Transformer

from analysis_utils import BASE_CATS,BASE_CONTROLS,FIPS_TO_STATE,M_PER_MILE,border_chainage_segment,dms_to_decimal,finite,fit_rd,fnum,merged_border_parts,parse_inspection_mmyy,strip_private_fit

ROOT=Path(__file__).resolve().parents[1]
YEARS=(2023,2024)
STATE_ZIP=ROOT/'data/raw/tl_2025_us_state.zip'
DISCOVERY=ROOT/'data/processed/AllEligible_Exact_Border_Bridge_Data_25mi.csv.gz'
OUT_JSON=ROOT/'results/generated/KS_MO_Temporal_Replication_Results.json'
OUT_CSV=ROOT/'results/generated/KS_MO_Temporal_Replication_Results.csv'
OUT_GZ=ROOT/'data/processed/KS_MO_2023_2025_Temporal_Bridge_Data_25mi.csv.gz'



def read_year(year,border,border_parts,tr):
    path=ROOT/f'data/raw/{year}HwyBridgesDelimitedAllStates.txt'
    if not path.exists():raise SystemExit(f'Missing temporal replication input: {path.relative_to(ROOT)}')
    raw=[]
    with open(path,'r',encoding='utf-8-sig',newline='',errors='replace') as f:
        for x in csv.DictReader(f,quotechar="'"):
            st=FIPS_TO_STATE.get(str(x.get('STATE_CODE_001','')).strip().zfill(2))
            if st not in {'KS','MO'}:continue
            dr,sr,ur=fnum(x.get('DECK_COND_058')),fnum(x.get('SUPERSTRUCTURE_COND_059')),fnum(x.get('SUBSTRUCTURE_COND_060'))
            if not all(finite(v) and 0<=v<=9 for v in (dr,sr,ur)):continue
            lat=dms_to_decimal(x.get('LAT_016'),False);lon=dms_to_decimal(x.get('LONG_017'),True)
            if not (finite(lat) and finite(lon) and 20 < lat < 55 and -130 < lon < -60):continue
            raw.append((x,st,dr,sr,ur,lat,lon))

    print(f"Temporal replication {year}: {len(raw):,} valid KS/MO candidate records",flush=True)
    if not raw:return []
    lons=np.asarray([q[6] for q in raw],dtype=float);lats=np.asarray([q[5] for q in raw],dtype=float)
    xs,ys=tr.transform(lons,lats);xs=np.asarray(xs,float);ys=np.asarray(ys,float)
    ok=np.isfinite(xs)&np.isfinite(ys)
    valid_idx=np.flatnonzero(ok)
    pts=shapely.points(xs[ok],ys[ok]) if len(valid_idx) else np.asarray([],dtype=object)
    dists=np.asarray(shapely.distance(pts,border),dtype=float)/M_PER_MILE if len(valid_idx) else np.asarray([],float)
    inrange=np.isfinite(dists)&(dists<=25.0001)
    chosen_idx=valid_idx[inrange];chosen_pts=np.asarray(pts,dtype=object)[inrange];chosen_dist=dists[inrange]
    rows=[]
    for idx,pt,dist in zip(chosen_idx,chosen_pts,chosen_dist):
        x,st,dr,sr,ur,lat,lon=raw[int(idx)]
        xx=float(xs[int(idx)]);yy=float(ys[int(idx)])
        yb=fnum(x.get('YEAR_BUILT_027'));adt=fnum(x.get('ADT_029'));iy,im,irec=parse_inspection_mmyy(x.get('DATE_OF_INSPECT_090'),year);seg,chain_m,part_i,part_chain_m=border_chainage_segment(pt,border_parts,'KS-MO',25);chain=chain_m/1000;part_chain=part_chain_m/1000
        side=1 if st=='MO' else 0
        rows.append({'inventory_year':year,'inventory_year_code':str(year),'border_pair':'KS-MO','state_a':'KS','state_b':'MO','state':st,'side_b':side,
                     'structure_number':str(x.get('STRUCTURE_NUMBER_008','')).strip(),'latitude':lat,'longitude':lon,'x_5070':xx,'y_5070':yy,
                     'distance_miles_exact':float(dist),'signed_distance_miles_exact':float(dist) if side else -float(dist),'nearest_border_segment_25km_exact':seg,'border_chainage_km_exact':chain,'border_part_index':part_i,'border_part_chainage_km_exact':part_chain,
                     'deck_rating':dr,'superstructure_rating':sr,'substructure_rating':ur,'lowest_rating':min(dr,sr,ur),
                     'year_built':yb,'age':year-yb if finite(yb) and 1800<=yb<=year else math.nan,'adt':adt,'log_adt':math.log1p(max(0.,adt)) if finite(adt) else math.nan,
                     'structure_length_m':fnum(x.get('STRUCTURE_LEN_MT_049')),'max_span_m':fnum(x.get('MAX_SPAN_LEN_MT_048')),'deck_width_m':fnum(x.get('DECK_WIDTH_MT_052')),
                     'inspection_year':iy,'inspection_month':im,'inspection_age_months':irec,'inspection_frequency_months':fnum(x.get('INSPECT_FREQ_MONTHS_091')),
                     'structure_kind_code':str(x.get('STRUCTURE_KIND_043A','')).strip(),'structure_type_code':str(x.get('STRUCTURE_TYPE_043B','')).strip(),
                     'functional_class_code':str(x.get('FUNCTIONAL_CLASS_026','')).strip(),'design_load_code':str(x.get('DESIGN_LOAD_031','')).strip(),'highway_system_code':str(x.get('HIGHWAY_SYSTEM_104','')).strip()})
    print(f"Temporal replication {year}: {len(rows):,} bridges within 25 mi",flush=True)
    return rows


def read_2025_discovery():
    out=[]
    with gzip.open(DISCOVERY,'rt',encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['border_pair']!='KS-MO':continue
            q=dict(r);q['inventory_year']=2025;q['inventory_year_code']='2025';q['side_b']=int(float(q.get('side_b',1 if q['state']=='MO' else 0)))
            for c in ['distance_miles_exact','signed_distance_miles_exact','lowest_rating','deck_rating','superstructure_rating','substructure_rating','age','log_adt','structure_length_m','max_span_m','deck_width_m','inspection_age_months','border_chainage_km_exact','border_part_index','border_part_chainage_km_exact']:
                q[c]=fnum(q.get(c))
            out.append(q)
    return out


def main():
    states=gpd.read_file('zip://'+str(STATE_ZIP)).to_crs(5070);g={r.STUSPS:r.geometry for _,r in states.iterrows()};border=g['KS'].boundary.intersection(g['MO'].boundary);border_parts=merged_border_parts(border);tr=Transformer.from_crs(4326,5070,always_xy=True)
    allrows=[]
    for y in YEARS:
        t0=time.perf_counter();allrows.extend(read_year(y,border,border_parts,tr));print(f"Temporal replication {y} geometry finished in {time.perf_counter()-t0:.1f}s",flush=True)
    allrows.extend(read_2025_discovery())
    results=[]
    for year in (2023,2024,2025):
        yr=[r for r in allrows if int(r['inventory_year'])==year]
        print(f'Temporal replication models {year}: n25={len(yr):,}',flush=True)
        for bw in (5,10,25):
            sub=[r for r in yr if r['distance_miles_exact']<=bw]
            for spec,cont in [('Core adjusted',BASE_CONTROLS),('Inspection-recency adjusted',BASE_CONTROLS+['inspection_age_months'])]:
                res=fit_rd(sub,'lowest_rating',cont_controls=cont,cat_controls=BASE_CATS)
                results.append({'inventory_year':year,'role':'discovery' if year==2025 else 'replication','bandwidth_miles':bw,'specification':spec,**strip_private_fit(res)})
    pooled=[r for r in allrows if r['distance_miles_exact']<=10]
    res=fit_rd(pooled,'lowest_rating',cont_controls=BASE_CONTROLS+['inspection_age_months'],cat_controls=BASE_CATS,extra_fixed_effect_fields=['inventory_year_code'])
    pooled_result={'inventory_year':'2023-2025 pooled','role':'pooled sensitivity','bandwidth_miles':10,'specification':'Inspection-recency adjusted + year fixed effects',**strip_private_fit(res)}
    results.append(pooled_result)
    result={'metadata':{'discovery_year':2025,'replication_years':[2023,2024],'border_geometry':'2025 TIGER/Line shared KS-MO boundary for all vintages','continuous_border_components':len(border_parts),'protocol':'same outcome, exact distance, merged-component 25-km segments, covariates and bandwidths across years'},'results':results}
    OUT_JSON.write_text(json.dumps(result,indent=2,allow_nan=True)+'\n')
    fields=[]
    for r in results:
        for k in r:
            if k not in fields:fields.append(k)
    with open(OUT_CSV,'w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)
    fields=[]
    for r in allrows:
        for k in r:
            if k not in fields:fields.append(k)
    with gzip.open(OUT_GZ,'wt',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(allrows)
    print(json.dumps({'ten_mile':[r for r in results if r['bandwidth_miles']==10 and r['specification'].startswith('Inspection')],'pooled':pooled_result},indent=2))

if __name__=='__main__':main()
