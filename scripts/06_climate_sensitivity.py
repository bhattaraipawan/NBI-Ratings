"""Kansas-Missouri climate sensitivity.

Preferred input is bridge-level gridded climate assigned by bridge coordinates:
  data/external/KS_MO_Bridge_Gridded_Climate.csv
with columns state, structure_number, annual_mean_temp_f,
winter_djf_mean_temp_f, annual_precip_in and optional freeze_thaw_days.

The final repository does not silently use county-based climate as a replacement.
If gridded climate is absent or incomplete, climate sensitivity is explicitly
marked incomplete and the manuscript freeze is blocked.
"""
from __future__ import annotations
import csv,gzip,json,math
from pathlib import Path
import numpy as np

from analysis_utils import finite,fit_rd,fnum,pooled_smd,strip_private_fit

ROOT=Path(__file__).resolve().parents[1]
BRIDGE_FILE=ROOT/'data/processed/KS_MO_Enriched_Bridge_Data_25mi.csv.gz'
NBI_FILE=ROOT/'data/raw/2025HwyBridgesDelimitedAllStates.txt'
GRIDDED=ROOT/'data/external/KS_MO_Bridge_Gridded_Climate.csv'
OUT_JSON=ROOT/'results/generated/KS_MO_Climate_Sensitivity_Results.json'
OUT_CSV=ROOT/'results/generated/KS_MO_Climate_Model_Results.csv'
OUT_BRIDGES=ROOT/'data/processed/KS_MO_Climate_Enriched_Bridges_10mi.csv.gz'
CONT=['age','log_adt','truck_pct','log_aadtt','structure_length_m','max_span_m','deck_width_m','lanes_on','inspection_age_months']
CATS=['structure_kind_code','structure_type_code','functional_class_code','design_load_code','highway_system_code','national_network','deck_protection_code']
STATE_FIPS={'20':'KS','29':'MO'}


def code(v):return str(v or '').strip()


def load_climate(bridge_rows):
    if not GRIDDED.exists():
        return 'missing PRISM gridded climate', len(bridge_rows)
    lookup={}
    with open(GRIDDED,encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            key=(r['state'].strip(),r['structure_number'].strip())
            if key in lookup:raise ValueError(f'Duplicate gridded climate bridge ID: {key}')
            lookup[key]={k:fnum(r.get(k)) for k in ['annual_mean_temp_f','winter_djf_mean_temp_f','annual_precip_in','freeze_thaw_days']}
    missing=0
    for r in bridge_rows:
        v=lookup.get((r['state'],r['structure_number'].strip()))
        if v is None or not all(finite(v.get(k)) for k in ['annual_mean_temp_f','winter_djf_mean_temp_f','annual_precip_in']):
            missing+=1
            continue
        r.update(v)
    return 'bridge-coordinate gridded climate (PRISM or documented external source)',missing


def main():
    rows=[]
    with gzip.open(BRIDGE_FILE,'rt',encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['border_pair']!='KS-MO' or fnum(r.get('distance_miles_exact'))>10:continue
            for c in ['distance_miles_exact','signed_distance_miles_exact','lowest_rating','deck_rating','superstructure_rating','substructure_rating']+CONT:r[c]=fnum(r.get(c))
            r['side_b']=1 if r['state']==r['state_b'] else 0;rows.append(r)
    source,missing=load_climate(rows)
    if missing:
        metadata={'records_10mi':len(rows),'climate_assignment':source,
                  'missing_climate_rows':missing,'preferred_final_analysis':False,
                  'reason':'PRISM bridge climate incomplete or not supplied; analysis not fitted'}
        OUT_JSON.write_text(json.dumps({'metadata':metadata,'models':[],
                                        'climate_balance':[],'climate_continuity':[]},indent=2)+'\n')
        print(json.dumps(metadata,indent=2));return
    primary_climate=['winter_djf_mean_temp_f','annual_precip_in'];annual_climate=['annual_mean_temp_f','annual_precip_in']
    if source.startswith('bridge-coordinate') and any(finite(r.get('freeze_thaw_days')) for r in rows):primary_climate.append('freeze_thaw_days')
    models=[];balance=[];continuity=[]
    for bw in (5,10):
        pr=[r for r in rows if r['distance_miles_exact']<=bw]
        for outcome,label in [('lowest_rating','Lowest rating'),('deck_rating','Deck rating'),('superstructure_rating','Superstructure rating'),('substructure_rating','Substructure rating')]:
            for spec,controls in [('Bridge/traffic/recency adjusted',CONT),('Climate adjusted',CONT+primary_climate)]:
                res=fit_rd(pr,outcome,cont_controls=controls,cat_controls=CATS)
                models.append({'border_pair':'KS-MO','bandwidth_miles':bw,'outcome':outcome,'outcome_label':label,'specification':spec,**strip_private_fit(res)})
        res=fit_rd(pr,'lowest_rating',cont_controls=CONT+annual_climate,cat_controls=CATS)
        models.append({'border_pair':'KS-MO','bandwidth_miles':bw,'outcome':'lowest_rating','outcome_label':'Lowest rating','specification':'Annual-temperature climate sensitivity',**strip_private_fit(res)})
        aa=[r for r in pr if r['side_b']==0];bb=[r for r in pr if r['side_b']==1]
        for v in ['annual_mean_temp_f','winter_djf_mean_temp_f','annual_precip_in']+(['freeze_thaw_days'] if 'freeze_thaw_days' in primary_climate else []):
            av=[r[v] for r in aa if finite(r.get(v))];bv=[r[v] for r in bb if finite(r.get(v))]
            balance.append({'border_pair':'KS-MO','bandwidth_miles':bw,'variable':v,'n_a':len(av),'n_b':len(bv),'mean_a':float(np.mean(av)) if av else math.nan,'mean_b':float(np.mean(bv)) if bv else math.nan,'smd':pooled_smd(av,bv)})
            res=fit_rd(pr,v,cont_controls=[],cat_controls=[])
            continuity.append({'border_pair':'KS-MO','bandwidth_miles':bw,'variable':v,**strip_private_fit(res)})
    result={'metadata':{'records_10mi':len(rows),'climate_assignment':source,'missing_climate_rows':missing,'preferred_final_analysis':source.startswith('bridge-coordinate'),'primary_climate_controls':primary_climate},'models':models,'climate_balance':balance,'climate_continuity':continuity}
    OUT_JSON.write_text(json.dumps(result,indent=2,allow_nan=True)+'\n')
    if models:
        fields=[]
        for r in models:
            for k in r:
                if k not in fields:fields.append(k)
        with open(OUT_CSV,'w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(models)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with gzip.open(OUT_BRIDGES,'wt',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(json.dumps(result['metadata'],indent=2))

if __name__=='__main__':main()
