"""Optional Kansas-Missouri subregion sensitivity.

Requires data/external/KS_MO_Border_Subregions.geojson in EPSG:4326 or with a
valid CRS. Each line feature must contain:
  group : a partition name, e.g. "boundary_type" or "metro"
  label : a level, e.g. "river", "land", "KC_metro", "nonmetro"
Features within each group should partition the shared border without overlap.

This hook allows river-vs-land and KC-metro-vs-nonmetro analyses without embedding
arbitrary hand-coded coordinate cutoffs in the scientific workflow.
"""
from __future__ import annotations
import csv,gzip,json,math
from pathlib import Path
import geopandas as gpd
import shapely
from analysis_utils import finite,fit_rd,fnum,strip_private_fit

ROOT=Path(__file__).resolve().parents[1]
SUB=ROOT/'data/external/KS_MO_Border_Subregions.geojson'
DATA=ROOT/'data/processed/KS_MO_Enriched_Bridge_Data_25mi.csv.gz'
OUT=ROOT/'results/generated/KS_MO_Subregion_Sensitivity_Results.json'
OUTCSV=ROOT/'results/generated/KS_MO_Subregion_Sensitivity_Results.csv'
CONT=['age','log_adt','truck_pct','log_aadtt','structure_length_m','max_span_m','deck_width_m','lanes_on','inspection_age_months']
CATS=['structure_kind_code','structure_type_code','functional_class_code','design_load_code','highway_system_code','national_network','deck_protection_code']

def main():
 if not SUB.exists():
  print(f'Optional subregion file not found: {SUB.relative_to(ROOT)}; skipping.');return
 g=gpd.read_file(SUB).to_crs(5070)
 if not {'group','label'}.issubset(g.columns):raise SystemExit('Subregion GeoJSON must contain group and label properties.')
 rows=[]
 with gzip.open(DATA,'rt',encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   if r['border_pair']!='KS-MO' or fnum(r.get('distance_miles_exact'))>10:continue
   for c in ['x_5070','y_5070','distance_miles_exact','signed_distance_miles_exact','lowest_rating']+CONT:r[c]=fnum(r.get(c))
   r['side_b']=1 if r['state']==r['state_b'] else 0;rows.append(r)
 results=[]
 for group in sorted(g['group'].astype(str).unique()):
  gg=g[g['group'].astype(str)==group]
  for r in rows:
   pt=shapely.Point(r['x_5070'],r['y_5070']);d=[pt.distance(x) for x in gg.geometry];i=int(min(range(len(d)),key=d.__getitem__));r[f'subregion_{group}']=str(gg.iloc[i]['label'])
  labels=sorted(set(r[f'subregion_{group}'] for r in rows))
  for label in labels:
   sub=[r for r in rows if r[f'subregion_{group}']==label]
   res=fit_rd(sub,'lowest_rating',cont_controls=CONT,cat_controls=CATS,min_n=40)
   results.append({'group':group,'label':label,'n_raw':len(sub),'n_a_raw':sum(r['side_b']==0 for r in sub),'n_b_raw':sum(r['side_b']==1 for r in sub),**strip_private_fit(res)})
 OUT.write_text(json.dumps({'metadata':{'source':str(SUB.relative_to(ROOT)),'assignment':'nearest documented subregion line within each partition group'},'results':results},indent=2,allow_nan=True)+'\n')
 if results:
  fields=[]
  for rr in results:
   for k in rr:
    if k not in fields:fields.append(k)
  with open(OUTCSV,'w',encoding='utf-8',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)
 print(json.dumps(results,indent=2,allow_nan=True))
if __name__=='__main__':main()
