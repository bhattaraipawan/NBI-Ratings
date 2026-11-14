"""Exact-distance RD analysis for every high-priority border.

Primary national family: 10-mile, adjusted, lowest-component-rating discontinuity
for all outcome-blind eligible borders. Holm-adjusted p-values control family-wise
error across that family. Exact 25-km border chainage segments replace approximate
screening segments for fixed effects and clustering.
"""
from __future__ import annotations

import csv,gzip,json,math,time
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from pyproj import Transformer
from scipy.stats import ttest_ind

from analysis_utils import BASE_CATS,BASE_CONTROLS,FIPS_TO_STATE,M_PER_MILE,binned_summary,border_chainage_segment,cramers_v,dms_to_decimal,finite,fit_rd,fnum,holm_adjust,merged_border_parts,parse_inspection_mmyy,pooled_smd,strip_private_fit

ROOT=Path(__file__).resolve().parents[1];YEAR=2025
NBI_FILE=ROOT/f"data/raw/{YEAR}HwyBridgesDelimitedAllStates.txt"
STATE_ZIP=ROOT/f"data/raw/tl_{YEAR}_us_state.zip"
HIGH=ROOT/f"results/generated/National_{YEAR}_NBI_High_Priority_Borders.csv"
OUTDIR=ROOT/"results/generated/all_exact_outputs";OUTDIR.mkdir(parents=True,exist_ok=True)
OUT_GZ=ROOT/"data/processed/AllEligible_Exact_Border_Bridge_Data_25mi.csv.gz"
OUT_JSON=ROOT/"results/generated/AllEligible_Exact_RD_Results.json"
OUTCOMES=[("lowest_rating","Lowest component rating"),("deck_rating","Deck rating"),("superstructure_rating","Superstructure rating"),("substructure_rating","Substructure rating")]
NUMERIC=["latitude","longitude","distance_miles_screening","signed_distance_miles_screening","deck_rating","superstructure_rating","substructure_rating","lowest_rating","year_built","age","adt","log_adt","structure_length_m","max_span_m","deck_width_m","inspection_year","inspection_month","inspection_age_months","inspection_frequency_months","year_reconstructed"]


def write_csv(path,rows):
    if not rows:return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)



def balance(rows,bw,pair):
    sub=[r for r in rows if r["distance_miles_exact"]<=bw];a=[r for r in sub if r["side_b"]==0];b=[r for r in sub if r["side_b"]==1]
    cont=[]
    for field in BASE_CONTROLS+["inspection_age_months"]:
        av=[r[field] for r in a if finite(r.get(field))];bv=[r[field] for r in b if finite(r.get(field))]
        cont.append({"border_pair":pair,"bandwidth":bw,"variable":field,"n_a":len(av),"n_b":len(bv),"mean_a":float(np.mean(av)) if av else math.nan,"mean_b":float(np.mean(bv)) if bv else math.nan,
                     "smd":pooled_smd(av,bv),"p_value":float(ttest_ind(av,bv,equal_var=False).pvalue) if len(av)>1 and len(bv)>1 else math.nan})
    cat=[]
    for field in BASE_CATS:
        cat.append({"border_pair":pair,"bandwidth":bw,"variable":field,"n_a":len(a),"n_b":len(b),"cramers_v":cramers_v([r.get(field) for r in sub],[r["side_b"] for r in sub])})
    return sub,a,b,cont,cat


def main():
    if not HIGH.exists():raise SystemExit("Run 01_screen_borders.py first")
    high=list(csv.DictReader(open(HIGH,encoding="utf-8-sig",newline="")))
    labels=[r["border_pair"] for r in high]
    if not labels:raise SystemExit("No high-priority borders passed screening")
    needed_states={st for pair in labels for st in pair.split('-')}
    records_by_state=defaultdict(list)
    with open(NBI_FILE,'r',encoding='utf-8-sig',newline='',errors='replace') as f:
        for x in csv.DictReader(f,quotechar="'"):
            st=FIPS_TO_STATE.get(str(x.get('STATE_CODE_001','')).strip().zfill(2))
            if st not in needed_states:continue
            lat=dms_to_decimal(x.get('LAT_016'),False);lon=dms_to_decimal(x.get('LONG_017'),True)
            dr,sr,ur=fnum(x.get('DECK_COND_058')),fnum(x.get('SUPERSTRUCTURE_COND_059')),fnum(x.get('SUBSTRUCTURE_COND_060'))
            if not (finite(lat) and finite(lon) and 20 < lat < 55 and -130 < lon < -60):continue
            if not all(finite(v) and 0<=v<=9 for v in (dr,sr,ur)):continue
            yb=fnum(x.get('YEAR_BUILT_027'));adt=fnum(x.get('ADT_029'));iy,im,irec=parse_inspection_mmyy(x.get('DATE_OF_INSPECT_090'),YEAR)
            records_by_state[st].append({'inventory_year':YEAR,'state':st,'structure_number':str(x.get('STRUCTURE_NUMBER_008','')).strip(),'latitude':lat,'longitude':lon,
                'deck_rating':dr,'superstructure_rating':sr,'substructure_rating':ur,'lowest_rating':min(dr,sr,ur),'year_built':yb,'age':YEAR-yb if finite(yb) and 1800<=yb<=YEAR else math.nan,
                'adt':adt,'log_adt':math.log1p(max(0.,adt)) if finite(adt) else math.nan,'structure_length_m':fnum(x.get('STRUCTURE_LEN_MT_049')),'max_span_m':fnum(x.get('MAX_SPAN_LEN_MT_048')),
                'deck_width_m':fnum(x.get('DECK_WIDTH_MT_052')),'inspection_year':iy,'inspection_month':im,'inspection_age_months':irec,'inspection_frequency_months':fnum(x.get('INSPECT_FREQ_MONTHS_091')),
                'year_reconstructed':fnum(x.get('YEAR_RECONSTRUCTED_106')),'structure_kind_code':str(x.get('STRUCTURE_KIND_043A','')).strip(),'structure_type_code':str(x.get('STRUCTURE_TYPE_043B','')).strip(),
                'functional_class_code':str(x.get('FUNCTIONAL_CLASS_026','')).strip(),'design_load_code':str(x.get('DESIGN_LOAD_031','')).strip(),'owner_code':str(x.get('OWNER_022','')).strip(),
                'maintenance_code':str(x.get('MAINTENANCE_021','')).strip(),'highway_system_code':str(x.get('HIGHWAY_SYSTEM_104','')).strip()})

    states=gpd.read_file("zip://"+str(STATE_ZIP)).to_crs(5070);geom={r.STUSPS:r.geometry for _,r in states.iterrows()}
    tr=Transformer.from_crs(4326,5070,always_xy=True)
    all_exact=[];rd=[];samples=[];balc=[];balk=[];covrd=[];placebo=[];bins=[];meta=[];fitted=[]
    projected_coordinate_skips=0
    total_state_records=sum(len(v) for v in records_by_state.values())
    print(f"Exact-analysis raw population loaded: {total_state_records:,} records across {len(records_by_state)} states",flush=True)
    overall_t0=time.perf_counter()
    for pair_i,pair in enumerate(labels,1):
        pair_t0=time.perf_counter()
        a,b=pair.split("-")
        border=geom[a].boundary.intersection(geom[b].boundary)
        if border.is_empty:
            raise RuntimeError(f"Shared TIGER boundary is empty for {pair}")
        border_parts=merged_border_parts(border)
        if not border_parts:
            raise RuntimeError(f"No line components after merging shared TIGER boundary for {pair}")
        exact=[]
        rs=records_by_state[a]+records_by_state[b]
        print(f"[exact {pair_i}/{len(labels)}] {pair}: {len(rs):,} candidate records from {a}+{b}",flush=True)

        lons=np.asarray([r["longitude"] for r in rs],dtype=float)
        lats=np.asarray([r["latitude"] for r in rs],dtype=float)
        xs,ys=tr.transform(lons,lats)
        xs=np.asarray(xs,dtype=float);ys=np.asarray(ys,dtype=float)
        ok=np.isfinite(xs)&np.isfinite(ys)
        projected_coordinate_skips += int((~ok).sum())
        valid_idx=np.flatnonzero(ok)
        if len(valid_idx):
            pts=shapely.points(xs[ok],ys[ok])
            dists=np.asarray(shapely.distance(pts,border),dtype=float)/M_PER_MILE
            inrange=np.isfinite(dists)&(dists<=25.0001)
            chosen_idx=valid_idx[inrange]
            chosen_pts=np.asarray(pts,dtype=object)[inrange]
            chosen_dist=dists[inrange]
        else:
            chosen_idx=np.asarray([],dtype=int);chosen_pts=np.asarray([],dtype=object);chosen_dist=np.asarray([],dtype=float)

        for idx,pt,d in zip(chosen_idx,chosen_pts,chosen_dist):
            r=rs[int(idx)]
            x=float(xs[int(idx)]);y=float(ys[int(idx)])
            q=dict(r);q["border_pair"]=pair;q["state_a"]=a;q["state_b"]=b;q["x_5070"]=x;q["y_5070"]=y;q["distance_miles_exact"]=float(d);q["side_b"]=1 if q["state"]==b else 0;q["signed_distance_miles_exact"]=float(d) if q["side_b"] else -float(d)
            seg,chain,part_i,part_chain=border_chainage_segment(pt,border_parts,pair,25);q["nearest_border_segment_25km_exact"]=seg;q["border_chainage_km_exact"]=chain/1000;q["border_part_index"]=part_i;q["border_part_chainage_km_exact"]=part_chain/1000;q["reported_state_geometry_match"]=bool(geom[q["state"]].covers(pt))
            exact.append(q);all_exact.append(q)
        mismatch=sum(not r["reported_state_geometry_match"] for r in exact)
        meta.append({"border_pair":pair,"state_a":a,"state_b":b,"records_25mi":len(exact),"border_length_km":float(border.length/1000),"continuous_border_components":len(border_parts),"reported_state_geometry_mismatches":mismatch})
        print(f"  exact geometry complete: {len(exact):,} bridges within 25 mi; merged border components={len(border_parts)}; geometry mismatches={mismatch}",flush=True)

        for z in binned_summary(exact,"lowest_rating",10,1):z.update({"border_pair":pair});bins.append(z)
        subfit=[r for r in exact if r["distance_miles_exact"]<=10]
        for side,st in [(0,a),(1,b)]:
            rr=[r for r in subfit if r["side_b"]==side]
            if len(rr)>=2:
                x=np.array([abs(r["signed_distance_miles_exact"]) for r in rr]);y=np.array([r["lowest_rating"] for r in rr]);coef=np.polyfit(x,y,1)
                for xx in np.linspace(0,10,101):fitted.append({"border_pair":pair,"state":st,"side_b":side,"signed_distance":(-xx if side==0 else xx),"fitted_mean":float(coef[0]*xx+coef[1])})
        for bw in (5,10,25):
            bw_t0=time.perf_counter()
            sub,ra,rb,bc,bk=balance(exact,bw,pair);balc.extend(bc);balk.extend(bk)
            print(f"  {bw:>2}-mi models: n={len(sub):,} ({a}={len(ra):,}, {b}={len(rb):,})",flush=True)
            samples.append({"border_pair":pair,"state_a":a,"state_b":b,"bandwidth":bw,"n_a":len(ra),"n_b":len(rb),"total_n":len(sub),"raw_difference_a_minus_b":float(np.mean([r["lowest_rating"] for r in ra])-np.mean([r["lowest_rating"] for r in rb])) if ra and rb else math.nan})
            for out,label in OUTCOMES:
                base=fit_rd(sub,out,cont_controls=[],cat_controls=[])
                adj=fit_rd(sub,out,cont_controls=BASE_CONTROLS,cat_controls=BASE_CATS)
                rec=fit_rd(sub,out,cont_controls=BASE_CONTROLS+["inspection_age_months"],cat_controls=BASE_CATS)
                row={"border_pair":pair,"state_a":a,"state_b":b,"bandwidth":bw,"outcome":out,"outcome_label":label}
                for pref,res in [("base",base),("adjusted",adj),("recency_adjusted",rec)]:
                    for k,v in strip_private_fit(res).items():row[f"{pref}_{k}"]=v
                rd.append(row)
            print(f"      completed in {time.perf_counter()-bw_t0:.1f}s",flush=True)
        sub10=[r for r in exact if r["distance_miles_exact"]<=10]
        print("  10-mi covariate-continuity models...",flush=True)
        for field in BASE_CONTROLS+["inspection_age_months"]:
            res=fit_rd(sub10,field,cont_controls=[],cat_controls=[])
            covrd.append({"border_pair":pair,"variable":field,**strip_private_fit(res)})
        print("  shifted-cutoff placebo models...",flush=True)
        for cutoff,label in [(-10,f"10 miles inside {a}"),(10,f"10 miles inside {b}")]:
            fake=[r for r in exact if abs(r["signed_distance_miles_exact"]-cutoff)<=5]
            res=fit_rd(fake,"lowest_rating",cont_controls=BASE_CONTROLS,cat_controls=BASE_CATS,cutoff=cutoff)
            placebo.append({"border_pair":pair,"placebo_cutoff":cutoff,"placebo_label":label,"window_miles":5,**strip_private_fit(res)})
        print(f"[exact {pair_i}/{len(labels)}] {pair} finished in {(time.perf_counter()-pair_t0)/60:.2f} min",flush=True)

    print(f"All exact-border models finished in {(time.perf_counter()-overall_t0)/60:.2f} min",flush=True)

    primary=[r for r in rd if r["bandwidth"]==10 and r["outcome"]=="lowest_rating"]
    p1=holm_adjust([r.get("adjusted_p_value",math.nan) for r in primary])
    p2=holm_adjust([r.get("recency_adjusted_p_value",math.nan) for r in primary])
    for r,padj1,padj2 in zip(primary,p1,p2):
        r["adjusted_p_holm_family"]=padj1
        r["recency_adjusted_p_holm_family"]=padj2

    fields=[]
    for r in all_exact:
        for k in r:
            if k not in fields:fields.append(k)
    with gzip.open(OUT_GZ,"wt",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(all_exact)
    for name,rows in [("sample_counts.csv",samples),("rd_results.csv",rd),("balance_continuous.csv",balc),("balance_categorical.csv",balk),("covariate_continuity.csv",covrd),("placebo_results.csv",placebo),("binned_means.csv",bins),("rd_fitted_lines.csv",fitted),("border_metadata.csv",meta)]:write_csv(OUTDIR/name,rows)
    result={"coordinate_qc":{"projected_coordinate_skips":projected_coordinate_skips,"accepted_geographic_bounds":"20 < latitude < 55 and -130 < longitude < -60"},"selection":{"rule":"all outcome-blind high-priority borders under frozen submitted eligibility rule","analysis_pairs":labels,"n_pairs":len(labels)},"primary_family":{"definition":"10-mi submitted-specification adjusted lowest-rating discontinuity across all eligible borders","multiplicity":"Holm family-wise correction"},"recency_enhancement_family":{"definition":"10-mi inspection-recency-adjusted lowest-rating discontinuity across all eligible borders","multiplicity":"separate Holm family-wise correction","status":"post-submission enhancement"},"metadata":meta,"sample_counts":samples,"rd_results":rd,"balance_continuous":balc,"balance_categorical":balk,"covariate_continuity":covrd,"placebo_results":placebo,"binned_means":bins,"bridge_records":len(all_exact)}
    OUT_JSON.write_text(json.dumps(result,indent=2,allow_nan=True)+"\n")
    print(json.dumps({"eligible_borders":labels,"n":len(labels),"primary_10mi":[{k:r.get(k) for k in ["border_pair","adjusted_estimate_a_minus_b","adjusted_ci_low","adjusted_ci_high","adjusted_p_value","adjusted_p_holm_family","recency_adjusted_estimate_a_minus_b","recency_adjusted_p_value","recency_adjusted_p_holm_family"]} for r in primary]},indent=2))

if __name__=="__main__":main()
