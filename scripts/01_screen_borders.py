"""Outcome-blind national screening of contiguous U.S. state borders.

Revision highlights
-------------------
* Keeps condition ratings out of all screening diagnostics and eligibility rules.
* Separates state separability into physical/roadway and institutional diagnostics.
* Preserves the ORIGINAL submitted eligibility AUC (physical/roadway + owner/maintenance)
  so the confirmatory 2025 border set is not redefined after outcomes were observed.
* Adds physical/roadway-only and institutional-only AUCs as diagnostic decompositions.
* Parses inspection date (Item 90, MMYY) for downstream recency diagnostics.
* Retains the composite screening score only as an ancillary descriptive field;
  downstream exact analysis includes every high-priority border, not a top-five.
"""
from __future__ import annotations

import csv, gzip, json, math
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from analysis_utils import (
    FIPS_TO_STATE, M_PER_MILE, cramers_v, dms_to_decimal, finite, fnum,
    parse_inspection_mmyy, pooled_smd
)

ROOT=Path(__file__).resolve().parents[1]
YEAR=2025
NBI_FILE=ROOT/f"data/raw/{YEAR}HwyBridgesDelimitedAllStates.txt"
STATE_ZIP=ROOT/f"data/raw/tl_{YEAR}_us_state.zip"
OUT_BRIDGES=ROOT/f"data/processed/National_{YEAR}_NBI_Border_Proximate_Bridges_25mi.csv.gz"
OUT_ALL=ROOT/f"results/generated/National_{YEAR}_NBI_Border_Screening.csv"
OUT_HIGH=ROOT/f"results/generated/National_{YEAR}_NBI_High_Priority_Borders.csv"

SAMPLE_STEP_M=250.0
BANDWIDTHS=(5,10,25)
NUMERIC_BALANCE=["age","log_adt","structure_length_m","max_span_m","deck_width_m"]
CORE_CATS=["structure_kind_code","structure_type_code","functional_class_code","highway_system_code"]
PHYSICAL_SEPARABILITY_CATS=CORE_CATS+["design_load_code"]
INSTITUTIONAL_SEPARABILITY_CATS=["owner_code","maintenance_code"]
ORIGINAL_ELIGIBILITY_CATS=PHYSICAL_SEPARABILITY_CATS+INSTITUTIONAL_SEPARABILITY_CATS


def _fold_one_hot(train_vals,test_vals,min_count):
    train_vals=[str(v or "MISSING") for v in train_vals]; test_vals=[str(v or "MISSING") for v in test_vals]
    counts=Counter(train_vals); keep={v for v,n in counts.items() if n>=min_count}
    train_vals=[v if v in keep else "OTHER" for v in train_vals]
    test_vals=[v if v in keep else "OTHER" for v in test_vals]
    levels=sorted(set(train_vals)); cols=levels[1:]
    if not cols: return np.zeros((len(train_vals),0)),np.zeros((len(test_vals),0))
    return (np.column_stack([[1.0 if v==lev else 0.0 for v in train_vals] for lev in cols]),
            np.column_stack([[1.0 if v==lev else 0.0 for v in test_vals] for lev in cols]))


def separability_auc(rows,numeric_fields,categorical_fields):
    """Fold-safe 3-fold out-of-fold AUC for predicting state side."""
    if len(rows)<200: return math.nan
    good=[r for r in rows if all(finite(r.get(c)) for c in numeric_fields)]
    if len(good)<200: return math.nan
    y=np.array([int(r["side_b"]) for r in good],int)
    if min(np.bincount(y,minlength=2))<40: return math.nan
    numeric=np.array([[r[c] for c in numeric_fields] for r in good],float) if numeric_fields else np.zeros((len(good),0))
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42); pred=np.full(len(good),np.nan)
    for tr,te in cv.split(np.zeros(len(good)),y):
        blocks_tr=[];blocks_te=[]
        if numeric_fields:
            scaler=StandardScaler().fit(numeric[tr]);blocks_tr.append(scaler.transform(numeric[tr]));blocks_te.append(scaler.transform(numeric[te]))
        min_count=max(5,int(.01*len(tr)))
        for c in categorical_fields:
            xtr,xte=_fold_one_hot([good[i].get(c,"") for i in tr],[good[i].get(c,"") for i in te],min_count)
            if xtr.shape[1]:blocks_tr.append(xtr);blocks_te.append(xte)
        if not blocks_tr:return .5
        model=LogisticRegression(max_iter=3000,C=1.0,solver="lbfgs")
        model.fit(np.column_stack(blocks_tr),y[tr]);pred[te]=model.predict_proba(np.column_stack(blocks_te))[:,1]
    return float(roc_auc_score(y,pred))


def sample_border(line,pair):
    parts=list(line.geoms) if hasattr(line,"geoms") else [line]
    xy=[];labels=[];grids=[];offset=0.0
    for part in parts:
        if part.length<=0:continue
        d=np.arange(0.0,part.length+SAMPLE_STEP_M,SAMPLE_STEP_M)
        pts=shapely.line_interpolate_point(part,d)
        xy.append(np.column_stack((shapely.get_x(pts),shapely.get_y(pts))))
        labels.extend([pair]*len(d));grids.extend([f"{pair}:{int((offset+di)//25000)}" for di in d]);offset+=part.length
    return np.vstack(xy),labels,grids


def read_nbi():
    rows=[]
    with open(NBI_FILE,"r",encoding="utf-8-sig",newline="",errors="replace") as f:
        rr=csv.DictReader(f,quotechar="'")
        for x in rr:
            st=FIPS_TO_STATE.get(str(x.get("STATE_CODE_001","")).strip().zfill(2))
            if not st:continue
            lat=dms_to_decimal(x.get("LAT_016"),False);lon=dms_to_decimal(x.get("LONG_017"),True)
            dr,sr,ur=fnum(x.get("DECK_COND_058")),fnum(x.get("SUPERSTRUCTURE_COND_059")),fnum(x.get("SUBSTRUCTURE_COND_060"))
            if not (finite(lat) and finite(lon) and 20<lat<55 and -130<lon<-60):continue
            if not all(finite(v) and 0<=v<=9 for v in (dr,sr,ur)):continue
            yb=fnum(x.get("YEAR_BUILT_027"));adt=fnum(x.get("ADT_029"));iy,im,irec=parse_inspection_mmyy(x.get("DATE_OF_INSPECT_090"),YEAR)
            rows.append({
                "inventory_year":YEAR,"state":st,"structure_number":str(x.get("STRUCTURE_NUMBER_008","")).strip(),
                "latitude":lat,"longitude":lon,"deck_rating":dr,"superstructure_rating":sr,"substructure_rating":ur,"lowest_rating":min(dr,sr,ur),
                "year_built":yb,"age":YEAR-yb if finite(yb) and 1800<=yb<=YEAR else math.nan,
                "adt":adt,"log_adt":math.log1p(max(0.0,adt)) if finite(adt) else math.nan,
                "structure_length_m":fnum(x.get("STRUCTURE_LEN_MT_049")),"max_span_m":fnum(x.get("MAX_SPAN_LEN_MT_048")),"deck_width_m":fnum(x.get("DECK_WIDTH_MT_052")),
                "inspection_year":iy,"inspection_month":im,"inspection_age_months":irec,"inspection_frequency_months":fnum(x.get("INSPECT_FREQ_MONTHS_091")),
                "year_reconstructed":fnum(x.get("YEAR_RECONSTRUCTED_106")),
                "structure_kind_code":str(x.get("STRUCTURE_KIND_043A","")).strip(),"structure_type_code":str(x.get("STRUCTURE_TYPE_043B","")).strip(),
                "functional_class_code":str(x.get("FUNCTIONAL_CLASS_026","")).strip(),"design_load_code":str(x.get("DESIGN_LOAD_031","")).strip(),
                "owner_code":str(x.get("OWNER_022","")).strip(),"maintenance_code":str(x.get("MAINTENANCE_021","")).strip(),"highway_system_code":str(x.get("HIGHWAY_SYSTEM_104","")).strip(),
            })
    return rows


def main():
    for p in [NBI_FILE,STATE_ZIP]:
        if not p.exists():raise SystemExit(f"Missing input: {p.relative_to(ROOT)}. See data/README.md")
    (ROOT/"data/processed").mkdir(parents=True,exist_ok=True);(ROOT/"results/generated").mkdir(parents=True,exist_ok=True)
    bridges=read_nbi();states=gpd.read_file("zip://"+str(STATE_ZIP)).to_crs(5070);states=states[states["STUSPS"].isin(set(FIPS_TO_STATE.values()))]
    geom={r.STUSPS:r.geometry for _,r in states.iterrows()}
    borders={};stlist=sorted(geom)
    for i,a in enumerate(stlist):
        for b in stlist[i+1:]:
            inter=geom[a].boundary.intersection(geom[b].boundary)
            if not inter.is_empty and inter.length>1000:borders[f"{a}-{b}"]=(a,b,inter)

    by_state_samples=defaultdict(lambda:{"xy":[],"pair":[],"grid":[]})
    for pair,(a,b,line) in borders.items():
        xy,labs,grids=sample_border(line,pair)
        for st in (a,b):by_state_samples[st]["xy"].append(xy);by_state_samples[st]["pair"].extend(labs);by_state_samples[st]["grid"].extend(grids)
    trees={}
    for st,d in by_state_samples.items():
        xy=np.vstack(d["xy"]);trees[st]=(cKDTree(xy),np.array(d["pair"],object),np.array(d["grid"],object))

    tr=Transformer.from_crs(4326,5070,always_xy=True);by_state=defaultdict(list)
    for r in bridges:by_state[r["state"]].append(r)
    prox=[]
    for st,rs in by_state.items():
        if st not in trees:continue
        xs,ys=tr.transform([r["longitude"] for r in rs],[r["latitude"] for r in rs]);dist,idx=trees[st][0].query(np.column_stack((xs,ys)),k=1)
        pair_labels,grid_labels=trees[st][1],trees[st][2]
        for r,x,y,d,j in zip(rs,xs,ys,dist,idx):
            miles=float(d/M_PER_MILE)
            if miles>25:continue
            pair=str(pair_labels[j]);a,b,_=borders[pair];q=dict(r)
            q.update({"border_pair":pair,"state_a":a,"state_b":b,"side_b":1 if st==b else 0,"x_5070":float(x),"y_5070":float(y),
                      "distance_miles_screening":miles,"signed_distance_miles_screening":miles if st==b else -miles,"nearest_border_grid_25km_screening":str(grid_labels[j])})
            prox.append(q)

    pair_rows=[]
    for pair,(a,b,line) in sorted(borders.items()):
        pr=[r for r in prox if r["border_pair"]==pair]
        for bw in BANDWIDTHS:
            sub=[r for r in pr if r["distance_miles_screening"]<=bw];aa=[r for r in sub if r["state"]==a];bb=[r for r in sub if r["state"]==b]
            smds=[abs(pooled_smd([r[c] for r in aa],[r[c] for r in bb])) for c in NUMERIC_BALANCE]
            cvs=[cramers_v([r[c] for r in sub],[r["side_b"] for r in sub]) for c in CORE_CATS] if sub else [math.nan]*len(CORE_CATS)
            ga={r["nearest_border_grid_25km_screening"] for r in aa};gb={r["nearest_border_grid_25km_screening"] for r in bb};den=min(len(ga),len(gb));coverage=len(ga&gb)/den if den else 0.0
            vs=[x for x in smds if finite(x)];vc=[x for x in cvs if finite(x)]
            row={"border_pair":pair,"state_a":a,"state_b":b,"bandwidth_miles":bw,"n_a":len(aa),"n_b":len(bb),"total_n":len(sub),"min_side_n":min(len(aa),len(bb)),
                 "mean_abs_smd":float(np.mean(vs)) if vs else math.nan,"max_abs_smd":float(np.max(vs)) if vs else math.nan,
                 "core_cat_mean_v":float(np.mean(vc)) if vc else math.nan,"core_cat_max_v":float(np.max(vc)) if vc else math.nan,
                 "min_side_cell_coverage":coverage,"border_length_km":float(line.length/1000)}
            if bw==10:
                elig=separability_auc(sub,NUMERIC_BALANCE,ORIGINAL_ELIGIBILITY_CATS)
                phys=separability_auc(sub,NUMERIC_BALANCE,PHYSICAL_SEPARABILITY_CATS)
                inst=separability_auc(sub,[],INSTITUTIONAL_SEPARABILITY_CATS)
                row["eligibility_separability_auc_10mi"]=elig
                row["physical_separability_auc_10mi"]=phys
                row["institutional_separability_auc_10mi"]=inst
                sample_score=min(1,row["min_side_n"]/1000);bal=max(0,1-min(row["mean_abs_smd"],.6)/.6) if finite(row["mean_abs_smd"]) else 0
                cat=max(0,1-min(row["core_cat_mean_v"],.8)/.8) if finite(row["core_cat_mean_v"]) else 0;aucscore=max(0,1-max(0,elig-.5)/.5) if finite(elig) else 0
                row["screening_score_ancillary"]=100*(.20*sample_score+.25*bal+.20*cat+.20*aucscore+.15*coverage)
                high=(row["min_side_n"]>=200 and row["mean_abs_smd"]<=.25 and elig<=.90 and row["core_cat_mean_v"]<=.50 and coverage>=.70)
                secondary=(row["min_side_n"]>=100 and row["mean_abs_smd"]<=.35 and elig<=.93 and row["core_cat_mean_v"]<=.60 and coverage>=.50)
                row["priority"]="High priority" if high else "Secondary candidate" if secondary else "Insufficient 10-mi sample" if row["min_side_n"]<100 else "Weak overlap/balance"
            pair_rows.append(row)

    ref={r["border_pair"]:r for r in pair_rows if r["bandwidth_miles"]==10}
    carry=["priority","screening_score_ancillary","eligibility_separability_auc_10mi","physical_separability_auc_10mi","institutional_separability_auc_10mi"]
    for r in pair_rows:
        for k in carry:r[k]=ref[r["border_pair"]].get(k,math.nan)

    with gzip.open(OUT_BRIDGES,"wt",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(prox[0].keys()));w.writeheader();w.writerows(prox)
    fields=list(pair_rows[0].keys())
    with open(OUT_ALL,"w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(pair_rows)
    high=sorted([r for r in pair_rows if r["bandwidth_miles"]==10 and r["priority"]=="High priority"],key=lambda r:r["border_pair"])
    with open(OUT_HIGH,"w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(high)

    meta={"inventory_year":YEAR,"primary_bandwidth_miles":10,"numeric_balance_variables":NUMERIC_BALANCE,"core_categorical_balance_variables":CORE_CATS,
          "eligibility_separability_variables":{"numeric":NUMERIC_BALANCE,"categorical":ORIGINAL_ELIGIBILITY_CATS},
          "eligibility_rule_status":"frozen from submitted 2025 screening; retained to avoid post-outcome redefinition",
          "physical_separability_variables":{"numeric":NUMERIC_BALANCE,"categorical":PHYSICAL_SEPARABILITY_CATS},
          "institutional_separability_variables":{"categorical":INSTITUTIONAL_SEPARABILITY_CATS},
          "priority_thresholds":{"high":{"min_side_n":200,"mean_abs_smd_max":.25,"eligibility_auc_max":.90,"core_cat_mean_v_max":.50,"coverage_min":.70},
                                 "secondary":{"min_side_n":100,"mean_abs_smd_max":.35,"eligibility_auc_max":.93,"core_cat_mean_v_max":.60,"coverage_min":.50}},
          "screening_score_role":"ancillary only; original weights retained descriptively and not used to select a top outcome subset",
          "outcome_used_in_screening":False}
    (ROOT/"results/generated/screening_method.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(f"Conventional bridges retained: {len(bridges):,}\nShared borders: {len(borders)}\nHigh-priority borders: {len(high)}")
    print("High-priority borders:",", ".join(r["border_pair"] for r in high))

if __name__=="__main__":main()
