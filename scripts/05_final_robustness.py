"""Focused robustness analysis for the Kansas-Missouri lead case.

Includes inspection-recency adjustment/restriction, donut exclusions, shared-state
and geometry-mismatch exclusions, leave-one-segment-out influence checks, exact
10/25/50-km segment sensitivity, shifted-cutoff placebos, and a restricted
wild-cluster bootstrap-t test.
"""
from __future__ import annotations
import csv,gzip,json,math
from pathlib import Path
import numpy as np
import statsmodels.api as sm

from analysis_utils import BASE_CATS,BASE_CONTROLS,finite,fit_rd,holm_adjust,strip_private_fit,wild_cluster_bootstrap_t

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/KS_MO_Enriched_Bridge_Data_25mi.csv.gz'
NBI=ROOT/'data/raw/2025HwyBridgesDelimitedAllStates.txt'
OUT=ROOT/'results/generated/KS_MO_Final_Robustness_Results.json'
OUTCSV=ROOT/'results/generated/KS_MO_Robustness_Models.csv'
PRIMARY_CONT=BASE_CONTROLS+['inspection_age_months']
PRIMARY_CATS=BASE_CATS
EXPANDED_CONT=['age','log_adt','truck_pct','log_aadtt','structure_length_m','max_span_m','deck_width_m','lanes_on','inspection_age_months']
EXPANDED_CATS=['structure_kind_code','structure_type_code','functional_class_code','design_load_code','highway_system_code','national_network','deck_protection_code']
STATE_FIPS={'KS':'20','MO':'29'}


def code(v):return str(v or '').strip()
def f(v):
    try:return float(v)
    except:return math.nan


def read_rows():
    rows=[]
    numeric=['distance_miles_exact','signed_distance_miles_exact','border_chainage_km_exact','border_part_index','border_part_chainage_km_exact','lowest_rating','deck_rating','superstructure_rating','substructure_rating']+list(dict.fromkeys(PRIMARY_CONT+EXPANDED_CONT))
    with gzip.open(DATA,'rt',encoding='utf-8-sig',newline='') as fh:
        for r in csv.DictReader(fh):
            if r['border_pair']!='KS-MO':continue
            for c in numeric:r[c]=f(r.get(c))
            r['side_b']=int(float(r.get('side_b',1 if r['state']==r['state_b'] else 0)))
            gm=code(r.get('reported_state_geometry_match')).lower();r['reported_state_geometry_match']=gm in {'true','1','yes'} if gm else True
            if finite(r.get('border_part_chainage_km_exact')):
                part=int(r['border_part_index']) if finite(r.get('border_part_index')) else 0
                for km in (10,25,50):
                    r[f'segment_{km}km']=f"KS-MO:p{part}:{int(r['border_part_chainage_km_exact']//km)}"
            elif finite(r.get('border_chainage_km_exact')):
                for km in (10,25,50):r[f'segment_{km}km']=f"KS-MO:{int(r['border_chainage_km_exact']//km)}"
            rows.append(r)
    key={(STATE_FIPS[r['state']],r['structure_number'].strip()):r for r in rows}
    with open(NBI,'r',encoding='utf-8-sig',newline='',errors='replace') as fh:
        for x in csv.DictReader(fh,quotechar="'"):
            k=(code(x.get('STATE_CODE_001')),code(x.get('STRUCTURE_NUMBER_008')))
            if k in key:
                v=code(x.get('OTHER_STATE_CODE_098A'));key[k]['shared_state_structure']=bool(v and v not in {'0','00'})
    for r in rows:r.setdefault('shared_state_structure',False)
    return rows


def model(rows,label,*,segment='segment_25km',cutoff=0.0,cont=None,cats=None):
    cont=PRIMARY_CONT if cont is None else cont;cats=PRIMARY_CATS if cats is None else cats
    res=fit_rd(rows,'lowest_rating',cont_controls=cont,cat_controls=cats,cutoff=cutoff,segment_field=segment,min_n=40)
    return {'specification':label,'segment_definition':segment,**strip_private_fit(res)}


def cumulative_gee(rows,threshold):
    from analysis_utils import build_rd_design
    used=[r for r in rows if finite(r.get('lowest_rating')) and all(finite(r.get(c)) for c in PRIMARY_CONT)]
    X,names,_,_=build_rd_design(used,PRIMARY_CONT,PRIMARY_CATS,segment_field='segment_25km');j=names.index('side_b')
    y=np.array([1 if r['lowest_rating']>=threshold else 0 for r in used]);groups=np.array([r['segment_25km'] for r in used])
    rr=sm.GEE(y,X,groups=groups,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Independence()).fit()
    beta=-float(rr.params[j]);se=float(rr.bse[j])
    return {'threshold':threshold,'n':len(used),'odds_ratio_a_vs_b':math.exp(beta),'ci_low':math.exp(beta-1.96*se),'ci_high':math.exp(beta+1.96*se),'p_value':float(rr.pvalues[j])}


def main():
    rows=read_rows();base=[r for r in rows if r['distance_miles_exact']<=10]
    models=[]
    models.append(model(base,'Primary recency-adjusted 10-mi model'))
    models.append(model(base,'Expanded exposure + recency sensitivity',cont=EXPANDED_CONT,cats=EXPANDED_CATS))
    models.append(model([r for r in base if finite(r.get('inspection_age_months')) and r['inspection_age_months']<=24],'Inspection within 24 months'))
    for donut in (.5,1.0):models.append(model([r for r in base if r['distance_miles_exact']>=donut],f'Donut: exclude <{donut:g} mile'))
    models.append(model([r for r in base if not r['shared_state_structure']],'Exclude shared-state structures'))
    models.append(model([r for r in base if r['reported_state_geometry_match']],'Exclude state-geometry mismatches'))

    segment_sensitivity=[]
    for km in (10,25,50):
        r=model(base,f'{km}-km border-segment definition',segment=f'segment_{km}km')
        if km == 50:
            raw=fit_rd(base,'lowest_rating',cont_controls=PRIMARY_CONT,
                       cat_controls=PRIMARY_CATS,segment_field='segment_50km',min_n=40)
            r['wild_cluster_bootstrap_t']=wild_cluster_bootstrap_t(raw,draws=50000,seed=42) if '_X' in raw else {'error':'fit unavailable'}
            r['inference_note']='Use exhaustive restricted wild-cluster bootstrap-t for 50-km sensitivity; HC3 p-value is descriptive only.'
        segment_sensitivity.append(r);models.append(r)

    loso=[];segs=sorted(set(r['segment_25km'] for r in base if r.get('segment_25km')))
    for seg in segs:
        rr=model([r for r in base if r['segment_25km']!=seg],f'Leave out {seg}')
        rr['omitted_segment']=seg;loso.append(rr)

    placebo=[]
    all25=[r for r in rows if r['distance_miles_exact']<=25]
    for cutoff in (-15,-10,-5,5,10,15):
        fake=[r for r in all25 if abs(r['signed_distance_miles_exact']-cutoff)<=5]
        rr=model(fake,f'Fake cutoff {cutoff:+g} mi',cutoff=cutoff);rr['cutoff']=cutoff;placebo.append(rr)
    ph=holm_adjust([r.get('p_value',math.nan) for r in placebo])
    for r,padj in zip(placebo,ph):r['p_holm_placebo_family']=padj

    primary_fit=fit_rd(base,'lowest_rating',cont_controls=PRIMARY_CONT,cat_controls=PRIMARY_CATS,segment_field='segment_25km',min_n=40)
    wild=wild_cluster_bootstrap_t(primary_fit,draws=50000,seed=42) if '_X' in primary_fit and primary_fit.get('clusters',0)>=2 else {'error':'bootstrap unavailable'}

    thresholds=[cumulative_gee(base,t) for t in (5,6,7)]
    result={'metadata':{'border_pair':'KS-MO','primary_bandwidth_miles':10,'continuous_controls':PRIMARY_CONT,'categorical_controls':PRIMARY_CATS,'expanded_continuous_controls':EXPANDED_CONT,'expanded_categorical_controls':EXPANDED_CATS,
                        'bootstrap':'restricted wild-cluster bootstrap-t with Rademacher multipliers'},
            'models':models,'segment_sensitivity':segment_sensitivity,'leave_one_segment_out':loso,'placebo':placebo,'wild_cluster_bootstrap_t':wild,'cumulative_threshold_logit_gee':thresholds}
    OUT.write_text(json.dumps(result,indent=2,allow_nan=True)+'\n')
    if models:
        fields=[]
        for r in models:
            for k in r:
                if k not in fields:fields.append(k)
        with open(OUTCSV,'w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(models)
    print(json.dumps({'primary':models[0],'wild_cluster_bootstrap_t':wild,'placebos':[{k:r.get(k) for k in ['cutoff','estimate_a_minus_b','p_value','p_holm_placebo_family']} for r in placebo]},indent=2))

if __name__=='__main__':main()
