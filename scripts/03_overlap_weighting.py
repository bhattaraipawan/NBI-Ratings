"""Overlap-weighted sensitivity analysis for all eligible borders."""
from __future__ import annotations
import csv,gzip,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from analysis_utils import BASE_CATS,BASE_CONTROLS,finite,fit_rd,one_hot,shared_categories,strip_private_fit

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/AllEligible_Exact_Border_Bridge_Data_25mi.csv.gz'
JSON=ROOT/'results/generated/AllEligible_Exact_RD_Results.json'
OUT=ROOT/'results/generated/all_exact_outputs/overlap_results.csv'
OUTCOMES=[('lowest_rating','Lowest component rating'),('deck_rating','Deck rating'),('superstructure_rating','Superstructure rating'),('substructure_rating','Substructure rating')]
NUMS=['distance_miles_exact','signed_distance_miles_exact','deck_rating','superstructure_rating','substructure_rating','lowest_rating']+BASE_CONTROLS


def as_float(v):
    try:return float(v)
    except:return math.nan

def ess(w):
    w=np.asarray(w,float);return float(w.sum()**2/(w@w)) if (w@w)>0 else 0.0

def propensity_matrix(rows):
    nums=np.array([[r[c] for c in BASE_CONTROLS] for r in rows],float);nums=StandardScaler().fit_transform(nums)
    blocks=[nums]
    for c in BASE_CATS:
        x,_=one_hot([r.get(c,'') for r in rows],shared_categories(rows,c,max(5,int(.01*len(rows)))),c)
        if x.shape[1]:blocks.append(x)
    x,_=one_hot([r.get('nearest_border_segment_25km_exact','') for r in rows],shared_categories(rows,'nearest_border_segment_25km_exact',2),'segment')
    if x.shape[1]:blocks.append(x)
    return np.column_stack(blocks)


def main():
    rows_by=defaultdict(list)
    with gzip.open(DATA,'rt',encoding='utf-8-sig',newline='') as fobj:
        for r in csv.DictReader(fobj):
            for c in NUMS:r[c]=as_float(r.get(c,''))
            r['side_b']=int(float(r.get('side_b',0)))
            rows_by[r['border_pair']].append(r)
    out=[]
    for pair,allrows in sorted(rows_by.items()):
        for bw in (5,10,25):
            base=[r for r in allrows if r['distance_miles_exact']<=bw and all(finite(r.get(c)) for c in BASE_CONTROLS)]
            if len(base)<100 or min(sum(r['side_b']==0 for r in base),sum(r['side_b']==1 for r in base))<30:continue
            X=propensity_matrix(base);ys=np.array([r['side_b'] for r in base],int)
            logit=LogisticRegression(max_iter=3000,C=1.0,solver='lbfgs').fit(X,ys)
            ps=np.clip(logit.predict_proba(X)[:,1],.01,.99);weights=np.where(ys==1,1-ps,ps)
            for r,w in zip(base,weights):r['_overlap_weight']=float(w)
            for outcome,label in OUTCOMES:
                res=fit_rd(base,outcome,cont_controls=BASE_CONTROLS,cat_controls=BASE_CATS,weight_field='_overlap_weight')
                rr={'border_pair':pair,'bandwidth':bw,'outcome':outcome,'outcome_label':label,'propensity_auc':float(roc_auc_score(ys,ps)),
                    'ps_min':float(ps.min()),'ps_max':float(ps.max()),'ess_a':ess(weights[ys==0]),'ess_b':ess(weights[ys==1]),**strip_private_fit(res)}
                out.append(rr)
    if out:
        with open(OUT,'w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(out[0].keys()));w.writeheader();w.writerows(out)
    j=json.loads(JSON.read_text());j['overlap_results']=out;JSON.write_text(json.dumps(j,indent=2,allow_nan=True)+'\n')
    print('Wrote',OUT)

if __name__=='__main__':main()
