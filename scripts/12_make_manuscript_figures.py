"""Generate revised manuscript figures from reviewed/frozen results when available."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results/frozen'
if not (RES/'AllEligible_Exact_RD_Results.json').exists():RES=ROOT/'results/generated'
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':11,'pdf.fonttype':42,'ps.fonttype':42})


def clean(ax):
 ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False);ax.grid(False)
def save(fig,out,name):
 fig.savefig(out/f'{name}.png',dpi=600,bbox_inches='tight');fig.savefig(out/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
def readcsv(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def flt(v):
 try:return float(v)
 except:return math.nan


def fig1(out):
    audited=RES/'National_2025_FewCluster_Holm_Sensitivity.csv'
    if not audited.exists():audited=ROOT/'results/generated/National_2025_FewCluster_Holm_Sensitivity.csv'
    if not audited.exists():
        raise FileNotFoundError('Missing verified national Holm sensitivity CSV; do not plot old adjusted model as Figure 1')
    rows=readcsv(audited)
    assert len(rows)==13 and len({r['border_pair'] for r in rows})==13
    rows=sorted(rows,key=lambda r:flt(r['recency_adjusted_estimate_a_minus_b']))
    y=np.arange(len(rows))
    est=np.array([flt(r['recency_adjusted_estimate_a_minus_b']) for r in rows])
    lo=np.array([flt(r['recency_adjusted_ci_low']) for r in rows])
    hi=np.array([flt(r['recency_adjusted_ci_high']) for r in rows])
    labels=[r['border_pair']+(' *' if flt(r['p_holm_fewcluster'])<.05 else '') for r in rows]
    fig,ax=plt.subplots(figsize=(7.3,max(4.8,.38*len(rows)+1.6)))
    ax.errorbar(est,y,xerr=np.vstack([est-lo,hi-est]),fmt='o',capsize=3,linewidth=1.0)
    ax.axvline(0,linestyle='--',linewidth=.9)
    ax.set_yticks(y,labels)
    ax.set_xlabel('State A - State B discontinuity (rating points)')
    ax.set_title('2025 NBI: recency-adjusted border discontinuities (10-mi)',pad=9)
    ax.text(.01,-.145,'* Holm p < 0.05; cluster-t (>=10 clusters) or exhaustive wild bootstrap (<10).\n95% bars are unadjusted model intervals.',transform=ax.transAxes,fontsize=8)
    clean(ax);fig.tight_layout();save(fig,out,'Fig1_All_Border_Forest')


def fig2(out):
    rows=[r for r in readcsv(RES/'National_2025_NBI_Border_Screening.csv') if flt(r.get('bandwidth_miles'))==10]
    hi=[r for r in rows if r.get('priority')=='High priority'];other=[r for r in rows if r.get('priority')!='High priority']
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter([flt(r['eligibility_separability_auc_10mi']) for r in other],[flt(r['mean_abs_smd']) for r in other],s=18,alpha=.55,label='Other borders')
    ax.scatter([flt(r['eligibility_separability_auc_10mi']) for r in hi],[flt(r['mean_abs_smd']) for r in hi],s=34,marker='D',label='Eligible borders')
    ax.axvline(.90,linestyle='--',linewidth=.8);ax.axhline(.25,linestyle='--',linewidth=.8)
    for r in hi:ax.annotate(r['border_pair'],(flt(r['eligibility_separability_auc_10mi']),flt(r['mean_abs_smd'])),xytext=(3,3),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Frozen eligibility state-separability AUC');ax.set_ylabel('Mean absolute standardized difference');ax.legend(frameon=False);clean(ax);fig.tight_layout();save(fig,out,'Fig2_Screening_Diagnostics')


def figs1(out):
    rows=[r for r in readcsv(RES/'National_2025_NBI_Border_Screening.csv') if flt(r.get('bandwidth_miles'))==10]
    rows=[r for r in rows if math.isfinite(flt(r.get('physical_separability_auc_10mi'))) and math.isfinite(flt(r.get('institutional_separability_auc_10mi')))]
    hi=[r for r in rows if r.get('priority')=='High priority'];other=[r for r in rows if r.get('priority')!='High priority']
    fig,ax=plt.subplots(figsize=(6.8,5));ax.scatter([flt(r['physical_separability_auc_10mi']) for r in other],[flt(r['institutional_separability_auc_10mi']) for r in other],s=18,alpha=.55,label='Other borders')
    ax.scatter([flt(r['physical_separability_auc_10mi']) for r in hi],[flt(r['institutional_separability_auc_10mi']) for r in hi],s=34,marker='D',label='Eligible borders')
    for r in hi:ax.annotate(r['border_pair'],(flt(r['physical_separability_auc_10mi']),flt(r['institutional_separability_auc_10mi'])),xytext=(3,3),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Physical/roadway separability AUC');ax.set_ylabel('Owner/maintenance separability AUC');ax.legend(frameon=False);clean(ax);fig.tight_layout();save(fig,out,'FigS1_AUC_Decomposition')


def fig3(out):
 j=json.loads((RES/'AllEligible_Exact_RD_Results.json').read_text());rows=[r for r in j['balance_continuous'] if r['border_pair']=='KS-MO' and r['bandwidth']==10]
 labels=[r['variable'].replace('_',' ') for r in rows];vals=np.array([abs(flt(r['smd'])) for r in rows]);y=np.arange(len(rows))
 fig,ax=plt.subplots(figsize=(7,4.6));ax.barh(y,vals);ax.axvline(.10,linestyle='--',linewidth=.8);ax.axvline(.25,linestyle=':',linewidth=.8);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('Absolute standardized mean difference');ax.set_title('Kansas-Missouri covariate balance within 10 mi',pad=9);clean(ax);fig.tight_layout();save(fig,out,'Fig3_KSMO_Balance')


def fig4(out):
 bins=[r for r in readcsv(RES/'all_exact_outputs/binned_means.csv') if r['border_pair']=='KS-MO'] if (RES/'all_exact_outputs/binned_means.csv').exists() else []
 fits=[r for r in readcsv(RES/'all_exact_outputs/rd_fitted_lines.csv') if r['border_pair']=='KS-MO'] if (RES/'all_exact_outputs/rd_fitted_lines.csv').exists() else []
 if not bins:
  bins=[r for r in readcsv(ROOT/'results/generated/all_exact_outputs/binned_means.csv') if r['border_pair']=='KS-MO'];fits=[r for r in readcsv(ROOT/'results/generated/all_exact_outputs/rd_fitted_lines.csv') if r['border_pair']=='KS-MO']
 fig,ax=plt.subplots(figsize=(7.2,4.8));x=np.array([flt(r['bin_center']) for r in bins]);m=np.array([flt(r['mean']) for r in bins]);lo=np.array([flt(r['ci_low']) for r in bins]);hi=np.array([flt(r['ci_high']) for r in bins])
 ax.errorbar(x,m,yerr=np.vstack([m-lo,hi-m]),fmt='o',capsize=2,linewidth=.8,markersize=4,label='1-mi binned means (95% CI)')
 for st in ['KS','MO']:
  rr=sorted([r for r in fits if r['state']==st],key=lambda r:flt(r['signed_distance']));ax.plot([flt(r['signed_distance']) for r in rr],[flt(r['fitted_mean']) for r in rr],linewidth=1.5,label=f'{st} local linear fit')
 ax.axvline(0,linestyle='--',linewidth=.9);ax.set_xlabel('Signed distance from KS-MO border (mi; Kansas negative)');ax.set_ylabel('Mean minimum condition rating');ax.legend(frameon=False,ncol=1);clean(ax);fig.tight_layout();save(fig,out,'Fig4_KSMO_RD')


def fig5(out):
    model_rows=readcsv(RES/'KS_MO_Model_Results.csv')
    rob=json.loads((RES/'KS_MO_Final_Robustness_Results.json').read_text())
    rows=[]
    specs=[
        ('Primary (25-km segments)','Primary recency-adjusted 10-mi model'),
        ('Expanded exposures + recency','Expanded exposure + recency sensitivity'),
        ('Inspection within 24 months','Inspection within 24 months'),
        ('Donut: exclude <0.8 km','Donut: exclude <0.5 mile'),
        ('Donut: exclude <1.6 km','Donut: exclude <1 mile'),
        ('Exclude shared-state structures','Exclude shared-state structures'),
        ('Exclude geometry mismatches','Exclude state-geometry mismatches'),
        ('10-km border segments','10-km border-segment definition'),
        ('50-km border segments','50-km border-segment definition')]
    for label,spec in specs:
        r=next((x for x in rob['models'] if x.get('specification')==spec),None)
        if r is None:raise ValueError('Missing robustness model: '+spec)
        rows.append((label,flt(r['estimate_a_minus_b']),flt(r['ci_low']),flt(r['ci_high'])))
    rr=next((r for r in model_rows if r.get('specification')=='Recency-adjusted overlap weighting' and r.get('outcome')=='lowest_rating'),None)
    if rr is None:raise ValueError('Missing recency-adjusted overlap-weighting estimate')
    rows.insert(3,('Overlap weighting',flt(rr['estimate_a_minus_b']),flt(rr['ci_low']),flt(rr['ci_high'])))
    y=np.arange(len(rows));est=np.array([x[1] for x in rows]);lo=np.array([x[2] for x in rows]);hi=np.array([x[3] for x in rows])
    fig,ax=plt.subplots(figsize=(7.4,max(5,.42*len(rows)+1.5)))
    ax.errorbar(est,y,xerr=np.vstack([est-lo,hi-est]),fmt='o',capsize=3,linewidth=1)
    ax.axvline(0,linestyle='--',linewidth=.9);ax.set_yticks(y,[x[0] for x in rows]);ax.invert_yaxis()
    ax.set_xlabel('Kansas - Missouri discontinuity (rating points)')
    ax.set_title('Kansas-Missouri: recency-adjusted robustness',pad=9)
    clean(ax);fig.tight_layout();save(fig,out,'Fig5_KSMO_Robustness')


def fig6(out):
    rob=json.loads((RES/'KS_MO_Final_Robustness_Results.json').read_text());rows=sorted(rob['placebo'],key=lambda r:flt(r.get('cutoff')))
    rows=[r for r in rows if all(math.isfinite(flt(r.get(k))) for k in ['cutoff','estimate_a_minus_b','ci_low','ci_high'])]
    x=np.array([flt(r['cutoff']) for r in rows]);e=np.array([flt(r['estimate_a_minus_b']) for r in rows]);lo=np.array([flt(r['ci_low']) for r in rows]);hi=np.array([flt(r['ci_high']) for r in rows])
    fig,ax=plt.subplots(figsize=(7,4.4));ax.errorbar(x,e,yerr=np.vstack([e-lo,hi-e]),fmt='o',capsize=3);ax.axhline(0,linestyle='--',linewidth=.9);ax.axvline(0,linestyle=':',linewidth=.8);ax.set_xlabel('Shifted cutoff relative to true border (mi)');ax.set_ylabel('Pseudo-discontinuity (rating points)');clean(ax);fig.tight_layout();save(fig,out,'Fig6_Placebos')


def fig7(out):
 rows=readcsv(RES/'KS_MO_Temporal_Replication_Results.csv');rows=[r for r in rows if str(r.get('inventory_year')) in {'2023','2024','2025'} and flt(r.get('bandwidth_miles'))==10 and r.get('specification')=='Inspection-recency adjusted'];rows=sorted(rows,key=lambda r:int(r['inventory_year']))
 x=np.arange(len(rows));e=np.array([flt(r['estimate_a_minus_b']) for r in rows]);lo=np.array([flt(r['ci_low']) for r in rows]);hi=np.array([flt(r['ci_high']) for r in rows])
 fig,ax=plt.subplots(figsize=(6.5,4.4));ax.errorbar(x,e,yerr=np.vstack([e-lo,hi-e]),fmt='o',capsize=4,linewidth=1.1);ax.axhline(0,linestyle='--',linewidth=.9);ax.set_xticks(x,[f"{r['inventory_year']}\n({'Discovery' if r['inventory_year']=='2025' else 'Replication'})" for r in rows]);ax.set_ylabel('Kansas - Missouri discontinuity (rating points)');ax.set_title('Temporal replication under a frozen specification',pad=9);clean(ax);fig.tight_layout();save(fig,out,'Fig7_Temporal_Replication')


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out-dir',default=str(ROOT/'outputs/figures'));args=ap.parse_args();out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
 for fn in [fig1,fig2,figs1,fig3,fig4,fig5,fig6,fig7]:fn(out)
 print('Wrote revised manuscript figures to',out)
if __name__=='__main__':main()
