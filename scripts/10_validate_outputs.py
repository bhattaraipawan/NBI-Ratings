"""Consistency checks for the revised analysis workflow."""
from pathlib import Path
import csv,json,math
ROOT=Path(__file__).resolve().parents[1];GEN=ROOT/'results/generated';OUT=GEN/'reproducibility_report.json'
required=[GEN/'input_manifest.json',GEN/'National_2025_NBI_Border_Screening.csv',GEN/'National_2025_NBI_High_Priority_Borders.csv',GEN/'screening_method.json',GEN/'AllEligible_Exact_RD_Results.json',GEN/'National_2025_FewCluster_Holm_Sensitivity.csv',GEN/'KS_MO_Confounder_Matching_Results.json',GEN/'KS_MO_Final_Robustness_Results.json',GEN/'KS_MO_Climate_Sensitivity_Results.json',GEN/'KS_MO_Temporal_Replication_Results.json']
missing=[str(p.relative_to(ROOT)) for p in required if not p.exists()];checks=[{'check':'required_files_present','passed':not missing,'details':missing}]
if (GEN/'National_2025_NBI_High_Priority_Borders.csv').exists() and (GEN/'AllEligible_Exact_RD_Results.json').exists():
 with open(GEN/'National_2025_NBI_High_Priority_Borders.csv',encoding='utf-8-sig',newline='') as f:high=sorted(r['border_pair'] for r in csv.DictReader(f))
 ex=json.loads((GEN/'AllEligible_Exact_RD_Results.json').read_text());anal=sorted(ex.get('selection',{}).get('analysis_pairs',[]))
 checks.append({'check':'all_eligible_borders_analyzed_exactly','passed':high==anal,'details':{'eligible':high,'analyzed':anal}})
 primary=[r for r in ex.get('rd_results',[]) if r.get('bandwidth')==10 and r.get('outcome')=='lowest_rating']
 checks.append({'check':'primary_family_matches_eligible_count','passed':len(primary)==len(high),'details':{'eligible_n':len(high),'primary_n':len(primary)}})
 checks.append({'check':'holm_adjustment_present','passed':all('adjusted_p_holm_family' in r and math.isfinite(float(r['adjusted_p_holm_family'])) for r in primary),'details':None})
 checks.append({'check':'recency_holm_adjustment_present','passed':all('recency_adjusted_p_holm_family' in r and math.isfinite(float(r['recency_adjusted_p_holm_family'])) for r in primary),'details':None})
 checks.append({'check':'ksmo_available_for_detailed_case','passed':'KS-MO' in high,'details':{'eligible':high}})

arch=None  # clean release: preserve the frozen selection rule as an explicit invariant
newhigh=GEN/'National_2025_NBI_High_Priority_Borders.csv'
if newhigh.exists():
 oldset=sorted('AL-MS CT-MA DE-MD IA-NE ID-MT IN-KY KS-MO MA-RI NC-VA NE-SD NJ-PA OK-TX OR-WA'.split())
 with open(newhigh,encoding='utf-8-sig',newline='') as f:newset=sorted(r['border_pair'] for r in csv.DictReader(f))
 checks.append({'check':'submitted_eligibility_set_preserved','passed':oldset==newset,'details':{'submitted':oldset,'revised':newset}})
if (GEN/'National_2025_FewCluster_Holm_Sensitivity.csv').exists():
 with open(GEN/'National_2025_FewCluster_Holm_Sensitivity.csv',newline='',encoding='utf-8-sig') as fh:national=list(csv.DictReader(fh))
 from analysis_utils import holm_adjust
 actual=[float(r['p_holm_fewcluster']) for r in national]
 expected=holm_adjust([float(r['p_fewcluster']) for r in national])
 checks.append({'check':'national_fewcluster_holm_recomputed',
                'passed':len(national)==13 and len(set(r['border_pair'] for r in national))==13 and all(abs(a-b)<1e-10 for a,b in zip(actual,expected)),
                'details':{'n_borders':len(national)}})
if (GEN/'KS_MO_Final_Robustness_Results.json').exists():
 r=json.loads((GEN/'KS_MO_Final_Robustness_Results.json').read_text());wb=r.get('wild_cluster_bootstrap_t',{})
 checks.append({'check':'wild_cluster_bootstrap_t_used','passed':'bootstrap_t' in wb.get('method',''),'details':wb})
 segs=r.get('segment_sensitivity',[])
 checks.append({'check':'three_segment_definitions_tested','passed':len(segs)==3,'details':segs})
 cl=[int(x.get('clusters',0) or 0) for x in segs]
 distinct=(len(set(cl))>=2 and len(cl)==3 and cl[0]>=cl[1]>=cl[2])
 checks.append({'check':'segment_definitions_change_cluster_structure','passed':distinct,'details':{'cluster_counts_10_25_50km':cl}})
 if len(segs)==3:
  wb50=segs[2].get('wild_cluster_bootstrap_t',{})
  expected_patterns=2**cl[2] if 0<cl[2]<=10 else None
  checks.append({'check':'50km_few_cluster_bootstrap',
                 'passed':wb50.get('method')=='exhaustive_rademacher_bootstrap_t' and wb50.get('patterns')==expected_patterns and 0<=wb50.get('p_value',-1)<=1,
                 'details':wb50})
 ph=[x.get('p_holm_placebo_family') for x in r.get('placebo',[])]
 checks.append({'check':'placebo_family_adjustment_present','passed':len(ph)==6 and all(x is not None and math.isfinite(float(x)) for x in ph),'details':ph})
if (GEN/'KS_MO_Temporal_Replication_Results.json').exists():
 r=json.loads((GEN/'KS_MO_Temporal_Replication_Results.json').read_text());yrs={x.get('inventory_year') for x in r.get('results',[]) if x.get('role')=='replication'}
 checks.append({'check':'2023_2024_replication_present','passed':{2023,2024}.issubset(yrs),'details':sorted(str(x) for x in yrs)})
if (GEN/'KS_MO_Climate_Sensitivity_Results.json').exists():
 r=json.loads((GEN/'KS_MO_Climate_Sensitivity_Results.json').read_text());preferred=bool(r.get('metadata',{}).get('preferred_final_analysis'))
 checks.append({'check':'climate_analysis_status','passed':True,
                 'details':{'preferred_final_analysis':preferred,'status':'bridge-level PRISM climate' if preferred else 'not generated'},
                 'warning':None if preferred else 'Run python run_all.py --prepare-external to obtain gridded PRISM normals.'})
report={'passed':all(c['passed'] for c in checks) and not missing,'manuscript_ready':(all(c['passed'] for c in checks) and not missing and (not (GEN/'KS_MO_Climate_Sensitivity_Results.json').exists() or bool(json.loads((GEN/'KS_MO_Climate_Sensitivity_Results.json').read_text()).get('metadata',{}).get('preferred_final_analysis')))),'checks':checks};OUT.write_text(json.dumps(report,indent=2,allow_nan=True)+'\n');print(json.dumps(report,indent=2,allow_nan=True))
if not report['passed']:raise SystemExit(2)
