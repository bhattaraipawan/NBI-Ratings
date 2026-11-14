"""Audit manuscript table values against frozen reproduction outputs."""
import csv, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results/frozen'
if not (RES/'KS_MO_Climate_Sensitivity_Results.json').exists(): RES=ROOT/'results/generated'
checks=[]
def check(label,actual,expected,decimals=3):
  actual=float(actual);expected=float(expected)
  ok=round(actual,decimals)==round(expected,decimals)
  checks.append({'name':label,'actual':actual,'manuscript':expected,'decimals':decimals,'passed':ok})
def lookup(rows,**filters):
  found=[r for r in rows if all(str(r.get(k))==str(v) for k,v in filters.items())]
  if len(found)!=1:raise ValueError(f'Expected one result for {filters}; got {len(found)}')
  return found[0]
with (RES/'National_2025_FewCluster_Holm_Sensitivity.csv').open(newline='') as f:national=list(csv.DictReader(f))
expected={'AL-MS':-.256,'CT-MA':-.366,'DE-MD':.278,'IA-NE':-.604,'ID-MT':-.289,'IN-KY':.656,'KS-MO':.401,'MA-RI':.496,'NC-VA':-.189,'NE-SD':.812,'NJ-PA':.239,'OK-TX':.209,'OR-WA':-.167}
for border,est in expected.items():
  row=lookup(national,border_pair=border)
  check(f'2025 {border} estimate',row['recency_adjusted_estimate_a_minus_b'],est)
passing=[r['border_pair'] for r in national if float(r['p_holm_fewcluster'])<.05]
assert sorted(passing)==['IA-NE','IN-KY','KS-MO','NE-SD']
check('MA-RI corrected Holm p',lookup(national,border_pair='MA-RI')['p_holm_fewcluster'],.3125,4)
climate=json.loads((RES/'KS_MO_Climate_Sensitivity_Results.json').read_text())
assert climate['metadata']['preferred_final_analysis'] and climate['metadata']['missing_climate_rows']==0
for spec,est,lo,hi in [('Bridge/traffic/recency adjusted',.415,.309,.521),('Climate adjusted',.428,.316,.540)]:
 r=lookup(climate['models'],bandwidth_miles=10,outcome='lowest_rating',specification=spec)
 assert int(r['n'])==2887 and r['clusters']==18
 for fld,val in [('estimate_a_minus_b',est),('ci_low',lo),('ci_high',hi)]:check('PRISM '+spec+' '+fld,r[fld],val)
bal=climate['climate_balance']
p=lookup(bal,bandwidth_miles=10,variable='annual_precip_in')
check('precipitation SMD',p['smd'],.026)
check('KS–MO precip difference mm',(p['mean_a']-p['mean_b'])*25.4,1.8,1)
t=lookup(bal,bandwidth_miles=10,variable='winter_djf_mean_temp_f')
check('KS–MO winter difference Celsius',(t['mean_a']-t['mean_b'])*5/9,.22,2)
tmp=json.loads((RES/'KS_MO_Temporal_Replication_Results.json').read_text())['results']
for year,n,est,lo,hi in [(2023,2877,.270,.153,.388),(2024,2875,.281,.175,.388),(2025,2887,.401,.311,.491)]:
 r=lookup(tmp,inventory_year=year,bandwidth_miles=10,specification='Inspection-recency adjusted'); assert r['n']==n
 for field,val in [('estimate_a_minus_b',est),('ci_low',lo),('ci_high',hi)]:check(f'{year} '+field,r[field],val)
r=lookup(tmp,inventory_year='2023-2025 pooled',bandwidth_miles=10,specification='Inspection-recency adjusted + year fixed effects')
assert r['n']==8639
for field,val in [('estimate_a_minus_b',.307),('ci_low',.211),('ci_high',.404)]:check('pooled '+field,r[field],val)
metro=json.loads((RES/'KS_MO_Subregion_Sensitivity_Results.json').read_text())['results']
for grp,n,val in [('KC_metro',1826,.382),('nonmetro',1061,.406)]:
 r=lookup(metro,label=grp); assert r['n']==n;check(grp+' estimate',r['estimate_a_minus_b'],val)
report={'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL', 'checks':checks,'passed':sum(x['passed'] for x in checks),'total':len(checks),'full_climate_data':climate['metadata']}
(RES/'manuscript_numeric_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['status'],report['passed'],'/',report['total'],'number checks passed')
if report['status']!='PASS':
 for x in checks:
  if not x['passed']:print('FAILED',x)
 sys.exit(2)
