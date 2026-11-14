"""Write hashes and provenance for raw/external inputs used in the revised rerun."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/generated/input_manifest.json'
FILES=[
 ('FHWA 2023 NBI',ROOT/'data/raw/2023HwyBridgesDelimitedAllStates.txt','https://www.fhwa.dot.gov/bridge/nbi/2023hwybronefiledel.zip',True),
 ('FHWA 2024 NBI',ROOT/'data/raw/2024HwyBridgesDelimitedAllStates.txt','https://www.fhwa.dot.gov/bridge/nbi/2024hwybronefiledel.zip',True),
 ('FHWA 2025 NBI',ROOT/'data/raw/2025HwyBridgesDelimitedAllStates.txt','https://www.fhwa.dot.gov/bridge/nbi/2025hwybronefiledel.zip',True),
 ('U.S. Census 2025 TIGER/Line States',ROOT/'data/raw/tl_2025_us_state.zip','https://www2.census.gov/geo/tiger/TIGER2025/STATE/tl_2025_us_state.zip',True),
 ('Bridge-level KS-MO climate normals',ROOT/'data/external/KS_MO_Bridge_Gridded_Climate.csv','PRISM 1991-2020 4-km normals; see provenance file',False),
 ('PRISM climate provenance',ROOT/'data/external/KS_MO_Bridge_Gridded_Climate_provenance.json','official PRISM raster download and sampling log',False),
 ('Census 2025 CBSA',ROOT/'data/external/tl_2025_us_cbsa.zip','https://www2.census.gov/geo/tiger/TIGER2025/CBSA/tl_2025_us_cbsa.zip',False),
 ('Metro subregion provenance',ROOT/'data/external/KS_MO_Subregion_Provenance.json','Census 2025 TIGER CBSA and state geometry',False),
 ('KS-MO documented border subregions',ROOT/'data/external/KS_MO_Border_Subregions.geojson','Census-derived geographic partition; see provenance JSON',False),
 ('KS-MO prepared FHWA element validation',ROOT/'data/external/KS_MO_Element_Validation.csv','user-prepared from FHWA 2025 NBI element data',False),
]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
items=[]
for label,path,url,required in FILES:
 if required and not path.exists():raise SystemExit(f'Missing required input: {path.relative_to(ROOT)}')
 if path.exists():items.append({'label':label,'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':sha(path),'source':url,'required':required})
OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps({'inputs':items},indent=2)+'\n');print(f'Wrote {OUT}')
