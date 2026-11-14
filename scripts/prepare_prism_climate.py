"""Extract 1991–2020 PRISM 4-km normals at actual KS–MO bridge coordinates.

Uses 5 official archived PRISM grids: annual tmean, winter tmean for
December/January/February, and annual total precipitation. Not interpolated:
cell-center value is taken from the grid cell containing each bridge.

PRISM normals are climatological baseline variables, NOT observed weather at
inspection or directly comparable annual exposures for construction years.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://ftp.prism.oregonstate.edu/normals/us/4km'
RAW = ROOT / 'data/external/prism_archives'
BRIDGES = ROOT / 'data/processed/KS_MO_Enriched_Bridge_Data_25mi.csv.gz'
OUTPUT = ROOT / 'data/external/KS_MO_Bridge_Gridded_Climate.csv'
PROVENANCE = ROOT / 'data/external/KS_MO_Bridge_Gridded_Climate_provenance.json'
FILES = {
    'annual_mean_c': ('tmean', '2020'),
    'winter_dec_c': ('tmean', '202012'),
    'winter_jan_c': ('tmean', '202001'),
    'winter_feb_c': ('tmean', '202002'),
    'annual_precip_mm': ('ppt', '2020'),
}
DAYS = (31, 31, 28)  # DJF 30-year normal uses nominal non-leap monthly weighting


def spec(kind, yearmonth):
    base = f'prism_{kind}_us_25m_{yearmonth}_avg_30y.zip'
    return f'{SOURCE}/{kind}/monthly/{base}', RAW / base


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def get_archive(kind, yearmonth, download):
    url, path = spec(kind, yearmonth)
    RAW.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        if not download:
            raise FileNotFoundError(f'Missing {path.relative_to(ROOT)}; rerun with --download to fetch official PRISM data.')
        tmp = path.with_suffix('.partial')
        print(f'Downloading {path.name}...', flush=True)
        try:
            urllib.request.urlretrieve(url, tmp)
            if not zipfile.is_zipfile(tmp):
                raise ValueError(f'PRISM did not return a valid ZIP: {url}')
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
    if not zipfile.is_zipfile(path):
        raise ValueError(f'Invalid PRISM archive: {path}')
    return url, path


def sample_archive(path, lon, lat):
    """Sample official PRISM GeoTIFF (COG) or historical BIL rasters.

    PRISM distributes current normals as COGs within zip archives; BIL support is
    retained for older cached archives. Values are read without interpolation.
    """
    import rasterio
    with tempfile.TemporaryDirectory(prefix='prism_nbi_') as scratch:
        with zipfile.ZipFile(path) as z:
            all_names = [n for n in z.namelist() if not n.endswith('/')]
            tiffs = [n for n in all_names if n.lower().endswith(('.tif', '.tiff'))]
            bils = [n for n in all_names if n.lower().endswith('.bil')]
            if len(tiffs) == 1:
                raster_name = tiffs[0]
                required = [raster_name]
                raster_format = 'COG/GeoTIFF'
            elif len(bils) == 1 and not tiffs:
                raster_name = bils[0]
                stem = raster_name.rsplit('.', 1)[0]
                required = [n for n in all_names if n.rsplit('.', 1)[0] == stem and
                            n.lower().endswith(('.bil', '.hdr', '.prj'))]
                if not any(n.lower().endswith('.hdr') for n in required):
                    raise ValueError(f'BIL raster header missing from {path.name}')
                raster_format = 'BIL/HDR'
            else:
                raise ValueError(f'Expected one GeoTIFF or BIL raster in {path.name}; '
                                 f'got {len(tiffs)} GeoTIFFs and {len(bils)} BILs')
            for name in required:
                member_path = Path(name)
                if member_path.is_absolute() or '..' in member_path.parts:
                    raise ValueError(f'Unsafe member path in {path.name}: {name}')
                z.extract(name, scratch)
        raster_path = Path(scratch) / raster_name
        with rasterio.open(raster_path) as src:
            if src.crs is None:
                raise ValueError(f'{raster_format} raster in {path.name} has no CRS')
            tr = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
            coords = [tr.transform(float(x), float(y)) for x, y in zip(lon, lat)]
            values = []
            for pixel in src.sample(coords, masked=True):
                val = pixel[0]
                values.append(np.nan if np.ma.is_masked(val) else float(val))
            samples = np.asarray(values, dtype=float)
            if src.nodata is not None and np.isfinite(src.nodata):
                samples[np.isclose(samples, src.nodata, equal_nan=False)] = np.nan
            return samples


def load_bridges(path):
    if not path.is_file():
        raise FileNotFoundError(f'Run 02_exact_all_rd.py first: {path.relative_to(ROOT)}')
    seen = set()
    rows = []
    with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r['border_pair'] != 'KS-MO' or float(r['distance_miles_exact']) > 10:
                continue
            key = (r['state'], r['structure_number'].strip())
            if key in seen:
                raise ValueError(f'Duplicate bridge ID in the KS–MO 10-mile sample: {key}')
            seen.add(key)
            rows.append(r)
    if not rows:
        raise ValueError('No KS–MO bridges found in exact-distance 10-mile input')
    return rows


def build(rows, grids, urls_and_paths):
    """Return climate rows and audit; pure mapping/conversion step for unit tests."""
    n = len(rows)
    if any(len(v) != n for v in grids.values()):
        raise ValueError('PRISM sample arrays do not all match the bridge count')
    for name, vals in grids.items():
        if not np.all(np.isfinite(vals)):
            raise ValueError(f'{name}: {int(np.count_nonzero(~np.isfinite(vals)))} missing/nonfinite raster values; do not silently impute')
    a = grids['annual_mean_c']; djf = (31*grids['winter_dec_c'] + 31*grids['winter_jan_c'] + 28*grids['winter_feb_c']) / 90
    mm = grids['annual_precip_mm']
    if np.any((a < -50) | (a > 50)) or np.any((mm < 0) | (mm > 10000)):
        raise ValueError('PRISM samples out of plausible source-unit range (C or mm)')
    output = []
    for i, r in enumerate(rows):
        output.append({'state':r['state'], 'structure_number':r['structure_number'].strip(),
                       'annual_mean_temp_f':round(float(a[i]*9/5+32), 6),
                       'winter_djf_mean_temp_f':round(float(djf[i]*9/5+32), 6),
                       'annual_precip_in':round(float(mm[i]/25.4), 6)})
    audit = {'dataset':'PRISM 1991-2020 30-year monthly/annual normals', 'resolution':'4 km (25m arc-second)',
             'sampling':'PRISM grid cell containing projected bridge point; no spatial interpolation',
             'year_basis':'1991-2020 climatological normal (the PRISM 2020 normal)',
             'temp_conversion':'degrees F = degrees C * 9/5 + 32',
             'precip_conversion':'inches = mm / 25.4',
             'winter_weighting':'DJF = (Dec*31 + Jan*31 + Feb*28)/90 (non-leap climatological weighting)',
             'freeze_thaw_days':'not computed; no daily normals used',
             'bridge_count':n,
             'archives':[{'url':url,'filename':path.name,'sha256':sha256(path)} for url,path in urls_and_paths]}
    return output, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download', action='store_true', help='Download official PRISM archives if not already cached')
    args = p.parse_args()
    rows = load_bridges(BRIDGES)
    lon = [float(r['longitude']) for r in rows]
    lat = [float(r['latitude']) for r in rows]
    grids, sources = {}, []
    for label, (kind, ym) in FILES.items():
        url, path = get_archive(kind, ym, args.download)
        sources.append((url,path))
        grids[label] = sample_archive(path, lon, lat)
        print(f'PRISM {label}: {len(grids[label]):,} bridge samples',flush=True)
    output, audit = build(rows, grids, sources)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', encoding='utf-8',newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(output[0]))
        w.writeheader(); w.writerows(output)
    PROVENANCE.write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print(f'Wrote {len(output):,} bridge-coordinate PRISM records and source provenance.', flush=True)


if __name__=='__main__':
    main()
