"""Small synthetic correctness tests; not substitutes for real-data results."""
import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_prism_climate import build, spec
from prepare_ksmo_subregions import build as subregion_build
from analysis_utils import fit_rd,wild_cluster_bootstrap_t


def test_prism_conversion_and_provenance():
    rows = [dict(state='KS', structure_number='A01'),dict(state='MO', structure_number='B02')]
    d = {
        'annual_mean_c':np.array([0.0,10.0]),
        'winter_dec_c':np.array([0.0,4.0]),
        'winter_jan_c':np.array([0.0,0.0]),
        'winter_feb_c':np.array([0.0,0.0]),
        'annual_precip_mm':np.array([254.0,508.0]),
    }
    out, prov = build(rows, d, [])
    assert len(out)==2
    assert out[0]['annual_mean_temp_f']==32.0
    assert out[1]['annual_mean_temp_f']==50.0
    assert abs(out[1]['winter_djf_mean_temp_f']-(32+((31*4)/90)*1.8))<1e-5
    assert out[0]['annual_precip_in']==10.0
    assert out[1]['annual_precip_in']==20.0
    assert prov['bridge_count']==2
    assert '1991-2020' in prov['dataset']
    assert spec('tmean','202001')[0].endswith('/tmean/monthly/prism_tmean_us_25m_202001_avg_30y.zip')


def test_prism_missing_grid_values_are_never_imputed():
    rows=[dict(state='KS',structure_number='A')]
    d={k:np.array([1.]) for k in ['annual_mean_c','winter_dec_c','winter_jan_c','winter_feb_c','annual_precip_mm']}
    d['annual_precip_mm'][0]=np.nan
    try:build(rows,d,[])
    except ValueError as exc:assert 'missing/nonfinite' in str(exc)
    else:raise AssertionError('Must fail when PRISM raster has missing cells')


def test_nine_cluster_exhaustive_bootstrap():
    rng=np.random.default_rng(77)
    rows=[]
    for g in range(9):
        for s in (0,1):
            for _ in range(8):
                z=float(rng.uniform(0.1,9))
                rows.append(dict(signed_distance_miles_exact=z if s else -z,
                                 segment_50km=f'S:{g}',lowest_rating=float(6+.2*(1-s)-.01*z+rng.normal(0,.3))))
    f=fit_rd(rows,'lowest_rating',segment_field='segment_50km',min_n=40)
    assert f['clusters']==9 and f['covariance']=='HC3'
    w=wild_cluster_bootstrap_t(f,draws=50000,seed=42)
    assert w['method']=='exhaustive_rademacher_bootstrap_t'
    assert w['patterns']==512
    assert 0<=w['p_value']<=1


def test_ksmo_metro_subregion_partition_on_synthetic_geography():
    # Synthetic geometry: no real geodata, verify two labels and consistent split.
    states=gpd.GeoDataFrame({'STUSPS':['KS','MO']},geometry=[box(-101,35,-100,36),box(-100,35,-99,36)],crs='EPSG:4326')
    cbsa=gpd.GeoDataFrame({'GEOID':['28140']},geometry=[box(-100.5,35.4,-99.5,35.7)],crs='EPSG:4326')
    result,prov=subregion_build(states,cbsa)
    assert set(result['label'])=={'KC_metro','nonmetro'}
    assert set(result['group'])=={'metro'}
    assert prov['metro_boundary_length_km']>0
    assert prov['nonmetro_boundary_length_km']>0
