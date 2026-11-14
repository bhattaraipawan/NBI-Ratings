import sys
from pathlib import Path
import unittest
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from analysis_utils import merged_border_parts,border_chainage_segment,fit_rd,holm_adjust,parse_inspection_mmyy,wild_cluster_bootstrap_t

class AnalysisUtilsTests(unittest.TestCase):
    def test_inspection_mmyy(self):
        self.assertEqual(parse_inspection_mmyy('0625',2025),(2025.0,6.0,6.0))
        self.assertEqual(parse_inspection_mmyy('1124',2025),(2024.0,11.0,13.0))

    def test_holm(self):
        out=holm_adjust([.001,.02,.04,.4])
        self.assertTrue(np.allclose(out,[.004,.06,.08,.4]))

    def test_rd_sign_convention(self):
        rng=np.random.default_rng(17);rows=[]
        for g in range(12):
            for side in (0,1):
                for _ in range(10):
                    d=float(rng.uniform(0,10));age=float(rng.normal(45,8));y=6.0+(0.5 if side==0 else 0)-.02*d+.01*age+float(rng.normal(0,.25))
                    rows.append({'side_b':side,'signed_distance_miles_exact':d if side else -d,'nearest_border_segment_25km_exact':f'X:{g}',
                                 'lowest_rating':y,'age':age,'log_adt':8.0,'structure_length_m':40.0,'max_span_m':18.0,'deck_width_m':9.0,
                                 'structure_kind_code':'1','structure_type_code':'2','functional_class_code':'3','design_load_code':'4','highway_system_code':'1'})
        r=fit_rd(rows,'lowest_rating',cont_controls=['age','log_adt','structure_length_m','max_span_m','deck_width_m'],cat_controls=['structure_kind_code','structure_type_code','functional_class_code','design_load_code','highway_system_code'])
        self.assertGreater(r['estimate_a_minus_b'],0.25)
        self.assertLess(r['estimate_a_minus_b'],0.75)

    def test_wild_cluster_bootstrap_matches_statsmodels(self):
        import itertools
        import statsmodels.api as sm
        rng=np.random.default_rng(31);rows=[]
        # Six clusters keeps the reference exhaustive test small (64 patterns).
        for g in range(6):
            for side in (0,1):
                for _ in range(12):
                    d=float(rng.uniform(0.2,8.0));age=float(rng.normal(48,7))
                    y=6.1+(0.45 if side==0 else 0)-.018*d+.008*age+float(rng.normal(0,.30))
                    rows.append({'side_b':side,'signed_distance_miles_exact':d if side else -d,'nearest_border_segment_25km_exact':f'X:{g}',
                                 'lowest_rating':y,'age':age,'log_adt':8.0,'structure_length_m':40.0,'max_span_m':18.0,'deck_width_m':9.0,
                                 'structure_kind_code':'1','structure_type_code':'2','functional_class_code':'3','design_load_code':'4','highway_system_code':'1'})
        base=fit_rd(rows,'lowest_rating',cont_controls=['age','log_adt','structure_length_m','max_span_m','deck_width_m'],
                    cat_controls=['structure_kind_code','structure_type_code','functional_class_code','design_load_code','highway_system_code'])
        fast=wild_cluster_bootstrap_t(base,draws=50000,seed=42)
        X=np.asarray(base['_X'],float); y=np.asarray(base['_y'],float); groups=np.asarray(base['_groups'],object); names=base['_names']; j=names.index('side_b')
        obs=sm.OLS(y,X).fit(); cov=obs.get_robustcov_results(cov_type='cluster',groups=groups,use_correction=True,df_correction=True)
        observed_t=abs(float(obs.params[j]/cov.bse[j]))
        self.assertAlmostEqual(fast['observed_abs_t'],observed_t,places=9)
        X0=np.delete(X,j,axis=1); null=sm.OLS(y,X0).fit(); fitted=np.asarray(null.fittedvalues); resid=np.asarray(null.resid)
        uniq=np.unique(groups); idx=[np.flatnonzero(groups==g) for g in uniq]; exceed=0; total=2**len(uniq)
        for pat in itertools.product((-1.0,1.0),repeat=len(uniq)):
            e=resid.copy()
            for sign,ii in zip(pat,idx): e[ii]*=sign
            boot=sm.OLS(fitted+e,X).fit(); bcov=boot.get_robustcov_results(cov_type='cluster',groups=groups,use_correction=True,df_correction=True)
            tb=abs(float(boot.params[j]/bcov.bse[j]))
            if tb>=observed_t-1e-9: exceed+=1
        self.assertAlmostEqual(fast['p_value'],exceed/total,places=12)
        self.assertEqual(fast['method'],'exhaustive_rademacher_bootstrap_t')

if __name__=='__main__':unittest.main()


def test_connected_border_fragments_are_merged_before_segmentation():
    raw=MultiLineString([LineString([(0,0),(5000,0)]),LineString([(5000,0),(15000,0)]),LineString([(15000,0),(60000,0)])])
    parts=merged_border_parts(raw)
    assert len(parts)==1
    assert round(parts[0].length)==60000
    p=Point(12000,100)
    s10,_,_,_=border_chainage_segment(p,parts,'AA-BB',10)
    s25,_,_,_=border_chainage_segment(p,parts,'AA-BB',25)
    assert s10.endswith(':1')
    assert s25.endswith(':0')
