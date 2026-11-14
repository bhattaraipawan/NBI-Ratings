"""Shared utilities for the revised NBI border-discontinuity workflow.

The revised workflow deliberately centralizes parsing, design-matrix construction,
inference, multiplicity correction, and plotting summaries so the same estimand is
used across the national, Kansas-Missouri, and temporal-replication analyses.
"""
from __future__ import annotations

import math
from collections import Counter
from datetime import date
from typing import Iterable

import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2_contingency, norm, t as student_t
from shapely.ops import linemerge, unary_union

M_PER_MILE = 1609.344

FIPS_TO_STATE = {
    "01":"AL","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","12":"FL","13":"GA",
    "16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD",
    "25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH",
    "34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA",
    "44":"RI","45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA",
    "54":"WV","55":"WI","56":"WY"
}

BASE_CONTROLS = ["age", "log_adt", "structure_length_m", "max_span_m", "deck_width_m"]
BASE_CATS = ["structure_kind_code", "structure_type_code", "functional_class_code", "design_load_code", "highway_system_code"]


def merged_border_parts(border):
    """Merge connected linework into stable continuous border components.

    TIGER polygon-boundary intersections are often MultiLineStrings split into
    many small connected pieces. Treating every raw piece as an independent
    component creates hundreds of pseudo-clusters and defeats 10/25/50-km
    segment sensitivity checks. This function first merges connected linework
    and returns only genuinely disconnected continuous components.
    """
    lines=[]
    def collect(g):
        if g is None or getattr(g, "is_empty", True):
            return
        gt=getattr(g, "geom_type", "")
        if gt in {"LineString", "LinearRing"}:
            lines.append(g)
        elif gt in {"MultiLineString", "GeometryCollection"}:
            for part in g.geoms:
                collect(part)
    collect(border)
    if not lines:
        return []
    u=unary_union(lines)
    if getattr(u, "geom_type", "") == "LineString":
        merged=u
    else:
        try:
            merged=linemerge(u)
        except Exception:
            merged=u
    out=[]
    def collect_out(g):
        if g is None or getattr(g, "is_empty", True):
            return
        gt=getattr(g, "geom_type", "")
        if gt == "LineString" and g.length > 0:
            out.append(g)
        elif gt in {"MultiLineString", "GeometryCollection"}:
            for part in g.geoms:
                collect_out(part)
    collect_out(merged)
    out.sort(key=lambda g:(round(g.bounds[0],3),round(g.bounds[1],3),round(g.bounds[2],3),round(g.bounds[3],3),round(g.length,3)))
    return out


def border_chainage_segment(point, parts, pair, segment_km=25):
    """Assign a point to the nearest continuous border component and segment."""
    if not parts:
        raise ValueError("No continuous border components available")
    best=None;best_local=0.0;best_part=0;offset=0.0;best_global=0.0
    for i,part in enumerate(parts):
        d=point.distance(part);proj=part.project(point)
        if best is None or d < best:
            best=d;best_local=float(proj);best_part=i;best_global=float(offset+proj)
        offset += float(part.length)
    width=float(segment_km)*1000.0
    seg=f"{pair}:p{best_part}:{int(best_local//width)}"
    return seg,best_global,best_part,best_local


def fnum(v):
    try:
        s = str(v).strip()
        return float(s) if s else math.nan
    except Exception:
        return math.nan


def finite(v):
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def clean_code(v):
    return "" if v is None else str(v).strip()


def dms_to_decimal(v, longitude=False):
    """Parse legacy NBI compact DMS coordinates."""
    s = str(v or "").strip()
    if not s.isdigit():
        return math.nan
    width = 9 if longitude else 8
    s = s.zfill(width)
    deg_n = 3 if longitude else 2
    try:
        deg = int(s[:deg_n])
        minute = int(s[deg_n:deg_n+2])
        sec = int(s[deg_n+2:deg_n+4]) + int(s[deg_n+4:deg_n+6]) / 100.0
        out = deg + minute / 60.0 + sec / 3600.0
        return -out if longitude else out
    except Exception:
        return math.nan


def parse_inspection_mmyy(value, inventory_year):
    """Parse legacy NBI Item 90 (MMYY) and compute recency at Dec. 31 of inventory year.

    Returns (inspection_year, inspection_month, inspection_age_months). Invalid or
    future dates return NaNs. The day is intentionally not imputed because the
    Coding Guide does not provide it.
    """
    s = str(value or "").strip()
    if not s.isdigit():
        return math.nan, math.nan, math.nan
    s = s.zfill(4)
    try:
        month = int(s[:2]); yy = int(s[2:])
    except Exception:
        return math.nan, math.nan, math.nan
    if not 1 <= month <= 12:
        return math.nan, math.nan, math.nan
    candidates = [1900 + yy, 2000 + yy]
    valid = [y for y in candidates if y <= inventory_year]
    if not valid:
        return math.nan, math.nan, math.nan
    year = max(valid)
    age_months = (inventory_year - year) * 12 + (12 - month)
    if age_months < 0 or age_months > 240:
        return math.nan, math.nan, math.nan
    return float(year), float(month), float(age_months)


def pooled_smd(a, b):
    a = np.asarray([float(x) for x in a if finite(x)], float)
    b = np.asarray([float(x) for x in b if finite(x)], float)
    if len(a) < 2 or len(b) < 2:
        return math.nan
    sp = math.sqrt(((len(a)-1)*np.var(a, ddof=1) + (len(b)-1)*np.var(b, ddof=1)) / (len(a)+len(b)-2))
    return float((np.mean(a)-np.mean(b))/sp) if sp > 1e-12 else 0.0


def cramers_v(vals, sides):
    vals = [clean_code(v) or "MISSING" for v in vals]
    sides = list(sides)
    levels = sorted(set(vals)); side_levels = sorted(set(sides))
    if len(levels) < 2 or len(side_levels) < 2:
        return 0.0
    tab = np.zeros((len(side_levels), len(levels)), dtype=float)
    li = {v:i for i,v in enumerate(levels)}; si = {v:i for i,v in enumerate(side_levels)}
    for v, s in zip(vals, sides):
        tab[si[s], li[v]] += 1.0
    tab = tab[:, tab.sum(axis=0) > 0]
    if tab.shape[1] < 2:
        return 0.0
    chi2, *_ = chi2_contingency(tab, correction=False)
    n = tab.sum(); denom = min(tab.shape[0]-1, tab.shape[1]-1)
    return math.sqrt((chi2/n)/denom) if n > 0 and denom > 0 else 0.0


def shared_categories(rows, field, min_each=5, sides=None):
    if sides is None:
        sides = [int(r["side_b"]) for r in rows]
    ca = Counter(clean_code(r.get(field)) or "MISSING" for r,s in zip(rows,sides) if int(s) == 0)
    cb = Counter(clean_code(r.get(field)) or "MISSING" for r,s in zip(rows,sides) if int(s) == 1)
    return {k for k in set(ca) | set(cb) if ca[k] >= min_each and cb[k] >= min_each}


def one_hot(values, keep=None, prefix="cat"):
    vals = [clean_code(v) or "MISSING" for v in values]
    if keep is not None:
        vals = [v if v in keep else "OTHER" for v in vals]
    levels = sorted(set(vals))
    if len(levels) <= 1:
        return np.zeros((len(vals),0)), []
    cols=[]; names=[]
    for lev in levels[1:]:
        cols.append(np.array([1.0 if v == lev else 0.0 for v in vals]))
        names.append(f"{prefix}={lev}")
    return np.column_stack(cols), names


def build_rd_design(rows, cont_controls, cat_controls, *, cutoff=0.0,
                    segment_field="nearest_border_segment_25km_exact",
                    extra_fixed_effect_fields=()):
    signed = np.array([float(r["signed_distance_miles_exact"]) - cutoff for r in rows], float)
    side = (signed >= 0).astype(float)
    dist = np.abs(signed)
    cols = [np.ones(len(rows)), side, dist, side*dist]
    names = ["Intercept", "side_b", "distance_abs", "side_x_distance"]

    if segment_field:
        keep = shared_categories(rows, segment_field, 2, sides=side)
        x, n = one_hot([r.get(segment_field, "") for r in rows], keep, "segment")
        if x.shape[1]: cols.extend([x[:,j] for j in range(x.shape[1])])
        names += n

    for field in extra_fixed_effect_fields:
        keep = set(clean_code(r.get(field)) or "MISSING" for r in rows)
        x,n = one_hot([r.get(field, "") for r in rows], keep, field)
        if x.shape[1]: cols.extend([x[:,j] for j in range(x.shape[1])])
        names += n

    for field in cont_controls:
        v = np.array([float(r[field]) for r in rows], float)
        sd = np.std(v, ddof=1)
        v = (v - np.mean(v))/sd if sd > 1e-12 else np.zeros(len(v))
        cols.append(v); names.append(field + "_z")

    for field in cat_controls:
        keep = shared_categories(rows, field, max(5, int(.01*len(rows))), sides=side)
        x,n = one_hot([r.get(field, "") for r in rows], keep, field)
        if x.shape[1]: cols.extend([x[:,j] for j in range(x.shape[1])])
        names += n

    X = np.column_stack(cols)
    keep_cols = [0] + [j for j in range(1, X.shape[1]) if np.nanstd(X[:,j]) > 1e-12]
    return X[:,keep_cols], [names[j] for j in keep_cols], side, dist


def fit_rd(rows, outcome, *, cont_controls=(), cat_controls=(), cutoff=0.0,
           segment_field="nearest_border_segment_25km_exact", weight_field=None,
           extra_fixed_effect_fields=(), min_n=40):
    required = [outcome] + list(cont_controls)
    used=[]; weights=[]
    for r in rows:
        if any(not finite(r.get(c)) for c in required):
            continue
        if weight_field is not None and (not finite(r.get(weight_field)) or float(r[weight_field]) <= 0):
            continue
        used.append(r); weights.append(float(r.get(weight_field, 1.0)))
    if len(used) < min_n:
        return {"n":len(used), "error":"insufficient sample"}

    X,names,side,_ = build_rd_design(
        used, list(cont_controls), list(cat_controls), cutoff=cutoff,
        segment_field=segment_field, extra_fixed_effect_fields=extra_fixed_effect_fields
    )
    if "side_b" not in names or min(np.bincount(side.astype(int), minlength=2)) == 0:
        return {"n":len(used), "error":"both sides not represented"}
    y = np.array([float(r[outcome]) for r in used], float)
    model = sm.WLS(y, X, weights=np.asarray(weights)) if weight_field else sm.OLS(y, X)
    j = names.index("side_b")
    groups = np.array([clean_code(r.get(segment_field)) or "MISSING" for r in used]) if segment_field else np.arange(len(used))
    nc = len(set(groups))

    if segment_field and nc >= 10:
        res = model.fit(cov_type="cluster", cov_kwds={"groups":groups, "use_correction":True})
        df = nc - 1
        est = -float(res.params[j]); se = float(res.bse[j])
        crit = student_t.ppf(.975, df)
        ci_low, ci_high = est-crit*se, est+crit*se
        p = float(2*student_t.sf(abs(est/se), df)) if se > 0 else math.nan
        reference = f"t({df})"
        covariance = f"cluster:{segment_field}"
    else:
        res = model.fit(cov_type="HC3"); df = None
        est = -float(res.params[j]); se = float(res.bse[j])
        ci = res.conf_int(alpha=.05)[j]
        ci_low, ci_high = -float(ci[1]), -float(ci[0])
        p = float(res.pvalues[j])
        reference = "normal robust"
        covariance = "HC3"

    return {
        "n":len(used), "clusters":nc if segment_field else None, "k":X.shape[1],
        "covariance":covariance, "inference_reference":reference,
        "estimate_a_minus_b":est, "se":se, "ci_low":ci_low, "ci_high":ci_high,
        "p_value":p, "r_squared":float(res.rsquared), "df":df,
        "_used":used, "_X":X, "_names":names, "_y":y, "_groups":groups,
        "_model":model, "_fit":res
    }


def strip_private_fit(result):
    return {k:v for k,v in result.items() if not k.startswith("_")}


def holm_adjust(pvalues: Iterable[float]):
    """Holm family-wise adjusted p-values, preserving original order."""
    p = np.asarray(list(pvalues), float)
    out = np.full(len(p), np.nan)
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return out.tolist()
    pv = p[valid]
    order = np.argsort(pv)
    m = len(pv); running = 0.0
    adj_sorted = np.zeros(m)
    for rank, pos in enumerate(order):
        val = min(1.0, (m-rank)*pv[pos])
        running = max(running, val)
        adj_sorted[pos] = running
    out[valid] = adj_sorted
    return out.tolist()


def binned_summary(rows, outcome="lowest_rating", bw=10.0, bin_width=1.0):
    out=[]
    edges = np.arange(-bw, bw+bin_width, bin_width)
    for lo, hi in zip(edges[:-1], edges[1:]):
        vals = np.asarray([float(r[outcome]) for r in rows
                           if finite(r.get(outcome)) and lo <= float(r["signed_distance_miles_exact"]) < hi], float)
        if len(vals) == 0:
            continue
        mean = float(np.mean(vals)); se = float(np.std(vals,ddof=1)/math.sqrt(len(vals))) if len(vals)>1 else math.nan
        out.append({"bin_low":float(lo), "bin_high":float(hi), "bin_center":float((lo+hi)/2),
                    "n":int(len(vals)), "mean":mean, "se":se,
                    "ci_low":mean-1.96*se if finite(se) else math.nan,
                    "ci_high":mean+1.96*se if finite(se) else math.nan})
    return out


def wild_cluster_bootstrap_t(base, draws=50000, seed=42):
    """Restricted Rademacher wild-cluster bootstrap-t for H0: jump = 0.

    The null-restricted model removes only the level-jump coefficient and keeps
    the side-specific slope interaction and all controls. The implementation
    uses the closed-form OLS/cluster-sandwich algebra rather than refitting a
    statsmodels model for every draw; the statistic is unchanged but the 50,000
    draw robustness check is much faster.
    """
    import itertools

    X=np.asarray(base["_X"],float); y=np.asarray(base["_y"],float)
    names=base["_names"]; groups=np.asarray(base["_groups"],object)
    j=names.index("side_b")
    n,k=X.shape
    uniq=np.array(sorted(set(groups)),dtype=object); G=len(uniq)
    if G<2 or n<=k:
        return {"error":"insufficient cluster/df support for bootstrap-t"}

    bread=np.linalg.pinv(X.T@X)
    beta_obs=bread@(X.T@y); resid_obs=y-X@beta_obs
    cvec=X@bread[:,j]
    index=[np.flatnonzero(groups==g) for g in uniq]
    correction=(G/(G-1))*((n-1)/(n-k))

    def coef_se_from_residual(resid):
        scores=np.array([float(cvec[ii]@resid[ii]) for ii in index])
        return math.sqrt(max(0.0,correction*float(scores@scores)))

    se_obs=coef_se_from_residual(resid_obs)
    if not finite(se_obs) or se_obs<=0:
        return {"error":"nonpositive observed cluster-robust standard error"}
    observed_t=abs(float(beta_obs[j]/se_obs))

    X0=np.delete(X,j,axis=1)
    beta0=np.linalg.pinv(X0.T@X0)@(X0.T@y)
    fitted0=X0@beta0; resid0=y-fitted0
    beta_fitted_full=bread@(X.T@fitted0)

    v=np.stack([X[ii].T@resid0[ii] for ii in index],axis=0)          # G x k
    a=np.array([float(cvec[ii]@resid0[ii]) for ii in index])        # G
    d=np.stack([cvec[ii]@X[ii] for ii in index],axis=0)             # G x k

    if G<=10:
        patterns=itertools.product((-1.0,1.0),repeat=G)
        total=2**G;method="exhaustive_rademacher_bootstrap_t";seed_out=None
    else:
        rng=np.random.default_rng(seed)
        patterns=(rng.choice([-1.0,1.0],size=G) for _ in range(draws))
        total=draws;method="monte_carlo_rademacher_bootstrap_t";seed_out=seed

    exceed=0;ts=[]
    for pat in patterns:
        signs=np.asarray(pat,float)
        delta=bread@(signs@v)
        beta_j=float(beta_fitted_full[j]+delta[j])
        scores=signs*a-d@delta
        se=math.sqrt(max(0.0,correction*float(scores@scores)))
        tb=abs(beta_j/se) if se>0 else math.inf
        ts.append(tb)
        if tb>=observed_t-1e-9:exceed+=1
    p=exceed/total if method.startswith("exhaustive") else (exceed+1)/(total+1)
    return {
        "p_value":float(p),"observed_abs_t":observed_t,"observed_cluster_se":se_obs,
        "method":method,"patterns":int(total),"seed":seed_out,
        "small_sample_correction":"G/(G-1) * (N-1)/(N-k)",
        "bootstrap_abs_t_q95":float(np.quantile(np.asarray(ts,float),.95))
    }

