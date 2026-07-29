#!/usr/bin/env python3
"""Robustness checks reported in the paper (Stage 4 and Results).

Pinned computation so every number in the text is reproducible:
  1. K-means (k=3) vs threshold-rule agreement (ARI) on the manual usage
     shares, plus silhouette scores for k = 2..5.
  2. Sensitivity of the 0.6/0.2 threshold rule (dominance cutoff +-0.05,
     gap cutoff +-0.05: all combinations).
  3. Pipeline facility-share validation: per-deployment shares vs manual,
     and type agreement when the classification rules run on the
     pipeline's own shares.
  4. Pairing sensitivity: Spearman rho with the second views
     (Loc_04-2, Loc_19-2) dropped.

Settings are fixed (sklearn KMeans, n_init=50, random_state=0, features =
the three facility shares) because silhouette values move with the
implementation; these are the settings behind the numbers in the text.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.join(HERE, "..", "data", "manual_counts_new.csv")
PIPE = os.path.join(HERE, "..", "outputs_new", "all_locations_summary.csv")

SMALL_N = {"02", "21", "25"}  # fewer than 10 manual events


def rule_type(sw, bl, rd):
    shares = sorted((("SW", sw), ("BL", bl), ("RD", rd)),
                    key=lambda kv: -kv[1])
    if shares[0][1] >= 0.6 and shares[0][1] - shares[1][1] >= 0.2:
        return shares[0][0]
    return "Mixed"


def rule_type_custom(sw, bl, rd, dom, gap):
    shares = sorted((("SW", sw), ("BL", bl), ("RD", rd)),
                    key=lambda kv: -kv[1])
    if shares[0][1] >= dom and shares[0][1] - shares[1][1] >= gap:
        return shares[0][0]
    return "Mixed"


man = pd.read_csv(MAN, dtype={"loc": str}).set_index("loc")
man["sw"] = (man.fwd_sw + man.ww_sw) / man.total
man["bl"] = (man.fwd_bl + man.ww_bl) / man.total
man["rd"] = (man.fwd_rd + man.ww_rd) / man.total
man["type"] = [rule_type(r.sw, r.bl, r.rd) for r in man.itertuples()]

# --- 1. K-means agreement and silhouettes -------------------------------
X = man[["sw", "bl", "rd"]].values
km3 = KMeans(n_clusters=3, n_init=50, random_state=0).fit(X)
ari = adjusted_rand_score(man["type"], km3.labels_)
print(f"K-means k=3 vs threshold rules: ARI = {ari:.2f}")
for k in (2, 3, 4, 5):
    km = KMeans(n_clusters=k, n_init=50, random_state=0).fit(X)
    print(f"  silhouette k={k}: {silhouette_score(X, km.labels_):.3f}")

# --- 2. threshold sensitivity -------------------------------------------
base = man["type"]
for dom in (0.55, 0.60, 0.65):
    for gap in (0.15, 0.20, 0.25):
        t = [rule_type_custom(r.sw, r.bl, r.rd, dom, gap)
             for r in man.itertuples()]
        changed = int((pd.Series(t, index=man.index) != base).sum())
        print(f"  dom={dom:.2f} gap={gap:.2f}: {changed} assignments change")

# --- 3. pipeline facility-share validation ------------------------------
pipe = pd.read_csv(PIPE)
pipe["loc"] = pipe.location.str.replace("loc_", "", regex=False)
pipe = pipe.set_index("loc")
pipe["sw"] = pipe.dom_sidewalk / pipe.riders
pipe["bl"] = pipe.dom_bike_lane / pipe.riders
pipe["rd"] = pipe.dom_roadway / pipe.riders
pipe["type"] = [rule_type(r.sw, r.bl, r.rd) for r in pipe.itertuples()]
both = man.join(pipe[["sw", "bl", "rd", "type"]], rsuffix="_p")
agree = (both["type"] == both["type_p"]).sum()
dis = both[both["type"] != both["type_p"]]
print(f"type agreement (rules on pipeline shares): {agree}/19; "
      f"disagreements: {list(dis.index)}")
big = both[~both.index.isin(SMALL_N)]
for label, sub in (("all 19", both), ("n>=10 (16)", big)):
    mv = np.concatenate([sub.sw, sub.bl, sub.rd])
    pv = np.concatenate([sub.sw_p, sub.bl_p, sub.rd_p])
    r = pearsonr(mv, pv).statistic
    mae = np.abs(mv - pv).mean()
    print(f"  shares {label}: r = {r:.3f}, MAE = {mae:.3f}")

# --- 4. pairing sensitivity ---------------------------------------------
sub = man[~man.index.isin(SMALL_N)]
rho_all = spearmanr(sub.sw, sub.ww_rate)
sub14 = sub[~sub.index.isin({"04-2", "19-2"})]
rho_14 = spearmanr(sub14.sw, sub14.ww_rate)
print(f"Spearman sw vs ww: n={len(sub)} rho={rho_all.statistic:.3f} "
      f"(p={rho_all.pvalue:.1e}); drop second views: n={len(sub14)} "
      f"rho={rho_14.statistic:.3f} (p={rho_14.pvalue:.1e})")
