import sys, nbformat
from nbclient import NotebookClient

# ---------- shared preamble inlined into every notebook (one-cell requirement) ----------
PRE = r'''
# ============================================================
# Shared setup (anonymized; paths are never printed)
# ============================================================
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr

# Grayscale, publication style; no colour, no titles/captions in saved images.
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"image.cmap": "gray", "axes.grid": True, "grid.alpha": 0.25,
                     "font.size": 11, "savefig.dpi": 600})

DATA = Path("../data"); FIG = Path("../results/figures"); TAB = Path("../results/tables")
RNG = np.random.default_rng(42)

# Dataset registry (public sources) and decision domain.
DATASETS = {"telco": "telco_churn.csv", "adult": "adult.parquet",
            "credit_risk": "credit_risk.csv", "gmsc": "gmsc.parquet",
            "taiwan": "taiwan.parquet"}
DOMAIN = {"telco": "non-credit", "adult": "non-credit",
          "credit_risk": "consumer credit", "gmsc": "consumer credit",
          "taiwan": "consumer credit"}
# Grayscale shade + marker per domain (colour-free, print-safe).
STYLE = {"consumer credit": ("0.15", "o"), "corporate credit": ("0.0", "*"),
         "non-credit": ("0.45", "s")}
# Paper-1 anchor (Polish corporate bankruptcy; values reported in the first paper).
POLISH = {"gini": 2*0.778-1, "pbar": 0.0482, "naive_gap": 2.72, "indep_gap": 0.33}

def load(name):
    f = DATA / DATASETS[name]
    return pd.read_parquet(f) if str(f).endswith(".parquet") else pd.read_csv(f)

def prep(name, df):
    # Return numeric design matrix, binary outcome, and two pd-independent drivers (b1,b2).
    df = df.copy()
    if name == "telco":
        y = (df["Churn"].astype(str).str.strip() == "Yes").astype(int).values
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        X = df.drop(columns=["customerID", "Churn"]); b2col = "TotalCharges"
    elif name == "adult":
        y = df["HIGH_INCOME"].astype(int).values
        drop = ["income", "HIGH_INCOME", "AGE_YEARS", "SEX_STD", "EDU_STD", "VIP", "fnlwgt"]
        X = df.drop(columns=[c for c in drop if c in df]); b2col = "hours-per-week"
    elif name == "credit_risk":
        y = df["loan_status"].astype(int).values
        X = df.drop(columns=["loan_status"]); b2col = "person_income"
    elif name == "gmsc":
        y = df["DEFAULT"].astype(int).values
        X = df.drop(columns=[c for c in ["DEFAULT", "VIP"] if c in df]); b2col = "INCOME"
    elif name == "taiwan":
        y = df["DEFAULT"].astype(int).values
        X = df.drop(columns=[c for c in ["DEFAULT", "VIP", "AGE_YEARS", "SEX_STD"] if c in df]); b2col = "LIMIT_BAL"
    b2 = pd.to_numeric(df[b2col], errors="coerce").values if b2col in df else None
    Xn = pd.get_dummies(X, dummy_na=False).apply(pd.to_numeric, errors="coerce")
    return Xn, y, b2

def system_pd(X, y):
    # L2 logistic system model; out-of-fold PD via 5-fold stratified CV; in-fold median imputation.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    pdh = np.zeros(len(y)); skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for tr, te in skf.split(X, y):
        Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
        med = Xtr.median(numeric_only=True); Xtr = Xtr.fillna(med); Xte = Xte.fillna(med)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
        clf.fit(sc.transform(Xtr), y[tr]); pdh[te] = clf.predict_proba(sc.transform(Xte))[:, 1]
    return pdh, roc_auc_score(y, pdh)

def pick_b1(X, pdh):
    # Near-orthogonal driver: numeric feature with the lowest |Spearman| vs PD (Attr43 analogue).
    best, br = None, 2.0
    for c in X.columns:
        v = X[c].values.astype(float); m = ~np.isnan(v)
        if m.sum() < 100 or np.nanstd(v) == 0: continue
        r = abs(spearmanr(v[m], pdh[m]).statistic)
        if not np.isnan(r) and r < br: br, best = r, v
    return best

QGRID = [0, 0.10, 0.20, 0.35, 0.55, 0.75, 0.90, 1.0]
def assign_grades(pdh):
    qs = np.quantile(pdh, QGRID); qs[0] -= 1e-9; qs[-1] += 1e-9
    return np.digitize(pdh, qs[1:-1])
SCHED = np.array([0.10, 0.15, 0.20, 0.35, 0.50, 0.70, 0.80])  # ord 0 best .. ord 6 worst
OVR = 0.35

def rank01(v):
    return pd.Series(v).rank(pct=True).fillna(0.5).values

def expected_gap(pi, y):
    # Exact expected upgrade-minus-none default-rate gap (percentage points).
    return (np.sum(pi * y) / np.sum(pi) - y.mean()) * 100.0

def sim_once(pi_firm, grade, G, rng):
    # One generative override draw; returns signed grade_diff (negative = upgrade) and masks.
    N = len(grade)
    overridden = rng.random(N) < OVR
    up = overridden & (rng.random(N) < pi_firm)
    down = overridden & ~up
    notch = np.where(rng.random(N) < 0.70, 1, 2)
    gd = np.zeros(N, int); gd[up] = -notch[up]; gd[down] = notch[down]
    final = np.clip(grade + gd, 0, G - 1)
    return final - grade, overridden, up

def save_fig(fig, name):
    # Save grayscale figure as PNG and PDF at 600 dpi; no caption; no path printed.
    fig.savefig(FIG / (name + ".png"), dpi=600, bbox_inches="tight")
    fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
'''

BODY = {}

BODY["00_data_prep"] = r'''
# 00_data_prep.ipynb -- load public datasets; summarize size, dimensionality, class balance
rows = []
for name in DATASETS:
    df = load(name); X, y, _ = prep(name, df)
    rows.append({"dataset": name, "domain": DOMAIN[name], "N": len(y),
                 "features": X.shape[1], "base_rate_pct": round(100*y.mean(), 2)})
summary = pd.DataFrame(rows)
summary.to_csv(TAB / "dataset_summary.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4))
order = summary.sort_values("base_rate_pct")
sns.barplot(data=order, x="dataset", y="base_rate_pct", color="0.4", ax=ax)
ax.set_xlabel(""); ax.set_ylabel("Positive-class rate (%)")
save_fig(fig, "fig00_class_balance")
summary
'''

BODY["01_system_model_grades"] = r'''
# 01_system_model_grades.ipynb -- L2 logistic system model, OOF PD, Gini, 7-grade quantiles
rows = []
for name in DATASETS:
    df = load(name); X, y, b2 = prep(name, df)
    pdh, auc = system_pd(X, y); gini = 2*auc - 1
    g = assign_grades(pdh); b1 = pick_b1(X, pdh)
    sc = pd.DataFrame({"y": y, "pd": pdh, "grade": g,
                       "b1": b1 if b1 is not None else np.nan,
                       "b2": b2 if b2 is not None else np.nan})
    sc.to_parquet(TAB / ("scores_" + name + ".parquet"))
    rows.append({"dataset": name, "domain": DOMAIN[name], "N": len(y),
                 "AUC": round(auc, 4), "Gini": round(gini, 4),
                 "base_rate_pct": round(100*y.mean(), 2)})
sysm = pd.DataFrame(rows); sysm.to_csv(TAB / "system_model.csv", index=False)
fig, axes = plt.subplots(1, len(DATASETS), figsize=(16, 3.0))
for ax, name in zip(axes, DATASETS):
    sc = pd.read_parquet(TAB / ("scores_" + name + ".parquet"))
    dr = sc.groupby("grade")["y"].mean() * 100
    sns.barplot(x=dr.index.astype(int), y=dr.values, color="0.4", ax=ax)
    ax.set_xlabel(name); ax.set_ylabel("Default rate (%)" if name == list(DATASETS)[0] else "")
save_fig(fig, "fig01_grade_default")
sysm
'''

BODY["02_dgm_replication"] = r'''
# 02_dgm_replication.ipynb -- three DGMs (naive/b1/b2), single-draw gaps, 500-rep gap + OR
from sklearn.linear_model import LogisticRegression
single, rep, orr = [], [], []
for name in DATASETS:
    sc = pd.read_parquet(TAB / ("scores_" + name + ".parquet"))
    y = sc["y"].values; pdh = sc["pd"].values; grade = sc["grade"].values.astype(int)
    G = 7; pbar = y.mean()
    pis = {"naive": SCHED[grade],
           "b1": 0.10 + 0.70*rank01(sc["b1"].values),
           "b2": 0.10 + 0.70*rank01(sc["b2"].values)}
    for dgm, pif in pis.items():
        # single draw
        rng = np.random.default_rng(7)
        ov = rng.random(len(y)) < OVR; up = ov & (rng.random(len(y)) < pif); none = ~ov
        single.append({"dataset": name, "dgm": dgm, "n_up": int(up.sum()),
                       "DR_none_pct": round(100*y[none].mean(), 2),
                       "DR_up_pct": round(100*y[up].mean(), 2),
                       "gap_pp": round(100*(y[up].mean()-y[none].mean()), 3)})
        # 500 replications of the gap
        gaps = np.empty(500)
        for i in range(500):
            ov = RNG.random(len(y)) < OVR; up = ov & (RNG.random(len(y)) < pif); none = ~ov
            gaps[i] = 100*(y[up].mean() - y[none].mean())
        rep.append({"dataset": name, "dgm": dgm, "gap_mean": round(gaps.mean(), 3),
                    "gap_lo": round(np.percentile(gaps, 2.5), 3),
                    "gap_hi": round(np.percentile(gaps, 97.5), 3),
                    "excludes_0": not (np.percentile(gaps,2.5) <= 0 <= np.percentile(gaps,97.5))})
        # 200 replications of grade_diff OR controlling for PD
        ors = np.empty(500)
        for i in range(500):
            gd, _, _ = sim_once(pif, grade, G, RNG)
            clf = LogisticRegression(solver="liblinear", max_iter=200).fit(
                np.column_stack([gd, pdh]), y)
            ors[i] = np.exp(clf.coef_[0, 0])
        orr.append({"dataset": name, "dgm": dgm, "OR_mean": round(ors.mean(), 3),
                    "OR_lo": round(np.percentile(ors, 2.5), 3),
                    "OR_hi": round(np.percentile(ors, 97.5), 3),
                    "excludes_1": not (np.percentile(ors,2.5) <= 1 <= np.percentile(ors,97.5))})
single = pd.DataFrame(single); rep = pd.DataFrame(rep); orr = pd.DataFrame(orr)
single.to_csv(TAB / "dgm_single_draw.csv", index=False)
rep.to_csv(TAB / "replication_gap.csv", index=False)
orr.to_csv(TAB / "replication_or.csv", index=False)
# grayscale: 500-rep gap by dataset x dgm (left) and OR by dgm (right)
greys = {"naive": "0.2", "b1": "0.55", "b2": "0.8"}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
x = np.arange(len(DATASETS)); w = 0.26
for k, dgm in enumerate(["naive", "b1", "b2"]):
    sub = rep[rep.dgm == dgm].set_index("dataset").loc[list(DATASETS)]
    ax.bar(x + (k-1)*w, sub["gap_mean"], w, color=greys[dgm], edgecolor="k", linewidth=0.5, label=dgm)
    ax.errorbar(x + (k-1)*w, sub["gap_mean"],
                yerr=[sub["gap_mean"]-sub["gap_lo"], sub["gap_hi"]-sub["gap_mean"]],
                fmt="none", ecolor="k", capsize=2, linewidth=0.7)
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(list(DATASETS), rotation=20)
ax.set_ylabel("Upgrade - none gap (pp)"); ax.legend(title="DGM", fontsize=8)
ax = axes[1]
for k, dgm in enumerate(["naive", "b1", "b2"]):
    sub = orr[orr.dgm == dgm].set_index("dataset").loc[list(DATASETS)]
    ax.errorbar(sub["OR_mean"], x + (k-1)*w,
                xerr=[sub["OR_mean"]-sub["OR_lo"], sub["OR_hi"]-sub["OR_mean"]],
                fmt="o", color=greys[dgm], ecolor="0.6", capsize=2, ms=5, label=dgm)
ax.axvline(1.0, color="k", lw=0.8, ls="--"); ax.set_yticks(x); ax.set_yticklabels(list(DATASETS))
ax.set_xlabel("grade_diff odds ratio (control PD)"); ax.legend(title="DGM", fontsize=8)
save_fig(fig, "fig02_dgm_effects")
rep
'''

BODY["03_gap_vs_gini_law"] = r'''
# 03_gap_vs_gini_law.ipynb -- the design-induced selection law: gap vs Gini and Gini*Var(Y)
sysm = pd.read_csv(TAB / "system_model.csv")
rep = pd.read_csv(TAB / "replication_gap.csv")
naive = rep[rep.dgm == "naive"].set_index("dataset")
indep = rep[rep.dgm == "b1"].set_index("dataset")
rows = []
for name in DATASETS:
    s = sysm[sysm.dataset == name].iloc[0]; pbar = s["base_rate_pct"]/100
    rows.append({"dataset": name, "domain": DOMAIN[name], "Gini": s["Gini"], "pbar": pbar,
                 "naive": naive.loc[name, "gap_mean"], "nlo": naive.loc[name, "gap_lo"],
                 "nhi": naive.loc[name, "gap_hi"], "indep": indep.loc[name, "gap_mean"],
                 "GiniVar": s["Gini"]*pbar*(1-pbar)*100})
R = pd.DataFrame(rows)
R = pd.concat([R, pd.DataFrame([{"dataset": "polish", "domain": "corporate credit",
    "Gini": POLISH["gini"], "pbar": POLISH["pbar"], "naive": POLISH["naive_gap"],
    "nlo": 1.94, "nhi": 3.36, "indep": POLISH["indep_gap"],
    "GiniVar": POLISH["gini"]*POLISH["pbar"]*(1-POLISH["pbar"])*100}])], ignore_index=True)
x = R["GiniVar"].values; yv = R["naive"].values
slope = np.sum(x*yv)/np.sum(x*x); r2 = 1 - np.sum((yv-slope*x)**2)/np.sum((yv-yv.mean())**2)
R.to_csv(TAB / "law_points.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
for _, r in R.iterrows():
    c, m = STYLE[r["domain"]]
    ax.errorbar(r["Gini"], r["naive"], yerr=[[r["naive"]-r["nlo"]], [r["nhi"]-r["naive"]]],
                fmt=m, color=c, ms=10 if m == "*" else 8, capsize=3)
    ax.annotate(r["dataset"], (r["Gini"], r["naive"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.scatter(R["Gini"], R["indep"], marker="x", color="0.6", s=40)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("System Gini (= 2*AUC - 1)"); ax.set_ylabel("Upgrade - none gap (pp)")
ax = axes[1]
xs = np.linspace(0, x.max()*1.08, 50)
ax.plot(xs, slope*xs, color="0.1", lw=1.5, label="gap = %.2f * (Gini*Var),  R2=%.3f" % (slope, r2))
for _, r in R.iterrows():
    c, m = STYLE[r["domain"]]
    ax.errorbar(r["GiniVar"], r["naive"], yerr=[[r["naive"]-r["nlo"]], [r["nhi"]-r["naive"]]],
                fmt=m, color=c, ms=14 if m == "*" else 9, capsize=3)
    ax.annotate(r["dataset"], (r["GiniVar"], r["naive"]), textcoords="offset points", xytext=(6, -10), fontsize=8)
ax.scatter(R["GiniVar"], R["indep"], marker="x", color="0.6", s=40, label="pd-independent (b1)")
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("Gini * base rate * (1 - base rate)  (pp)"); ax.set_ylabel("Upgrade - none gap (pp)")
ax.legend(fontsize=8, loc="upper left")
save_fig(fig, "fig03_gap_vs_gini_law")
R
'''

BODY["04_theory_verification"] = r'''
# 04_theory_verification.ipynb -- backbone identity, sigma_DR, closed-form C, multi-schedule, general-G
def wmean(x, w): return np.sum(w*x)/np.sum(w)
def wcov(x, y, w):
    mx, my = wmean(x, w), wmean(y, w); return np.sum(w*(x-mx)*(y-my))/np.sum(w)
def wstd(x, w): return np.sqrt(max(wcov(x, x, w), 0))
def grades_G(pdh, G):
    qs = np.quantile(pdh, np.linspace(0, 1, G+1)); qs[0] -= 1e-9; qs[-1] += 1e-9
    return np.digitize(pdh, qs[1:-1])
ANCH = (np.arange(7)+0.5)/7.0
def sched_on(G, s7=SCHED):
    return np.interp((np.arange(G)+0.5)/G, ANCH, s7)
SCHEDS = {"paper": SCHED, "steep": np.array([0.05,0.08,0.12,0.25,0.55,0.80,0.95]),
          "flat": np.array([0.20,0.25,0.30,0.40,0.50,0.60,0.70]),
          "linear": np.linspace(0.10,0.80,7)}
Gs = [7, 14, 28, 70, 200]
ident, sched_tab, G_tab = [], [], []
for name in DATASETS:
    sc = pd.read_parquet(TAB / ("scores_" + name + ".parquet"))
    y = sc["y"].values; pdh = sc["pd"].values; pbar = y.mean()
    gini = 2*__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, pdh) - 1
    U = pd.Series(pdh).rank(pct=True).values
    ident.append({"dataset": name, "cov_emp": round(np.cov(y, U)[0,1], 5),
                  "cov_pred": round(pbar*(1-pbar)*gini/2, 5)})
    for sname, sch in SCHEDS.items():
        g = assign_grades(pdh); w = np.array([(g == k).mean() for k in range(7)])
        DR = np.array([y[g == k].mean() if (g == k).any() else np.nan for k in range(7)])
        gap = expected_gap(sch[g], y); C_emp = (gap/100)/(gini*pbar*(1-pbar))
        rpd = wcov(sch, DR, w)/(wstd(sch, w)*wstd(DR, w))
        Umid = (np.arange(7)+0.5)/7; rmu = wcov(DR, Umid, w)/(wstd(DR, w)*wstd(Umid, w))
        C_pred = np.sqrt(3)*rpd*wstd(sch, w)/(rmu*wmean(sch, w))
        sched_tab.append({"schedule": sname, "dataset": name,
                          "C_emp": round(C_emp, 3), "C_pred": round(C_pred, 3)})
    row = {"dataset": name, "Gini": round(gini, 4)}
    for G in Gs:
        g = grades_G(pdh, G); gap = expected_gap(sched_on(G)[g], y)
        row["C_G%d" % G] = round((gap/100)/(gini*pbar*(1-pbar)), 3)
    G_tab.append(row)
ident = pd.DataFrame(ident); sched_tab = pd.DataFrame(sched_tab); G_tab = pd.DataFrame(G_tab)
ident.to_csv(TAB / "theory_identity.csv", index=False)
sched_tab.to_csv(TAB / "theory_C_by_schedule.csv", index=False)
G_tab.to_csv(TAB / "theory_C_by_G.csv", index=False)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax = axes[0]
ax.plot([0, ident[["cov_emp","cov_pred"]].max().max()*1.1]*1, [0, ident[["cov_emp","cov_pred"]].max().max()*1.1], "k--", lw=1)
ax.scatter(ident["cov_pred"], ident["cov_emp"], color="0.2", s=55)
ax.set_xlabel("predicted  0.5*p(1-p)*Gini"); ax.set_ylabel("empirical  Cov(Y, U)")
ax = axes[1]
greys = {"paper": "0.0", "steep": "0.3", "flat": "0.55", "linear": "0.75"}
for s in SCHEDS:
    sub = sched_tab[sched_tab.schedule == s]
    ax.scatter(sub["C_pred"], sub["C_emp"], color=greys[s], s=45, label=s)
lim = [0.5, 1.6]; ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("predicted C (closed form)"); ax.set_ylabel("empirical C"); ax.legend(fontsize=8, title="schedule")
ax = axes[2]
for _, r in G_tab.iterrows():
    ax.plot(Gs, [r["C_G%d" % G] for G in Gs], marker="o", color="0.3", lw=1)
    ax.annotate(r["dataset"], (Gs[-1], r["C_G%d" % Gs[-1]]), textcoords="offset points", xytext=(4, 0), fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("number of grades G (log)"); ax.set_ylabel("empirical C")
save_fig(fig, "fig04_theory_verification")
sched_tab.groupby("schedule").agg(C_emp=("C_emp","mean"), C_pred=("C_pred","mean")).round(3)
'''

BODY["05_parameter_sensitivity"] = r'''
# 05_parameter_sensitivity.ipynb -- gap over an override-rate x upgrade-spread grid, per DGM
rates = [0.20, 0.30, 0.40, 0.50]; uppers = [0.50, 0.65, 0.80, 0.95]
recs = []
for name in DATASETS:
    sc = pd.read_parquet(TAB / ("scores_" + name + ".parquet"))
    y = sc["y"].values; grade = sc["grade"].values.astype(int)
    drivers = {"naive": None, "b1": rank01(sc["b1"].values), "b2": rank01(sc["b2"].values)}
    for dgm, r01 in drivers.items():
        for orr in rates:
            for up in uppers:
                if dgm == "naive":
                    sch = np.interp(np.linspace(0.071, 0.929, 7), [0.071, 0.929], [0.10, up])
                    pif = sch[grade]
                else:
                    pif = 0.10 + (up-0.10)*r01
                g = expected_gap(pif, y)  # override-rate cancels in the expected gap; kept for grid labelling
                recs.append({"dataset": name, "dgm": dgm, "rate": orr, "upper": up, "gap_pp": round(g, 3)})
grid = pd.DataFrame(recs); grid.to_csv(TAB / "sensitivity_grid.csv", index=False)
summ = grid.groupby("dgm")["gap_pp"].agg(["min", "max", "mean"]).round(3)
summ.to_csv(TAB / "sensitivity_summary.csv")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, dgm in zip(axes, ["naive", "b1", "b2"]):
    piv = grid[grid.dgm == dgm].groupby(["rate", "upper"])["gap_pp"].mean().unstack()
    sns.heatmap(piv, ax=ax, cmap="Greys", annot=True, fmt=".1f", cbar=False,
                vmin=grid["gap_pp"].min(), vmax=grid["gap_pp"].max(), linewidths=0.5, linecolor="w")
    ax.set_xlabel("upgrade-prob upper (" + dgm + ")"); ax.set_ylabel("override rate" if dgm == "naive" else "")
save_fig(fig, "fig05_sensitivity")
summ
'''

BODY["06_leakage_aware_validation"] = r'''
# 06_leakage_aware_validation.ipynb -- XGBoost nested feature sets, fold balance, SHAP (one dataset)
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import shap
XGB = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=0)
def cv_auc(Xv, y, spw):
    skf = StratifiedKFold(5, shuffle=True, random_state=42); a, p = [], []
    for tr, te in skf.split(Xv, y):
        m = XGBClassifier(scale_pos_weight=spw, **XGB)
        m.fit(Xv[tr], y[tr]); pr = m.predict_proba(Xv[te])[:, 1]
        a.append(roc_auc_score(y[te], pr)); p.append(average_precision_score(y[te], pr))
    return np.mean(a), np.mean(p)
auc_rows, fold_rows = [], []
for name in DATASETS:
    df = load(name); X, y, _ = prep(name, df)
    sc = pd.read_parquet(TAB / ("scores_" + name + ".parquet"))
    pdh = sc["pd"].values; grade = sc["grade"].values.astype(int)
    pif = 0.10 + 0.70*rank01(sc["b1"].values)  # leakage-safe b1 override draw
    gd, ovflag, _ = sim_once(pif, grade, 7, np.random.default_rng(1))
    Xf = X.fillna(X.median(numeric_only=True)).values
    spw = (y == 0).sum()/(y == 1).sum()
    base = Xf
    ovr = np.column_stack([Xf, gd, ovflag.astype(int)])
    full = np.column_stack([Xf, gd, ovflag.astype(int), pdh, grade])
    for lab, Xv in [("financial", base), ("financial+override", ovr), ("full(+pd)", full)]:
        a, p = cv_auc(Xv, y, spw)
        auc_rows.append({"dataset": name, "feature_set": lab, "ROC_AUC": round(a, 4), "PR_AUC": round(p, 4)})
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for k, (tr, te) in enumerate(skf.split(Xf, y)):
        fold_rows.append({"dataset": name, "fold": k,
                          "train_dr": round(y[tr].mean(), 4), "test_dr": round(y[te].mean(), 4)})
auct = pd.DataFrame(auc_rows); foldt = pd.DataFrame(fold_rows)
auct.to_csv(TAB / "featureset_auc.csv", index=False); foldt.to_csv(TAB / "fold_balance.csv", index=False)
# delta vs financial
piv = auct.pivot(index="dataset", columns="feature_set", values="ROC_AUC")
piv["delta_override"] = (piv["financial+override"] - piv["financial"]).round(4)
piv.to_csv(TAB / "featureset_delta.csv")
# SHAP for one representative dataset (credit_risk), full model
rep_name = "credit_risk"
dfr = load(rep_name); Xr, yr, _ = prep(rep_name, dfr)
scr = pd.read_parquet(TAB / ("scores_" + rep_name + ".parquet"))
gdr, ovr_, _ = sim_once(0.10+0.70*rank01(scr["b1"].values), scr["grade"].values.astype(int), 7, np.random.default_rng(1))
Xrf = Xr.fillna(Xr.median(numeric_only=True)).copy()
Xrf["grade_diff"] = gdr; Xrf["override_flag"] = ovr_.astype(int)
Xrf["pd_system"] = scr["pd"].values; Xrf["grade_ordinal"] = scr["grade"].values
mdl = XGBClassifier(scale_pos_weight=(yr == 0).sum()/(yr == 1).sum(), **XGB).fit(Xrf.values, yr)
idx = np.random.default_rng(0).choice(len(Xrf), size=min(2000, len(Xrf)), replace=False)
sv = shap.TreeExplainer(mdl).shap_values(Xrf.iloc[idx].values)
imp = pd.Series(np.abs(sv).mean(0), index=Xrf.columns).sort_values(ascending=True).tail(10)
fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"wspace": 0.5})
ax = axes[0]
x = np.arange(len(DATASETS)); w = 0.26; greys = {"financial": "0.2", "financial+override": "0.5", "full(+pd)": "0.78"}
for k, fs in enumerate(["financial", "financial+override", "full(+pd)"]):
    sub = auct[auct.feature_set == fs].set_index("dataset").loc[list(DATASETS)]
    ax.bar(x + (k-1)*w, sub["ROC_AUC"], w, color=greys[fs], edgecolor="k", linewidth=0.5, label=fs)
ax.set_xticks(x); ax.set_xticklabels(list(DATASETS), rotation=20); ax.set_ylim(0.5, 1.0)
ax.set_ylabel("5-fold ROC-AUC"); ax.legend(fontsize=8)
ax = axes[1]
ax.barh(np.arange(len(imp)), imp.values, color="0.4", edgecolor="k", linewidth=0.5)
ax.set_yticks(np.arange(len(imp))); ax.set_yticklabels([(str(t) if len(str(t))<=18 else str(t)[:17]+chr(8230)) for t in imp.index], fontsize=8)
ax.set_xlabel("mean |SHAP| (credit_risk, full model)")
save_fig(fig, "fig06_leakage_validation")
piv
'''

BODY["07_credit_risk_real_vs_sim_grade"] = r'''
# 07_credit_risk_real_vs_sim_grade.ipynb -- real loan_grade vs simulated system grade (credit_risk)
df = load("credit_risk")
sc = pd.read_parquet(TAB / "scores_credit_risk.parquet")
order = {g: i for i, g in enumerate(["A", "B", "C", "D", "E", "F", "G"])}
real = df["loan_grade"].map(order).values
y = sc["y"].values; simg = sc["grade"].values.astype(int)
dr_real = pd.DataFrame({"g": real, "y": y}).groupby("g")["y"].mean()*100
dr_sim = pd.DataFrame({"g": simg, "y": y}).groupby("g")["y"].mean()*100
# agreement: Spearman between real letter grade and simulated quantile grade
rho = spearmanr(real, simg).statistic
pd.DataFrame({"spearman_real_vs_sim": [round(rho, 4)]}).to_csv(TAB / "real_vs_sim_grade.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
sns.barplot(x=dr_real.index, y=dr_real.values, color="0.3", ax=axes[0])
axes[0].set_xlabel("real loan_grade (A..G)"); axes[0].set_ylabel("Default rate (%)")
sns.barplot(x=dr_sim.index.astype(int), y=dr_sim.values, color="0.6", ax=axes[1])
axes[1].set_xlabel("simulated system grade (0=best..6=worst)"); axes[1].set_ylabel("")
save_fig(fig, "fig07_real_vs_sim_grade")
pd.DataFrame({"real_grade_default_%": dr_real.round(2)})
'''

BODY["08_corporate_split_rating"] = r'''
# 08_corporate_split_rating.ipynb -- appendix: inter-agency rating disagreement (corporate_rating)
df = pd.read_csv(DATA / "corporate_rating.csv")
scale = ["AAA","AA+","AA","AA-","A+","A","A-","BBB+","BBB","BBB-","BB+","BB","BB-",
         "B+","B","B-","CCC+","CCC","CCC-","CC","C","D"]
omap = {g: i for i, g in enumerate(scale)}
df["ord"] = df["Rating"].map(omap)
df = df.dropna(subset=["ord"])
# firms rated by more than one agency (same ticker): notch disagreement = max-min ordinal
g = df.groupby("Symbol")["ord"]
disag = (g.max() - g.min()); disag = disag[g.count() > 1]
by_ag = df.groupby("Rating Agency Name")["ord"].mean().sort_values()
pd.DataFrame({"n_firms_multi_agency": [int((g.count() > 1).sum())],
             "mean_notch_disagreement": [round(disag.mean(), 3) if len(disag) else np.nan],
             "max_notch_disagreement": [int(disag.max()) if len(disag) else 0]}
            ).to_csv(TAB / "split_rating_summary.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
if len(disag):
    sns.histplot(disag.values, bins=range(0, int(disag.max())+2), color="0.4", ax=axes[0])
axes[0].set_xlabel("inter-agency notch disagreement"); axes[0].set_ylabel("firms")
sns.barplot(x=by_ag.values, y=by_ag.index, color="0.5", ax=axes[1])
axes[1].set_xlabel("mean rating ordinal (higher = riskier)"); axes[1].set_ylabel("")
save_fig(fig, "fig08_split_rating")
by_ag.round(2)
'''

def run(name, timeout=1800):
    src = PRE + "\n" + BODY[name]
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(src)]
    nb.metadata = {"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"}}
    NotebookClient(nb, timeout=timeout, kernel_name="python3",
                   resources={"metadata": {"path": "notebooks"}}).execute()
    nbformat.write(nb, "notebooks/" + name + ".ipynb")
    for c in nb.cells:
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                print("ERROR", name, "\n", "\n".join(o.get("traceback", []))[-2000:]); return False
    print("OK", name); return True

if __name__ == "__main__":
    for nm in sys.argv[1:]:
        if not run(nm):
            sys.exit(1)
