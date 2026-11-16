# Design-Induced Selection in Simulated Credit-Rating Overrides — Cross-Domain Evidence and a Discriminatory-Power Law

> Replication code for a manuscript **under anonymous peer review**.
> Author and affiliation information has been removed for double-blind review.

This repository reproduces every figure and table of the study. It extends an
earlier, single-dataset analysis into a quantitative law: the spurious
"upgrade-risk" gap manufactured by circular override simulations is, to first
order, the product of the system model's discriminatory power and the outcome
variance.

## Headline result

For a simulated two-stage rating system in which upgrade probability is keyed to
the system grade (the *naive* design), the upgrade-minus-none default-rate gap
obeys

```
gap  ≈  C · Gini · p̄ · (1 − p̄)        with  C ≈ 1  for the paper's schedule
```

equivalently **gap ≈ Gini × Var(Y)**. The closed-form constant
`C = √3 · ρ(π,DR) · σ_π / (ρ(m,U) · π̄)` is validated across four upgrade
schedules and five datasets (empirical vs predicted `C`, correlation 1.000).
When the upgrade rule is decoupled from the system-estimated default
probability, the gap collapses to within sampling error of zero in every
dataset and every decision domain.

## Repository layout

```
.
├── data/              # public datasets (not committed — see "Data" below)
├── notebooks/         # one self-contained notebook per analysis stage
├── results/
│   ├── figures/       # grayscale PNG + PDF, 600 dpi, no embedded captions
│   └── tables/        # CSV result tables and intermediate score files
├── requirements.txt
└── README.md
```

## Notebooks (run in order)

| # | Notebook | Produces |
|---|----------|----------|
| 00 | `00_data_prep` | dataset size / class-balance summary |
| 01 | `01_system_model_grades` | L2-logistic out-of-fold PD, Gini, 7-grade quantiles |
| 02 | `02_dgm_replication` | three DGMs; 500-replication gap and grade_diff odds ratio |
| 03 | `03_gap_vs_gini_law` | the Gap-vs-Gini and Gap-vs-(Gini·Var) law figure |
| 04 | `04_theory_verification` | backbone identity, σ_DR, closed-form C, multi-schedule, general-G |
| 05 | `05_parameter_sensitivity` | gap over an override-rate × upgrade-spread grid |
| 06 | `06_leakage_aware_validation` | XGBoost nested feature sets, fold balance, SHAP |
| 07 | `07_credit_risk_real_vs_sim_grade` | real loan grade vs simulated system grade |
| 08 | `08_corporate_split_rating` | appendix: inter-agency rating disagreement |

Each notebook is a single executable cell with English comments. Notebooks 02–08
read the per-row score files written by notebook 01, so run the notebooks in the
order above.

## Data

All datasets are public. Place the raw files in `data/` under the names expected
by the notebooks:

| File in `data/` | Public source |
|-----------------|----------------|
| `telco_churn.csv` | Telco customer churn (IBM sample dataset) |
| `adult.parquet` | Adult / Census Income (UCI ML Repository) |
| `credit_risk.csv` | Credit-risk loan dataset (public Kaggle dataset) |
| `gmsc.parquet` | Give Me Some Credit (public Kaggle competition) |
| `taiwan.parquet` | Default of Credit Card Clients (UCI ML Repository) |
| `corporate_rating.csv` | Corporate credit ratings (public Kaggle dataset) |

The Polish Companies Bankruptcy dataset (UCI ML Repository, id 365) is used as a
corporate-credit anchor point from the earlier study; its summary statistics are
embedded in notebook 03 and the raw file is not required to reproduce the law.

Raw data and generated score files are intentionally excluded from version
control (see `.gitignore`).

## Reproducing

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/00_data_prep.ipynb
# ... repeat for 01 through 08, in order
```

All results are deterministic under fixed seeds. Figures are written to
`results/figures/` as 600-dpi grayscale PNG and PDF; tables to
`results/tables/`.

## Notes on validity

The override step is **simulated**, not observed. All findings are statements
about simulation designs and about models fitted to simulated overrides, not
direct evidence about real evaluator behaviour. The datasets labelled
"non-credit" (income, churn) are included to test whether the design-induced
selection effect generalizes to model-conditioned human-adjustment settings
beyond credit; they are not credit-override analyses.
