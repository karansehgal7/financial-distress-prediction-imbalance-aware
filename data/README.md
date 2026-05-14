# Dataset Documentation

## Overview

The experimental workflow utilises structured enterprise financial distress prediction datasets derived from publicly available Taiwan Economic Journal (TEJ) financial records and related bankruptcy prediction research repositories hosted through Kaggle and associated open-access financial research ecosystems.

The dataset supports imbalance-aware machine learning experimentation for enterprise bankruptcy-event prediction under severe minority-class sparsity conditions.

The broader experimental framework investigates:

- enterprise financial distress prediction,
- imbalance-aware optimisation,
- minority-class sensitivity analysis,
- explainability-oriented evaluation,
- and reproducibility-aware machine learning workflows.

---

# Dataset Characteristics

| Property | Value |
|---|---|
| Total observations | ~6,819 |
| Financial attributes | 96 |
| Prediction task | Binary classification |
| Minority-class prevalence | ~1.8% |
| Imbalance condition | Severe minority-class sparsity |
| Domain | Enterprise financial distress prediction |

---

# Binary Classification Formulation

The predictive task is formulated as:

\[
f_\theta : \mathbb{R}^d \rightarrow \{0,1\}
\]

where:

- \(y=1\) denotes financially distressed/bankrupt firms,
- \(y=0\) denotes non-distressed firms,
- \(d\) represents the dimensionality of financial feature space.

The dataset exhibits severe class imbalance:

\[
P(y=1) \ll P(y=0)
\]

which introduces optimisation instability and minority-event sensitivity challenges during conventional model training.

---

# Financial Feature Categories

The dataset contains structured enterprise financial indicators associated with:

- leverage ratios,
- retained earnings,
- liquidity metrics,
- solvency indicators,
- profitability measures,
- operational efficiency,
- debt exposure,
- cash-flow stability,
- and enterprise financial health.

These variables collectively support predictive modelling of bankruptcy-event behaviour under highly heterogeneous financial conditions.

---

# Imbalance Characteristics

The dataset exhibits extreme minority-class sparsity:

\[
IR =
\frac{|D_{majority}|}
{|D_{minority}|}
\]

where:

- \(D_{minority}\) represents bankrupt firms,
- \(D_{majority}\) represents non-bankrupt firms.

Under such imbalance conditions, aggregate classification accuracy may become misleading due to majority-class dominance.

Consequently, the broader experimentation framework prioritises:

- recall,
- F1-score,
- ROC-AUC,
- false-negative analysis,
- and threshold-sensitive evaluation.

---

# Preprocessing and Reproducibility Workflow

The dataset is processed using reproducibility-oriented machine learning engineering procedures including:

- deterministic train-test partitioning,
- stratified sampling,
- feature scaling,
- correlation-aware feature inspection,
- and controlled random-state initialisation.

SMOTE-based oversampling procedures are applied exclusively to training partitions in order to minimise information leakage into holdout evaluation datasets.

---

# Dataset Access Information

Due to licensing, redistribution, and repository compliance considerations, the complete raw dataset is not directly redistributed within this repository.

Researchers may obtain equivalent publicly accessible datasets through:

- Taiwan Economic Journal research references,
- Kaggle financial distress prediction repositories,
- publicly available bankruptcy prediction datasets,
- and associated open-access financial machine learning resources.

Example search references include:

- Taiwan Bankruptcy Prediction Dataset
- Company Bankruptcy Prediction Dataset
- Financial Distress Prediction Data Repository

---

# Repository Integration

The dataset supports the broader experimentation pipeline implemented across:

```text
src/
notebooks/
results/
figures/
reproducibility/
```

The repository includes:

- preprocessing workflows,
- feature-engineering pipelines,
- imbalance-mitigation procedures,
- comparative ensemble-learning experimentation,
- explainability-oriented evaluation,
- and reproducibility-aware computational infrastructure.

---

# Engineering and Research Orientation

The dataset infrastructure reflects an applied computational systems engineering approach toward:

- imbalance-aware machine learning,
- enterprise financial risk modelling,
- governance-oriented AI evaluation,
- explainability-aware experimentation,
- and reproducibility-oriented machine learning engineering.

The broader research direction additionally explores:

- deterministic workflow systems,
- trustworthy enterprise AI infrastructure,
- orchestration-aware evaluation pipelines,
- and governance-oriented intelligent decision-support systems.
