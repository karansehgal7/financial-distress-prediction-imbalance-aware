# Comparative Evaluation of Machine Learning Approaches for Minority-Class Financial Distress Prediction Under Class Imbalance Constraints

## Overview

This repository contains the research artefacts, reproducibility workflows, experimental pipelines, visualisation infrastructure, and evaluation frameworks associated with the study:

> **Comparative Evaluation of Machine Learning Approaches for Minority-Class Financial Distress Prediction Under Class Imbalance Constraints**

The project investigates imbalance-aware machine learning methodologies for enterprise financial distress prediction under severe minority-class sparsity conditions using ensemble learning architectures, SMOTE-based oversampling, explainability-oriented evaluation, and reproducibility-focused experimentation workflows.

The repository is structured as an applied computational systems engineering framework designed to support:

- imbalance-aware optimisation,
- minority-event sensitivity analysis,
- explainable machine learning evaluation,
- governance-oriented experimentation,
- reproducibility-aware enterprise AI workflows,
- and trustworthy financial decision-support evaluation systems.

---

# Research Motivation

Enterprise financial distress prediction represents a highly imbalanced classification problem in which bankruptcy events typically constitute only a small fraction of total firm observations.

Under severe imbalance conditions, conventional machine learning models may achieve deceptively high aggregate classification accuracy while exhibiting poor minority-class sensitivity and elevated false-negative rates.

Given the asymmetric operational risk associated with failure to detect financially distressed firms, imbalance-aware optimisation becomes a critical engineering requirement for enterprise risk modelling systems.

This repository investigates reproducibility-oriented machine learning workflows designed to improve minority-class detection performance through:

- structured preprocessing pipelines,
- correlation-aware feature filtering,
- SMOTE-based oversampling,
- comparative ensemble learning evaluation,
- SHAP explainability analysis,
- threshold-sensitive evaluation,
- and governance-oriented interpretability frameworks.

---

# Mathematical Formulation and Computational Framework

## Binary Classification Objective

The enterprise financial distress prediction task is formulated as a supervised binary classification problem:

\[
f_\theta : \mathbb{R}^d \rightarrow \{0,1\}
\]

where:

- \(d\) denotes the dimensionality of financial feature space,
- \(f_\theta\) represents a parameterised classifier,
- \(y=1\) denotes financially distressed firms,
- \(y=0\) denotes non-distressed firms.

Given feature vectors:

\[
X = \{x_1, x_2, ..., x_n\}
\]

with associated labels:

\[
Y = \{y_1, y_2, ..., y_n\}
\]

the optimisation objective is to minimise classification error under severe imbalance constraints while preserving minority-class sensitivity.

---

## Class Imbalance Formalisation

The dataset exhibits substantial minority-class sparsity:

\[
|D_{minority}| \ll |D_{majority}|
\]

with imbalance ratio:

\[
IR =
\frac{|D_{majority}|}
{|D_{minority}|}
\]

where:

- \(D_{minority}\) denotes bankrupt observations,
- \(D_{majority}\) denotes non-bankrupt observations.

Under such conditions, naïve optimisation may bias classifiers toward majority-class dominance, leading to elevated false-negative behaviour.

---

## Risk-Sensitive Classification Objective

The operational objective extends beyond aggregate accuracy optimisation.

Instead, the workflow prioritises minority-event sensitivity using asymmetric risk-aware evaluation:

\[
\mathcal{L}_{risk}
=
\alpha \cdot FN
+
\beta \cdot FP
\]

where:

- \(FN\) represents false negatives,
- \(FP\) represents false positives,
- \(\alpha > \beta\) reflects increased operational cost associated with bankruptcy-event omission.

This formulation reflects enterprise financial risk environments in which failure to detect distressed firms may contribute toward delayed intervention and increased downstream exposure.

---

# Dataset Description

The experimental workflow utilises structured financial indicators derived from publicly available enterprise bankruptcy prediction datasets associated with Taiwan Economic Journal financial records and related research repositories.

## Dataset Characteristics

| Property | Value |
|---|---|
| Total observations | ~6,819 |
| Financial attributes | 96 |
| Prediction task | Binary classification |
| Bankruptcy prevalence | ~1.8% |
| Imbalance condition | Severe minority sparsity |

The dataset incorporates variables associated with:

- leverage,
- liquidity,
- retained earnings,
- profitability,
- operational solvency,
- and enterprise financial stability.

---

# Data Preprocessing and Feature Engineering

Preprocessing workflows were designed to improve reproducibility, reduce optimisation instability, and preserve evaluation integrity under imbalance-aware experimentation conditions.

The preprocessing pipeline incorporates:

- missing-value handling,
- median imputation,
- feature normalisation,
- StandardScaler transformation,
- correlation-aware feature filtering,
- and controlled train-test partitioning.

Feature redundancy reduction was performed using correlation-threshold analysis in order to minimise multicollinearity and duplicated informational influence during optimisation.

---

# SMOTE-Based Imbalance Mitigation

To improve minority-class representation during optimisation, the Synthetic Minority Oversampling Technique (SMOTE) was incorporated into the training workflow.

Synthetic observations are generated using nearest-neighbour interpolation:

\[
x_{new}
=
x_i
+
\lambda(x_{nn}-x_i)
\]

where:

- \(x_i\) represents a minority-class sample,
- \(x_{nn}\) denotes a nearest-neighbour minority observation,
- \(\lambda \sim U(0,1)\).

This procedure improves minority-class representation while preserving local feature-space structure.

Oversampling procedures were applied exclusively to training partitions in order to preserve evaluation integrity and minimise information leakage into holdout datasets.

---

# Machine Learning Models Evaluated

The comparative experimentation framework evaluates:

- Logistic Regression
- Random Forest
- AdaBoost
- XGBoost
- CatBoost
- LightGBM

Exploratory temporal modelling approaches involving:

- LSTM,
- ARIMA,
- and SARIMA

are additionally discussed within the broader research framework.

---

## Logistic Regression Decision Boundary

Baseline statistical classification is represented using logistic transformation:

\[
P(y=1|x)
=
\frac{1}
{1 + e^{-(w^Tx+b)}}
\]

where:

- \(w\) denotes model coefficients,
- \(b\) represents intercept bias,
- \(x\) denotes feature vectors.

Although computationally interpretable, linear decision boundaries may demonstrate reduced flexibility under heterogeneous financial environments.

---

## Random Forest Approximation

Random Forest classification is represented through aggregated ensemble voting:

\[
\hat{y}
=
\operatorname{mode}
\left(
T_1(x),
T_2(x),
...,
T_k(x)
\right)
\]

where:

- \(T_k(x)\) denotes decision-tree estimators,
- \(k\) represents total ensemble trees.

Bootstrap aggregation improves variance reduction and generalisation stability.

---

## Ensemble Learning Approximation

Gradient-boosting architectures iteratively optimise residual prediction behaviour through additive ensemble construction:

\[
F_M(x)
=
\sum_{m=1}^{M}
\gamma_m h_m(x)
\]

where:

- \(h_m(x)\) denotes weak learners,
- \(\gamma_m\) represents ensemble weighting coefficients,
- \(M\) denotes total boosting iterations.

Sequential optimisation improves nonlinear feature interaction modelling under imbalance-aware environments.

---

## XGBoost Objective Function

The XGBoost optimisation framework minimises:

\[
\mathcal{L}^{(t)}
=
\sum_{i=1}^{n}
l(y_i,\hat{y}_i^{(t-1)} + f_t(x_i))
+
\Omega(f_t)
\]

where:

- \(l(\cdot)\) denotes differentiable loss,
- \(f_t\) represents boosting tree functions,
- \(\Omega(f_t)\) denotes regularisation penalty.

Regularisation contributes toward improved generalisation under imbalance-aware optimisation conditions.

---

# Evaluation Framework

The evaluation pipeline prioritises minority-class sensitivity metrics due to the operational significance of false negatives within enterprise financial risk systems.

Conventional aggregate accuracy metrics were not treated as primary evaluation criteria due to their limited interpretability under highly imbalanced classification environments.

---

## Precision

\[
\text{Precision}
=
\frac{TP}{TP+FP}
\]

---

## Recall

\[
\text{Recall}
=
\frac{TP}{TP+FN}
\]

---

## F1-Score

\[
F1
=
2
\cdot
\frac{
\text{Precision}
\cdot
\text{Recall}
}
{
\text{Precision}
+
\text{Recall}
}
\]

where:

- \(TP\) = true positives,
- \(FP\) = false positives,
- \(FN\) = false negatives.

---

## ROC-AUC Interpretation

ROC-AUC evaluates discriminatory capability across varying threshold conditions:

\[
AUC
=
\int_0^1 TPR(FPR^{-1}(x))dx
\]

where:

- \(TPR\) denotes true-positive rate,
- \(FPR\) denotes false-positive rate.

---

## Threshold-Sensitive Decision Function

Prediction outputs are threshold-dependent:

\[
\hat{y}
=
\begin{cases}
1 & \text{if } P(y=1|x) \ge \tau \\
0 & \text{otherwise}
\end{cases}
\]

where:

- \(\tau\) denotes classification threshold.

Threshold calibration becomes particularly important under severe imbalance conditions due to precision-recall trade-offs.

---

# Confusion Matrix and Minority-Class Sensitivity

Confusion-matrix-oriented evaluation was incorporated to improve operational interpretation of predictive behaviour under severe imbalance conditions.

Particular emphasis was placed on:

- false-negative analysis,
- minority-event omission rates,
- threshold sensitivity,
- and precision-recall trade-offs.

Within enterprise financial distress environments, elevated false-negative rates may contribute toward increased financial exposure and delayed risk intervention.

Consequently, imbalance-aware evaluation prioritises minority-event detection sensitivity over aggregate classification accuracy alone.

---

# Explainability and Governance-Oriented Evaluation

To improve interpretability and auditability, SHAP-based explainability analysis was applied to the best-performing ensemble learning models.

SHAP estimates feature-level contribution behaviour using cooperative game-theoretic principles:

\[
\phi_i
=
\sum_{S \subseteq F \setminus \{i\}}
\frac{|S|!(|F|-|S|-1)!}{|F|!}
\left[
f(S \cup \{i\}) - f(S)
\right]
\]

where:

- \(F\) denotes the feature set,
- \(S\) represents feature subsets,
- \(\phi_i\) measures marginal feature contribution.

The explainability workflow supports:

- transparent model interpretation,
- feature-level auditability,
- governance-oriented AI evaluation,
- trustworthy enterprise decision-support analysis,
- and reproducibility-oriented experimentation.

---

# Reproducibility Constraints

Controlled random-state initialisation was applied throughout preprocessing, oversampling, and model training stages:

\[
\theta_{run}^{(1)}
\approx
\theta_{run}^{(2)}
\]

to minimise stochastic variance across repeated experimentation workflows.

Stratified 5-fold cross-validation procedures were additionally incorporated:

\[
D
=
\bigcup_{i=1}^{5}
D_i
\]

while preserving minority-class distribution consistency across validation partitions.

---

# Proposition 1 — Imbalance-Aware Sensitivity Observation

Under severe minority-class sparsity conditions:

\[
P(y=1) \ll P(y=0)
\]

ensemble learning architectures combined with imbalance-aware oversampling procedures demonstrate improved minority-event sensitivity relative to conventional linear classification baselines.

This observation is empirically supported through comparative recall and ROC-AUC evaluation across gradient-boosting model families.

---

# Proposition 2 — Explainability-Oriented Governance Observation

Feature-attribution mechanisms capable of decomposing prediction behaviour into local and global contribution structures improve interpretability and auditability within enterprise-oriented machine learning evaluation workflows.

Consequently, explainability-aware evaluation contributes toward broader governance-oriented AI reliability considerations under operational financial risk environments.

---

# Repository Structure

```text
financial-distress-prediction-imbalance-aware/

├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore

├── figures/
├── notebooks/
├── src/
├── docs/
├── reproducibility/
├── results/
└── paper/
```

---

# Figures and Visualisations

Included visual artefacts:

- ROC-AUC comparative analysis
- SHAP feature-attribution plots
- SMOTE class-distribution analysis
- confusion matrix evaluation
- CRISP-DM-oriented workflow architecture
- imbalance-aware optimisation pipeline diagrams

---

# Reproducibility Framework

The repository is structured using reproducibility-oriented machine learning engineering practices including:

- modular preprocessing pipelines,
- deterministic train-test partitioning,
- fixed random-state initialisation,
- explainability-aware evaluation,
- structured experimentation workflows,
- and documented configuration procedures.

Cross-validation workflows utilise stratified sampling procedures to preserve minority-class distributions across validation partitions.

---

# Engineering and Research Orientation

This repository reflects an applied computational systems engineering approach toward:

- imbalance-aware machine learning,
- enterprise financial risk modelling,
- explainable AI,
- governance-oriented experimentation,
- reproducibility engineering,
- and trustworthy decision-support infrastructure.

The broader research direction additionally explores:

- deterministic orchestration systems,
- governance-aware enterprise AI workflows,
- interoperability engineering,
- explainability-oriented intelligent systems evaluation,
- and reproducibility-aware computational infrastructure.

---

# arXiv Status

Current arXiv submission is under moderation review.

Planned categories:

- `cs.LG` — Machine Learning
- `stat.ML` — Statistical Machine Learning

---

# Author

## Karan Sehgal

Kent Business School  
University of Kent  
Canterbury, United Kingdom

---

# License

This repository is released under the MIT License for research, educational, and reproducibility-oriented experimentation purposes.
