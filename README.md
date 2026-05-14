# Comparative Evaluation of Machine Learning Approaches for Minority-Class Financial Distress Prediction Under Class Imbalance Constraints

## Overview

This repository contains the implementation framework, research artefacts, visualisations, and reproducibility-oriented experimentation workflow associated with the research study:

**“Comparative Evaluation of Machine Learning Approaches for Minority-Class Financial Distress Prediction Under Class Imbalance Constraints”**

The project investigates imbalance-aware machine learning workflows for enterprise financial distress prediction under highly skewed bankruptcy-event distributions using ensemble learning architectures, SMOTE-based oversampling, and SHAP explainability analysis.

The study is positioned as an applied machine learning engineering evaluation focused on reproducibility, interpretability, governance-oriented AI evaluation, and minority-class sensitivity within enterprise financial risk environments.

---

## Research Motivation

Financial distress prediction represents a critical challenge in enterprise risk management and insolvency monitoring systems due to the rarity of bankruptcy events in real-world financial datasets.

Under severe class imbalance conditions, conventional machine learning models may exhibit inflated aggregate accuracy while failing to detect minority-class distress events with acceptable sensitivity.

This repository explores imbalance-aware optimisation workflows designed to improve minority-event detection performance using:

* structured preprocessing pipelines,
* SMOTE-based oversampling,
* comparative ensemble learning evaluation,
* explainability-oriented feature attribution analysis,
* and reproducibility-focused experimentation practices.

---

## Research Objectives

* Evaluate comparative machine learning performance under severe class imbalance conditions.
* Improve minority-class bankruptcy-event sensitivity using imbalance-aware optimisation strategies.
* Investigate ensemble learning architectures for enterprise financial distress prediction.
* Incorporate SHAP-based explainability analysis for interpretability and governance-oriented model evaluation.
* Support reproducible experimentation and transparent machine learning engineering workflows.

---

## Dataset Description

The experimental workflow utilises structured financial and operational indicators associated with enterprise bankruptcy prediction derived from publicly available Taiwan Economic Journal financial records and related Kaggle research repositories.

### Dataset Characteristics

* ~6,819 firm observations
* 96 financial attributes
* Binary classification task
* Bankruptcy-event prevalence: ~1.8%
* Severe minority-class imbalance environment

---

## Machine Learning Models Evaluated

The comparative evaluation framework includes:

* Logistic Regression
* Random Forest
* AdaBoost
* XGBoost
* CatBoost
* LightGBM

Exploratory temporal modelling considerations involving LSTM, ARIMA, and SARIMA architectures are additionally discussed within the broader research framework.

---

## Imbalance Mitigation Strategy

The workflow incorporates the Synthetic Minority Oversampling Technique (SMOTE) to improve minority-class representation during model optimisation.

SMOTE-based oversampling is applied exclusively to training partitions in order to preserve evaluation integrity and minimise information leakage into holdout datasets.

---

## Evaluation Framework

The evaluation strategy prioritises minority-class sensitivity metrics due to the asymmetric operational risk associated with false negatives in enterprise financial distress prediction systems.

### Metrics Evaluated

* Precision
* Recall
* F1-score
* ROC-AUC

Additional analysis includes:

* confusion matrix interpretation,
* threshold sensitivity considerations,
* and precision-recall trade-off analysis.

---

## Explainability and Governance

The workflow integrates SHAP-based explainability analysis to improve:

* interpretability,
* auditability,
* governance-oriented evaluation,
* and transparency of predictive model behaviour.

Feature attribution analysis is used to identify the financial indicators contributing most strongly toward bankruptcy-event prediction.

---

## Repository Structure

```text
financial-distress-prediction-imbalance-aware/

├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore

├── figures/
├── paper/
├── notebooks/
├── src/
├── docs/
├── reproducibility/
└── results/
```

---

## Figures and Visualisations

The repository includes:

* ROC-AUC comparative analysis
* SHAP summary visualisations
* SMOTE class-distribution analysis
* Confusion matrix analysis
* Workflow architecture diagrams

---

## Reproducibility

The repository is being structured using reproducibility-oriented machine learning engineering practices including:

* modular preprocessing workflows,
* controlled train-test partitioning,
* fixed random-state configurations,
* explainability pipelines,
* and documented experimentation procedures.

---

## Research Paper

The accompanying research paper is included within the `paper/` directory.

### arXiv Status

Current arXiv submission is under moderation review.

Planned arXiv category:

* `cs.LG` — Machine Learning
* `stat.ML` — Machine Learning

---

## Engineering and Research Orientation

This repository reflects an applied computational systems engineering approach toward:

* imbalance-aware machine learning,
* explainable AI,
* governance-oriented AI evaluation,
* reproducibility-focused experimentation,
* and enterprise-oriented financial risk modelling.

The broader research direction additionally explores:

* deterministic orchestration systems,
* trustworthy enterprise AI workflows,
* interoperability engineering,
* and governance-aware intelligent decision-support infrastructure.

---

## Author

### Karan Sehgal

Kent Business School
University of Kent
Canterbury, United Kingdom

---

## License

This repository is released for research, educational, and reproducibility-oriented experimentation purposes.
