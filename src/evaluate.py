"""
evaluation.py

Research-oriented evaluation framework
for imbalance-aware financial distress prediction.

This module supports:

- minority-class sensitivity analysis
- threshold-aware evaluation
- ROC-AUC analysis
- confusion matrix interpretation
- false-negative risk analysis
- reproducibility-oriented experimentation
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)


def evaluate_model(
    y_true,
    y_pred,
    y_prob,
    model_name="Model",
    threshold=0.5,
    save_dir="../figures"
):
    """
    Evaluate predictive performance under
    imbalance-aware financial risk conditions.
    """

    # -----------------------------
    # Core Evaluation Metrics
    # -----------------------------

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    # -----------------------------
    # Confusion Matrix Components
    # -----------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    # -----------------------------
    # Print Evaluation Summary
    # -----------------------------

    print("\n===================================")
    print(f"{model_name} Evaluation Summary")
    print("===================================")

    print("\nClassification Report")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_true, y_pred))

    print("\nMinority-Class Sensitivity Analysis")

    print(f"True Positives  (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives  (TN): {tn}")

    print("\nEvaluation Metrics")

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print(f"\nDecision Threshold: {threshold}")

    # -----------------------------
    # Structured Metrics Table
    # -----------------------------

    results_df = pd.DataFrame({
        "Metric": [
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
            "False Negatives",
            "False Positives"
        ],
        "Score": [
            precision,
            recall,
            f1,
            roc_auc,
            fn,
            fp
        ]
    })

    # -----------------------------
    # Save Metrics CSV
    # -----------------------------

    os.makedirs("../results", exist_ok=True)

    results_df.to_csv(
        f"../results/{model_name.lower()}_evaluation_metrics.csv",
        index=False
    )

    # -----------------------------
    # Save Confusion Matrix Figure
    # -----------------------------

    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred),
        display_labels=["Non-Bankrupt", "Bankrupt"]
    )

    disp.plot(ax=ax)

    plt.title(
        f"{model_name} Confusion Matrix\n"
        "Minority-Class Financial Distress Prediction"
    )

    plt.savefig(
        f"{save_dir}/{model_name.lower()}_confusion_matrix.png",
        bbox_inches="tight"
    )

    plt.close()

    return results_df
