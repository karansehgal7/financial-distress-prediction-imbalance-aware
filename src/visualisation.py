"""
visualisation.py

Visualisation utilities for
imbalance-aware financial distress prediction.

Supports:

- confusion matrix analysis
- ROC curve generation
- class imbalance visualisation
- publication-oriented figure generation
"""

import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    model_name="Model",
    save_dir="../figures"
):
    """
    Generate and save confusion matrix plot.
    """

    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["Non-Bankrupt", "Bankrupt"],
        ax=ax
    )

    plt.title(
        f"{model_name} Confusion Matrix\n"
        "Minority-Class Bankruptcy Classification"
    )

    plt.savefig(
        f"{save_dir}/{model_name.lower()}_confusion_matrix.png",
        bbox_inches="tight"
    )

    plt.close()


def plot_roc_curve(
    model,
    X_test,
    y_test,
    model_name="Model",
    save_dir="../figures"
):
    """
    Generate and save ROC-AUC curve.
    """

    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax
    )

    plt.title(
        f"{model_name} ROC-AUC Curve"
    )

    plt.savefig(
        f"{save_dir}/{model_name.lower()}_roc_curve.png",
        bbox_inches="tight"
    )

    plt.close()
