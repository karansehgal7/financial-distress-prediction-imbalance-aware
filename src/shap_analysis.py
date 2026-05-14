"""
shap_analysis.py

SHAP-based explainability framework
for imbalance-aware financial distress prediction.

Supports:

- global feature attribution
- explainability-oriented evaluation
- governance-aware interpretability analysis
- publication-oriented SHAP visualisation
"""

import os
import shap
import matplotlib.pyplot as plt


def generate_shap_summary(
    model,
    X_train,
    model_name="Model",
    save_dir="../figures"
):
    """
    Generate SHAP summary plot
    for feature attribution analysis.
    """

    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------
    # SHAP Explainer
    # ---------------------------------

    explainer = shap.Explainer(model)

    shap_values = explainer(X_train)

    # ---------------------------------
    # SHAP Summary Plot
    # ---------------------------------

    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        X_train,
        show=False
    )

    plt.title(
        f"{model_name} SHAP Feature Attribution Analysis"
    )

    plt.savefig(
        f"{save_dir}/{model_name.lower()}_shap_summary.png",
        bbox_inches="tight"
    )

    plt.close()

    return shap_values
