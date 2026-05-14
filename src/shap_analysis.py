"""
shap_analysis.py

SHAP-based explainability analysis
for ensemble financial distress models.
"""

import shap
import matplotlib.pyplot as plt


def generate_shap_summary(model, X_train):

    explainer = shap.Explainer(model)

    shap_values = explainer(X_train)

    shap.summary_plot(shap_values, X_train)

    plt.title("SHAP Summary Plot")
    plt.show()
