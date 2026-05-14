"""
visualisation.py

Visualisation utilities for
financial distress prediction workflows.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(model, X_test, y_test):

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test
    )

    plt.title("Confusion Matrix")
    plt.show()
