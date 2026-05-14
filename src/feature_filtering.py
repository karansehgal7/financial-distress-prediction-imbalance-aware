"""
feature_filtering.py

Correlation-aware feature filtering pipeline for
financial distress prediction under severe class imbalance.

This module removes highly correlated financial variables
to reduce multicollinearity and improve downstream
ensemble model stability.
"""

import pandas as pd
import numpy as np


def correlation_filter(df, threshold=0.90):
    """
    Remove highly correlated features.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input feature dataframe.

    threshold : float
        Correlation threshold.

    Returns:
    --------
    filtered_df : pandas.DataFrame
        Reduced dataframe.
    """

    corr_matrix = df.corr().abs()

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    drop_columns = [
        column for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]

    filtered_df = df.drop(columns=drop_columns)

    print(f"Removed {len(drop_columns)} highly correlated features.")

    return filtered_df
