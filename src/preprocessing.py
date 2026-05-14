import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class FinancialDataPreprocessor:

    def __init__(self):

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def load_dataset(self, file_path):

        df = pd.read_csv(file_path)

        print(f"Dataset loaded successfully.")
        print(f"Dataset shape: {df.shape}")

        return df

    def separate_features_and_target(self, df, target_column):

        X = df.drop(columns=[target_column])
        y = df[target_column]

        return X, y

    def preprocess_features(self, X):

        numerical_columns = X.select_dtypes(include=[np.number]).columns

        X[numerical_columns] = self.imputer.fit_transform(
            X[numerical_columns]
        )

        X[numerical_columns] = self.scaler.fit_transform(
            X[numerical_columns]
        )

        return X

    def perform_train_test_split(
        self,
        X,
        y,
        test_size=0.2,
        random_state=42
    ):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        print("Train-test split completed.")

        print(f"Training samples: {X_train.shape[0]}")
        print(f"Testing samples: {X_test.shape[0]}")

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    print("Financial distress preprocessing pipeline initialised.")
