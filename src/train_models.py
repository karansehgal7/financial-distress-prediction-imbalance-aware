from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class ModelTrainer:

    def __init__(self):

        self.models = {

            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ),

            "AdaBoost": AdaBoostClassifier(
                n_estimators=200,
                random_state=42
            ),

            "XGBoost": XGBClassifier(
                eval_metric="logloss",
                random_state=42
            ),

            "LightGBM": LGBMClassifier(
                random_state=42
            ),

            "CatBoost": CatBoostClassifier(
                verbose=0,
                random_state=42
            )
        }

    def train_models(self, X_train, y_train):

        trained_models = {}

        for name, model in self.models.items():

            print(f"Training {name}...")

            model.fit(X_train, y_train)

            trained_models[name] = model

            print(f"{name} training completed.")

        return trained_models


if __name__ == "__main__":

    print("Model training pipeline initialised.")
