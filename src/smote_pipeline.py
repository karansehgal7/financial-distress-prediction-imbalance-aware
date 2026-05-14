from imblearn.over_sampling import SMOTE


class SMOTEPipeline:

    def __init__(self, random_state=42):

        self.smote = SMOTE(random_state=random_state)

    def apply_smote(self, X_train, y_train):

        X_resampled, y_resampled = self.smote.fit_resample(
            X_train,
            y_train
        )

        print("SMOTE oversampling completed.")

        print("Class distribution after SMOTE:")

        print(y_resampled.value_counts())

        return X_resampled, y_resampled


if __name__ == "__main__":

    print("SMOTE imbalance mitigation pipeline initialised.")
