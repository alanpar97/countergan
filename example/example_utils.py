import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _to_one_hot(y: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Convert integer labels to one-hot encoding."""
    if n_classes is None:
        n_classes = int(y.max()) + 1
    return np.eye(n_classes)[y]


def load_data(base_path: str, random_seed: int = 42):
    """Load the Pima Indians Diabetes dataset and return train/test splits."""
    df = pd.read_csv(os.path.join(base_path, "diabetes.csv"), index_col=False)
    target_column = "Outcome"
    immutable_features = {"Pregnancies", "DiabetesPedigreeFunction", "Age"}

    features = set(df.columns) - {target_column}
    mutable_features = features - immutable_features
    # Mutable features first, then immutable — important for the projection step
    ordered_features = list(mutable_features) + list(immutable_features)

    x = df[ordered_features].values
    y = df[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = _to_one_hot(y_train)
    y_test = _to_one_hot(y_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        ordered_features,
        mutable_features,
        immutable_features,
        scaler,
    )
