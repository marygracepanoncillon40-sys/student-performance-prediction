"""
Train the Student Performance Prediction System.

Dataset and target follow the laboratory handout:
- UCI Student Performance dataset (ID 320)
- G3 < 10  -> AT RISK (1)
- G3 >= 10 -> NOT AT RISK (0)
- G3 is never used as an input feature.
"""

from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_data.csv"
MODEL_PATH = BASE_DIR / "student_model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"
METADATA_PATH = BASE_DIR / "model_metadata.json"
RANDOM_STATE = 42


def download_dataset() -> pd.DataFrame:
    """Download UCI dataset and save the records as student_data.csv."""
    print("Downloading the UCI Student Performance dataset (ID 320)...")
    try:
        student = fetch_ucirepo(id=320)
        features = student.data.features
        targets = student.data.targets
        df = pd.concat([features, targets], axis=1)
    except Exception as exc:
        raise RuntimeError(
            "Could not download the UCI dataset. Check your internet connection "
            "and try again. Original error: " + str(exc)
        ) from exc

    if "G3" not in df.columns:
        raise ValueError(
            "The downloaded dataset does not contain the required G3 target column. "
            "Check the UCI download and dataset version."
        )

    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved dataset: {DATA_PATH.name}")
    print("Rows and columns:", df.shape)
    print("\nFirst five rows:")
    print(df.head())
    print("\nColumn data types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    return df


def build_and_train(df: pd.DataFrame):
    """Prepare data, train Random Forest, evaluate it, and save artifacts."""
    if "G3" not in df.columns:
        raise ValueError("Required target column 'G3' was not found.")

    # Target: 1 = at risk (G3 < 10); 0 = not at risk (G3 >= 10)
    y = (pd.to_numeric(df["G3"], errors="coerce") < 10).astype(int)

    # Prevent target leakage: G3 must not be included in X.
    X = df.drop(columns=["G3"]).copy()

    if X.empty:
        raise ValueError("No predictor columns remain after removing G3.")

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipe, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not at risk", "At risk"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision_at_risk": round(float(precision), 4),
        "recall_at_risk": round(float(recall), 4),
        "f1_at_risk": round(float(f1), 4),
        "classification_report": report,
        "confusion_matrix_labels": ["Not at risk", "At risk"],
        "confusion_matrix": matrix.tolist(),
        "test_records": int(len(y_test)),
        "training_records": int(len(y_train)),
    }

    metadata = {
        "target_rule": "G3 < 10 = AT RISK; G3 >= 10 = NOT AT RISK",
        "target_column": "G3",
        "excluded_from_features": ["G3"],
        "feature_columns": X.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "model": "RandomForestClassifier",
        "random_state": RANDOM_STATE,
        "dataset_source": "UCI Machine Learning Repository, Student Performance (ID 320)",
    }

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nMODEL EVALUATION")
    print("----------------")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision (At risk): {precision:.3f}")
    print(f"Recall (At risk):    {recall:.3f}")
    print(f"F1-score (At risk):  {f1:.3f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Not at risk", "At risk"],
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(matrix)
    print(f"\nSaved model: {MODEL_PATH.name}")
    print(f"Saved metrics: {METRICS_PATH.name}")
    print(f"Saved metadata: {METADATA_PATH.name}")
    print("\nTraining completed successfully.")


def main():
    try:
        df = download_dataset()
        build_and_train(df)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
