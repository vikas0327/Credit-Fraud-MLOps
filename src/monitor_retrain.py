
from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.model_utils import load_bundle, ARTIFACTS


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
INCOMING = ROOT / "datasets" / "incoming"

REPORTS = ROOT / "reports"
CANDIDATES = ARTIFACTS / "candidates"

REPORTS.mkdir(parents=True, exist_ok=True)
CANDIDATES.mkdir(parents=True, exist_ok=True)


CONFIG = {
    "credit": {
        "old_file": RAW / "GermanCredit.csv",
        "new_file": INCOMING / "credit_new.csv",
        "target": "credit_risk",
        "experiment": "credit-monitoring-retraining",
    },
    "fraud": {
        "old_file": RAW / "creditcard_2023.csv",
        "new_file": INCOMING / "fraud_new.csv",
        "target": "Class",
        "experiment": "fraud-monitoring-retraining",
    },
}


def get_features(df, target, model_name):
    drop_columns = [target]

    if model_name == "fraud":
        drop_columns.append("id")

    return df.drop(columns=drop_columns, errors="ignore")


def get_labels(df, target, model_name):
    labels = df[target].astype(int)

    # In GermanCredit.csv:
    # 1 = good credit, 0 = bad credit.
    # Our model predicts bad credit risk as positive class (1).
    if model_name == "credit":
        labels = (labels == 0).astype(int)

    return labels


def calculate_metrics(model, X, y):
    predictions = model.predict(X)

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y, predictions, zero_division=0)
        ),
    }


def validate_data(df, old_df, target, model_name):
    print("\nDATA VALIDATION")

    print("Incoming dataset shape:", df.shape)
    print("Missing values:")
    print(df.isnull().sum())

    duplicates = int(df.duplicated().sum())
    print("Duplicate rows:", duplicates)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing.")

    if df[target].isnull().any():
        raise ValueError("Incoming dataset contains missing target values.")

    if model_name == "credit":
        if not set(df[target].astype(int).unique()).issubset({0, 1}):
            raise ValueError("Credit target must contain only 0 and 1.")
    else:
        if not set(df[target].astype(int).unique()).issubset({0, 1}):
            raise ValueError("Fraud target must contain only 0 and 1.")

    if duplicates > 0:
        raise ValueError("Duplicate rows found in incoming dataset.")

    old_features = set(
        get_features(old_df, target, model_name).columns
    )
    new_features = set(
        get_features(df, target, model_name).columns
    )

    if old_features != new_features:
        raise ValueError(
            f"Feature mismatch. Missing: {old_features - new_features}; "
            f"Extra: {new_features - old_features}"
        )

    if model_name == "fraud" and "id" in df.columns:
        if "id" in old_df.columns:
            overlap = set(old_df["id"]) & set(df["id"])
            if overlap:
                raise ValueError(
                    f"Fraud transaction IDs overlap: {len(overlap)}"
                )

    print("\nTarget distribution:")
    print(df[target].value_counts())

    print("\nData validation passed.")


def run_workflow(model_name, config):
    print("\n" + "=" * 60)
    print(f"STARTING {model_name.upper()} WORKFLOW")
    print("=" * 60)

    old_df = pd.read_csv(config["old_file"])
    new_df = pd.read_csv(config["new_file"])

    target = config["target"]

    validate_data(new_df, old_df, target, model_name)

    # Prepare incoming data
    X_new = get_features(new_df, target, model_name)
    y_new = get_labels(new_df, target, model_name)

    # Load existing champion
    champion = load_bundle(model_name)

    # Task B: Evaluate champion on the complete incoming batch
    incoming_metrics = calculate_metrics(champion, X_new, y_new)

    print("\nCHAMPION EVALUATION ON NEW DATA")
    print(json.dumps(incoming_metrics, indent=4))

    # Split incoming data for fair candidate comparison
    X_train_new, X_test_new, y_train_new, y_test_new = (
        train_test_split(
            X_new,
            y_new,
            test_size=0.25,
            random_state=42,
            stratify=y_new,
        )
    )

    # Task C: Combine old data and incoming training portion
    X_old = get_features(old_df, target, model_name)
    y_old = get_labels(old_df, target, model_name)

    X_combined = pd.concat(
        [X_old, X_train_new],
        ignore_index=True
    )
    y_combined = pd.concat(
        [y_old, y_train_new],
        ignore_index=True
    )

    candidate = clone(champion)
    candidate.fit(X_combined, y_combined)

    # Task D: Compare both models on exactly the same holdout
    champion_test_metrics = calculate_metrics(
        champion, X_test_new, y_test_new
    )
    candidate_test_metrics = calculate_metrics(
        candidate, X_test_new, y_test_new
    )

    print("\nCHAMPION HOLDOUT METRICS")
    print(json.dumps(champion_test_metrics, indent=4))

    print("\nCANDIDATE HOLDOUT METRICS")
    print(json.dumps(candidate_test_metrics, indent=4))

    # Quality gate: candidate must not be worse on either metric
    passed = (
        candidate_test_metrics["f1_score"]
        >= champion_test_metrics["f1_score"]
        and
        candidate_test_metrics["accuracy"]
        >= champion_test_metrics["accuracy"]
    )

    decision = (
        "PASS - Candidate eligible for manual promotion"
        if passed
        else "FAIL - Keep existing champion"
    )

    print("\nQUALITY GATE:", decision)

    # Save candidate separately; never overwrite champion
    candidate_path = (
        CANDIDATES / f"{model_name}_candidate.joblib"
    )
    joblib.dump(candidate, candidate_path)

    report = {
        "model": model_name,
        "incoming_rows": len(new_df),
        "incoming_target_distribution": {
            str(k): int(v)
            for k, v in new_df[target].value_counts().items()
        },
        "champion_incoming_metrics": incoming_metrics,
        "champion_holdout_metrics": champion_test_metrics,
        "candidate_holdout_metrics": candidate_test_metrics,
        "quality_gate": decision,
        "candidate_artifact": str(candidate_path),
        "champion_automatically_replaced": False,
    }

    report_path = REPORTS / f"{model_name}_retraining_report.json"
    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    # Task C/D: Track candidate and decision in MLflow
    mlflow.set_experiment(config["experiment"])

    with mlflow.start_run(run_name=f"{model_name}_candidate"):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("old_training_rows", len(old_df))
        mlflow.log_param("new_training_rows", len(X_train_new))
        mlflow.log_param("candidate_rows", len(X_combined))
        mlflow.log_param("quality_gate", decision)

        for key, value in champion_test_metrics.items():
            mlflow.log_metric(f"champion_{key}", value)

        for key, value in candidate_test_metrics.items():
            mlflow.log_metric(f"candidate_{key}", value)

        mlflow.log_artifact(str(candidate_path))
        mlflow.log_artifact(str(report_path))

    print("\nSaved candidate:", candidate_path)
    print("Saved report:", report_path)
    print("Existing champion remains unchanged.")

    return report


def main():
    for model_name, config in CONFIG.items():
        try:
            run_workflow(model_name, config)
        except Exception as error:
            print(
                f"\nERROR in {model_name.upper()} workflow: {error}"
            )
            raise

    print("\nALL MONITORING AND RETRAINING WORKFLOWS COMPLETED.")


if __name__ == "__main__":
    main()
