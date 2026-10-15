
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from src.data_loader import load_transactions


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"


def predict_baseline(df: pd.DataFrame) -> pd.Series:
    """Flag all TRANSFER and CASH_OUT transactions as suspicious."""

    suspicious_types = {"TRANSFER", "CASH_OUT"}

    predictions = df["type"].isin(suspicious_types).astype(int)
    predictions.name = "baseline_prediction"

    return predictions


def evaluate_baseline(
    df: pd.DataFrame,
    predictions: pd.Series,
) -> dict:
    """Evaluate predictions against actual fraud labels."""

    actual = pd.to_numeric(df["isFraud"], errors="coerce")

    valid = actual.isin([0, 1])

    y_true = actual.loc[valid].astype(int)
    y_pred = predictions.loc[valid].astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
        if tp + tn + fp + fn
        else 0.0
    )

    return {
        "evaluated_transactions": len(y_true),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "accuracy": round(accuracy, 6),
    }


def main():
    df = load_transactions()

    required = {"type", "isFraud"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    predictions = predict_baseline(df)
    results = evaluate_baseline(df, predictions)

    print("=" * 50)
    print("BASELINE FRAUD DETECTION RESULTS")
    print("=" * 50)

    for metric, value in results.items():
        print(f"{metric}: {value}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([results]).to_csv(
        REPORT_DIR / "baseline_metrics.csv",
        index=False,
    )

    print("\nSaved: reports/baseline_metrics.csv")


if __name__ == "__main__":
    main()
