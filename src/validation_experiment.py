
from pathlib import Path

import pandas as pd

from src.data_loader import load_transactions
from src.baseline import predict_baseline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"

THRESHOLDS = [
    50_000, 100_000, 150_000, 200_000, 250_000,
    300_000, 400_000, 500_000, 750_000, 1_000_000,
]

FP_COST = 5.0
FN_COST = 150.0


def calculate_metrics(df, predictions):
    actual = pd.to_numeric(df["isFraud"], errors="coerce")
    valid = actual.isin([0, 1])

    actual = actual.loc[valid].astype(int)
    predicted = predictions.loc[valid].astype(int)

    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tp = int(((actual == 1) & (predicted == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0

    cost = fp * FP_COST + fn * FN_COST

    return {
        "transactions_evaluated": len(actual),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "precision_percent": round(precision * 100, 4),
        "recall_percent": round(recall * 100, 4),
        "false_positive_rate_percent": round(fpr * 100, 4),
        "estimated_cost": round(cost, 2),
    }


def predict_threshold(df, threshold):
    amount = pd.to_numeric(df["amount"], errors="coerce")
    eligible_type = df["type"].isin({"TRANSFER", "CASH_OUT"})

    return (
        eligible_type & (amount >= threshold)
    ).astype(int)


def main():
    df = load_transactions()

    required = {"step", "type", "amount", "isFraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Split by chronological step, not random rows.
    steps = sorted(df["step"].dropna().unique())

    if len(steps) < 3:
        raise ValueError("Not enough unique time steps for a split.")

    first_cut = int(len(steps) * 0.70)
    second_cut = int(len(steps) * 0.85)

    validation_steps = set(steps[first_cut:second_cut])
    test_steps = set(steps[second_cut:])

    validation = df[df["step"].isin(validation_steps)].copy()
    test = df[df["step"].isin(test_steps)].copy()

    # Select the threshold using validation data only.
    validation_results = []

    for threshold in THRESHOLDS:
        predictions = predict_threshold(validation, threshold)
        metrics = calculate_metrics(validation, predictions)

        validation_results.append({
            "threshold": threshold,
            **metrics,
        })

    validation_report = pd.DataFrame(validation_results)
    validation_report = validation_report.sort_values(
        "estimated_cost"
    ).reset_index(drop=True)

    selected_threshold = int(
        validation_report.iloc[0]["threshold"]
    )

    # Test data is evaluated only after threshold selection.
    baseline_predictions = predict_baseline(test)
    selected_predictions = predict_threshold(
        test, selected_threshold
    )

    test_results = pd.DataFrame([
        {
            "rule": "Baseline",
            "threshold": None,
            **calculate_metrics(test, baseline_predictions),
        },
        {
            "rule": "Selected threshold rule",
            "threshold": selected_threshold,
            **calculate_metrics(test, selected_predictions),
        },
    ])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    validation_path = REPORT_DIR / "validation_threshold_results.csv"
    test_path = REPORT_DIR / "final_test_results.csv"

    validation_report.to_csv(validation_path, index=False)
    test_results.to_csv(test_path, index=False)

    print("\nVALIDATION RESULTS — sorted by estimated cost")
    print("=" * 95)
    print(validation_report.to_string(index=False))

    print("\nSELECTED THRESHOLD")
    print("=" * 95)
    print(f"Threshold: {selected_threshold:,}")
    print("Selection basis: lowest validation estimated cost")
    print(f"Assumed FP cost: ${FP_COST}")
    print(f"Assumed FN cost: ${FN_COST}")

    print("\nFINAL TEST RESULTS")
    print("=" * 95)
    print(test_results.to_string(index=False))

    print(f"\nSaved: {validation_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {test_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
