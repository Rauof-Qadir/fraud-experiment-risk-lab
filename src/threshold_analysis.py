
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_transactions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"

# Experimental thresholds; amounts are in dataset currency units.
THRESHOLDS = [
    50_000,
    100_000,
    150_000,
    200_000,
    250_000,
    300_000,
    400_000,
    500_000,
    750_000,
    1_000_000,
]

# Illustrative assumptions, not verified bank costs.
FP_COST = 5.0
FN_COST = 150.0


def evaluate_threshold(df, threshold):
    actual = pd.to_numeric(df["isFraud"], errors="coerce")
    amount = pd.to_numeric(df["amount"], errors="coerce")

    valid = actual.isin([0, 1]) & amount.notna() & df["type"].notna()
    actual = actual.loc[valid].astype(int)

    eligible = df.loc[valid, "type"].isin({"TRANSFER", "CASH_OUT"})
    values = amount.loc[valid]

    predicted = (eligible & (values >= threshold)).astype(int)

    y = actual.to_numpy()
    p = predicted.to_numpy()

    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tp = int(np.sum((y == 1) & (p == 1)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    estimated_cost = fp * FP_COST + fn * FN_COST

    return {
        "threshold": threshold,
        "transactions_evaluated": len(actual),
        "flagged_transactions": tp + fp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "precision_percent": round(precision * 100, 4),
        "recall_percent": round(recall * 100, 4),
        "specificity_percent": round(specificity * 100, 4),
        "f1_score": round(f1, 6),
        "estimated_cost_usd": round(estimated_cost, 2),
    }


def main():
    df = load_transactions()

    required = {"type", "amount", "isFraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    results = [
        evaluate_threshold(df, threshold)
        for threshold in THRESHOLDS
    ]

    report = pd.DataFrame(results)

    # Lower estimated cost is better under our assumed cost model.
    report = report.sort_values("estimated_cost_usd")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "threshold_sensitivity.csv"
    report.to_csv(output_path, index=False)

    print("\nTHRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 100)
    print(report.to_string(index=False))
    print("\nLowest estimated cost under current assumptions:")
    print(report.iloc[0].to_string())
    print(f"\nSaved: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()