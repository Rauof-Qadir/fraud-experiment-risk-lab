
from pathlib import Path

import pandas as pd

from src.data_loader import load_transactions
from src.baseline import predict_baseline, evaluate_baseline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"


def main():
    df = load_transactions()

    predictions = predict_baseline(df)
    metrics = evaluate_baseline(df, predictions)

    tn = metrics["TN"]
    fp = metrics["FP"]
    fn = metrics["FN"]
    tp = metrics["TP"]

    total_genuine = tn + fp
    total_fraud = tp + fn
    flagged_total = tp + fp

    false_positive_rate = (
        fp / total_genuine if total_genuine else 0
    )

    fraud_miss_rate = (
        fn / total_fraud if total_fraud else 0
    )

    genuine_flag_rate = (
        fp / total_genuine if total_genuine else 0
    )

    frauds_per_1000_flags = (
        tp / flagged_total * 1000 if flagged_total else 0
    )

    analysis = {
        "total_transactions": len(df),
        "flagged_transactions": flagged_total,
        "genuine_transactions_flagged": fp,
        "fraud_transactions_detected": tp,
        "false_positive_rate_percent": round(
            false_positive_rate * 100, 4
        ),
        "fraud_miss_rate_percent": round(
            fraud_miss_rate * 100, 4
        ),
        "genuine_flag_rate_percent": round(
            genuine_flag_rate * 100, 4
        ),
        "frauds_per_1000_flags": round(
            frauds_per_1000_flags, 2
        ),
    }

    print("=" * 55)
    print("BASELINE: BUSINESS IMPACT ANALYSIS")
    print("=" * 55)

    for key, value in analysis.items():
        print(f"{key}: {value}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([analysis]).to_csv(
        REPORT_DIR / "baseline_business_analysis.csv",
        index=False,
    )

    print("\nSaved: reports/baseline_business_analysis.csv")


if __name__ == "__main__":
    main()
