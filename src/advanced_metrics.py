
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from src.data_loader import load_transactions
from src.baseline import predict_baseline
from src.new_rule import predict_new_rule

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"


def calculate_metrics(y_true, y_pred):
    actual = pd.to_numeric(y_true, errors="coerce")
    valid = actual.isin([0, 1])

    actual = actual.loc[valid].astype(int)
    predicted = y_pred.loc[valid].astype(int)

    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tp = int(((actual == 1) & (predicted == 1)).sum())

    precision = precision_score(actual, predicted, zero_division=0)
    recall = recall_score(actual, predicted, zero_division=0)
    f1 = f1_score(actual, predicted, zero_division=0)

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "transactions_evaluated": len(actual),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "precision_percent": round(precision * 100, 4),
        "recall_percent": round(recall * 100, 4),
        "specificity_percent": round(specificity * 100, 4),
        "false_positive_rate_percent": round(fpr * 100, 4),
        "f1_score": round(f1, 6),
    }


def main():
    df = load_transactions()

    actual = df["isFraud"]
    predictions = {
        "Baseline": predict_baseline(df),
        "New Rule": predict_new_rule(df),
    }

    results = []

    for rule_name, prediction in predictions.items():
        metrics = calculate_metrics(actual, prediction)
        results.append({"rule": rule_name, **metrics})

    report = pd.DataFrame(results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "advanced_metrics_comparison.csv"
    report.to_csv(output_path, index=False)

    print("\nADVANCED METRICS COMPARISON")
    print("=" * 75)
    print(report.to_string(index=False))
    print(f"\nSaved: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
