
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_loader import load_transactions
from src.baseline import predict_baseline, evaluate_baseline
from src.new_rule import predict_new_rule

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"


def main():
    df = load_transactions()

    rules = {
        "Baseline": predict_baseline(df),
        "New Rule": predict_new_rule(df),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for rule_name, predictions in rules.items():
        metrics = evaluate_baseline(df, predictions)

        # Confusion matrix values
        tn = metrics["TN"]
        fp = metrics["FP"]
        fn = metrics["FN"]
        tp = metrics["TP"]

        matrix = [[tn, fp], [fn, tp]]

        plt.figure(figsize=(7, 5))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=",",
            cmap="Blues",
            xticklabels=["Predicted Genuine", "Predicted Fraud"],
            yticklabels=["Actually Genuine", "Actually Fraud"],
        )

        plt.title(f"{rule_name}: Confusion Matrix")
        plt.xlabel("Model Prediction")
        plt.ylabel("Actual Label")
        plt.tight_layout()

        filename = rule_name.lower().replace(" ", "_")
        output_path = REPORT_DIR / f"{filename}_confusion_matrix.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"\n{rule_name}")
        print(f"True Negatives (TN): {tn:,}")
        print(f"False Positives (FP): {fp:,}")
        print(f"False Negatives (FN): {fn:,}")
        print(f"True Positives (TP): {tp:,}")
        print(f"Saved: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
