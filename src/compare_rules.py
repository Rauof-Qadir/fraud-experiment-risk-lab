
from pathlib import Path

import pandas as pd

from src.data_loader import load_transactions
from src.baseline import predict_baseline, evaluate_baseline
from src.new_rule import predict_new_rule
from src.cost_analysis import calculate_business_cost


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"


def main():
    df = load_transactions()

    # Same dataset, same actual labels, fair initial comparison.
    baseline_predictions = predict_baseline(df)
    new_predictions = predict_new_rule(df)

    baseline_metrics = evaluate_baseline(
        df, baseline_predictions
    )

    new_metrics = evaluate_baseline(
        df, new_predictions
    )

    baseline_cost = calculate_business_cost(
        baseline_metrics,
        fp_cost=5.0,
        fn_cost=150.0,
    )

    new_cost = calculate_business_cost(
        new_metrics,
        fp_cost=5.0,
        fn_cost=150.0,
    )

    results = pd.DataFrame([
        {
            "rule": "Baseline",
            **baseline_metrics,
            "estimated_cost_usd":
                baseline_cost["total_estimated_cost_usd"],
        },
        {
            "rule": "New rule",
            **new_metrics,
            "estimated_cost_usd":
                new_cost["total_estimated_cost_usd"],
        },
    ])

    print("=" * 65)
    print("BASELINE VS NEW RULE")
    print("=" * 65)
    print(results.to_string(index=False))

    cost_saving = (
        baseline_cost["total_estimated_cost_usd"]
        - new_cost["total_estimated_cost_usd"]
    )

    print("\nEstimated cost saving (USD):", round(cost_saving, 2))

    if baseline_cost["total_estimated_cost_usd"] > 0:
        saving_percent = (
            cost_saving
            / baseline_cost["total_estimated_cost_usd"]
            * 100
        )
    else:
        saving_percent = 0.0

    print("Estimated cost saving (%):", round(saving_percent, 2))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        REPORT_DIR / "rule_comparison.csv",
        index=False,
    )

    print("\nSaved: reports/rule_comparison.csv")


if __name__ == "__main__":
    main()
