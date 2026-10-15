
from pathlib import Path

import pandas as pd

from src.data_loader import load_transactions
from src.baseline import predict_baseline, evaluate_baseline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"

# Illustrative assumptions, not verified bank costs.
FALSE_POSITIVE_COST = 5.0
FALSE_NEGATIVE_COST = 150.0


def calculate_business_cost(
    metrics: dict,
    fp_cost: float,
    fn_cost: float,
) -> dict:
    """Calculate the estimated financial cost of classification errors."""

    if fp_cost < 0 or fn_cost < 0:
        raise ValueError("Costs must be non-negative.")

    fp = metrics["FP"]
    fn = metrics["FN"]

    total_fp_cost = fp * fp_cost
    total_fn_cost = fn * fn_cost
    total_cost = total_fp_cost + total_fn_cost

    return {
        "false_positives": fp,
        "false_negatives": fn,
        "cost_per_false_positive_usd": fp_cost,
        "cost_per_false_negative_usd": fn_cost,
        "total_false_positive_cost_usd": round(total_fp_cost, 2),
        "total_false_negative_cost_usd": round(total_fn_cost, 2),
        "total_estimated_cost_usd": round(total_cost, 2),
    }


def main():
    df = load_transactions()

    predictions = predict_baseline(df)
    metrics = evaluate_baseline(df, predictions)

    costs = calculate_business_cost(
        metrics,
        fp_cost=FALSE_POSITIVE_COST,
        fn_cost=FALSE_NEGATIVE_COST,
    )

    print("=" * 55)
    print("FRAUD EXPERIMENT: COST ANALYSIS")
    print("=" * 55)

    for key, value in costs.items():
        print(f"{key}: {value}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([costs]).to_csv(
        REPORT_DIR / "baseline_cost_analysis.csv",
        index=False,
    )

    print("\nSaved: reports/baseline_cost_analysis.csv")


if __name__ == "__main__":
    main()
