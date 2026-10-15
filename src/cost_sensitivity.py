
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"

INPUT_FILE = REPORT_DIR / "threshold_sensitivity.csv"

# Illustrative cost assumptions, not verified bank costs.
FP_COSTS = [1, 5, 10, 25]
FN_COSTS = [50, 150, 500, 1000]

# Example business constraints for minimum fraud recall.
MIN_RECALL_REQUIREMENTS = [0.50, 0.80, 0.90]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Run threshold_analysis first: {INPUT_FILE}"
        )

    thresholds = pd.read_csv(INPUT_FILE)
    results = []

    for _, row in thresholds.iterrows():
        for fp_cost in FP_COSTS:
            for fn_cost in FN_COSTS:
                estimated_cost = (
                    row["FP"] * fp_cost
                    + row["FN"] * fn_cost
                )

                results.append({
                    "threshold": int(row["threshold"]),
                    "fp_cost": fp_cost,
                    "fn_cost": fn_cost,
                    "precision_percent": row["precision_percent"],
                    "recall_percent": row["recall_percent"],
                    "FP": int(row["FP"]),
                    "FN": int(row["FN"]),
                    "estimated_cost": round(estimated_cost, 2),
                })

    detailed = pd.DataFrame(results)

    # Best threshold for each cost assumption.
    best_by_cost = (
        detailed.sort_values("estimated_cost")
        .groupby(["fp_cost", "fn_cost"], as_index=False)
        .first()
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    detailed.to_csv(
        REPORT_DIR / "cost_sensitivity_all.csv",
        index=False,
    )
    best_by_cost.to_csv(
        REPORT_DIR / "cost_sensitivity_best.csv",
        index=False,
    )

    print("\nBEST THRESHOLD BY COST ASSUMPTION")
    print("=" * 85)
    print(best_by_cost.to_string(index=False))

    # Apply example minimum-recall constraints.
    print("\nBEST THRESHOLD WITH MINIMUM RECALL REQUIREMENTS")
    print("=" * 85)

    for minimum_recall in MIN_RECALL_REQUIREMENTS:
        eligible = detailed[
            (detailed["fp_cost"] == 5)
            & (detailed["fn_cost"] == 150)
            & (detailed["recall_percent"] >= minimum_recall * 100)
        ]

        print(f"\nMinimum recall: {minimum_recall * 100:.0f}%")

        if eligible.empty:
            print("No tested threshold meets this requirement.")
            continue

        best = eligible.loc[eligible["estimated_cost"].idxmin()]

        print(f"Best threshold: {int(best['threshold']):,}")
        print(f"Recall: {best['recall_percent']:.4f}%")
        print(f"False positives: {int(best['FP']):,}")
        print(f"Missed frauds: {int(best['FN']):,}")
        print(f"Estimated cost: ${best['estimated_cost']:,.2f}")

    print("\nSaved:")
    print("reports/cost_sensitivity_all.csv")
    print("reports/cost_sensitivity_best.csv")


if __name__ == "__main__":
    main()
