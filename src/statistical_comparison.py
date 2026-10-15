
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.data_loader import load_transactions
from src.baseline import predict_baseline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"

SELECTED_THRESHOLD = 750_000
FP_COST = 5.0
FN_COST = 150.0
BOOTSTRAP_REPETITIONS = 2000
RANDOM_SEED = 42


def predict_threshold(df, threshold):
    amount = pd.to_numeric(df["amount"], errors="coerce")
    eligible = df["type"].isin({"TRANSFER", "CASH_OUT"})
    return (eligible & (amount >= threshold)).astype(int)


def main():
    df = load_transactions()

    # Reproduce the same chronological split.
    steps = sorted(df["step"].dropna().unique())
    first_cut = int(len(steps) * 0.70)
    second_cut = int(len(steps) * 0.85)

    test_steps = set(steps[second_cut:])
    test = df[df["step"].isin(test_steps)].copy()

    actual = pd.to_numeric(test["isFraud"], errors="coerce")
    valid = actual.isin([0, 1])

    test = test.loc[valid].copy()
    actual = actual.loc[valid].astype(int).to_numpy()

    baseline = predict_baseline(test).astype(int).to_numpy()
    selected = predict_threshold(
        test, SELECTED_THRESHOLD
    ).astype(int).to_numpy()

    # Per-transaction cost for each rule.
    baseline_fp = (actual == 0) & (baseline == 1)
    baseline_fn = (actual == 1) & (baseline == 0)
    selected_fp = (actual == 0) & (selected == 1)
    selected_fn = (actual == 1) & (selected == 0)

    baseline_cost = (
        baseline_fp.astype(float) * FP_COST
        + baseline_fn.astype(float) * FN_COST
    )

    selected_cost = (
        selected_fp.astype(float) * FP_COST
        + selected_fn.astype(float) * FN_COST
    )

    savings = baseline_cost - selected_cost

    print("\nPAIRED STATISTICAL COMPARISON")
    print("=" * 65)
    print(f"Test transactions: {len(test):,}")
    print(f"Baseline estimated cost: ${baseline_cost.sum():,.2f}")
    print(f"Selected rule estimated cost: ${selected_cost.sum():,.2f}")
    print(f"Estimated saving: ${savings.sum():,.2f}")

    # McNemar's exact test on paired correct/incorrect decisions.
    baseline_error = baseline != actual
    selected_error = selected != actual

    baseline_wrong_selected_right = int(
        np.sum(baseline_error & ~selected_error)
    )
    baseline_right_selected_wrong = int(
        np.sum(~baseline_error & selected_error)
    )

    discordant = (
        baseline_wrong_selected_right
        + baseline_right_selected_wrong
    )

    if discordant:
        test_result = binomtest(
            baseline_wrong_selected_right,
            n=discordant,
            p=0.5,
            alternative="two-sided",
        )
        p_value = test_result.pvalue
    else:
        p_value = 1.0

    print("\nMcNEMAR EXACT TEST")
    print(f"Baseline wrong, selected right: {baseline_wrong_selected_right:,}")
    print(f"Baseline right, selected wrong: {baseline_right_selected_wrong:,}")
    print(f"Discordant decisions: {discordant:,}")
    print(f"Exact p-value: {p_value:.8f}")

    # Cluster bootstrap: resample test time steps, not individual rows.
    per_step = pd.DataFrame({
        "step": test["step"].to_numpy(),
        "saving": savings,
    }).groupby("step")["saving"].sum()

    step_savings = per_step.to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)

    bootstrap_savings = np.empty(BOOTSTRAP_REPETITIONS)

    for i in range(BOOTSTRAP_REPETITIONS):
        sampled = rng.choice(
            step_savings,
            size=len(step_savings),
            replace=True,
        )
        bootstrap_savings[i] = sampled.sum()

    lower, upper = np.percentile(
        bootstrap_savings, [2.5, 97.5]
    )

    print("\n95% CLUSTER BOOTSTRAP CONFIDENCE INTERVAL")
    print(f"Estimated saving: ${savings.sum():,.2f}")
    print(f"95% CI lower bound: ${lower:,.2f}")
    print(f"95% CI upper bound: ${upper:,.2f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([{
        "selected_threshold": SELECTED_THRESHOLD,
        "test_transactions": len(test),
        "baseline_cost": baseline_cost.sum(),
        "selected_rule_cost": selected_cost.sum(),
        "estimated_saving": savings.sum(),
        "mcnemar_p_value": p_value,
        "bootstrap_95_ci_lower": lower,
        "bootstrap_95_ci_upper": upper,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
    }])

    output_path = REPORT_DIR / "statistical_comparison.csv"
    summary.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
