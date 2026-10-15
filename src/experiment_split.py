
from pathlib import Path

import pandas as pd

from src.data_loader import load_transactions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"


def main():
    df = load_transactions()

    if "step" not in df.columns:
        raise ValueError(
            "Time-based split ke liye dataset mein 'step' column chahiye."
        )

    steps = sorted(df["step"].dropna().unique())

    if len(steps) < 3:
        raise ValueError("Dataset mein enough unique time steps nahi hain.")

    first_cut = int(len(steps) * 0.70)
    second_cut = int(len(steps) * 0.85)

    train_steps = steps[:first_cut]
    validation_steps = steps[first_cut:second_cut]
    test_steps = steps[second_cut:]

    partitions = {
        "train": train_steps,
        "validation": validation_steps,
        "test": test_steps,
    }

    results = []

    for name, partition_steps in partitions.items():
        part = df[df["step"].isin(partition_steps)]
        labels = pd.to_numeric(part["isFraud"], errors="coerce")
        valid_labels = labels.isin([0, 1])
        valid = labels[valid_labels].astype(int)

        results.append({
            "partition": name,
            "first_step": int(partition_steps[0]),
            "last_step": int(partition_steps[-1]),
            "unique_steps": len(partition_steps),
            "transactions": len(part),
            "valid_labels": len(valid),
            "fraud_count": int((valid == 1).sum()),
            "genuine_count": int((valid == 0).sum()),
        })

    report = pd.DataFrame(results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "temporal_split_report.csv"
    report.to_csv(output_path, index=False)

    print("\nTIME-BASED EXPERIMENT SPLIT")
    print("=" * 80)
    print(report.to_string(index=False))
    print(f"\nSaved: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
