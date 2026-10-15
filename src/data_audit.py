
from pathlib import Path
import json

import pandas as pd

from src.data_loader import load_transactions


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"


def audit_transactions(df: pd.DataFrame) -> dict:
    """Inspect transaction data without modifying the original data."""

    required_columns = [
        "amount",
        "isFraud",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    fraud_labels = df["isFraud"]

    valid_labels = fraud_labels.isin([0, 1])

    fraud_count = int((fraud_labels == 1).sum())
    genuine_count = int((fraud_labels == 0).sum())
    invalid_label_count = int((~valid_labels).sum())

    total_rows = len(df)
    valid_label_rows = int(valid_labels.sum())

    fraud_rate = (
        fraud_count / valid_label_rows * 100
        if valid_label_rows else 0.0
    )

    report = {
        "total_rows": int(total_rows),
        "total_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values_by_column": {
            col: int(count)
            for col, count in df.isnull().sum().items()
        },
        "missing_cells_total": int(df.isnull().sum().sum()),
        "fraud_transactions": fraud_count,
        "genuine_transactions": genuine_count,
        "invalid_fraud_labels": invalid_label_count,
        "fraud_rate_percent": round(fraud_rate, 4),
        "amount_negative_count": (
            int((df["amount"] < 0).sum())
            if pd.api.types.is_numeric_dtype(df["amount"])
            else None
        ),
        "amount_zero_count": (
            int((df["amount"] == 0).sum())
            if pd.api.types.is_numeric_dtype(df["amount"])
            else None
        ),
        "column_types": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },
    }

    return report


def save_report(report: dict) -> Path:
    """Save the audit report as a JSON file."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "data_audit.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return report_path


def main():
    df = load_transactions()

    print("=" * 55)
    print("FRAUD EXPERIMENT & RISK STATISTICS LAB")
    print("DAY 2: DATA AUDIT")
    print("=" * 55)

    print("\nDataset shape:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    report = audit_transactions(df)

    print("\nDATA QUALITY REPORT")
    print("-" * 55)

    for key, value in report.items():
        print(f"\n{key}:")
        print(value)

    report_path = save_report(report)

    print("\nAudit report saved to:", report_path)


if __name__ == "__main__":
    main()
