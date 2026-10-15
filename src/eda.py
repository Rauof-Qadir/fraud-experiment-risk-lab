
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_loader import load_transactions


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"

sns.set_theme(style="whitegrid")


def prepare_eda_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a copy of the data for exploratory analysis."""

    required = ["amount", "isFraud", "type"]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    data = df.copy()

    # Convert relevant columns to numeric where possible.
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
    data["isFraud"] = pd.to_numeric(data["isFraud"], errors="coerce")

    # Keep rows with valid binary fraud labels and usable amounts.
    data = data[
        data["isFraud"].isin([0, 1])
        & data["amount"].notna()
        & data["type"].notna()
    ].copy()

    data["isFraud"] = data["isFraud"].astype(int)
    data["fraud_status"] = data["isFraud"].map({
        0: "Genuine",
        1: "Fraud",
    })

    return data


def plot_fraud_distribution(df: pd.DataFrame) -> None:
    counts = (
        df["fraud_status"]
        .value_counts()
        .reindex(["Genuine", "Fraud"], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        legend=False,
        ax=ax,
    )

    ax.set_title("Genuine vs Fraud Transactions")
    ax.set_xlabel("Transaction Status")
    ax.set_ylabel("Number of Transactions")

    fig.tight_layout()
    fig.savefig(REPORT_DIR / "fraud_distribution.png", dpi=150)
    plt.close(fig)


def plot_amount_distribution(df: pd.DataFrame) -> None:
    # Limit extreme visual clutter by using a reproducible sample.
    sample = df.sample(
        n=min(100_000, len(df)),
        random_state=42,
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        data=sample,
        x="amount",
        hue="fraud_status",
        bins=60,
        stat="density",
        common_norm=False,
        element="step",
        log_scale=(False, True),
        ax=ax,
    )

    ax.set_title("Transaction Amount Distribution")
    ax.set_xlabel("Transaction Amount")
    ax.set_ylabel("Density (log scale)")

    fig.tight_layout()
    fig.savefig(REPORT_DIR / "amount_distribution.png", dpi=150)
    plt.close(fig)


def plot_fraud_by_type(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("type", observed=True)
        .agg(
            total_transactions=("isFraud", "size"),
            fraud_transactions=("isFraud", "sum"),
        )
        .reset_index()
    )

    summary["fraud_rate_percent"] = (
        summary["fraud_transactions"]
        / summary["total_transactions"]
        * 100
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=summary,
        x="type",
        y="fraud_rate_percent",
        color="steelblue",
        ax=ax,
    )

    ax.set_title("Fraud Rate by Transaction Type")
    ax.set_xlabel("Transaction Type")
    ax.set_ylabel("Fraud Rate (%)")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(REPORT_DIR / "fraud_by_type.png", dpi=150)
    plt.close(fig)

    return summary


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = load_transactions()
    df = prepare_eda_data(raw_data)

    print("=" * 55)
    print("DAY 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 55)

    print("\nOriginal rows:", len(raw_data))
    print("Rows used in EDA:", len(df))
    print("Rows excluded:", len(raw_data) - len(df))

    print("\nFraud summary:")
    print(df["fraud_status"].value_counts())

    print("\nFraud percentage:")
    print(round(df["isFraud"].mean() * 100, 4), "%")

    print("\nFraud rate by transaction type:")
    summary = plot_fraud_by_type(df)
    print(summary.to_string(index=False))

    plot_fraud_distribution(df)
    plot_amount_distribution(df)

    summary.to_csv(
        REPORT_DIR / "fraud_by_type.csv",
        index=False,
    )

    print("\nCharts and summary saved in:", REPORT_DIR)


if __name__ == "__main__":
    main()
