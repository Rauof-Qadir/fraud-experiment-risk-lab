
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "transactions.csv"


def load_transactions() -> pd.DataFrame:
    """Load the raw transaction dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return df


if __name__ == "__main__":
    transactions = load_transactions()

    print("Dataset loaded successfully!")
    print("Shape:", transactions.shape)
    print("\nFirst 5 rows:")
    print(transactions.head())

    print("\nColumn names:")
    print(transactions.columns.tolist())

    print("\nMissing values:")
    print(transactions.isnull().sum())
