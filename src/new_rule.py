
import pandas as pd


AMOUNT_THRESHOLD = 200_000


def predict_new_rule(df: pd.DataFrame) -> pd.Series:
    """
    Experimental rule:
    Flag TRANSFER or CASH_OUT transactions
    when amount is at least the configured threshold.
    """

    required = {"type", "amount"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    amount = pd.to_numeric(df["amount"], errors="coerce")

    eligible_type = df["type"].isin(
        {"TRANSFER", "CASH_OUT"}
    )

    predictions = (
        eligible_type
        & (amount >= AMOUNT_THRESHOLD)
    ).astype(int)

    predictions.name = "new_rule_prediction"

    return predictions
