
import pandas as pd

from src.statistical_comparison import predict_threshold


def test_flags_transfer_above_threshold():
    df = pd.DataFrame({
        "type": ["TRANSFER"],
        "amount": [800_000],
    })

    result = predict_threshold(df, 750_000)

    assert result.tolist() == [1]


def test_flags_cash_out_above_threshold():
    df = pd.DataFrame({
        "type": ["CASH_OUT"],
        "amount": [900_000],
    })

    result = predict_threshold(df, 750_000)

    assert result.tolist() == [1]


def test_flags_amount_equal_to_threshold():
    df = pd.DataFrame({
        "type": ["TRANSFER"],
        "amount": [750_000],
    })

    result = predict_threshold(df, 750_000)

    assert result.tolist() == [1]


def test_does_not_flag_amount_below_threshold():
    df = pd.DataFrame({
        "type": ["TRANSFER"],
        "amount": [749_999],
    })

    result = predict_threshold(df, 750_000)

    assert result.tolist() == [0]


def test_does_not_flag_other_transaction_types():
    df = pd.DataFrame({
        "type": ["PAYMENT", "CASH_IN", "DEBIT"],
        "amount": [900_000, 900_000, 900_000],
    })

    result = predict_threshold(df, 750_000)

    assert result.tolist() == [0, 0, 0]


def test_handles_invalid_amounts_safely():
    df = pd.DataFrame({
        "type": ["TRANSFER", "CASH_OUT"],
        "amount": ["invalid", None],
    })

    result = predict_threshold(df, 750_000)

    assert result.tolist() == [0, 0]


def test_handles_numeric_strings():
    df = pd.DataFrame({
        "type": ["TRANSFER"],
        "amount": ["800000"],
    })

    result = predict_threshold(df, 750_000)

    assert result.tolist() == [1]
