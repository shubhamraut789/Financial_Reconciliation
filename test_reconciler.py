"""Unit tests for the reconciliation system."""

import numpy as np
import pandas as pd
import pytest
from reconciler import (
    tokenize_description,
    transaction_tokens,
    match_unique_amounts,
    SVDReconciler,
    evaluate,
    load_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_bank():
    return pd.DataFrame({
        "transaction_id": ["B001", "B002", "B003"],
        "date": pd.to_datetime(["2023-01-01", "2023-01-05", "2023-01-10"]),
        "description": ["KROGER #1234", "AMAZON.COM", "BP GAS #5678"],
        "amount": [50.00, 25.00, 50.00],
        "type": ["DEBIT", "DEBIT", "DEBIT"],
        "balance": [950, 925, 875],
        "type_norm": ["DR", "DR", "DR"],
    })


@pytest.fixture
def sample_reg():
    return pd.DataFrame({
        "transaction_id": ["R001", "R002", "R003"],
        "date": pd.to_datetime(["2022-12-30", "2023-01-03", "2023-01-09"]),
        "description": ["Grocery store", "Online purchase", "Fill up"],
        "amount": [50.00, 25.00, 50.00],
        "type": ["DR", "DR", "DR"],
        "category": ["Grocery", "Online Purchase", "Gas Station"],
        "notes": ["", "", ""],
        "type_norm": ["DR", "DR", "DR"],
    })


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def test_tokenize_description():
    assert tokenize_description("KROGER #1234") == ["kroger"]
    assert tokenize_description("BP GAS #5678") == ["bp", "gas"]
    assert tokenize_description("") == []


def test_transaction_tokens_bank(sample_bank):
    tokens = transaction_tokens(sample_bank.iloc[0], "bank")
    assert "B_kroger" in tokens
    assert any(t.startswith("B_type_") for t in tokens)


def test_transaction_tokens_register(sample_reg):
    tokens = transaction_tokens(sample_reg.iloc[0], "register")
    assert "R_grocery" in tokens
    assert "R_store" in tokens
    assert "R_cat_grocery" in tokens


# ---------------------------------------------------------------------------
# Unique-amount matching
# ---------------------------------------------------------------------------

def test_unique_amount_matches_unique(sample_bank, sample_reg):
    """Amount 25.00 is unique in both; 50.00 is not."""
    matches = match_unique_amounts(sample_bank, sample_reg)
    assert len(matches) == 1
    assert matches[0]["bank_id"] == "B002"
    assert matches[0]["register_id"] == "R002"


def test_unique_amount_confidence(sample_bank, sample_reg):
    matches = match_unique_amounts(sample_bank, sample_reg)
    assert matches[0]["confidence"] > 0.5


# ---------------------------------------------------------------------------
# SVD Reconciler
# ---------------------------------------------------------------------------

def test_svd_reconciler_train_and_project(sample_bank, sample_reg):
    training = [{
        "bank_id": "B002", "register_id": "R002",
        "bank_idx": 1, "register_idx": 1,
        "amount_diff": 0, "day_diff": 2,
        "confidence": 1.0, "method": "unique_amount", "flag": None,
    }]
    rec = SVDReconciler(n_components=2)
    rec.train(training, sample_bank, sample_reg)
    assert rec.U is not None
    vec = rec.project(sample_bank.iloc[0], "bank")
    assert vec.shape[0] == 2
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-6  # unit vector


def test_svd_reconciler_match(sample_bank, sample_reg):
    training = [{
        "bank_id": "B002", "register_id": "R002",
        "bank_idx": 1, "register_idx": 1,
        "amount_diff": 0, "day_diff": 2,
        "confidence": 1.0, "method": "unique_amount", "flag": None,
    }]
    rec = SVDReconciler(n_components=2)
    rec.train(training, sample_bank, sample_reg)
    matches = rec.match(sample_bank, sample_reg, [0, 2], [0, 2])
    assert len(matches) == 2
    assert all("confidence" in m for m in matches)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_evaluate_perfect():
    matches = [
        {"bank_id": "B001", "register_id": "R001"},
        {"bank_id": "B002", "register_id": "R002"},
    ]
    bank = pd.DataFrame({"transaction_id": ["B001", "B002"]})
    reg = pd.DataFrame({"transaction_id": ["R001", "R002"]})
    metrics = evaluate(matches, bank, reg)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_evaluate_with_errors():
    matches = [
        {"bank_id": "B001", "register_id": "R001"},
        {"bank_id": "B002", "register_id": "R003"},  # wrong
    ]
    bank = pd.DataFrame({"transaction_id": ["B001", "B002", "B003"]})
    reg = pd.DataFrame({"transaction_id": ["R001", "R002", "R003"]})
    metrics = evaluate(matches, bank, reg)
    assert metrics["precision"] == 0.5
    assert metrics["correct"] == 1


# ---------------------------------------------------------------------------
# Integration: full dataset
# ---------------------------------------------------------------------------

def test_full_dataset_accuracy():
    """The system should achieve >= 95% F1 on the provided dataset."""
    from reconciler import run_reconciliation
    _, metrics = run_reconciliation(verbose=False)
    assert metrics["f1"] >= 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])