"""
Financial Reconciliation System Using Unsupervised Learning

Inspired by: Peter A. Chew, "Unsupervised-Learning Financial Reconciliation:
a Robust, Accurate Approach Inspired by Machine Translation" (ICAIF '20).

Approach: Hybrid SVD + feature-based matching.
- Phase 1: Unique-amount matching (high-confidence seed matches)
- Phase 2: SVD-based cross-language retrieval for remaining transactions

The system treats bank statements and check registers as two "languages"
describing the same transactions, builds a shared semantic space via SVD,
and finds nearest-neighbor matches using cosine similarity combined with
date-proximity and amount-similarity features.
"""

import re
import json
import sys
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(bank_path="bank_statements.csv", register_path="check_register.csv"):
    """Load and normalize the two data sources."""
    bank = pd.read_csv(bank_path, parse_dates=["date"])
    reg = pd.read_csv(register_path, parse_dates=["date"])
    bank["type_norm"] = bank["type"].map({"DEBIT": "DR", "CREDIT": "CR"})
    reg["type_norm"] = reg["type"]
    return bank, reg


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_description(desc):
    """Split a description into lowercase alpha tokens."""
    return re.findall(r"[a-z]+", desc.lower())


def transaction_tokens(row, source):
    """
    Build prefixed token bag for a transaction.
    B_ prefix for bank, R_ for register — creates separate vocabulary
    spaces for cross-language retrieval.
    """
    prefix = "B_" if source == "bank" else "R_"
    tokens = [prefix + t for t in tokenize_description(str(row["description"]))]
    tokens.append(prefix + "type_" + str(row.get("type_norm", "")))
    if source == "register" and pd.notna(row.get("category")):
        tokens.append(prefix + "cat_" + str(row["category"]).lower().replace(" ", "_"))
    return tokens


# ---------------------------------------------------------------------------
# Phase 1: Unique-amount matching
# ---------------------------------------------------------------------------

def match_unique_amounts(bank, reg, tolerance=0.10):
    """
    Match transactions whose amounts are unique (within tolerance) across
    both datasets. Each match gets a confidence score based on date gap,
    amount difference, and type agreement.
    """
    matches = []
    bank_remaining = set(bank.index)
    reg_remaining = set(reg.index)

    reg_by_amt = {}
    for idx, row in reg.iterrows():
        reg_by_amt.setdefault(round(row["amount"], 2), []).append(idx)

    bank_amt_counts = Counter(round(bank.loc[i, "amount"], 2) for i in bank_remaining)

    for bidx in list(bank_remaining):
        bamt = round(bank.loc[bidx, "amount"], 2)
        if bank_amt_counts[bamt] != 1:
            continue

        candidates = []
        for ramt, ridxs in reg_by_amt.items():
            if abs(bamt - ramt) <= tolerance:
                candidates.extend(ri for ri in ridxs if ri in reg_remaining)

        if len(candidates) != 1:
            continue

        ridx = candidates[0]
        brow, rrow = bank.loc[bidx], reg.loc[ridx]
        day_diff = abs((brow["date"] - rrow["date"]).days)
        amt_diff = abs(brow["amount"] - rrow["amount"])
        type_match = brow["type_norm"] == rrow["type_norm"]

        # Confidence: penalize anomalies from a 1.0 baseline
        confidence = 1.0
        confidence -= min(day_diff / 30.0, 0.3)       # up to -0.3 for 9+ day gap
        confidence -= min(amt_diff / brow["amount"], 0.2)  # up to -0.2 for amount diff
        if not type_match:
            confidence -= 0.2
        confidence = round(max(confidence, 0.0), 3)

        flag = None
        if day_diff > 5:
            flag = "date_gap"
        elif amt_diff > 0.05:
            flag = "amount_rounding"

        matches.append({
            "bank_id": brow["transaction_id"],
            "register_id": rrow["transaction_id"],
            "bank_idx": int(bidx),
            "register_idx": int(ridx),
            "amount_diff": round(amt_diff, 4),
            "day_diff": int(day_diff),
            "confidence": confidence,
            "method": "unique_amount",
            "flag": flag,
        })
        bank_remaining.discard(bidx)
        reg_remaining.discard(ridx)

    return matches


# ---------------------------------------------------------------------------
# Phase 2: SVD-based cross-language matching
# ---------------------------------------------------------------------------

class SVDReconciler:
    """
    Cross-language LSI approach (Dumais et al. 1996 / Chew et al. 2011):
    builds a shared semantic space from aligned transaction pairs via SVD,
    then projects unmatched transactions and finds nearest neighbors.
    """

    def __init__(self, n_components=40):
        self.n_components = n_components
        self.vocab = {}
        self.U = None

    def _build_vocab(self, bank, reg):
        vocab = {}
        for _, row in bank.iterrows():
            for t in transaction_tokens(row, "bank"):
                vocab.setdefault(t, len(vocab))
        for _, row in reg.iterrows():
            for t in transaction_tokens(row, "register"):
                vocab.setdefault(t, len(vocab))
        self.vocab = vocab

    def _make_vector(self, row, source):
        vec = np.zeros(len(self.vocab))
        for t in transaction_tokens(row, source):
            if t in self.vocab:
                vec[self.vocab[t]] += 1.0
        return np.log1p(vec)

    def _build_alignment_matrix(self, matched_pairs, bank, reg):
        """PMI-based term alignment from matched pairs (the 'parallel corpus')."""
        V = len(self.vocab)
        N = len(matched_pairs)
        if N == 0:
            return np.eye(V)

        cooccur = np.zeros((V, V))
        bank_count = np.zeros(V)
        reg_count = np.zeros(V)

        for m in matched_pairs:
            btokens = transaction_tokens(bank.loc[m["bank_idx"]], "bank")
            rtokens = transaction_tokens(reg.loc[m["register_idx"]], "register")
            for bt in btokens:
                if bt in self.vocab:
                    bank_count[self.vocab[bt]] += 1
            for rt in rtokens:
                if rt in self.vocab:
                    reg_count[self.vocab[rt]] += 1
            for bt in btokens:
                for rt in rtokens:
                    if bt in self.vocab and rt in self.vocab:
                        cooccur[self.vocab[bt], self.vocab[rt]] += 1

        alignment = np.zeros((V, V))
        for i in range(V):
            for j in range(V):
                if cooccur[i, j] > 0 and bank_count[i] > 0 and reg_count[j] > 0:
                    pmi = np.log((cooccur[i, j] * N) / (bank_count[i] * reg_count[j]))
                    alignment[i, j] = max(pmi, 0)
        return alignment

    def train(self, matched_pairs, bank, reg):
        """
        Train from matched pairs:
        1. Build vocabulary
        2. PMI term-alignment matrix
        3. Term-by-document matrix from parallel corpus
        4. Multiply alignment into term-doc matrix
        5. SVD → U matrix (shared semantic space)
        """
        self._build_vocab(bank, reg)
        V = len(self.vocab)
        n_pairs = len(matched_pairs)

        if n_pairs < 2:
            self.U = np.eye(V, min(self.n_components, V))
            return

        alignment = self._build_alignment_matrix(matched_pairs, bank, reg)

        td_matrix = np.zeros((V, n_pairs))
        for di, m in enumerate(matched_pairs):
            td_matrix[:, di] = (
                self._make_vector(bank.loc[m["bank_idx"]], "bank")
                + self._make_vector(reg.loc[m["register_idx"]], "register")
            )

        product = alignment @ td_matrix
        k = min(self.n_components, min(product.shape) - 1)
        if k < 1:
            self.U = np.eye(V, min(self.n_components, V))
            return

        U, S, Vt = svds(csr_matrix(product), k=k)
        order = np.argsort(-S)
        self.U = U[:, order]

    def project(self, row, source):
        """Project a transaction into the SVD semantic space."""
        vec = self._make_vector(row, source)
        projected = vec @ self.U
        norm = np.linalg.norm(projected)
        return projected / norm if norm > 0 else projected

    def match(self, bank, reg, bank_indices, reg_indices,
              date_weight=0.3, amount_weight=0.4):
        """
        Match unreconciled transactions using combined scoring:
        - SVD cosine similarity (text matching)
        - Date proximity (Gaussian, σ=5 days)
        - Amount similarity (Gaussian, σ=1% relative)

        Returns matches with calibrated confidence scores.
        """
        if not bank_indices or not reg_indices:
            return []

        bank_vecs = np.array([self.project(bank.loc[i], "bank") for i in bank_indices])
        reg_vecs = np.array([self.project(reg.loc[i], "register") for i in reg_indices])
        sim_svd = cosine_similarity(bank_vecs, reg_vecs)

        bank_dates = np.array([(bank.loc[i, "date"] - pd.Timestamp("2023-01-01")).days
                                for i in bank_indices])
        reg_dates = np.array([(reg.loc[i, "date"] - pd.Timestamp("2023-01-01")).days
                               for i in reg_indices])
        date_diff = np.abs(bank_dates[:, None] - reg_dates[None, :])
        sim_date = np.exp(-date_diff**2 / (2 * 5**2))

        bank_amts = np.array([bank.loc[i, "amount"] for i in bank_indices])
        reg_amts = np.array([reg.loc[i, "amount"] for i in reg_indices])
        amt_diff = np.abs(bank_amts[:, None] - reg_amts[None, :])
        max_amt = np.maximum(bank_amts[:, None], reg_amts[None, :])
        rel_diff = amt_diff / np.maximum(max_amt, 1e-6)
        sim_amount = np.exp(-rel_diff**2 / (2 * 0.01**2))

        text_weight = 1.0 - date_weight - amount_weight
        combined = (text_weight * sim_svd
                    + date_weight * sim_date
                    + amount_weight * sim_amount)

        # Greedy assignment by descending combined score
        matches = []
        used_bank, used_reg = set(), set()
        for flat_idx in np.argsort(-combined.ravel()):
            bi = flat_idx // len(reg_indices)
            ri = flat_idx % len(reg_indices)
            bidx, ridx = bank_indices[bi], reg_indices[ri]
            if bidx in used_bank or ridx in used_reg:
                continue
            used_bank.add(bidx)
            used_reg.add(ridx)

            brow, rrow = bank.loc[bidx], reg.loc[ridx]
            raw_score = float(combined[bi, ri])

            # Calibrated confidence: how much better is this match than
            # the next-best alternative for this bank transaction?
            row_scores = np.sort(combined[bi])[::-1]
            if len(row_scores) > 1:
                margin = raw_score - row_scores[1]
            else:
                margin = raw_score
            # Blend raw score with margin for calibrated confidence
            confidence = round(0.5 * raw_score + 0.5 * min(margin * 2, 1.0), 3)

            day_diff_val = abs((brow["date"] - rrow["date"]).days)
            amt_diff_val = abs(brow["amount"] - rrow["amount"])

            flag = None
            if confidence < 0.4:
                flag = "low_confidence"
            elif day_diff_val > 5:
                flag = "date_gap"
            elif amt_diff_val > 0.05:
                flag = "amount_rounding"

            matches.append({
                "bank_id": brow["transaction_id"],
                "register_id": rrow["transaction_id"],
                "bank_idx": int(bidx),
                "register_idx": int(ridx),
                "amount_diff": round(amt_diff_val, 4),
                "day_diff": int(day_diff_val),
                "confidence": confidence,
                "method": "svd_ml",
                "flag": flag,
                "svd_sim": round(float(sim_svd[bi, ri]), 3),
                "date_sim": round(float(sim_date[bi, ri]), 3),
                "amount_sim": round(float(sim_amount[bi, ri]), 3),
            })
            if len(matches) == min(len(bank_indices), len(reg_indices)):
                break

        return matches


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(matches, bank, reg):
    """Precision, recall, F1 using ground-truth ID correspondence (B0047 <-> R0047)."""
    total = len(bank)
    correct = sum(1 for m in matches if m["bank_id"][1:] == m["register_id"][1:])
    proposed = len(matches)
    precision = correct / proposed if proposed else 0
    recall = correct / total if total else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        "correct": correct, "proposed": proposed, "total": total,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def analyze_errors(matches, bank, reg):
    """Print details of incorrect matches for debugging."""
    errors = [m for m in matches if m["bank_id"][1:] != m["register_id"][1:]]
    if not errors:
        print("  No errors!")
        return
    for m in errors:
        brow, rrow = bank.loc[m["bank_idx"]], reg.loc[m["register_idx"]]
        print(f"  WRONG: {m['bank_id']} -> {m['register_id']} "
              f"(conf={m['confidence']:.3f}, method={m['method']})")
        print(f"    Bank:     {brow['date'].date()} | {brow['description']:30s} | ${brow['amount']:.2f}")
        print(f"    Register: {rrow['date'].date()} | {rrow['description']:30s} | ${rrow['amount']:.2f}")
        correct_id = "R" + m["bank_id"][1:]
        cr = reg[reg["transaction_id"] == correct_id]
        if not cr.empty:
            c = cr.iloc[0]
            print(f"    Correct:  {c['date'].date()} | {c['description']:30s} | ${c['amount']:.2f}")


def analyze_difficulty(matches, bank, reg):
    """Analyze which transaction types are hardest to match."""
    print("\n  Difficulty analysis by transaction category:")
    # Group phase2 matches by register category
    phase2 = [m for m in matches if m["method"] == "svd_ml"]
    if not phase2:
        print("    (no SVD matches to analyze)")
        return

    by_cat = {}
    for m in phase2:
        rid = m["register_id"]
        rrow = reg[reg["transaction_id"] == rid]
        cat = rrow.iloc[0]["category"] if not rrow.empty else "Unknown"
        by_cat.setdefault(cat, []).append(m)

    print(f"    {'Category':<20s} {'Count':>5s} {'Correct':>7s} {'Avg Conf':>8s} {'Avg Margin':>10s}")
    for cat in sorted(by_cat, key=lambda c: len(by_cat[c]), reverse=True):
        ms = by_cat[cat]
        correct = sum(1 for m in ms if m["bank_id"][1:] == m["register_id"][1:])
        avg_conf = np.mean([m["confidence"] for m in ms])
        # Lower confidence = harder
        print(f"    {cat:<20s} {len(ms):>5d} {correct:>5d}/{len(ms):<2d} {avg_conf:>8.3f}")

    # Also show: transactions with multiple close-amount candidates
    print("\n  Hardest individual matches (lowest confidence in Phase 2):")
    for m in sorted(phase2, key=lambda x: x["confidence"])[:5]:
        brow = bank.loc[m["bank_idx"]]
        rrow = reg.loc[m["register_idx"]]
        ok = "✓" if m["bank_id"][1:] == m["register_id"][1:] else "✗"
        print(f"    {ok} {m['bank_id']} conf={m['confidence']:.3f} "
              f"| {brow['description'][:25]:25s} ${brow['amount']:.2f} "
              f"-> {rrow['description'][:25]:25s} ${rrow['amount']:.2f}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_reconciliation(bank_path="bank_statements.csv",
                       register_path="check_register.csv",
                       n_components=40, seed_fraction=0.0, verbose=True):
    """Run the full two-phase reconciliation pipeline."""
    bank, reg = load_data(bank_path, register_path)

    if verbose:
        print(f"Loaded {len(bank)} bank and {len(reg)} register transactions.\n")

    # Optional: simulate prior ground-truth matches
    seed_matches = []
    if seed_fraction > 0:
        n_seed = int(len(bank) * seed_fraction)
        np.random.seed(42)
        for i in np.random.choice(len(bank), n_seed, replace=False):
            brow = bank.iloc[i]
            rid = "R" + brow["transaction_id"][1:]
            rrow_df = reg[reg["transaction_id"] == rid]
            if not rrow_df.empty:
                seed_matches.append({
                    "bank_id": brow["transaction_id"], "register_id": rid,
                    "bank_idx": int(bank.index[i]),
                    "register_idx": int(rrow_df.index[0]),
                    "amount_diff": 0, "day_diff": 0,
                    "confidence": 1.0, "method": "seed", "flag": None,
                })
        if verbose:
            print(f"Phase 0: Seeded {len(seed_matches)} ground-truth matches "
                  f"({seed_fraction:.0%})\n")

    # Phase 1
    phase1 = match_unique_amounts(bank, reg)
    if verbose:
        e1 = evaluate(phase1, bank, reg)
        print(f"Phase 1 (unique amounts): {len(phase1)} matches")
        print(f"  Precision={e1['precision']:.2%}  Recall={e1['recall']:.2%}  "
              f"F1={e1['f1']:.2%}")
        flagged = [m for m in phase1 if m["flag"]]
        if flagged:
            print(f"  Flagged: {len(flagged)} ({', '.join(set(m['flag'] for m in flagged))})")
        print()

    # Deduplicate seeds + phase1
    seen = set()
    all_confirmed = []
    for m in seed_matches + phase1:
        if m["bank_idx"] not in seen:
            seen.add(m["bank_idx"])
            all_confirmed.append(m)

    matched_bank = {m["bank_idx"] for m in all_confirmed}
    matched_reg = {m["register_idx"] for m in all_confirmed}
    remaining_bank = [i for i in bank.index if i not in matched_bank]
    remaining_reg = [i for i in reg.index if i not in matched_reg]

    if verbose:
        print(f"Remaining after Phase 1: {len(remaining_bank)} bank, "
              f"{len(remaining_reg)} register\n")

    # Phase 2
    reconciler = SVDReconciler(n_components=n_components)
    reconciler.train(all_confirmed, bank, reg)
    phase2 = reconciler.match(bank, reg, remaining_bank, remaining_reg)

    if verbose:
        e2 = evaluate(phase2, bank, reg)
        print(f"Phase 2 (SVD ML): {len(phase2)} matches")
        print(f"  Precision={e2['precision']:.2%}  Recall={e2['recall']:.2%}  "
              f"F1={e2['f1']:.2%}")
        print()

    all_matches = all_confirmed + phase2
    if verbose:
        total_eval = evaluate(all_matches, bank, reg)
        print("=" * 60)
        print(f"OVERALL: {total_eval['proposed']}/{total_eval['total']} matches")
        print(f"  Correct:   {total_eval['correct']}")
        print(f"  Precision: {total_eval['precision']:.2%}")
        print(f"  Recall:    {total_eval['recall']:.2%}")
        print(f"  F1:        {total_eval['f1']:.2%}")

        errors = [m for m in all_matches if m["bank_id"][1:] != m["register_id"][1:]]
        if errors:
            print(f"\nError analysis ({len(errors)} mismatches):")
            analyze_errors(all_matches, bank, reg)

        analyze_difficulty(all_matches, bank, reg)
        print()

    return all_matches, evaluate(all_matches, bank, reg)


# ---------------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------------

def demonstrate_learning_curve(bank_path="bank_statements.csv",
                               register_path="check_register.csv"):
    """
    Show how performance improves with more training data.

    To properly isolate the ML learning effect, we seed ONLY from the 44
    hard-to-match transactions (those with non-unique amounts) rather than
    from the full dataset. This avoids the problem of seeds overlapping
    with Phase 1 unique-amount matches, which would mask the learning curve.

    We test three scenarios:
    1. Full system (amount+date+SVD) — production accuracy
    2. SVD + date only (no amount feature) — shows date+text learning
    3. SVD text-only — isolates pure ML text matching improvement
    """
    print("=" * 60)
    print("LEARNING CURVE: Performance vs. seed data from hard transactions")
    print("=" * 60)

    bank, reg = load_data(bank_path, register_path)
    phase1 = match_unique_amounts(bank, reg)
    phase1_bank = {m["bank_idx"] for m in phase1}
    phase1_reg = {m["register_idx"] for m in phase1}

    # The 44 hard transactions (non-unique amounts, not matched by Phase 1)
    hard_bank = [i for i in bank.index if i not in phase1_bank]
    hard_reg = [i for i in reg.index if i not in phase1_reg]

    print(f"\n  Phase 1 matched {len(phase1)}/308 (unique amounts)")
    print(f"  Hard transactions (for SVD): {len(hard_bank)}\n")

    configs = [
        ("Full system (amt+date+SVD)", 0.4, 0.3),
        ("SVD + date (no amount)",     0.0, 0.5),
        ("SVD text-only",              0.0, 0.0),
    ]

    for label, amt_w, date_w in configs:
        print(f"  [{label}]")
        for frac in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70]:
            # Seed from HARD transactions only
            n_seed = int(len(hard_bank) * frac)
            np.random.seed(42)
            seed_indices = np.random.choice(len(hard_bank), n_seed, replace=False) if n_seed > 0 else []

            seed_matches = []
            for si in seed_indices:
                bidx = hard_bank[si]
                brow = bank.loc[bidx]
                rid = "R" + brow["transaction_id"][1:]
                rrow_df = reg[reg["transaction_id"] == rid]
                if not rrow_df.empty:
                    seed_matches.append({
                        "bank_id": brow["transaction_id"], "register_id": rid,
                        "bank_idx": int(bidx),
                        "register_idx": int(rrow_df.index[0]),
                        "amount_diff": 0, "day_diff": 0,
                        "confidence": 1.0, "method": "seed", "flag": None,
                    })

            # Training data = phase1 + seeds from hard set
            training = phase1 + seed_matches
            seeded_bank = {m["bank_idx"] for m in seed_matches}
            seeded_reg = {m["register_idx"] for m in seed_matches}

            # Remaining = hard transactions minus seeded ones
            rem_bank = [i for i in hard_bank if i not in seeded_bank]
            rem_reg = [i for i in hard_reg if i not in seeded_reg]

            reconciler = SVDReconciler(n_components=40)
            reconciler.train(training, bank, reg)
            phase2 = reconciler.match(bank, reg, rem_bank, rem_reg,
                                      amount_weight=amt_w, date_weight=date_w)

            all_matches = phase1 + seed_matches + phase2
            metrics = evaluate(all_matches, bank, reg)
            svd_metrics = evaluate(phase2, bank, reg)
            svd_correct = svd_metrics["correct"]
            svd_total = svd_metrics["proposed"]

            print(f"    Seed={frac:5.0%} ({n_seed:2d}/{len(hard_bank)} hard)  "
                  f"Overall F1={metrics['f1']:.2%}  "
                  f"SVD-only={svd_correct}/{svd_total} correct  "
                  f"Total={metrics['correct']}/{metrics['total']}")
        print()


# ---------------------------------------------------------------------------
# Interactive review
# ---------------------------------------------------------------------------

def interactive_review(bank_path="bank_statements.csv",
                       register_path="check_register.csv"):
    """
    CLI-based match → review → improve cycle:
    1. Auto-match unique amounts
    2. SVD-match remaining, present for human review
    3. Accepted matches feed back into training → retrain → re-match
    """
    bank, reg = load_data(bank_path, register_path)
    print(f"Loaded {len(bank)} bank / {len(reg)} register transactions.\n")

    confirmed = []

    # Phase 1
    phase1 = match_unique_amounts(bank, reg)
    print(f"Phase 1: {len(phase1)} unique-amount matches (auto-accepted).\n")
    confirmed.extend(phase1)

    iteration = 0
    resp = ""
    while True:
        iteration += 1
        matched_bank = {m["bank_idx"] for m in confirmed}
        matched_reg = {m["register_idx"] for m in confirmed}
        remaining_bank = [i for i in bank.index if i not in matched_bank]
        remaining_reg = [i for i in reg.index if i not in matched_reg]

        if not remaining_bank or not remaining_reg:
            print("All transactions matched!")
            break

        print(f"--- Iteration {iteration}: {len(remaining_bank)} unmatched ---")
        reconciler = SVDReconciler(n_components=40)
        reconciler.train(confirmed, bank, reg)
        proposals = reconciler.match(bank, reg, remaining_bank, remaining_reg)
        proposals.sort(key=lambda m: -m["confidence"])

        newly_confirmed = 0
        for m in proposals:
            brow = bank.loc[m["bank_idx"]]
            rrow = reg.loc[m["register_idx"]]
            print(f"\n  Bank:     {brow['transaction_id']} | {brow['date'].date()} | "
                  f"{brow['description']:30s} | ${brow['amount']:.2f}")
            print(f"  Register: {rrow['transaction_id']} | {rrow['date'].date()} | "
                  f"{rrow['description']:30s} | ${rrow['amount']:.2f}")
            print(f"  Confidence: {m['confidence']:.3f}  "
                  f"Amount diff: ${m['amount_diff']:.2f}  Day diff: {m['day_diff']}")
            if m.get("flag"):
                print(f"  ⚠ Flag: {m['flag']}")

            resp = input("  Accept? [Y/n/q] ").strip().lower()
            if resp == "q":
                break
            elif resp in ("", "y"):
                confirmed.append(m)
                newly_confirmed += 1

        if resp == "q" or newly_confirmed == 0:
            break
        print(f"\n  Accepted {newly_confirmed} new matches. Retraining...\n")

    metrics = evaluate(confirmed, bank, reg)
    print(f"\nFinal: {metrics['correct']}/{metrics['total']} correct  "
          f"P={metrics['precision']:.2%} R={metrics['recall']:.2%} "
          f"F1={metrics['f1']:.2%}")

    save_path = Path("matched_pairs.json")
    with open(save_path, "w") as f:
        json.dump(confirmed, f, indent=2, default=str)
    print(f"Saved {len(confirmed)} matches to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"

    if mode == "run":
        run_reconciliation()
    elif mode == "curve":
        demonstrate_learning_curve()
    elif mode == "review":
        interactive_review()
    else:
        print("Usage: python reconciler.py [run|curve|review]")
        print("  run    - Full automated reconciliation with evaluation")
        print("  curve  - Show learning curve (performance vs training data)")
        print("  review - Interactive match-review-improve cycle")