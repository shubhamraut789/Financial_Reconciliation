# Financial Reconciliation System Using Unsupervised Learning

An ML-based system that automatically matches transactions between bank statements and check registers, inspired by [Chew 2020](https://doi.org/10.1145/3383455.3422517) which reframes reconciliation as a cross-language information retrieval problem.

## Quick Start

```bash
pip install numpy pandas scikit-learn scipy

# Full automated reconciliation with evaluation
python reconciler.py run

# Learning curve analysis
python reconciler.py curve

# Interactive match → review → improve cycle
python reconciler.py review
```

## Approach

### Phase 1: Unique-Amount Matching
Transactions with amounts that are unique across the entire dataset are matched automatically with high confidence. This matches 264/308 transactions at 100% precision and serves as the "seed" parallel corpus for the ML phase — analogous to how the paper uses unique amounts to bootstrap training data.

Each match receives a confidence score (0–1) penalized for:
- Date gap (up to −0.3 for 9+ day gaps)
- Amount rounding differences (proportional to relative diff)
- Type code mismatches (−0.2)

Matches are flagged for `date_gap` (>5 days) or `amount_rounding` (>$0.05).

### Phase 2: SVD-Based Cross-Language Retrieval
For the remaining 44 transactions (those with non-unique or near-duplicate amounts), the system follows the paper's cross-language IR methodology:

1. **Tokenization**: Each transaction becomes a bag of prefixed tokens (`B_kroger`, `R_grocery`, `B_type_DR`, `R_cat_grocery`). The prefix separation creates distinct vocabulary spaces for each source — exactly as in cross-language IR.

2. **Term Alignment via PMI**: From matched pairs (the "parallel corpus"), we compute pointwise mutual information between bank tokens and register tokens. This learns associations like `B_kroger ↔ R_grocery`, `B_amazon ↔ R_online`.

3. **SVD Projection**: We build a term-by-document matrix from the parallel corpus, multiply by the alignment matrix to reinforce cross-source co-occurrence patterns, then apply truncated SVD. This produces a shared semantic space where both bank and register transactions can be compared.

4. **Nearest-Neighbor Matching**: Unmatched transactions are projected into the SVD space. We compute a combined similarity score:
   - **SVD cosine similarity** (text/descriptor matching) — weight 0.3
   - **Date proximity** (Gaussian kernel, σ=5 days) — weight 0.3
   - **Amount similarity** (Gaussian kernel, σ=1% relative) — weight 0.4

   Greedy assignment picks the best overall match for each bank transaction.

5. **Confidence Scoring**: Each Phase 2 match gets a calibrated confidence score that blends the raw combined similarity with the *margin* over the next-best candidate. A match with score 0.9 where the runner-up is 0.3 gets higher confidence than one with score 0.9 where the runner-up is 0.85. Matches below 0.4 confidence are flagged as `low_confidence`.

### Human-in-the-Loop Workflow
The `review` mode implements the paper's monthly cycle:
- Automated matching → human QC → retrain → re-match
- Accepted matches feed back into training data
- Rejected matches return to the unmatched pool

## Results

### Full System Performance
```
Phase 1 (unique amounts): 264/308 — Precision 100%, Recall 85.7%
Phase 2 (SVD ML):          44/44  — Precision 100%
OVERALL: 308/308 — Precision 100%, Recall 100%, F1 100%
```

The dataset has 308 transactions with only 3 duplicate amount values (6 transactions total). Phase 1 handles the easy 264. For the remaining 44, the amount-similarity Gaussian (σ=1%) is extremely discriminative — even small rounding differences create enough signal to disambiguate most pairs. The SVD text component helps break ties but isn't the primary driver on this dataset.

This is visible in the learning curve: the full system achieves 100% regardless of seed data, while text-only SVD tops out at ~96%.

### Difficulty Analysis
Hardest matches (lowest confidence in Phase 2):
```
B0110 conf=0.338 | ONLINE PMT WATER $147.29 → Electric bill $147.33
B0184 conf=0.423 | WHOLE FOODS #3931 $147.30 → Grocery store $147.30
```
These are hard because $147.29, $147.30, and $147.33 are all within rounding tolerance, and the descriptions share no vocabulary. The system resolves them correctly using the combined signal.

### Learning Curve
Seeds drawn from the 44 hard transactions only (to properly isolate the ML effect):

```
[Full system (amt+date+SVD)]
  Seed=  0% → F1=100%  (amount feature handles it)
  Seed= 50% → F1=100%

[SVD + date (no amount)]
  Seed=  0% → F1=97.7%  (7 errors in 44 hard matches)
  Seed= 50% → F1=100%   (learning eliminates errors)

[SVD text-only]
  Seed=  0% → F1=87.0%  (40 errors in 44 hard matches)
  Seed= 70% → F1=95.8%  (improving but still limited)
```

The middle row is the most informative: without the amount feature, the system starts at 97.7% and reaches 100% as it learns from more examples — confirming the paper's central claim.

## Design Decisions

**Why hybrid SVD + features instead of pure SVD?** Financial transactions have strong numerical signals (amounts, dates) that would be wasteful to ignore. The combined scoring lets the SVD handle "translation" of descriptions while numerical features handle what they're best at. On this dataset, the amount feature alone nearly solves the problem — but on noisier data with more duplicate amounts, the SVD component would matter much more.

**Why not embeddings (BERT/sentence-transformers)?** Transaction descriptions are short fragments ("BP GAS #1775", "Fill up"), not natural sentences. Pre-trained models are overkill and add heavy dependencies. The SVD approach is lightweight, interpretable, and genuinely learns from the specific data.

**Confidence calibration**: Raw combined similarity isn't a good confidence measure because a score of 0.8 means different things depending on whether the next-best candidate scored 0.2 or 0.79. We blend raw score with margin-over-runner-up to produce more meaningful confidence values.

**Greedy vs. optimal assignment**: We use greedy nearest-neighbor rather than Hungarian algorithm. Greedy is simpler and works well when confidence scores are well-separated. For production use with many ambiguous matches, Hungarian assignment would be better.

## Limitations & Future Work

- **Scale**: The PMI alignment matrix is O(V²) where V is vocabulary size. For large datasets, sparse representations or sampling would be needed.
- **Amount dominance**: On this dataset, the amount feature does most of the work. A harder dataset with more duplicate amounts would better exercise the SVD component.
- **Temporal modeling**: The paper uses date-lag distributions; we use a simpler Gaussian kernel. Learning the actual lag distribution from data would be more robust.
- **Recovery from errors**: Incorrect matches in training data can reinforce bad patterns. Active learning that prioritizes uncertain matches for human review would help.
- **One-to-many matching**: The current system assumes 1:1 correspondence. Real reconciliation often involves split transactions or aggregations.

## Testing

```bash
pip install pytest
python -m pytest test_reconciler.py -v
```

10 tests covering tokenization, unique-amount matching, SVD training/projection/matching, evaluation metrics, and a full-dataset integration test.

## Files

- `reconciler.py` — Core system: data loading, matching, SVD, evaluation, CLI
- `test_reconciler.py` — Unit and integration tests
- `bank_statements.csv` — 308 bank transactions
- `check_register.csv` — 308 check register entries