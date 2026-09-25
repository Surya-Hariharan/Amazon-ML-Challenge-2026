# ML Challenge 2026 — Evaluation, Leaderboard & Fair-Play Guidelines

> Source: verbatim excerpt from the organizer-provided `student_resource/README.md`
> (preserved in full at `resources/student_resource.zip`). Reproduced here unedited; do
> not paraphrase away from it. See also [`problem_statement.md`](problem_statement.md)
> and [`submission_requirements.md`](submission_requirements.md).

### Evaluation Criteria

Submissions are evaluated using **F_β Score (β = 0.5)** — a precision-heavy metric that
penalizes false merges (matching two different businesses) more than missed matches.

**Formula:**

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Computed as a **macro-average**: F_0.5 is calculated per Source 1 entity, then averaged
across **all** Source 1 entities in the evaluation set.

Singletons are included in that average. A Source 1 entity with no true matches scores
1.0 when you correctly predict an empty list, and 0.0 when you predict any match for it.
Correctly identifying singletons therefore earns credit, and false merges on them are
penalised.

**Why precision-heavy?** In real-world entity resolution, merging two distinct
businesses (false positive) is more damaging than missing a link (false negative). F_0.5
weights precision 2× over recall.

**Example:**

- Your model predicts S1-00001 matches [S2-00047, S2-00193, S3-00812]
- Ground truth says S1-00001 matches [S2-00047, S3-00812]
- Precision = 2/3, Recall = 2/2 = 1.0
- F_0.5 = (1.25 × 0.667 × 1.0) / (0.25 × 0.667 + 1.0) = **0.714**

### Leaderboard Information

- **Public Leaderboard:** During the challenge, rankings will be based on a subset of
  the test set to provide real-time feedback on your model's performance.
- **Private Leaderboard:** After the challenge ends, the private leaderboard will be
  revealed, which uses the remaining portion of the test set for evaluation.
- **Final Rankings:** The final decision will be based on the private leaderboard.

You submit predictions for the full test set in both cases; the split is applied during
scoring.

### Academic Integrity and Fair Play

**⚠️ STRICTLY PROHIBITED: External Data Lookup**

Participants are **STRICTLY NOT ALLOWED** to use external databases, APIs, or services
to look up business identities or resolve entities. This includes but is not limited to:

- Using commercial entity resolution APIs or services
- Looking up business registrations from government databases
- Using geocoding APIs to normalize addresses
- Any external data augmentation from internet sources

**Enforcement:**

- All submitted approaches, methodologies, and code pipelines will be thoroughly
  reviewed and verified
- Any evidence of external data lookup will result in **immediate disqualification**

**Fair Play:** This challenge is designed to test your machine learning and data science
skills using only the provided training data.

### Tips for Success

- Invest in a strong blocking/candidate generation strategy — it determines the upper
  bound of your recall
- Explore string similarity features (Jaccard, Levenshtein, TF-IDF cosine) for name and
  address matching
- Pay attention to country specific address patterns
- Consider the precision-recall trade-off carefully — F_0.5 rewards precision more than
  recall
- Do not neglect singletons — correctly predicting "no match" is worth a full 1.0 on
  that entity
- Validate your own output format against the rules above before submitting
