# Pretrained models used

Every pretrained model in the final pipeline must be MIT or Apache-2.0 licensed and
≤ 8B parameters (CLAUDE.md §2.2). Add a row *before* the model is used.

| Model | Licence | Parameters | Model card URL | Used in | Verified by / date |
|-------|---------|------------|----------------|---------|--------------------|
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (revision e8f8c211226b894fcb81acc59f3b34ba3efd5f42) | Apache-2.0 | 117,654,272 (≈118M) | https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | `blocking.py` embedding pass (pass 2); `features.py` embedding cosine | HF model API (`cardData.license`, `safetensors.total`), 2026-09-25 |

## Models trained from scratch (no pretrained weights)

Listed for the licence audit; these are libraries whose models we train only on the
provided training data.

| Library | Licence | Model | Used in | Verified by / date |
|---------|---------|-------|---------|--------------------|
| LightGBM 4.7.0 | MIT | gradient-boosted trees, binary pair classifier (size set by `config.LGB_PARAMS`, far below 8B) | `model.py` | package metadata `License-Expression: MIT`, 2026-09-25 |

## Non-model libraries (for reference)

rapidfuzz (MIT), scikit-learn (BSD-3-Clause; TF-IDF only), sentence-transformers
(Apache-2.0; loader for the model above).
