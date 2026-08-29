# NLP Pipeline on Sentiment140 (UC53564_41)

End-to-end NLP pipeline covering text normalization, N-gram language modelling,
POS tagging (HMM/Viterbi), CFG/CKY + dependency parsing, lexical semantics/WSD,
and vector semantics, built on the **Sentiment140** Kaggle tweet-sentiment
dataset. Random seed fixed at **42** throughout for reproducibility.

## Contents

```
nlp_project/
├── README.md
├── requirements.txt
├── run_all.ipynb          # Main Colab notebook — run this top to bottom
└── src/
    ├── __init__.py
    ├── utils.py            # seeding, edge-case helpers, generic utilities
    ├── data_loader.py       # Stage i  — download, describe, split
    ├── preprocessing.py     # Stage ii — normalization, vocab, stemmer/lemmatizer
    ├── ngram_lm.py           # Stage iii — unigram/bigram/trigram + smoothing
    ├── pos_tagging.py        # Stage iv — HMM Viterbi tagger vs spaCy
    ├── parsing.py            # Stage v — CFG→CNF, CKY, spaCy dependency parses
    ├── lexical_semantics.py  # Stage vi — WordNet + simplified Lesk WSD
    └── vector_semantics.py   # Stage vii — TF-IDF, PPMI, Word2Vec comparison
```

## How to run (Google Colab)

1. Upload the `nlp_project` folder to Colab (or `git clone` your GitHub repo).
2. Open `run_all.ipynb` in Colab.
3. Run the first cell — it installs dependencies, downloads NLTK corpora and the
   spaCy English model, and sets `RANDOM_SEED = 42` everywhere.
4. To download Sentiment140 you need a Kaggle API token:
   - Kaggle → Account → **Create New API Token** → download `kaggle.json`.
   - In Colab: `from google.colab import files; files.upload()` and select
     `kaggle.json`, or use `kagglehub` (no token needed, used by default in
     `data_loader.py`).
5. Run cells top-to-bottom. Each stage prints its required tables and saves
   plots/figures to `outputs/`. Copy these into the report PDF.

## Design decisions / assumptions

- **Dataset**: Sentiment140 (`kazanova/sentiment140` on Kaggle), 1.6M tweets,
  2 classes (negative=0, positive=4), auto-labelled by emoticon. For Colab
  runtime feasibility we **stratified-subsample** `SAMPLE_SIZE=60,000` tweets
  (configurable in `data_loader.py`) — this is stated explicitly in the report
  as a scoping decision, not a hidden shortcut. All splits use `random_state=42`.
- **Split**: 80% train / 10% validation / 10% test, stratified by class.
- **Vocabulary cutoff**: tokens with frequency < `MIN_FREQ=5` in the training
  split are mapped to `<UNK>`.
- **Edge cases** (empty input, single token, OOV, punctuation/specials, mixed
  case, duplicates, very long input, non-ASCII) are each handled by a small
  guard function in `utils.py` (`safe_tokenize`, `is_empty_or_whitespace`, etc.)
  and unit-style asserted in the notebook's "Edge case tests" cell.

## Reproducibility

`utils.set_global_seed(42)` seeds `random`, `numpy`, and `PYTHONHASHSEED` and is
called at the top of every stage.

## Submission checklist

- [ ] Report PDF: `UC53564_41_RollNo1_RollNo2.pdf` (tables, plots, parse trees)
- [ ] Source code: this repo (GitHub link) or `nlp_project.zip`
- [ ] Executed Colab notebook: `run_all.ipynb` (with outputs, not cleared)
