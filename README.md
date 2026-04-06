# Reinforcement Learning Trading Project

This project trains and evaluates RL trading agents on Q3-Q4 2025 data for `NVDA`, `GOOG`, and `TSLA`.

The current dataset spans July 1, 2025 through December 31, 2025. Article coverage is split across 2,181 Q3 articles and 2,644 Q4 articles.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build Data

```bash
python build_dataset.py
```

This writes `data/daily_features.csv` and `data/article_sentiments.csv`.

## Files

- `src/dataset_builder.py`: dataset preparation
- `src/trading_env.py`: trading environment
- `notebooks/`: data prep, training, and evaluation
- `results/`: exported charts and tables
