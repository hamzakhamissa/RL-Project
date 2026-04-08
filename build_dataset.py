import argparse
from pathlib import Path

from src.dataset_builder import DatasetPaths, build_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finbert", action="store_true", help="Use FinBERT scorer instead of VADER")
    args = parser.parse_args()

    scorer = "finbert" if args.finbert else "vader"

    project_root = Path(__file__).resolve().parent
    files_dir = project_root / "files"
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    ticker_file_map = {
        "NVDA": files_dir / "NVDA_q4_2025.csv",
        "GOOG": files_dir / "GOOG_q4_2025.csv",
        "TSLA": files_dir / "TSLA_q4_2025.csv",
    }

    suffix = "_finbert" if args.finbert else ""
    paths = DatasetPaths(
        market_dir=files_dir,
        articles_csv=files_dir / "q4_articles" / "raw_articles_q4_2025.csv",
        output_daily=data_dir / f"daily_features{suffix}.csv",
        output_article_sentiments=data_dir / f"article_sentiments{suffix}.csv",
    )

    print(f"Scorer: {scorer}")
    daily, article_sentiments = build_dataset(paths, ticker_file_map, scorer=scorer)
    daily.to_csv(paths.output_daily, index=False)
    article_sentiments.to_csv(paths.output_article_sentiments, index=False)

    print(f"Saved {len(daily)} daily rows to {paths.output_daily}")
    print(f"Saved {len(article_sentiments)} article rows to {paths.output_article_sentiments}")


if __name__ == "__main__":
    main()
