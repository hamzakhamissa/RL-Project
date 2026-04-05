from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MARKET_TIMEZONE = "America/New_York"
PREMARKET_END_MINUTE = 9 * 60 + 30
MARKET_CLOSE_MINUTE = 16 * 60

BASIC_SENTIMENT_COLUMNS = [
    "article_count",
    "sentiment_mean",
    "sentiment_max",
    "sentiment_min",
    "sentiment_std",
    "positive_count",
    "negative_count",
]

ENHANCED_SENTIMENT_COLUMNS = BASIC_SENTIMENT_COLUMNS + [
    "sentiment_abs_mean",
    "sentiment_range",
    "net_sentiment_count",
    "sentiment_balance",
    "article_count_change_1d",
    "sentiment_mean_change_1d",
    "sentiment_mean_lag_1d",
    "sentiment_mean_lag_3d",
    "sentiment_abs_mean_lag_1d",
    "sentiment_std_lag_1d",
    "premarket_article_count",
    "intraday_article_count",
    "after_hours_article_count",
    "premarket_sentiment_mean",
    "intraday_sentiment_mean",
    "after_hours_sentiment_mean",
]


@dataclass(frozen=True)
class DatasetPaths:
    market_dir: Path
    articles_csv: Path
    output_daily: Path
    output_article_sentiments: Path


def compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy().sort_values("date")
    result["return_1d"] = result["close"].pct_change(1)
    result["return_3d"] = result["close"].pct_change(3)
    result["rolling_volatility_5d"] = result["return_1d"].rolling(5).std()
    return result


def score_articles(articles: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    result = articles.copy()

    def score_row(row: pd.Series) -> float:
        headline = str(row.get("headline") or "")
        summary = str(row.get("summary") or "")
        return analyzer.polarity_scores(f"{headline}. {summary}")["compound"]

    result["sentiment_vader"] = result.apply(score_row, axis=1)
    return result


def align_articles_to_sessions(
    articles: pd.DataFrame,
    market_dates: pd.DataFrame,
) -> pd.DataFrame:
    result = articles.copy()
    result["published_at_utc"] = pd.to_datetime(result["published_at_utc"], utc=True)
    result["ticker"] = result["ticker"].replace({"GOOGL": "GOOG"})
    result["published_at_et"] = result["published_at_utc"].dt.tz_convert(MARKET_TIMEZONE)
    result["local_date"] = (
        result["published_at_et"].dt.tz_localize(None).dt.normalize()
    )
    minutes = (
        result["published_at_et"].dt.hour * 60 + result["published_at_et"].dt.minute
    )
    result["news_session"] = np.select(
        [minutes < PREMARKET_END_MINUTE, minutes < MARKET_CLOSE_MINUTE],
        ["premarket", "intraday"],
        default="after_hours",
    )
    result["candidate_date"] = result["local_date"] + pd.to_timedelta(
        (result["news_session"] == "after_hours").astype(int),
        unit="D",
    )

    effective_dates = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")
    for ticker, group in result.groupby("ticker", sort=False):
        trading_days = np.array(
            pd.to_datetime(
                market_dates.loc[market_dates["ticker"] == ticker, "date"]
            ).sort_values(),
            dtype="datetime64[ns]",
        )
        if trading_days.size == 0:
            continue
        candidates = np.array(group["candidate_date"], dtype="datetime64[ns]")
        positions = np.searchsorted(trading_days, candidates, side="left")
        mapped = np.full(len(group), np.datetime64("NaT"), dtype="datetime64[ns]")
        valid = positions < trading_days.size
        mapped[valid] = trading_days[positions[valid]]
        effective_dates.loc[group.index] = mapped

    result["effective_date"] = pd.to_datetime(effective_dates)
    result = result.dropna(subset=["effective_date"]).copy()
    return result


def aggregate_sentiment_features(aligned_articles: pd.DataFrame) -> pd.DataFrame:
    grouped = aligned_articles.groupby(["ticker", "effective_date"])
    daily = grouped["sentiment_vader"].agg(
        article_count="count",
        sentiment_mean="mean",
        sentiment_max="max",
        sentiment_min="min",
        sentiment_std="std",
        sentiment_abs_mean=lambda s: s.abs().mean(),
    ).reset_index()
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)
    daily["sentiment_range"] = daily["sentiment_max"] - daily["sentiment_min"]

    pos = (
        aligned_articles[aligned_articles["sentiment_vader"] > 0.05]
        .groupby(["ticker", "effective_date"])
        .size()
        .rename("positive_count")
        .reset_index()
    )
    neg = (
        aligned_articles[aligned_articles["sentiment_vader"] < -0.05]
        .groupby(["ticker", "effective_date"])
        .size()
        .rename("negative_count")
        .reset_index()
    )
    daily = daily.merge(pos, on=["ticker", "effective_date"], how="left")
    daily = daily.merge(neg, on=["ticker", "effective_date"], how="left")
    daily[["positive_count", "negative_count"]] = (
        daily[["positive_count", "negative_count"]].fillna(0).astype(int)
    )
    daily["net_sentiment_count"] = daily["positive_count"] - daily["negative_count"]
    daily["sentiment_balance"] = np.where(
        daily["article_count"] > 0,
        daily["net_sentiment_count"] / daily["article_count"],
        0.0,
    )

    session_counts = (
        aligned_articles.pivot_table(
            index=["ticker", "effective_date"],
            columns="news_session",
            values="article_id",
            aggfunc="count",
            fill_value=0,
        )
        .rename(
            columns={
                "premarket": "premarket_article_count",
                "intraday": "intraday_article_count",
                "after_hours": "after_hours_article_count",
            }
        )
        .reset_index()
    )

    session_means = (
        aligned_articles.pivot_table(
            index=["ticker", "effective_date"],
            columns="news_session",
            values="sentiment_vader",
            aggfunc="mean",
        )
        .rename(
            columns={
                "premarket": "premarket_sentiment_mean",
                "intraday": "intraday_sentiment_mean",
                "after_hours": "after_hours_sentiment_mean",
            }
        )
        .reset_index()
    )

    daily = daily.merge(session_counts, on=["ticker", "effective_date"], how="left")
    daily = daily.merge(session_means, on=["ticker", "effective_date"], how="left")

    session_cols = [
        "premarket_article_count",
        "intraday_article_count",
        "after_hours_article_count",
        "premarket_sentiment_mean",
        "intraday_sentiment_mean",
        "after_hours_sentiment_mean",
    ]
    for col in session_cols:
        daily[col] = daily[col].fillna(0.0)

    daily = daily.rename(columns={"effective_date": "date"})
    return daily


def merge_market_and_sentiment(
    market: pd.DataFrame,
    sentiment_daily: pd.DataFrame,
) -> pd.DataFrame:
    daily = market.merge(sentiment_daily, on=["ticker", "date"], how="left")
    fill_cols = [col for col in ENHANCED_SENTIMENT_COLUMNS if col in daily.columns]
    daily[fill_cols] = daily[fill_cols].fillna(0.0)
    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)

    def add_lags(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        group["article_count_change_1d"] = group["article_count"] - group["article_count"].shift(1)
        group["sentiment_mean_change_1d"] = (
            group["sentiment_mean"] - group["sentiment_mean"].shift(1)
        )
        group["sentiment_mean_lag_1d"] = group["sentiment_mean"].shift(1)
        group["sentiment_mean_lag_3d"] = group["sentiment_mean"].shift(3)
        group["sentiment_abs_mean_lag_1d"] = group["sentiment_abs_mean"].shift(1)
        group["sentiment_std_lag_1d"] = group["sentiment_std"].shift(1)
        return group

    daily = daily.groupby("ticker", group_keys=False).apply(add_lags)

    lag_cols = [
        "article_count_change_1d",
        "sentiment_mean_change_1d",
        "sentiment_mean_lag_1d",
        "sentiment_mean_lag_3d",
        "sentiment_abs_mean_lag_1d",
        "sentiment_std_lag_1d",
    ]
    daily[lag_cols] = daily[lag_cols].fillna(0.0)
    return daily.reset_index(drop=True)


def build_dataset(paths: DatasetPaths, ticker_file_map: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_frames = []
    for ticker, csv_path in ticker_file_map.items():
        frame = pd.read_csv(csv_path, parse_dates=["Date"]).rename(columns={"Date": "date"})
        frame["ticker"] = ticker
        frame = frame[["date", "ticker", "open", "high", "low", "close", "volume"]]
        market_frames.append(frame)

    market = pd.concat(market_frames, ignore_index=True)
    market = market.sort_values(["ticker", "date"]).reset_index(drop=True)
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    market = market.groupby("ticker", group_keys=False).apply(compute_market_features)
    market = market.reset_index(drop=True)

    articles = pd.read_csv(paths.articles_csv)
    articles = articles.copy()
    articles = articles[pd.to_datetime(articles["published_at_utc"], utc=True) >= pd.Timestamp("2025-10-01", tz="UTC")]
    articles = score_articles(articles)
    aligned_articles = align_articles_to_sessions(articles, market[["ticker", "date"]])
    daily_sentiment = aggregate_sentiment_features(aligned_articles)
    daily = merge_market_and_sentiment(market, daily_sentiment)

    article_export_cols = [
        "article_id",
        "ticker",
        "headline",
        "source_tier",
        "quality_score",
        "published_at_utc",
        "published_at_et",
        "news_session",
        "effective_date",
        "sentiment_vader",
    ]
    article_export = aligned_articles[article_export_cols].copy()
    article_export = article_export.rename(columns={"effective_date": "aligned_trading_date"})
    return daily, article_export
