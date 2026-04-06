#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
except Exception:
    trafilatura = None


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

FINNHUB_BASE = "https://finnhub.io/api/v1/company-news"
GDELT_DOC_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

SYMBOL_CONFIG = {
    "NVDA": {
        "company": "NVIDIA",
        "queries": [
            '"NVIDIA"',
            'NVDA',
            '"Jensen Huang"',
            'Blackwell',
            'Hopper',
        ],
    },
    "GOOGL": {
        "company": "Alphabet",
        "queries": [
            '"Alphabet"',
            '"Google"',
            'GOOGL',
            'GOOG',
            '"Sundar Pichai"',
        ],
    },
    "TSLA": {
        "company": "Tesla",
        "queries": [
            '"Tesla"',
            'TSLA',
            '"Elon Musk"',
            'Cybercab',
            'Robotaxi',
        ],
    },
}

HIGH_VALUE_DOMAINS = {
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com", "cnbc.com",
    "marketwatch.com", "barrons.com", "investors.com", "seekingalpha.com",
    "finance.yahoo.com", "yahoo.com", "businessinsider.com", "benzinga.com",
    "theinformation.com", "techcrunch.com", "theverge.com"
}

BLOCKED_DOMAINS = {
    "youtube.com", "youtu.be", "x.com", "twitter.com", "instagram.com",
    "facebook.com", "reddit.com", "linkedin.com", "tiktok.com"
}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_local_env(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values

    bare_values: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                values[key] = value
        else:
            bare_values.append(line)

    if "FINNHUB_API_KEY" not in values and len(bare_values) == 1:
        values["FINNHUB_API_KEY"] = bare_values[0]

    return values


def clean_text(x: str) -> str:
    if not isinstance(x, str):
        return ""
    return re.sub(r"\s+", " ", x).strip()


def normalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        p = urlparse(url)
        if p.netloc.lower().endswith("finnhub.io") and p.path == "/api/news":
            p = p._replace(fragment="")
        else:
            p = p._replace(query="", fragment="")
        return urlunparse(p).rstrip("/")
    except Exception:
        return url.strip()


def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def is_blocked_domain(url: str) -> bool:
    d = get_domain(url)
    return d in BLOCKED_DOMAINS


def source_tier(source: str, url: str) -> str:
    domain = get_domain(url)
    src = (source or "").strip().lower()
    if domain in HIGH_VALUE_DOMAINS or src in {"reuters", "bloomberg", "cnbc", "financial times", "the wall street journal"}:
        return "high"
    if domain:
        return "mid"
    return "low"


def parse_any_datetime(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not math.isnan(value):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except Exception:
            return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


def article_id(ticker: str, url: str, headline: str, published_at: Optional[datetime]) -> str:
    payload = "||".join([
        ticker or "",
        normalize_url(url),
        clean_text(headline).lower(),
        published_at.isoformat() if published_at else "",
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def sleep_with_jitter(base=1.5, jitter=0.8):
    time.sleep(base + random.random() * jitter)


def iter_date_windows(start: str, end: str, window_days: int) -> Iterable[tuple[str, str]]:
    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end, "%Y-%m-%d").date()
    cursor = start_dt
    step = max(1, window_days)

    while cursor <= end_dt:
        window_end = min(cursor + timedelta(days=step - 1), end_dt)
        yield cursor.isoformat(), window_end.isoformat()
        cursor = window_end + timedelta(days=1)


def request_with_backoff(session: requests.Session, url: str, *, params=None, timeout=30, max_retries=7):
    err = None
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if resp.status_code == 429:
                wait = min(90, (2 ** attempt) + random.random() * 2)
                log(f"Rate limited, retrying in {wait:.1f}s")
                time.sleep(wait)
                continue
            if 500 <= resp.status_code < 600:
                wait = min(45, (2 ** attempt) + random.random() * 2)
                log(f"Server error {resp.status_code}, retrying in {wait:.1f}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:
            err = e
            wait = min(45, (2 ** attempt) + random.random() * 2)
            log(f"Request failed ({type(e).__name__}: {e}), retrying in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Request failed after retries: {err}")


def scrape_full_text(session: requests.Session, url: str) -> str:
    if not url or is_blocked_domain(url):
        return ""
    try:
        if trafilatura is not None:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
                if text:
                    return clean_text(text)
        resp = request_with_backoff(session, url, timeout=25, max_retries=3)
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(" ", strip=True)
        text = clean_text(text)
        return text[:25000]
    except Exception:
        return ""


def extract_author_from_html(session: requests.Session, url: str) -> str:
    if not url or is_blocked_domain(url):
        return ""
    try:
        resp = request_with_backoff(session, url, timeout=20, max_retries=2)
        soup = BeautifulSoup(resp.text, "html.parser")
        for key in [
            ("meta", {"name": "author"}),
            ("meta", {"property": "author"}),
            ("meta", {"property": "article:author"}),
        ]:
            tag = soup.find(*key)
            if tag and tag.get("content"):
                return clean_text(tag["content"])
    except Exception:
        pass
    return ""


def fetch_finnhub_company_news(
    session: requests.Session,
    api_key: str,
    symbol: str,
    start: str,
    end: str,
    window_days: int = 30,
) -> List[dict]:
    out = []
    for window_start, window_end in iter_date_windows(start, end, window_days):
        log(f"Fetching {symbol} news for {window_start} to {window_end}")
        params = {"symbol": symbol, "from": window_start, "to": window_end, "token": api_key}
        resp = request_with_backoff(session, FINNHUB_BASE, params=params)
        data = resp.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected Finnhub response for {symbol}: {data}")
        sleep_with_jitter(1.2, 0.6)

        for item in data:
            dt = parse_any_datetime(item.get("datetime"))
            url = item.get("url") or ""
            headline = item.get("headline") or ""
            out.append({
                "article_id": article_id(symbol, url, headline, dt),
                "ticker": symbol,
                "company": SYMBOL_CONFIG[symbol]["company"],
                "published_at_utc": dt.isoformat() if dt else None,
                "date_utc": dt.date().isoformat() if dt else None,
                "headline": clean_text(headline),
                "source": clean_text(item.get("source") or ""),
                "source_tier": source_tier(item.get("source") or "", url),
                "url": url,
                "author": "",
                "category": clean_text(item.get("category") or ""),
                "tags": clean_text(item.get("related") or ""),
                "summary": clean_text(item.get("summary") or ""),
                "full_text": "",
                "collector_source": "finnhub",
            })
    return out


def fetch_gdelt_articles(session: requests.Session, ticker: str, query: str, start: str, end: str, max_records: int = 250) -> List[dict]:
    start_g = start.replace("-", "") + "000000"
    end_g = end.replace("-", "") + "235959"
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(max_records),
        "startdatetime": start_g,
        "enddatetime": end_g,
        "sort": "DateDesc",
    }
    resp = request_with_backoff(session, GDELT_DOC_BASE, params=params, timeout=45, max_retries=5)
    data = resp.json()
    articles = data.get("articles", []) if isinstance(data, dict) else []
    sleep_with_jitter(2.0, 1.0)

    out = []
    for item in articles:
        url = item.get("url") or ""
        title = item.get("title") or ""
        dt = parse_any_datetime(item.get("seendate") or item.get("socialimage"))
        if dt is None:
            raw = item.get("seendate")
            if isinstance(raw, str):
                try:
                    dt = pd.to_datetime(raw, utc=True).to_pydatetime()
                except Exception:
                    dt = None

        domain = item.get("domain") or get_domain(url)
        out.append({
            "article_id": article_id(ticker, url, title, dt),
            "ticker": ticker,
            "company": SYMBOL_CONFIG[ticker]["company"],
            "published_at_utc": dt.isoformat() if dt else None,
            "date_utc": dt.date().isoformat() if dt else None,
            "headline": clean_text(title),
            "source": clean_text(domain),
            "source_tier": source_tier(domain, url),
            "url": url,
            "author": "",
            "category": "",
            "tags": "",
            "summary": clean_text(item.get("excerpt") or ""),
            "full_text": "",
            "collector_source": "gdelt",
        })
    return out


def dedupe_articles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["normalized_url"] = df["url"].fillna("").map(normalize_url)
    df["normalized_headline"] = df["headline"].fillna("").str.lower().map(clean_text)

    tier_rank = {"high": 0, "mid": 1, "low": 2}
    df["tier_rank"] = df["source_tier"].map(lambda x: tier_rank.get(x, 3))
    df["text_len"] = df["full_text"].fillna("").str.len() + df["summary"].fillna("").str.len()

    df = df.sort_values(
        ["ticker", "normalized_headline", "normalized_url", "tier_rank", "text_len", "published_at_utc"],
        ascending=[True, True, True, True, False, False]
    )

    df = df.drop_duplicates(subset=["ticker", "normalized_url"], keep="first")
    df = df.drop_duplicates(subset=["ticker", "normalized_headline", "date_utc"], keep="first")
    df = df.reset_index(drop=True)
    return df.drop(columns=["tier_rank", "text_len"], errors="ignore")


def enrich_articles(
    session: requests.Session,
    df: pd.DataFrame,
    scrape_text: bool,
    fetch_authors: bool,
    per_ticker_cap: Optional[int],
    save_every: int = 0,
    save_callback: Optional[Callable[[pd.DataFrame], None]] = None,
) -> pd.DataFrame:
    df = df.copy()

    tier_rank = {"high": 0, "mid": 1, "low": 2}
    df["tier_rank"] = df["source_tier"].map(lambda x: tier_rank.get(x, 3))
    df["summary_len"] = df["summary"].fillna("").str.len()
    df = df.sort_values(["ticker", "tier_rank", "summary_len", "published_at_utc"], ascending=[True, True, False, False])

    if per_ticker_cap:
        df = df.groupby("ticker", group_keys=False).head(per_ticker_cap).reset_index(drop=True)

    if not scrape_text and not fetch_authors:
        return df.drop(columns=["tier_rank", "summary_len"], errors="ignore")

    full_texts = df.get("full_text", pd.Series([""] * len(df))).fillna("").tolist()
    existing_authors = df.get("author", pd.Series([""] * len(df))).fillna("").tolist()
    authors = existing_authors[:]
    writes_since_save = 0

    for i, row in df.iterrows():
        log(f"[{i + 1}/{len(df)}] {row['ticker']} {row['source']}: {row['headline'][:90]}")
        url = row.get("url", "")
        current_text = full_texts[i] if i < len(full_texts) else ""
        current_author = authors[i] if i < len(authors) else ""

        if scrape_text and not current_text:
            current_text = scrape_full_text(session, url)
            full_texts[i] = current_text
            writes_since_save += 1

        if fetch_authors and not current_author:
            current_author = extract_author_from_html(session, url)
            authors[i] = current_author
            writes_since_save += 1

        if save_every and save_callback and writes_since_save >= save_every:
            checkpoint_df = df.copy()
            checkpoint_df["full_text"] = full_texts
            checkpoint_df["author"] = authors
            save_callback(checkpoint_df)
            writes_since_save = 0

        sleep_with_jitter(1.6, 1.0)

    df["full_text"] = full_texts
    df["author"] = authors

    if save_every and save_callback and writes_since_save:
        save_callback(df)

    return df.drop(columns=["tier_rank", "summary_len"], errors="ignore")


def quality_score(row: pd.Series) -> int:
    score = 0
    if row.get("source_tier") == "high":
        score += 4
    elif row.get("source_tier") == "mid":
        score += 2
    if isinstance(row.get("full_text"), str) and len(row["full_text"]) > 1200:
        score += 4
    elif isinstance(row.get("summary"), str) and len(row["summary"]) > 200:
        score += 2
    if row.get("author"):
        score += 1
    if row.get("published_at_utc"):
        score += 1
    return score


def build_summary(df: pd.DataFrame) -> dict:
    by_ticker = {}
    for ticker in sorted(df["ticker"].dropna().unique()):
        sub = df[df["ticker"] == ticker]
        by_ticker[ticker] = {
            "rows": int(len(sub)),
            "rows_with_full_text": int((sub["full_text"].fillna("").str.len() > 300).sum()),
            "high_tier_rows": int((sub["source_tier"] == "high").sum()),
            "top_sources": sub["source"].value_counts().head(12).to_dict(),
        }
    return {
        "total_rows": int(len(df)),
        "rows_with_full_text": int((df["full_text"].fillna("").str.len() > 300).sum()),
        "rows_high_tier": int((df["source_tier"] == "high").sum()),
        "by_ticker": by_ticker,
    }


def write_outputs(df: pd.DataFrame, outdir: Path, date_range: dict, symbols: List[str]) -> None:
    cols = [
        "article_id",
        "ticker",
        "company",
        "published_at_utc",
        "date_utc",
        "headline",
        "source",
        "source_tier",
        "url",
        "author",
        "category",
        "tags",
        "summary",
        "full_text",
        "quality_score",
        "collector_source",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]

    csv_path = outdir / "raw_articles_q4_2025.csv"
    jsonl_path = outdir / "raw_articles_q4_2025.jsonl"
    summary_path = outdir / "collection_summary.json"

    df.to_csv(csv_path, index=False)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    summary = build_summary(df)
    summary["date_range"] = date_range
    summary["symbols"] = symbols
    summary["outputs"] = {
        "csv": str(csv_path),
        "jsonl": str(jsonl_path),
        "summary": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", help="Use an existing CSV as the input article list instead of fetching")
    p.add_argument("--symbols", nargs="+", default=["NVDA", "GOOGL", "TSLA"])
    p.add_argument("--start", default="2025-07-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--outdir", default="q4_article_dump")
    p.add_argument("--use-gdelt", action="store_true", help="Add broader GDELT search results")
    p.add_argument("--no-scrape", action="store_true", help="Skip full text scraping")
    p.add_argument("--fetch-authors", action="store_true", help="Try to fetch author metadata")
    p.add_argument("--per-ticker-cap", type=int, default=None, help="Optional max records kept per ticker before enrichment")
    p.add_argument("--finnhub-window-days", type=int, default=30, help="Sequential Finnhub request window size in days")
    p.add_argument("--gdelt-max-records", type=int, default=250, help="Max records requested per GDELT query")
    p.add_argument("--save-every", type=int, default=25, help="Write checkpoint output every N newly enriched rows")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    if args.input_csv:
        input_csv = Path(args.input_csv)
        if not input_csv.exists():
            raise RuntimeError(f"Input CSV not found: {input_csv}")
        df = pd.read_csv(input_csv)
        date_range = {"start": args.start, "end": args.end}
        if "date_utc" in df.columns and df["date_utc"].notna().any():
            date_range = {
                "start": str(df["date_utc"].dropna().min()),
                "end": str(df["date_utc"].dropna().max()),
            }
        symbols = sorted(df["ticker"].dropna().astype(str).unique().tolist()) if "ticker" in df.columns else args.symbols
    else:
        env_values = load_local_env(Path(".env"))
        api_key = os.getenv("FINNHUB_API_KEY", "").strip() or env_values.get("FINNHUB_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing FINNHUB_API_KEY. Set it in the shell or in .env.")

        rows: List[dict] = []

        for symbol in args.symbols:
            log(f"Collecting Finnhub news for {symbol}")
            rows.extend(fetch_finnhub_company_news(
                session,
                api_key,
                symbol,
                args.start,
                args.end,
                window_days=args.finnhub_window_days,
            ))

        if args.use_gdelt:
            for symbol in args.symbols:
                for query in SYMBOL_CONFIG[symbol]["queries"]:
                    log(f"Collecting GDELT matches for {symbol}: {query}")
                    try:
                        rows.extend(fetch_gdelt_articles(
                            session,
                            symbol,
                            query,
                            args.start,
                            args.end,
                            max_records=args.gdelt_max_records,
                        ))
                    except Exception as e:
                        log(f"GDELT failed for {symbol}: {e}")

        if not rows:
            raise RuntimeError("No articles found.")

        df = pd.DataFrame(rows)

        df = df[~df["url"].fillna("").map(is_blocked_domain)].copy()
        df["headline"] = df["headline"].fillna("").map(clean_text)
        df = df[df["headline"].str.len() > 10].copy()

        df = dedupe_articles(df)
        date_range = {"start": args.start, "end": args.end}
        symbols = args.symbols

    def save_checkpoint(df_to_save: pd.DataFrame) -> None:
        checkpoint_df = df_to_save.copy()
        checkpoint_df["quality_score"] = checkpoint_df.apply(quality_score, axis=1)
        checkpoint_df = checkpoint_df.sort_values(
            ["ticker", "quality_score", "source_tier", "published_at_utc"],
            ascending=[True, False, True, False],
        ).reset_index(drop=True)
        write_outputs(checkpoint_df, outdir, date_range, symbols)
        log(f"Saved checkpoint to {outdir}")

    df = enrich_articles(
        session=session,
        df=df,
        scrape_text=(not args.no_scrape),
        fetch_authors=args.fetch_authors,
        per_ticker_cap=args.per_ticker_cap,
        save_every=args.save_every,
        save_callback=save_checkpoint,
    )

    df["quality_score"] = df.apply(quality_score, axis=1)
    df = df.sort_values(
        ["ticker", "quality_score", "source_tier", "published_at_utc"],
        ascending=[True, False, True, False]
    ).reset_index(drop=True)

    write_outputs(df, outdir, date_range, symbols)
    summary = build_summary(df)
    summary["date_range"] = date_range
    summary["symbols"] = symbols
    summary["outputs"] = {
        "csv": str(outdir / "raw_articles_q4_2025.csv"),
        "jsonl": str(outdir / "raw_articles_q4_2025.jsonl"),
        "summary": str(outdir / "collection_summary.json"),
    }

    log("\nDONE")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
