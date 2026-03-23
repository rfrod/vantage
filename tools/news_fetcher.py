import yfinance as yf
import feedparser
import finnhub
import time
import os
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import List, Dict, Any, Optional


class NewsFetcher:
    """
    Fetches financial news for a given ticker combining multiple sources:
      1. Finnhub API (if FINNHUB_API_KEY env var is set)
      2. yfinance .news property
      3. Yahoo Finance RSS feed (fallback / supplement)

    Features:
    - Configurable Finnhub rate limiter (default 60 calls/min)
    - max_age_days filter: articles older than N days are discarded across all sources
    - Normalised publication timestamps across all sources
    - age_label field on every article (e.g. "2h ago", "3d ago", "STALE")
    """

    # RSS feeds use RFC 2822 date strings; we normalise them to UTC-aware datetimes
    _RSS_DATE_FORMATS = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
    ]

    def __init__(self, finnhub_calls_per_min: int = 60, max_age_days: int = 7):
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
        self.finnhub_client = (
            finnhub.Client(api_key=self.finnhub_api_key) if self.finnhub_api_key else None
        )

        # Rate limiting
        self.calls_per_min = finnhub_calls_per_min
        self.min_interval = 60.0 / self.calls_per_min
        self.last_finnhub_call_time = 0.0

        # Staleness cutoff
        self.max_age_days = max_age_days

    # ── Internal helpers ──────────────────────────────────────────────────

    def _enforce_finnhub_rate_limit(self):
        """Sleep if necessary to stay within the configured Finnhub call rate."""
        now = time.time()
        elapsed = now - self.last_finnhub_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_finnhub_call_time = time.time()

    def _parse_rss_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse an RSS/Atom date string into a UTC-aware datetime.
        Tries email.utils.parsedate_to_datetime first (handles RFC 2822),
        then falls back to manual format parsing.
        """
        if not date_str:
            return None
        try:
            dt = parsedate_to_datetime(date_str)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
        for fmt in self._RSS_DATE_FORMATS:
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue
        return None

    def _age_label(self, pub_dt: Optional[datetime]) -> str:
        """
        Return a human-readable age label relative to now.
        Examples: '15m ago', '4h ago', '2d ago', 'STALE (12d ago)'
        Returns 'unknown age' if the date could not be parsed.
        """
        if pub_dt is None:
            return "unknown age"
        now = datetime.now(timezone.utc)
        delta = now - pub_dt
        total_seconds = int(delta.total_seconds())
        if total_seconds < 0:
            return "just now"
        if total_seconds < 3600:
            label = f"{total_seconds // 60}m ago"
        elif total_seconds < 86400:
            label = f"{total_seconds // 3600}h ago"
        else:
            days = total_seconds // 86400
            label = f"{days}d ago"
            if days > self.max_age_days:
                label = f"STALE ({days}d ago)"
        return label

    def _is_within_age_limit(self, pub_dt: Optional[datetime]) -> bool:
        """Return True if the article is within the max_age_days window."""
        if pub_dt is None:
            # If we cannot determine the date, include the article to avoid silently
            # dropping potentially valid content; the LLM will see 'unknown age'.
            return True
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.max_age_days)
        return pub_dt >= cutoff

    def _build_article(
        self,
        source: str,
        title: str,
        summary: str,
        url: str,
        pub_dt: Optional[datetime],
    ) -> Dict[str, Any]:
        """Build a normalised article dict with all standard fields."""
        published_at = pub_dt.strftime("%Y-%m-%d %H:%M UTC") if pub_dt else "unknown"
        return {
            "source": source,
            "title": title.strip(),
            "summary": summary.strip() if summary else "",
            "url": url,
            "published_at": published_at,
            "pub_dt": pub_dt,          # kept for internal sorting/filtering
            "age_label": self._age_label(pub_dt),
        }

    # ── Source fetchers ───────────────────────────────────────────────────

    def fetch_yfinance_news(self, ticker: str) -> List[Dict[str, Any]]:
        """Fetch recent news using yfinance built-in .news property."""
        try:
            raw_news = yf.Ticker(ticker).news
            articles = []
            for item in raw_news:
                ts = item.get("providerPublishTime")
                pub_dt = (
                    datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
                )
                if not self._is_within_age_limit(pub_dt):
                    continue
                articles.append(
                    self._build_article(
                        source="Yahoo Finance (yfinance)",
                        title=item.get("title", ""),
                        summary=item.get("summary", ""),
                        url=item.get("link", ""),
                        pub_dt=pub_dt,
                    )
                )
            return articles
        except Exception as e:
            print(f"[NewsFetcher] yfinance error for {ticker}: {e}")
            return []

    def fetch_yahoo_rss_news(self, ticker: str) -> List[Dict[str, Any]]:
        """Fetch news from Yahoo Finance RSS feed."""
        try:
            url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
            feed = feedparser.parse(url)
            articles = []
            for entry in feed.entries:
                pub_dt = self._parse_rss_date(entry.get("published", ""))
                if not self._is_within_age_limit(pub_dt):
                    continue
                articles.append(
                    self._build_article(
                        source="Yahoo Finance (RSS)",
                        title=entry.get("title", ""),
                        summary=entry.get("summary", ""),
                        url=entry.get("link", ""),
                        pub_dt=pub_dt,
                    )
                )
            return articles
        except Exception as e:
            print(f"[NewsFetcher] Yahoo RSS error for {ticker}: {e}")
            return []

    def fetch_finnhub_news(self, ticker: str) -> List[Dict[str, Any]]:
        """Fetch structured company news from Finnhub (requires FINNHUB_API_KEY)."""
        if not self.finnhub_client:
            return []
        try:
            self._enforce_finnhub_rate_limit()
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=self.max_age_days)
            raw_news = self.finnhub_client.company_news(
                ticker,
                _from=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
            )
            articles = []
            for item in raw_news:
                ts = item.get("datetime")
                pub_dt = (
                    datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
                )
                # Finnhub already filters by date range, but we double-check
                if not self._is_within_age_limit(pub_dt):
                    continue
                articles.append(
                    self._build_article(
                        source=f"Finnhub ({item.get('source', 'Unknown')})",
                        title=item.get("headline", ""),
                        summary=item.get("summary", ""),
                        url=item.get("url", ""),
                        pub_dt=pub_dt,
                    )
                )
            return articles
        except Exception as e:
            print(f"[NewsFetcher] Finnhub error for {ticker}: {e}")
            return []

    # ── Public API ────────────────────────────────────────────────────────

    def get_consolidated_news(
        self, ticker: str, max_articles: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Fetch, merge, deduplicate, filter by age, and sort news from all sources.
        Returns at most `max_articles` items, newest first.
        Each article contains:
            source, title, summary, url, published_at, age_label
        """
        all_news: List[Dict[str, Any]] = []

        # Priority order: Finnhub > yfinance > RSS
        all_news.extend(self.fetch_finnhub_news(ticker))
        all_news.extend(self.fetch_yfinance_news(ticker))
        all_news.extend(self.fetch_yahoo_rss_news(ticker))

        # Deduplicate by normalised title
        seen: set = set()
        unique: List[Dict[str, Any]] = []
        for article in all_news:
            key = article["title"].lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(article)

        # Sort newest first (articles with unknown date go to the end)
        unique.sort(
            key=lambda a: a["pub_dt"] or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        # Strip internal pub_dt before returning (consumers use published_at / age_label)
        for a in unique:
            a.pop("pub_dt", None)

        return unique[:max_articles]
