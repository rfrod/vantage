import requests
import subprocess
import time
import re
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Lightweight keyword-based sentiment classifier for Reddit posts
# (StockTwits already provides explicit Bullish/Bearish tags)
# ---------------------------------------------------------------------------
_BULLISH_WORDS = {
    "bull", "bullish", "buy", "long", "calls", "moon", "rocket", "breakout",
    "upside", "rally", "squeeze", "undervalued", "cheap", "strong", "beat",
    "upgrade", "outperform", "accumulate", "hold", "growth", "positive",
    "gains", "profit", "winner", "surge", "soar", "explode", "yolo",
}
_BEARISH_WORDS = {
    "bear", "bearish", "sell", "short", "puts", "crash", "dump", "breakdown",
    "downside", "decline", "overvalued", "expensive", "weak", "miss",
    "downgrade", "underperform", "avoid", "cut", "loss", "loser", "drop",
    "fall", "collapse", "bubble", "risk", "danger", "warning", "red",
}


def _keyword_sentiment(text: str) -> str:
    """Return 'Bullish', 'Bearish', or 'Neutral' based on keyword counts."""
    words = set(re.findall(r"\b\w+\b", text.lower()))
    bull_hits = len(words & _BULLISH_WORDS)
    bear_hits = len(words & _BEARISH_WORDS)
    if bull_hits > bear_hits:
        return "Bullish"
    if bear_hits > bull_hits:
        return "Bearish"
    return "Neutral"


class SentimentFetcher:
    """
    Fetches retail sentiment for a given ticker from two free sources:
      1. StockTwits public API  (no key required)
      2. Reddit JSON API        (no key required)

    Configurable parameters:
    - stocktwits_limit (int)  : Max messages to fetch from StockTwits. Default 30.
    - reddit_subreddits (list): Subreddits to search. Default covers the four main
                                investing communities.
    - reddit_posts_per_sub (int): Max posts per subreddit. Default 10.
    - reddit_time_filter (str): Reddit time filter ('day','week','month'). Default 'week'.
    - max_age_hours (int)     : Discard StockTwits messages older than N hours. Default 48.
    - request_timeout (int)   : HTTP timeout in seconds. Default 10.
    """

    # StockTwits blocks Python requests due to TLS fingerprinting; we use curl subprocess.
    _STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json?limit={limit}"
    _REDDIT_SEARCH_URL = (
        "https://www.reddit.com/r/{sub}/search.json"
        "?q={query}&sort={sort}&limit={limit}&t={time}&restrict_sr=1"
    )
    _HEADERS = {"User-Agent": "VantageResearchBot/1.0 (investment research tool)"}

    def __init__(
        self,
        stocktwits_limit: int = 30,
        reddit_subreddits: Optional[List[str]] = None,
        reddit_posts_per_sub: int = 10,
        reddit_time_filter: str = "week",
        max_age_hours: int = 48,
        request_timeout: int = 10,
    ):
        self.stocktwits_limit    = stocktwits_limit
        self.reddit_subreddits   = reddit_subreddits or [
            "stocks", "investing", "wallstreetbets", "options"
        ]
        self.reddit_posts_per_sub = reddit_posts_per_sub
        self.reddit_time_filter  = reddit_time_filter
        self.max_age_hours       = max_age_hours
        self.request_timeout     = request_timeout

    # ── Internal helpers ──────────────────────────────────────────────────

    def _age_label(self, dt: Optional[datetime]) -> str:
        if dt is None:
            return "unknown age"
        delta = datetime.now(timezone.utc) - dt
        s = int(delta.total_seconds())
        if s < 3600:
            return f"{s // 60}m ago"
        if s < 86400:
            return f"{s // 3600}h ago"
        return f"{s // 86400}d ago"

    def _is_within_age(self, dt: Optional[datetime]) -> bool:
        if dt is None:
            return True  # include if we can't determine age
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.max_age_hours)
        return dt >= cutoff

    # ── StockTwits ────────────────────────────────────────────────────────

    def fetch_stocktwits(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch recent messages for a ticker from StockTwits.
        Returns a structured dict with counts, ratios, watchlist count, and samples.
        """
        url = self._STOCKTWITS_URL.format(
            ticker=ticker.upper(), limit=self.stocktwits_limit
        )
        try:
            # StockTwits blocks Python requests via TLS fingerprinting; curl works fine.
            result = subprocess.run(
                ["curl", "-s", "--max-time", str(self.request_timeout), url],
                capture_output=True, text=True, timeout=self.request_timeout + 5
            )
            if result.returncode != 0 or not result.stdout.strip():
                return {"error": "curl returned no data", "source": "StockTwits"}
            data = json.loads(result.stdout)
        except Exception as e:
            return {"error": str(e), "source": "StockTwits"}

        symbol_info = data.get("symbol", {})
        watchlist_count = symbol_info.get("watchlist_count", "N/A")
        messages = data.get("messages", [])

        bullish_msgs, bearish_msgs, neutral_msgs = [], [], []
        for m in messages:
            body = m.get("body", "").strip()
            created_str = m.get("created_at", "")
            try:
                pub_dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except Exception:
                pub_dt = None

            if not self._is_within_age(pub_dt):
                continue

            sentiment_tag = (
                m.get("entities", {}).get("sentiment") or {}
            ).get("basic")  # 'Bullish', 'Bearish', or None

            age = self._age_label(pub_dt)
            entry = {"body": body, "age": age, "tagged": sentiment_tag is not None}

            if sentiment_tag == "Bullish":
                bullish_msgs.append(entry)
            elif sentiment_tag == "Bearish":
                bearish_msgs.append(entry)
            else:
                neutral_msgs.append(entry)

        total = len(bullish_msgs) + len(bearish_msgs) + len(neutral_msgs)
        bull_pct = len(bullish_msgs) / total * 100 if total else 0
        bear_pct = len(bearish_msgs) / total * 100 if total else 0
        tagged_total = len(bullish_msgs) + len(bearish_msgs)
        tagged_bull_pct = len(bullish_msgs) / tagged_total * 100 if tagged_total else 0

        return {
            "source": "StockTwits",
            "watchlist_count": watchlist_count,
            "total_messages": total,
            "tagged_messages": tagged_total,
            "bullish_count": len(bullish_msgs),
            "bearish_count": len(bearish_msgs),
            "neutral_count": len(neutral_msgs),
            "bull_pct_all": bull_pct,
            "bear_pct_all": bear_pct,
            "bull_pct_tagged": tagged_bull_pct,
            "bear_pct_tagged": 100 - tagged_bull_pct if tagged_total else 0,
            "bullish_samples": bullish_msgs[:3],
            "bearish_samples": bearish_msgs[:3],
            "neutral_samples": neutral_msgs[:3],
        }

    # ── Reddit ────────────────────────────────────────────────────────────

    def _fetch_reddit_subreddit(self, ticker: str, subreddit: str, sort: str = "new") -> List[Dict]:
        url = self._REDDIT_SEARCH_URL.format(
            sub=subreddit,
            query=ticker.upper(),
            sort=sort,
            limit=self.reddit_posts_per_sub,
            time=self.reddit_time_filter,
        )
        try:
            resp = requests.get(url, headers=self._HEADERS, timeout=self.request_timeout)
            resp.raise_for_status()
            children = resp.json().get("data", {}).get("children", [])
        except Exception:
            return []

        posts = []
        for child in children:
            p = child.get("data", {})
            title = p.get("title", "").strip()
            body  = p.get("selftext", "").strip()
            # Skip deleted/removed posts
            if body in ("[deleted]", "[removed]", ""):
                body = ""

            created_utc = p.get("created_utc")
            pub_dt = (
                datetime.fromtimestamp(created_utc, tz=timezone.utc)
                if created_utc else None
            )
            if not self._is_within_age(pub_dt):
                continue

            full_text = f"{title} {body}"
            sentiment = _keyword_sentiment(full_text)

            posts.append({
                "subreddit": subreddit,
                "title": title,
                "body": body[:300] if body else "",
                "score": p.get("score", 0),
                "upvote_ratio": p.get("upvote_ratio", 0),
                "num_comments": p.get("num_comments", 0),
                "age": self._age_label(pub_dt),
                "sentiment": sentiment,
                "url": f"https://reddit.com{p.get('permalink', '')}",
            })
        return posts

    def fetch_reddit(self, ticker: str) -> Dict[str, Any]:
        """
        Search configured subreddits for the ticker and aggregate results.
        """
        all_posts: List[Dict] = []
        for sub in self.reddit_subreddits:
            posts = self._fetch_reddit_subreddit(ticker, sub, sort="new")
            all_posts.extend(posts)
            time.sleep(0.5)  # polite delay between subreddit requests

        if not all_posts:
            return {"source": "Reddit", "error": "No posts found", "posts": []}

        # Deduplicate by title
        seen, unique = set(), []
        for p in all_posts:
            key = p["title"].lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(p)

        # Sort by score descending
        unique.sort(key=lambda x: x["score"], reverse=True)

        bull_posts = [p for p in unique if p["sentiment"] == "Bullish"]
        bear_posts = [p for p in unique if p["sentiment"] == "Bearish"]
        neut_posts = [p for p in unique if p["sentiment"] == "Neutral"]

        total = len(unique)
        avg_score = sum(p["score"] for p in unique) / total if total else 0
        avg_upvote_ratio = sum(p["upvote_ratio"] for p in unique) / total if total else 0

        return {
            "source": "Reddit",
            "total_posts": total,
            "subreddits_searched": self.reddit_subreddits,
            "bullish_count": len(bull_posts),
            "bearish_count": len(bear_posts),
            "neutral_count": len(neut_posts),
            "bull_pct": len(bull_posts) / total * 100 if total else 0,
            "bear_pct": len(bear_posts) / total * 100 if total else 0,
            "avg_score": avg_score,
            "avg_upvote_ratio": avg_upvote_ratio,
            "top_posts": unique[:5],
            "bullish_samples": bull_posts[:2],
            "bearish_samples": bear_posts[:2],
        }

    # ── Aggregation & formatting ──────────────────────────────────────────

    def _overall_bias(self, st: dict, rd: dict) -> str:
        """Derive an overall sentiment bias from both sources."""
        signals = []
        if "error" not in st and st.get("tagged_messages", 0) >= 3:
            signals.append(st["bull_pct_tagged"])
        if "error" not in rd and rd.get("total_posts", 0) >= 2:
            signals.append(rd["bull_pct"])

        if not signals:
            return "INSUFFICIENT DATA"
        avg_bull = sum(signals) / len(signals)
        if avg_bull >= 65:
            return "BULLISH"
        if avg_bull <= 35:
            return "BEARISH"
        if 45 <= avg_bull <= 55:
            return "NEUTRAL"
        return "MIXED"

    def _format_stocktwits(self, st: dict, ticker: str) -> str:
        if "error" in st:
            return f"--- STOCKTWITS ---\n  Error: {st['error']}"

        lines = [f"--- STOCKTWITS (${ticker.upper()}) ---"]
        lines.append(f"  Watchlist Count    : {st['watchlist_count']:,}" if isinstance(st['watchlist_count'], int) else f"  Watchlist Count    : {st['watchlist_count']}")
        lines.append(f"  Messages Fetched   : {st['total_messages']}  (tagged: {st['tagged_messages']})")
        lines.append(f"  Bullish Tagged     : {st['bullish_count']}  ({st['bull_pct_tagged']:.1f}% of tagged)")
        lines.append(f"  Bearish Tagged     : {st['bearish_count']}  ({st['bear_pct_tagged']:.1f}% of tagged)")
        lines.append(f"  Untagged           : {st['neutral_count']}")

        if st["bullish_samples"]:
            lines.append("\n  Bullish samples:")
            for s in st["bullish_samples"]:
                lines.append(f"    [{s['age']}] {s['body'][:120]}")
        if st["bearish_samples"]:
            lines.append("\n  Bearish samples:")
            for s in st["bearish_samples"]:
                lines.append(f"    [{s['age']}] {s['body'][:120]}")
        if st["neutral_samples"]:
            lines.append("\n  Untagged samples:")
            for s in st["neutral_samples"]:
                lines.append(f"    [{s['age']}] {s['body'][:120]}")
        return "\n".join(lines)

    def _format_reddit(self, rd: dict, ticker: str) -> str:
        if "error" in rd:
            return f"--- REDDIT ---\n  Error: {rd['error']}"

        lines = [f"--- REDDIT (${ticker.upper()}) ---"]
        lines.append(f"  Subreddits         : {', '.join('r/' + s for s in rd['subreddits_searched'])}")
        lines.append(f"  Posts Found        : {rd['total_posts']}")
        lines.append(f"  Bullish (keyword)  : {rd['bullish_count']}  ({rd['bull_pct']:.1f}%)")
        lines.append(f"  Bearish (keyword)  : {rd['bearish_count']}  ({rd['bear_pct']:.1f}%)")
        lines.append(f"  Neutral            : {rd['neutral_count']}")
        lines.append(f"  Avg Post Score     : {rd['avg_score']:.1f}")
        lines.append(f"  Avg Upvote Ratio   : {rd['avg_upvote_ratio']:.2f}")

        if rd.get("top_posts"):
            lines.append("\n  Top posts by score:")
            for p in rd["top_posts"]:
                lines.append(
                    f"    [{p['age']}] [{p['sentiment']:^8}] r/{p['subreddit']}  "
                    f"score={p['score']}  ratio={p['upvote_ratio']:.2f}  "
                    f"comments={p['num_comments']}"
                )
                lines.append(f"    Title: {p['title'][:110]}")
                if p["body"]:
                    lines.append(f"    Body : {p['body'][:110]}")
        return "\n".join(lines)

    def get_formatted_context(self, ticker: str) -> str:
        """
        Fetch sentiment from all sources and return a single pre-formatted
        string ready to be injected into an LLM prompt.
        """
        st = self.fetch_stocktwits(ticker)
        rd = self.fetch_reddit(ticker)
        bias = self._overall_bias(st, rd)

        meta = (
            f"Ticker: {ticker.upper()} | "
            f"Overall Bias: {bias} | "
            f"Max Age Filter: {self.max_age_hours}h | "
            f"Data as of: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )

        sections = [
            f"=== RETAIL SENTIMENT DATA: {ticker.upper()} ===",
            meta,
            "",
            self._format_stocktwits(st, ticker),
            "",
            self._format_reddit(rd, ticker),
            "",
            f"--- AGGREGATED SIGNAL ---",
            f"  Overall Retail Bias: {bias}",
            (
                f"  StockTwits Bull%: {st.get('bull_pct_tagged', 'N/A'):.1f}%  "
                f"(of {st.get('tagged_messages', 0)} tagged msgs)"
                if "error" not in st else "  StockTwits: unavailable"
            ),
            (
                f"  Reddit Bull%: {rd.get('bull_pct', 'N/A'):.1f}%  "
                f"(of {rd.get('total_posts', 0)} posts, keyword-based)"
                if "error" not in rd else "  Reddit: unavailable"
            ),
            "  NOTE: StockTwits tags are user-declared; Reddit sentiment is keyword-inferred.",
            "        Low tagged-message counts reduce StockTwits signal reliability.",
        ]
        return "\n".join(sections)
