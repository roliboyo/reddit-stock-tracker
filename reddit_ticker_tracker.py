#!/usr/bin/env python3
"""
Reddit ticker-mention tracker: separates sustained interest from one-day spikes.

SETUP
  pip install praw vaderSentiment
  Create a "script" app at https://www.reddit.com/prefs/apps (check Reddit's current
  API terms; access rules change). Then set:
    export REDDIT_CLIENT_ID=...  REDDIT_CLIENT_SECRET=...  REDDIT_USER_AGENT="isa-tracker/0.1 by u/yourname"

USAGE
  python reddit_ticker_tracker.py collect            # run daily (cron / Task Scheduler)
  python reddit_ticker_tracker.py report             # after ~3+ weeks of data
  python reddit_ticker_tracker.py report --subs UKInvesting ValueInvesting --min 8

Reddit's API only returns recent items, so history builds up from your own daily runs.
Trend labels are not meaningful until you have ~3 weeks of data.
"""
import argparse
import os
import re
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone

# wallstreetbets is included as a hype gauge; filter it out with --subs if you prefer.
SUBREDDITS = ["UKInvesting", "investing", "ValueInvesting", "stocks", "wallstreetbets"]

# Bare uppercase words only count if listed here. $CASHTAGS always count.
# Edit freely (these are examples, not recommendations).
WATCHLIST = {"SGRO", "BBOX", "BLND", "LAND", "IUKP", "LMP", "NVDA", "MSFT", "AAPL", "VWRL", "VUSA"}
BLACKLIST = {"USD", "GBP", "EUR", "ETF", "ISA", "SIPP", "IRA", "CEO", "IPO", "ATH", "YOLO", "DD", "EPS", "PE"}

CASHTAG = re.compile(r"(?<![\w$])\$([A-Za-z]{1,5})\b")
BARE = re.compile(r"\b([A-Z]{2,5})\b")
SPAN = 35  # days analysed: 3 recent weeks + 4-week baseline window overlap


def connect(path):
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS mentions (
            item_id TEXT, ticker TEXT, day TEXT, subreddit TEXT, author TEXT, sentiment REAL,
            PRIMARY KEY (item_id, ticker))"""
    )
    return conn


def extract(text):
    found = {m.upper() for m in CASHTAG.findall(text)} - BLACKLIST
    found |= {m for m in BARE.findall(text) if m in WATCHLIST}
    return found


def collect(conn, limit):
    import praw
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "isa-tracker/0.1"),
    )
    analyzer = SentimentIntensityAnalyzer()
    added = 0
    for name in SUBREDDITS:
        sub = reddit.subreddit(name)
        for item in list(sub.new(limit=limit)) + list(sub.comments(limit=limit)):
            text = f"{getattr(item, 'title', '')} {getattr(item, 'selftext', '')} {getattr(item, 'body', '')}"
            tickers = extract(text)
            if not tickers:
                continue
            day = datetime.fromtimestamp(item.created_utc, tz=timezone.utc).date().isoformat()
            author = str(item.author) if item.author else "[deleted]"
            score = analyzer.polarity_scores(text)["compound"]
            for t in tickers:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO mentions VALUES (?,?,?,?,?,?)",
                    (item.fullname, t, day, name, author, score),
                )
                added += cur.rowcount
    conn.commit()
    print(f"Stored {added} new ticker mentions.")


def label(counts, week_authors, min_mentions):
    wk3, wk2, wk1 = sum(counts[-7:]), sum(counts[-14:-7]), sum(counts[-21:-14])
    baseline = sum(counts[-35:-7]) / 4 or 0.0
    ratio = wk3 / max(baseline, 1.0)
    base = counts[-29:-1]
    z = (counts[-1] - statistics.mean(base)) / max(statistics.stdev(base), 1.0)
    active_days = sum(1 for c in counts[-7:] if c > 0)
    top_day_share = max(counts[-7:]) / wk3 if wk3 else 0

    if wk3 < min_mentions:
        return wk3, baseline, ratio, "low volume"
    if wk1 < wk2 < wk3 and active_days >= 4 and week_authors >= 5:
        return wk3, baseline, ratio, "SUSTAINED RISE"
    if z >= 3 or top_day_share >= 0.6:
        return wk3, baseline, ratio, "SPIKE (treat with caution)"
    if ratio < 0.5:
        return wk3, baseline, ratio, "fading"
    return wk3, baseline, ratio, "steady"


def report(conn, min_mentions, subs):
    today = datetime.now(timezone.utc).date()
    days = [(today - timedelta(days=i)).isoformat() for i in range(SPAN - 1, -1, -1)]
    filt, params = "", [days[0]]
    if subs:
        filt = f" AND subreddit IN ({','.join('?' * len(subs))})"
        params += subs

    first = conn.execute("SELECT MIN(day) FROM mentions").fetchone()[0]
    if first:
        history = (today - datetime.fromisoformat(first).date()).days + 1
        if history < 21:
            print(f"WARNING: only {history} days of history; trend labels aren't reliable until ~21+.\n")

    series = defaultdict(lambda: dict.fromkeys(days, 0))
    for t, d, n in conn.execute(
        f"SELECT ticker, day, COUNT(*) FROM mentions WHERE day >= ?{filt} GROUP BY ticker, day", params
    ):
        series[t][d] = n

    wk = {
        t: (a, s)
        for t, a, s in conn.execute(
            f"SELECT ticker, COUNT(DISTINCT author), AVG(sentiment) FROM mentions WHERE day >= ?{filt} GROUP BY ticker",
            [days[-7]] + params[1:],
        )
    }

    rows = []
    for t, by_day in series.items():
        counts = [by_day[d] for d in days]
        authors, sent = wk.get(t, (0, 0.0))
        wk3, baseline, ratio, tag = label(counts, authors, min_mentions)
        rows.append((ratio, t, wk3, baseline, authors, sent or 0.0, tag))

    print(f"{'TICKER':8}{'7d':>5}{'base/wk':>9}{'ratio':>7}{'authors':>9}{'sent':>7}  LABEL")
    for ratio, t, wk3, base, authors, sent, tag in sorted(rows, reverse=True):
        if tag != "low volume":
            print(f"{t:8}{wk3:>5}{base:>9.1f}{ratio:>7.1f}{authors:>9}{sent:>7.2f}  {tag}")
    print("\nsent = average VADER score (-1 to +1) of whole posts/comments mentioning the ticker.")
    print("Few unique authors or a very high 'ratio' on thin volume usually means hype, not conviction.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="mentions.db")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--limit", type=int, default=500)
    r = sub.add_parser("report")
    r.add_argument("--min", type=int, default=5, help="minimum mentions in last 7 days")
    r.add_argument("--subs", nargs="*", default=None, help="restrict to these subreddits")
    args = p.parse_args()

    conn = connect(args.db)
    if args.cmd == "collect":
        collect(conn, args.limit)
    else:
        report(conn, args.min, args.subs)


if __name__ == "__main__":
    main()
