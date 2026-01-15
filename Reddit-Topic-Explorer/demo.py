# -----------------------------
# SocialPulse Flask backend
# - Fetches Reddit posts (count mode or time window mode)
# - Runs BERTopic topic modeling + analytics + optional BotBuster filtering
# - Serves dashboard pages + exportable PDF report
# -----------------------------

from flask import Flask, request, jsonify, render_template, abort, make_response, redirect, url_for
from flask_cors import CORS

import json                    # read/write config + topic metadata JSON
from pathlib import Path        # file paths for plots + saved topic metadata
import re                      # input sanitization + URL detection + text normalization
import traceback               # print stack traces for debugging

# Topic modeling + NLP stack
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

# Reddit API calls
import requests
import requests.auth

# Visualization
import plotly.express as px
from plotly.io import to_html 

import os
from datetime import datetime, timezone
from collections import Counter, defaultdict
from dotenv import load_dotenv

# LLM / AI helpers 
from ai_summary import generate_topic_summary, summarize_subreddit, describe_image

# PDF export
from weasyprint import HTML

# BotBuster CSV parsing + execution
import csv
import subprocess

# Sentiment scoring
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Dedup / hashing utilities
import hashlib
from itertools import combinations

# Timezone helper (store-local)
from zoneinfo import ZoneInfo
STORE_TZ = ZoneInfo("America/Chicago")  

# Load .env (Reddit + OpenAI keys, etc.)
load_dotenv()

# ---- Reddit credentials / client configuration ----
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
username = os.getenv("REDDIT_USERNAME")
password = os.getenv("REDDIT_PASSWORD")
user_agent = os.getenv("USER_AGENT")  # required by Reddit API
if not user_agent:
    raise RuntimeError("USER_AGENT environment variable is required by the Reddit API")

# ---- Global defaults for analysis ----
LIMIT = 300  # default post limit in "count" mode if not overridden

# Directory where you write plotly HTML + PNG snapshots + topics JSON
OUT_DIR = Path("static/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- BotBuster integration ----
REPO_ROOT = Path(__file__).resolve().parent  # Reddit-Topic-Explorer folder
BOTBUSTER_BASE = (REPO_ROOT / ".." / "BotBuster-Universe" / "Botbuster").resolve()
BOTBUSTER_ENV = "botbusterEnv"

# BotBuster expects input/output files in a specific folder layout
BOTBUSTER_INPUT = BOTBUSTER_BASE.parent / "test_data" / "test_reddit.json"
BOTBUSTER_OUTPUT = BOTBUSTER_BASE.parent / "test_data" / "test_reddit_bots.json"

# ---- Flask app ----
app = Flask(__name__, static_folder='static')
CORS(app)  # allow frontend on a different origin (dev) to call the API

# VADER sentiment analyzer is lightweight + fast for post-level sentiment
sent_analyzer = SentimentIntensityAnalyzer()

# ---- Settings persistence ----
# DEFAULT_CONFIG is what the Settings page should reset to (and initial state).
# NOTE: current_config.json must be valid JSON (no true comments); use python comments here.
DEFAULT_CONFIG = {
    "limit": 300,              # number of posts to fetch in count mode
    "min_topic_size": 2,        # BERTopic: minimum docs per topic
    "min_df": 5,               # CountVectorizer: minimum doc frequency for terms
    "max_df": 0.9,             # CountVectorizer: max document frequency (filter very common terms)
    "ngram_range": 2,          # you treat as max n-gram (vectorizer uses (1, ngram_range))
    "diversity": 0.3,          # MMR diversity (higher -> more diverse keywords)
    "fetch_mode": "count",     # "count" uses limit; "time" uses interval_start/end
    "interval_start": "",      # datetime-local string 'YYYY-MM-DDTHH:MM' (interpreted as UTC in your code)
    "interval_end": "",        # datetime-local string 'YYYY-MM-DDTHH:MM' (interpreted as UTC in your code)
    "bot_threshold": 0.5,      # BotBuster: flag user as bot if botprobability >= threshold
}

# In-memory config used by /settings + /analyze
CURRENT_CONFIG = DEFAULT_CONFIG.copy()

# Attempt to load previously saved settings from disk
cfg_path = Path('current_config.json')
try:
    if cfg_path.exists():
        with cfg_path.open('r') as f:
            loaded = json.load(f)
            # Only accept known keys (ignore any extra keys like "_comment")
            for k in DEFAULT_CONFIG:
                if k in loaded:
                    CURRENT_CONFIG[k] = loaded[k]
except Exception:
    # Don't crash server if config file is malformed; keep defaults instead
    traceback.print_exc()

def pretty_utc(dt_str):
    """Convert an ISO timestamp into a readable string (still UTC)."""
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y · %I:%M %p UTC")
    except Exception:
        return dt_str

def load_botbuster_results(csv_path, bot_threshold=0.5):
    """
    Parse BotBuster output CSV and split users into humans/bots.

    CSV expected columns:
      userid, humanprobability, botprobability, botornot

    You ignore "botornot" and apply your own threshold on botprobability so users can tune it.
    """
    human_ids = set()
    bot_ids = set()

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # BotBuster user IDs usually come in lowercase-ish; normalize here
            uid = (row.get("userid") or "").strip().lower()
            if not uid:
                continue

            # Be defensive: missing / non-float values become 0.0
            try:
                bot_prob = float(row.get("botprobability") or 0.0)
            except Exception:
                bot_prob = 0.0

            # Decision rule: adjustable threshold
            if bot_prob >= bot_threshold:
                bot_ids.add(uid)
            else:
                human_ids.add(uid)

    return human_ids, bot_ids

def run_botbuster_on_posts(posts):
    """
    Run BotBuster on a list of Reddit post dicts.
    Returns the output CSV path (string) if successful, else None.

    BotBuster expects a JSONL file where each line is a post object.
    """
    if not posts:
        return None

    # Ensure expected folder exists
    BOTBUSTER_INPUT.parent.mkdir(parents=True, exist_ok=True)

    # Write posts as JSON Lines (JSONL)
    with BOTBUSTER_INPUT.open("w", encoding="utf-8") as f:
        for p in posts:
            json.dump(p, f)
            f.write("\n")

    # Execute botbuster_reddit.py inside the correct conda env + repo directory
    try:
        subprocess.run(
            ["conda", "run", "-n", BOTBUSTER_ENV, "python", "botbuster_reddit.py"],
            check=True,
            cwd=str(BOTBUSTER_BASE),
        )
    except subprocess.CalledProcessError as e:
        print("BotBuster failed:", e)
        return None

    # Verify output exists where we expect it
    if not BOTBUSTER_OUTPUT.exists():
        print("BotBuster output not found:", BOTBUSTER_OUTPUT)
        return None

    return str(BOTBUSTER_OUTPUT)

def normalize_reddit_userid(s: str) -> str:
    """Normalize Reddit user fullnames like 't2_xxxxx' -> 'xxxxx' (lowercase)."""
    s = (s or "").strip().lower()
    if s.startswith("t2_"):
        s = s[3:]
    return s

def normalize_reddit_id(s: str) -> str:
    """Normalize Reddit post fullnames like 't3_xxxxx' -> 'xxxxx' (lowercase)."""
    s = (s or "").strip().lower()
    if s.startswith("t3_"):
        s = s[3:]
    return s

def normalize_dtlocal(s: str) -> str:
    """
    Normalize UI datetime strings into datetime-local format: 'YYYY-MM-DDTHH:MM'

    Accepts:
      - '2025-12-14T14:00'
      - '2025-12-14T14:00:00'
      - '2025-12-14T14:00:00Z'
      - '2025-12-14T14:00:00+00:00'

    NOTE: This does *not* apply timezone conversion. It just strips seconds/zone info.
    """
    s = (s or "").strip()
    if not s:
        return ""
    s = s.replace("Z", "").replace("+00:00", "")
    return s[:16]

def parse_dt_utc(s: str) -> float:
    """
    Interpret a datetime-local string as UTC and return epoch seconds.

    IMPORTANT: This assumes the UI sends UTC-ish times (or you intentionally treat inputs as UTC).
    If your UI actually sends America/Chicago local time, you'd want to convert using STORE_TZ.
    """
    s = normalize_dtlocal(s)
    dt = datetime.fromisoformat(s)
    dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()

def sanitize_subreddit(s: str) -> str:
    """
    Sanitize subreddit input:
    - remove leading 'r/' or '/r/'
    - keep only alphanumerics, underscore, hyphen
    """
    s = (s or "").strip()
    s = re.sub(r'^/r/|^r/', '', s, flags=re.IGNORECASE)
    s = re.sub(r'[^A-Za-z0-9_-]', '', s)
    return s

def normalize_url(u: str) -> str:
    """Normalize URLs for 'shared links' comparison: strip whitespace + fragment."""
    if not u:
        return ""
    u = u.strip()
    u = u.split("#", 1)[0]  # drop anchors so same link counts as same
    return u

# ---- Duplicate detection helpers ----
_dup_ws_re = re.compile(r"\s+")
_dup_url_re = re.compile(r"(https?://\S+)", re.IGNORECASE)
_dup_xpost_re = re.compile(r"\b(x[- ]?post(ed)?|crosspost(ed)?|repost(ed)?)\b", re.IGNORECASE)

def normalize_for_dup(text: str) -> str:
    """
    Normalize text for exact/near-duplicate detection.
    - lowercase
    - drop URLs
    - remove common 'xpost' markers
    - remove punctuation
    - collapse whitespace
    """
    if not text:
        return ""
    t = text.lower().strip()
    t = _dup_url_re.sub("", t)
    t = t.replace("&amp;", "&")
    t = _dup_xpost_re.sub("", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = _dup_ws_re.sub(" ", t).strip()
    return t

def sha256_hex(s: str) -> str:
    """Stable hash used for exact duplicate bucketing."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def char_ngrams(s: str, n: int = 5) -> set:
    """
    Character n-grams for near-duplicate detection.
    Useful when titles are nearly identical but with small edits.
    """
    s = s or ""
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}

def jaccard_set(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

def post_text_key(p: dict) -> str:
    """
    Choose which text fields to compare for duplicates:
    - always include title
    - include selftext if present (so identical title but different body won't match)
    """
    title = (p.get("title") or "").strip()
    body = (p.get("selftext") or "").strip()
    if body:
        return f"{title}\n{body}"
    return title

def lightweight_post_view(p: dict, subreddit_fallback: str | None = None) -> dict:
    """
    Downsample post dict to only the fields needed for:
    - duplicate detection
    - cross-subreddit linking in UI
    """
    created = p.get("created_utc")

    # Make permalinks absolute so frontend can link directly
    permalink = p.get("permalink") or ""
    if permalink and not permalink.startswith("http"):
        permalink = "https://www.reddit.com" + permalink

    return {
        "id": p.get("id"),
        "subreddit": (p.get("subreddit") or subreddit_fallback or "").lower(),
        "author": (p.get("author") or "").strip(),
        "created_utc": float(created) if created is not None else None,
        "title": (p.get("title") or "").strip(),
        "permalink": permalink,
        "url": permalink,  
        "text_key": post_text_key(p),
    }

def compute_exact_duplicate_clusters(posts_light: list[dict], min_subreddits: int = 2):
    """
    Exact duplicates:
    - Normalize text_key
    - Hash
    - Group by hash
    - Keep only clusters appearing in >= min_subreddits
    """
    buckets = defaultdict(list)
    for p in posts_light:
        norm = normalize_for_dup(p.get("text_key") or "")
        if not norm:
            continue
        h = sha256_hex(norm)
        buckets[h].append((p, norm))

    clusters = []
    for h, items in buckets.items():
        # Require cross-subreddit presence (coordination signal)
        subs = {it[0].get("subreddit") for it in items if it[0].get("subreddit")}
        if len(subs) < min_subreddits:
            continue

        posts = [it[0] for it in items]
        canonical = items[0][1]  # first normalized text as representative
        clusters.append({
            "cluster_id": f"exact:{h[:16]}",
            "type": "exact",
            "canonical_text": canonical[:280],  # truncate for UI
            "post_count": len(posts),
            "subreddits": sorted(list(subs)),
            "unique_authors": sorted(list({p.get("author") for p in posts if p.get("author")})),
            "posts": sorted(posts, key=lambda x: (x.get("created_utc") or 0.0)),
        })

    return clusters

def compute_near_duplicate_pairs(posts_light: list[dict], threshold: float = 0.80, ngram_n: int = 5):
    """
    Near duplicates:
    - Use character n-grams + Jaccard similarity
    - Block by prefix of normalized text to avoid O(N^2) over all posts
    - Only keep cross-subreddit pairs (since that's your coordination goal)
    """
    prepared = []
    for p in posts_light:
        norm = normalize_for_dup(p.get("text_key") or "")
        if not norm:
            continue
        grams = char_ngrams(norm, n=ngram_n)
        prepared.append((p, norm, grams))

    # Simple blocking by first 16 characters of normalized string
    block = defaultdict(list)
    for p, norm, grams in prepared:
        key = norm[:16]
        block[key].append((p, norm, grams))

    pairs = []
    for _, group in block.items():
        if len(group) < 2:
            continue

        for (p1, _, g1), (p2, _, g2) in combinations(group, 2):
            if (p1.get("subreddit") or "") == (p2.get("subreddit") or ""):
                continue
            s = jaccard_set(g1, g2)
            if s >= threshold:
                pairs.append((p1, p2, float(s)))

    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs

def pairs_to_clusters(pairs: list[tuple], max_clusters: int = 50):
    """
    Convert near-dup pairs into clusters via connected components:
    - Build adjacency list by post id
    - DFS/BFS to get connected components
    """
    adj = defaultdict(set)
    by_id = {}

    for a, b, _score in pairs:
        ida = a.get("id"); idb = b.get("id")
        if not ida or not idb:
            continue
        by_id[ida] = a
        by_id[idb] = b
        adj[ida].add(idb)
        adj[idb].add(ida)

    visited = set()
    clusters = []
    for start in adj.keys():
        if start in visited:
            continue

        # DFS component
        stack = [start]
        comp = []
        visited.add(start)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in visited:
                    visited.add(y)
                    stack.append(y)

        posts = [by_id[i] for i in comp if i in by_id]
        if len(posts) < 2:
            continue

        subs = sorted(list({p.get("subreddit") for p in posts if p.get("subreddit")}))
        clusters.append({
            "cluster_id": f"near:{start}",
            "type": "near",
            "post_count": len(posts),
            "subreddits": subs,
            "unique_authors": sorted(list({p.get("author") for p in posts if p.get("author")})),
            "posts": sorted(posts, key=lambda x: (x.get("created_utc") or 0.0)),
        })

        if len(clusters) >= max_clusters:
            break

    return clusters

def add_user_time_gap_metrics(cluster: dict):
    """
    Add coordination-style metrics:
    - cluster spread (minutes between earliest and latest post)
    - per-author min gap between duplicate posts
    These help rank suspicious coordination bursts.
    """
    posts = cluster.get("posts", [])
    times = [p.get("created_utc") for p in posts if p.get("created_utc") is not None]
    if times:
        tmin, tmax = min(times), max(times)
        cluster["spread_minutes"] = round((tmax - tmin) / 60.0, 2)
    else:
        cluster["spread_minutes"] = None

    by_author = defaultdict(list)
    for p in posts:
        a = p.get("author")
        if not a:
            continue
        if p.get("created_utc") is None:
            continue
        by_author[a].append(p)

    signals = []
    for author, plist in by_author.items():
        plist_sorted = sorted(plist, key=lambda x: x["created_utc"])
        gaps = []
        for i in range(1, len(plist_sorted)):
            gaps.append(plist_sorted[i]["created_utc"] - plist_sorted[i-1]["created_utc"])
        min_gap = min(gaps) / 60.0 if gaps else None

        signals.append({
            "author": author,
            "post_count": len(plist_sorted),
            "min_gap_minutes": round(min_gap, 2) if min_gap is not None else None,
            "posts": [
                {
                    "subreddit": p.get("subreddit"),
                    "created_utc": p.get("created_utc"),
                    "title": p.get("title"),
                    "permalink": p.get("permalink"),
                }
                for p in plist_sorted
            ]
        })

    # sort: smallest min gap (fast reposting) then highest post count
    signals.sort(
        key=lambda d: (
            float("inf") if d["min_gap_minutes"] is None else d["min_gap_minutes"],
            -d["post_count"]
        )
    )
    cluster["user_signals"] = signals

def coordination_score(cluster: dict) -> int:
    """
    Heuristic score for ranking coordination:
    - more subreddits => higher
    - more posts => higher
    - same author posting multiple times + small time gaps => higher
    """
    subs = cluster.get("subreddits") or []
    posts = cluster.get("posts") or []
    user_signals = cluster.get("user_signals") or []

    num_subs = len(subs)
    num_posts = len(posts)

    same_author_cross = any(
        (s.get("post_count", 0) >= 2 and s.get("min_gap_minutes") is not None)
        for s in user_signals
    )

    min_gap = None
    for s in user_signals:
        g = s.get("min_gap_minutes")
        if g is not None:
            min_gap = g if min_gap is None else min(min_gap, g)

    score = 0
    score += 3 * max(0, num_subs - 1)
    score += 2 * max(0, num_posts - 2)
    if same_author_cross:
        score += 4
    if min_gap is not None and min_gap <= 30:
        score += 4
    elif min_gap is not None and min_gap <= 120:
        score += 2

    return int(score)

# -----------------------------
# Settings API
# -----------------------------

@app.route('/settings', methods=['POST'])
def settings_route():
    """
    Accepts JSON config from settings page, validates it, saves it to disk,
    and updates CURRENT_CONFIG (the in-memory runtime config).
    """
    try:
        data = request.get_json() or {}

        # Parse incoming values (fallback to CURRENT_CONFIG)
        limit = int(data.get('limit', CURRENT_CONFIG['limit']))
        min_topic_size = int(data.get('min_topic_size', CURRENT_CONFIG['min_topic_size']))
        min_df = int(data.get('min_df', CURRENT_CONFIG['min_df']))
        max_df = float(data.get('max_df', CURRENT_CONFIG['max_df']))
        ngram_range = int(data.get('ngram_range', CURRENT_CONFIG['ngram_range']))
        diversity = float(data.get('diversity', CURRENT_CONFIG['diversity']))
        bot_threshold = float(data.get('bot_threshold', CURRENT_CONFIG.get('bot_threshold', 0.5)))

        # Validate ranges to prevent crashes / weird model configs
        if not (50 <= limit <= 2000):
            return jsonify({"success": False, "error": "limit out of range (50-2000)"}), 400
        if not (1 <= min_topic_size <= 200):
            return jsonify({"success": False, "error": "min_topic_size out of range"}), 400
        if not (1 <= min_df <= 100):
            return jsonify({"success": False, "error": "min_df out of range"}), 400
        if not (0.01 <= max_df <= 1.0):
            return jsonify({"success": False, "error": "max_df out of range"}), 400
        if ngram_range not in (1, 2):
            return jsonify({"success": False, "error": "ngram_range must be 1 or 2"}), 400
        if not (0.0 <= diversity <= 1.0):
            return jsonify({"success": False, "error": "diversity out of range"}), 400
        if not (0.0 <= bot_threshold <= 1.0):
            return jsonify({"success": False, "error": "bot_threshold out of range (0.0-1.0)"}), 400

        # Mode switch: count-based fetch vs time-window fetch
        fetch_mode = data.get('fetch_mode', CURRENT_CONFIG.get('fetch_mode', 'count'))
        if fetch_mode not in ("count", "time"):
            fetch_mode = "count"

        # Normalize UI datetime fields (trim seconds / zone suffixes)
        interval_start = normalize_dtlocal(data.get('interval_start') or "")
        interval_end   = normalize_dtlocal(data.get('interval_end') or "")

        # If time-window mode, require valid start/end and ensure start < end
        if fetch_mode == "time":
            if not interval_start or not interval_end:
                return jsonify({
                    "success": False,
                    "error": "Time window mode requires BOTH interval_start and interval_end."
                }), 400
            try:
                start_ts = parse_dt_utc(interval_start)
                end_ts = parse_dt_utc(interval_end)

                # Debug prints to verify UI -> UTC conversion assumptions
                print("TIME WINDOW (local):", interval_start, "→", interval_end)
                print("TIME WINDOW (utc):",
                      datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
                      "→",
                      datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat())
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"Could not parse time interval. Use YYYY-MM-DDTHH:MM. Details: {e}"
                }), 400

            if start_ts >= end_ts:
                return jsonify({
                    "success": False,
                    "error": "Time interval is invalid: interval_start must be BEFORE interval_end."
                }), 400

        # Apply new settings to runtime config
        CURRENT_CONFIG.update({
            "limit": limit,
            "min_topic_size": min_topic_size,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": ngram_range,
            "diversity": diversity,
            "fetch_mode": fetch_mode,
            "interval_start": interval_start,
            "interval_end": interval_end,
            "bot_threshold": bot_threshold,
        })

        # Persist to disk so refresh/restart keeps user settings
        with open('current_config.json', 'w') as f:
            json.dump(CURRENT_CONFIG, f)

        return jsonify({"success": True, "config": CURRENT_CONFIG})

    except Exception as e:
        # If anything unexpected happens, return 500 but keep server alive
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/settings/reset', methods=['POST'])
def settings_reset():
    """Reset CURRENT_CONFIG to defaults and overwrite current_config.json."""
    global CURRENT_CONFIG
    CURRENT_CONFIG = DEFAULT_CONFIG.copy()
    with open('current_config.json', 'w') as f:
        json.dump(CURRENT_CONFIG, f)
    return jsonify({"success": True, "config": CURRENT_CONFIG})

# -----------------------------
# Reddit API helpers
# -----------------------------

def get_token():
    """
    Get OAuth token for Reddit API using password grant.
    NOTE: This is okay for personal/demo use but not ideal for production multi-user apps.
    """
    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    data = {"grant_type":"password", "username":username, "password":password}
    headers = {"User-Agent": user_agent}
    try:
        r = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=auth,
            data=data,
            headers=headers,
            timeout=30
        )
        r.raise_for_status()
        return r.json()["access_token"]
    except requests.RequestException:
        traceback.print_exc()
        raise

# Regex used for turning plain URLs into <a> links in the rendered HTML bodies
url_pattern = re.compile(r"(https?://[^\s]+)", re.IGNORECASE)

def linkify(text):
    """Wrap detected URLs in anchor tags for export/topic pages."""
    if not text:
        return text
    return url_pattern.sub(
        r'<a href="\1" target="_blank" rel="noopener" style="color: var(--accent-2); text-decoration: underline;">\1</a>',
        text
    )

def bucket_for_rate(posts_per_week: float) -> str:
    """
    Convert a user's posting rate into a bucket label for UI plots:
      <1, ~1, ~2, 3–4, 5–6, 7–9, 10+ posts/week
    """
    if posts_per_week < 1:
        return "< 1 post/week"
    elif posts_per_week < 1.5:
        return "≈1 post/week"
    elif posts_per_week < 2.5:
        return "≈2 posts/week"
    elif posts_per_week < 4.5:
        return "3–4 posts/week"
    elif posts_per_week < 6.5:
        return "5–6 posts/week"
    elif posts_per_week < 9.5:
        return "7–9 posts/week"
    else:
        return "10+ posts/week (superspreaders)"

def fetch_reddit_docs_timewindow(subreddit: str, time_start: str, time_end: str, hard_cap: int = 5000):
    """
    Best-effort time window fetch:
    - Page through /new (descending)
    - Keep posts in [start_ts, end_ts]
    - Stop once posts get older than start_ts

    LIMITATION:
    /new pagination only goes back so far. For very active subs, you might not reach start_ts.
    """
    token = get_token()
    headers = {"User-Agent": user_agent, "Authorization": f"bearer {token}"}

    start_ts = int(parse_dt_utc(time_start))
    end_ts   = int(parse_dt_utc(time_end))

    docs, posts = [], []
    seen_ids = set()

    after = None
    pages = 0
    max_pages = 50  # safety cap to avoid hammering API / waiting too long
    reached_oldest = None

    while pages < max_pages and len(posts) < hard_cap:
        params = {"limit": 100}
        if after:
            params["after"] = after

        r = requests.get(
            f"https://oauth.reddit.com/r/{subreddit}/new",
            headers=headers,
            params=params,
            timeout=30,
        )
        r.raise_for_status()

        data = r.json().get("data", {})
        children = data.get("children", [])
        after = data.get("after")
        pages += 1

        if not children:
            break

        # Track oldest timestamp reached (useful to warn if we couldn't reach start_ts)
        page_times = [c.get("data", {}).get("created_utc") for c in children]
        page_times = [t for t in page_times if t is not None]
        if page_times:
            reached_oldest = min(page_times)

        for ch in children:
            d = ch.get("data", {})
            post_id = d.get("id")
            if not post_id or post_id in seen_ids:
                continue

            created_utc = d.get("created_utc")
            if created_utc is None:
                continue

            # Too new => ignore, but keep paging because /new is descending overall
            if created_utc > end_ts:
                continue

            # Too old => we can stop early (everything after this page will be older)
            if created_utc < start_ts:
                return docs, posts

            title = (d.get("title") or "").strip()
            selftext = (d.get("selftext") or "").strip()
            text = title + ("\n" + selftext if selftext else "")

            # Keep doc only if title exists (avoid empty docs)
            if title:
                docs.append(text)
                posts.append(d)
                seen_ids.add(post_id)

        if not after:
            break

    # Warn if we never paged back far enough to include the requested start time
    if reached_oldest is not None and reached_oldest > start_ts:
        print(
            "WARNING: Could not page back far enough to reach interval_start. "
            f"Oldest reached={datetime.fromtimestamp(reached_oldest, tz=timezone.utc).isoformat()} "
            f"but start={datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat()}."
        )

    return docs, posts

def fetch_reddit_docs(subreddit: str, total_limit: int | None = 300, time_start: str | None = None, time_end: str | None = None,):
    """
    Count-based fetch:
    - Sample from /hot, /new, /top (week)
    - De-dupe by post ID
    This is not a clean time slice; it's a "representative mix" snapshot.
    """
    token = get_token()
    headers = {"User-Agent": user_agent, "Authorization": f"bearer {token}"}

    docs = []
    posts = []
    seen_ids = set()

    if total_limit is None:
        total_limit = 300

    # Split limit across 3 endpoints; clamp per-request to <=100
    per = min(100, max(10, total_limit // 3))
    endpoints = [
        (f"https://oauth.reddit.com/r/{subreddit}/hot", {"limit": per}),
        (f"https://oauth.reddit.com/r/{subreddit}/new", {"limit": per}),
        (f"https://oauth.reddit.com/r/{subreddit}/top", {"limit": per, "t": "week"}),
    ]

    for url, params in endpoints:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        children = r.json().get("data", {}).get("children", [])
        for ch in children:
            d = ch.get("data", {})
            post_id = d.get("id")
            if not post_id or post_id in seen_ids:
                continue

            # Combine title + body into one doc for BERTopic
            text = f"{d.get('title','')}\n{d.get('selftext','')}".strip()
            if text:
                docs.append(text)
                posts.append(d)
                seen_ids.add(post_id)

    return docs, posts

def fetch_subreddit_info(subreddit: str):
    """Fetch metadata for the subreddit (subscribers + active user count)."""
    token = get_token()
    headers = {"User-Agent": user_agent, "Authorization": f"bearer {token}"}
    r = requests.get(f"https://oauth.reddit.com/r/{subreddit}/about", headers=headers, timeout=30)
    r.raise_for_status()
    d = r.json().get("data", {})
    return {
        "subscribers": d.get("subscribers"),
        "active_user_count": d.get("active_user_count") or d.get("accounts_active")
    }

def extract_image_url(p: dict) -> str | None:
    """
    Try to extract a direct image URL from a Reddit post dict.
    Used only in "with_images" mode (to generate captions).
    """
    # Prefer Reddit preview image if present
    preview = p.get("preview")
    if preview and isinstance(preview, dict):
        images = preview.get("images") or []
        if images:
            src = images[0].get("source", {})
            url = src.get("url")
            if url:
                return url.replace("&amp;", "&")

    # Fallback to dest URL if it looks like an image file
    url = p.get("url_overridden_by_dest") or p.get("url")
    if url and any(url.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp")):
        return url

    return None

def augment_docs_with_image_captions(docs, posts):
    """
    Append LLM-generated image captions to each doc when an image is present.
    This can improve topic clustering for image-heavy subreddits.
    """
    new_docs = []
    for doc, p in zip(docs, posts):
        image_url = extract_image_url(p)
        if image_url:
            caption = describe_image(image_url)  # your OpenAI helper
            if caption:
                doc = f"{doc}\n\n[Image description]: {caption}"
        new_docs.append(doc)
    return new_docs

# -----------------------------
# Main analysis pipeline (BERTopic + analytics + export payload)
# -----------------------------

def analyze_subreddit(subreddit: str, limit: int = LIMIT, filter_bots: bool = False, with_images: bool = False, bot_threshold: float = 0.5):
    """
    Pipeline:
      1) fetch posts/docs (count or time window)
      2) optional BotBuster filtering (remove bot-like authors)
      3) optional image caption augmentation
      4) compute analytics (sentiment, activity, top users, etc.)
      5) run BERTopic and build topic metadata payload
      6) write plot HTML + optional PNG snapshot
      7) return path + analytics + compare_payload
    """
    # Defensive check (prevents path traversal etc.)
    if not re.fullmatch(r"[A-Za-z0-9_-]{2,30}", subreddit):
        raise ValueError("Invalid subreddit name")

    # Read current settings
    cfg = CURRENT_CONFIG
    fetch_mode = cfg.get("fetch_mode", "count")
    cfg_limit = int(cfg.get("limit", limit))

    # Time window strings (datetime-local style)
    interval_start = (cfg.get("interval_start") or "").strip() or None
    interval_end   = (cfg.get("interval_end") or "").strip() or None

    # Pre-compute requested window length (used later for posts/week rates)
    requested_window_days = None
    if fetch_mode == "time" and interval_start and interval_end:
        start_ts = parse_dt_utc(interval_start)
        end_ts   = parse_dt_utc(interval_end)
        requested_window_days = max((end_ts - start_ts) / 86400.0, 1.0)

    # Fetch docs/posts based on mode
    if fetch_mode == "time":
        if not interval_start or not interval_end:
            raise ValueError("Time window mode is enabled, but interval_start/interval_end are missing.")
        docs, posts = fetch_reddit_docs_timewindow(subreddit, time_start=interval_start, time_end=interval_end)
    else:
        docs, posts = fetch_reddit_docs(subreddit, total_limit=cfg_limit)

    # Abort early if nothing fetched (private sub / API issue / etc.)
    if not docs:
        raise ValueError("No documents fetched for that subreddit (it may not exist or is private)")

    # Track how many posts were originally fetched (for reporting)
    original_total_posts = len(posts)
    bots_filtered_out = 0

    # Optional: filter bot-like authors using BotBuster
    if filter_bots:
        if not BOTBUSTER_BASE.exists():
            print("BotBuster not found; skipping bot filtering.")
            filter_bots = False

        print("Bot filter enabled: running BotBuster...")
        csv_path = run_botbuster_on_posts(posts)
        if csv_path:
            _, bot_user_ids = load_botbuster_results(csv_path, bot_threshold=bot_threshold)

            # Normalize BotBuster IDs (it may output bare ids without 't2_')
            bot_user_ids = {normalize_reddit_userid(x) for x in bot_user_ids}

            filtered_docs, filtered_posts = [], []
            for doc, p in zip(docs, posts):
                author_fullname = normalize_reddit_userid(p.get("author_fullname"))
                author_name = (p.get("author") or "").strip().lower()

                # Primary match: author_fullname-based ids
                if author_fullname and author_fullname in bot_user_ids:
                    continue
                # Fallback match: username-based
                if author_name and author_name in bot_user_ids:
                    continue

                filtered_docs.append(doc)
                filtered_posts.append(p)

            bots_filtered_out = original_total_posts - len(filtered_posts)
            docs, posts = filtered_docs, filtered_posts

            print(f"Filtered out {bots_filtered_out} posts from users flagged by BotBuster.")
        else:
            print("BotBuster did not produce a CSV; skipping bot filtering.")

    # BERTopic needs enough docs to form meaningful topics
    if len(docs) < 5:
        raise ValueError(
            f"Too few documents after bot filtering ({len(docs)} left). "
            "Try lowering the bot threshold, disabling bot filtering, or increasing Limit."
        )

    # Optional: add image captions to docs
    if with_images:
        print("Image-caption mode enabled: augmenting documents with image descriptions via OpenAI…")
        docs = augment_docs_with_image_captions(docs, posts)

    # --- Analytics section ---
    # now is used to compute last-day/last-week counts relative to current time
    now = datetime.now(timezone.utc).timestamp()

    total_posts = len(posts)
    posts_last_day = 0
    posts_last_week = 0
    scores = []
    created_times = []

    post_lengths = []
    comment_counts = []

    hour_counts = Counter()  # counts by UTC hour
    dow_counts = Counter()   # counts by weekday (Mon=0)
    top_post = None
    top_score = float("-inf")
    user_post_counts = Counter()

    sent_scores = []
    sent_hist = {"neg": 0, "neu": 0, "pos": 0}

    # Walk over posts once to compute aggregate metrics + store per-post sentiment
    for p in posts:
        created_utc = p.get("created_utc")
        if created_utc is None:
            continue
        created_times.append(created_utc)

        # Score (Reddit "score" is net upvotes; not perfect but useful)
        score = p.get("score", 0) or 0
        scores.append(score)

        # Text length (title + selftext)
        title = p.get("title", "") or ""
        selftext = p.get("selftext", "") or ""
        full_text = f"{title}\n{selftext}".strip()
        post_lengths.append(len(full_text))

        # VADER sentiment per post (compound in [-1, 1])
        vs = sent_analyzer.polarity_scores(full_text)
        compound = vs["compound"]
        sent_scores.append(compound)
        p["_sentiment"] = compound  # store back into post dict for per-topic aggregation

        # Histogram buckets for UI
        if compound > 0.05:
            sent_hist["pos"] += 1
        elif compound < -0.05:
            sent_hist["neg"] += 1
        else:
            sent_hist["neu"] += 1

        # Comments count
        num_comments = p.get("num_comments", 0) or 0
        comment_counts.append(num_comments)

        # Activity over recent windows (relative to now)
        if created_utc >= now - 24 * 3600:
            posts_last_day += 1
        if created_utc >= now - 7 * 24 * 3600:
            posts_last_week += 1

        # UTC hour + weekday distribution
        dt = datetime.fromtimestamp(created_utc, tz=timezone.utc)
        hour_counts[dt.hour] += 1
        dow_counts[dt.weekday()] += 1

        # Track most-upvoted post
        if score > top_score:
            top_score = score
            top_post = p

        # Per-user activity counts (skip deleted + automod)
        author = (p.get("author") or "").strip()
        if author and author not in ("[deleted]", "AutoModerator"):
            user_post_counts[author] += 1

    # Compute averages (guard against empty lists)
    avg_score = sum(scores) / len(scores) if scores else None
    avg_post_length = sum(post_lengths) / len(post_lengths) if post_lengths else None
    avg_comments = sum(comment_counts) / len(comment_counts) if comment_counts else None

    # Determine actual time range covered by fetched posts
    if created_times:
        t_min_ts = min(created_times)
        t_max_ts = max(created_times)
        time_range = {
            "min": datetime.fromtimestamp(t_min_ts, tz=timezone.utc).isoformat(),
            "max": datetime.fromtimestamp(t_max_ts, tz=timezone.utc).isoformat(),
        }
        effective_days = max((t_max_ts - t_min_ts) / 86400.0, 1.0)
    else:
        time_range = None
        effective_days = 7.0

    # Choose the window length used to compute posts/week:
    # - In time mode, base on requested window (what user asked for)
    # - In count mode, use a fixed heuristic (7 days) because it's not a real time slice
    if fetch_mode == "time" and interval_start and interval_end:
        start_ts = parse_dt_utc(interval_start)
        end_ts   = parse_dt_utc(interval_end)
        window_days = max((end_ts - start_ts) / 86400.0, 1.0)
    else:
        window_days = 7.0

    # Peak hour in UTC (you could convert to STORE_TZ later if desired)
    most_active_hour = None
    if hour_counts:
        most_active_hour = max(hour_counts.items(), key=lambda kv: kv[1])[0]

    # Build posting-rate buckets for users
    bucket_counts = Counter()
    user_freq_detail = []
    if user_post_counts:
        for user, count in user_post_counts.items():
            posts_per_week = (count / window_days) * 7.0
            bucket = bucket_for_rate(posts_per_week)
            bucket_counts[bucket] += 1
            user_freq_detail.append({"user": user, "posts": count, "posts_per_week": posts_per_week})

    BUCKET_ORDER = [
        "< 1 post/week",
        "≈1 post/week",
        "≈2 posts/week",
        "3–4 posts/week",
        "5–6 posts/week",
        "7–9 posts/week",
        "10+ posts/week (superspreaders)",
    ]
    user_freq_buckets = [{"bucket": b, "user_count": bucket_counts.get(b, 0)} for b in BUCKET_ORDER]

    # Top "superspreaders" by posting rate (posts/week)
    top_users = sorted(user_freq_detail, key=lambda d: d["posts_per_week"], reverse=True)[:3]
    top_users_payload = [
        {"name": d["user"], "post_count": d["posts"], "posts_per_week": d["posts_per_week"], "url": f"https://www.reddit.com/user/{d['user']}"}
        for d in top_users
    ]

    # Day-of-week distribution (Mon..Sun)
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_list = [{"day": day_names[i], "count": dow_counts.get(i, 0)} for i in range(7)]

    # Most-upvoted post info for UI
    top_post_info = None
    if top_post is not None:
        top_post_info = {
            "title": top_post.get("title", "(no title)"),
            "score": top_score if top_score != float("-inf") else None,
            "url": "https://www.reddit.com" + (top_post.get("permalink") or "")
        }

    # Subreddit metadata (subscribers / active users); ignore failures
    try:
        sub_info = fetch_subreddit_info(subreddit)
    except Exception:
        sub_info = None

    # Custom stopwords to reduce junk tokens in topic names/keywords
    custom_stops = {
        "she","her","hers","he","him","his","they","them","their","it","its",
        "im","you","your","we","our","us","one","two","also","like","just",
        "amp","https","http"
    }

    # Re-read config values (ensure we use CURRENT_CONFIG, not function args)
    cfg = CURRENT_CONFIG
    limit = int(cfg.get('limit', cfg_limit))
    min_topic_size = int(cfg.get('min_topic_size', 2))

    # Safety: clamp min_df relative to corpus size so vectorizer doesn't drop everything
    min_df_cfg = int(cfg.get('min_df', 5))
    min_df = min(min_df_cfg, max(1, len(docs) // 2))

    max_df = float(cfg.get('max_df', 0.9))
    ngram_range = int(cfg.get('ngram_range', 2))
    diversity = float(cfg.get('diversity', 0.3))

    # Build CountVectorizer for BERTopic c-TF-IDF
    vectorizer_model = CountVectorizer(
        stop_words=list(set(CountVectorizer(stop_words="english").get_stop_words()) | custom_stops),
        ngram_range=(1, ngram_range),
        min_df=min_df,
        max_df=max_df
    )

    # Fit vectorizer once to check vocab size (helps diagnose too-aggressive filters)
    try:
        X_tmp = vectorizer_model.fit_transform(docs)
        vocab_size = len(vectorizer_model.get_feature_names_out())
        print("Vectorizer vocab_size =", vocab_size, "X shape =", X_tmp.shape)
    except Exception as e:
        raise ValueError(f"Vectorizer failed (likely too aggressive filters): {e}")

    # If vocab collapses, relax filters to avoid BERTopic crashing
    if vocab_size < 5:
        print("Vocab too small; relaxing vectorizer settings for stability...")
        vectorizer_model = CountVectorizer(
            stop_words=None,
            ngram_range=(1, ngram_range),
            min_df=1,
            max_df=1.0
        )
        X_tmp = vectorizer_model.fit_transform(docs)
        vocab_size = len(vectorizer_model.get_feature_names_out())
        print("Relaxed vocab_size =", vocab_size, "X shape =", X_tmp.shape)
        if vocab_size < 2:
            raise ValueError(
                "Still too few terms after relaxing vectorizer settings. "
                "Try increasing Limit, lowering bot_threshold, or disabling bot filtering."
            )

    # BERTopic c-TF-IDF transformer
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    # Build and train BERTopic model
    topic_model = BERTopic(
        min_topic_size=min_topic_size,
        calculate_probabilities=True,
        embedding_model="all-MiniLM-L6-v2",  # sentence-transformers model
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=[KeyBERTInspired(), MaximalMarginalRelevance(diversity=diversity)],
        top_n_words=10,
    )
    topics, probs = topic_model.fit_transform(docs)

    # Map topic_id -> posts (for examples + sentiment per topic)
    topic_to_posts = defaultdict(list)
    for p, topic_id_i in zip(posts, topics):
        if topic_id_i == -1:
            continue  # -1 is BERTopic "outlier" bucket
        topic_to_posts[int(topic_id_i)].append(p)

    # For each topic, pick a high-score post as an "example link" for UI
    topic_examples = {}
    for t_id, t_posts in topic_to_posts.items():
        if not t_posts:
            continue
        best = max(t_posts, key=lambda p: (p.get("score", 0) or 0))
        topic_examples[int(t_id)] = {
            "title": best.get("title", "(no title)"),
            "url": "https://www.reddit.com" + (best.get("permalink") or ""),
            "score": best.get("score", 0) or 0,
        }

    # Average sentiment per topic (using per-post VADER compound stored earlier)
    topic_sentiment = {}
    for t_id, t_posts in topic_to_posts.items():
        vals = [p.get("_sentiment") for p in t_posts if "_sentiment" in p]
        topic_sentiment[t_id] = (sum(vals) / len(vals)) if vals else None

    # Analytics payload returned to frontend and used in export
    analytics = {
        "total_posts": total_posts,
        "original_total_posts": original_total_posts,
        "bots_filtered_out": bots_filtered_out,
        "bot_filter_active": filter_bots,
        "image_augmentation_active": with_images,

        "posts_last_day": posts_last_day,
        "posts_last_week": posts_last_week,
        "avg_score": avg_score,
        "avg_post_length": avg_post_length,
        "avg_comments": avg_comments,
        "time_range": time_range,
        "most_active_hour_utc": most_active_hour,

        "dow_counts": dow_list,
        "top_post": top_post_info,
        "subreddit_info": sub_info,

        "user_freq_buckets": user_freq_buckets,
        "user_posts_per_week": [d["posts_per_week"] for d in user_freq_detail],
        "top_users": top_users_payload,

        "sentiment_hist": sent_hist,
        "sentiment_per_topic": topic_sentiment,

        "bot_threshold": bot_threshold,
    }

    # BERTopic topic info table; drop outlier row Topic=-1
    info = topic_model.get_topic_info()
    valid = info[info["Topic"] != -1]

    # Build rich topic metadata for topic_details page + export
    topic_meta = []
    for _, row in valid.iterrows():
        topic_id = int(row["Topic"])
        topic_name = row["Name"]

        words_scores = topic_model.get_topic(topic_id) or []
        keywords = [w for (w, _) in words_scores]

        # Representative docs from BERTopic (strings of doc text)
        rep_texts = topic_model.get_representative_docs(topic_id)[:3]
        rep_items = []

        # Try to map representative doc strings back to actual posts by matching title line
        for txt in rep_texts:
            title_line, body_text = (txt.split("\n", 1) + [""])[:2]
            title_line = title_line.strip()
            title_key = title_line.lower()

            match = next(
                (p for p in posts if title_key == (p.get("title", "").strip().lower())),
                None
            )

            if match:
                body_raw = (match.get("selftext") or "")
            else:
                body_raw = body_text or ""

            # Clean up body formatting for HTML display
            body_raw = re.sub(r'^\s*\n+', '', body_raw).lstrip()
            clean_body = linkify(body_raw)

            rep_items.append({
                "title": (match.get("title", "(no title)") if match else (title_line or (txt[:80] + "..."))),
                "url": ("https://www.reddit.com" + (match.get("permalink") or "") if match else None),
                "body": clean_body,
                "score": (match.get("score", 0) or 0) if match else 0,
                "num_comments": (match.get("num_comments", 0) or 0) if match else 0,
                "image_url": extract_image_url(match) if match else None,
            })

        # Top 3 posts by score within topic
        topic_posts = topic_to_posts.get(topic_id, [])
        top_by_score = sorted(topic_posts, key=lambda p: (p.get("score", 0) or 0), reverse=True)[:3]
        rep_top_score = []
        for p in top_by_score:
            body_raw = (p.get("selftext") or "")
            body_raw = re.sub(r'^\s*\n+', '', body_raw).lstrip()
            rep_top_score.append({
                "title": p.get("title", "(no title)"),
                "url": "https://www.reddit.com" + (p.get("permalink") or ""),
                "body": linkify(body_raw),
                "score": p.get("score", 0) or 0,
                "num_comments": p.get("num_comments", 0) or 0,
                "image_url": extract_image_url(p),
            })

        # Top 3 posts by comment count within topic
        top_by_comments = sorted(topic_posts, key=lambda p: (p.get("num_comments", 0) or 0), reverse=True)[:3]
        rep_top_comments = []
        for p in top_by_comments:
            body_raw = (p.get("selftext") or "")
            body_raw = re.sub(r'^\s*\n+', '', body_raw).lstrip()
            rep_top_comments.append({
                "title": p.get("title", "(no title)"),
                "url": "https://www.reddit.com" + (p.get("permalink") or ""),
                "body": linkify(body_raw),
                "score": p.get("score", 0) or 0,
                "num_comments": p.get("num_comments", 0) or 0,
                "image_url": extract_image_url(p),
            })

        topic_meta.append({
            "topic_id": topic_id,
            "topic_name": topic_name,
            "keywords": keywords,
            "rep_docs": rep_items,
            "rep_top_score": rep_top_score,
            "rep_top_comments": rep_top_comments,
            "avg_sentiment": topic_sentiment.get(topic_id),
        })

    # Top posts overall (by score) for subreddit-level summary
    top_posts_sorted = sorted(posts, key=lambda p: p.get("score", 0) or 0, reverse=True)
    top_posts_for_summary = top_posts_sorted[:3]

    analytics["top_posts"] = [
        {"title": p.get("title", "(no title)"),
         "score": p.get("score", 0) or 0,
         "url": "https://www.reddit.com" + (p.get("permalink") or "")}
        for p in top_posts_sorted[:3]
    ]

    # Generate high-level natural language summary (LLM)
    try:
        subreddit_summary = summarize_subreddit(subreddit=subreddit, topics=topic_meta, top_posts=top_posts_for_summary)
    except Exception as e:
        print("subreddit summary error:", e)
        subreddit_summary = "Subreddit overview unavailable (generation error)."

    analytics["subreddit_summary"] = subreddit_summary

    # Persist topics + summary to JSON for topic_details + export route
    meta_path = OUT_DIR / f"{subreddit}_topics.json"
    with meta_path.open('w', encoding='utf-8') as f:
        json.dump(
            {"subreddit": subreddit, "topics": topic_meta, "subreddit_summary": subreddit_summary},
            f,
            ensure_ascii=False,
            indent=2
        )

    # Compare payload fields used in cross-subreddit comparison
    users_map = dict(user_post_counts)

    urls = []
    for p in posts:
        u = p.get("url_overridden_by_dest") or p.get("url") or ""
        u = normalize_url(u)
        if u:
            urls.append(u)

    topics_light = []
    for t in topic_meta:
        tid = int(t["topic_id"])
        topics_light.append({
            "topic_id": tid,
            "topic_name": t.get("topic_name"),
            "keywords": (t.get("keywords", [])[:10]),
            "avg_sentiment": t.get("avg_sentiment"),
            "example_post": topic_examples.get(tid),
        })

    posts_light = [lightweight_post_view(p, subreddit_fallback=subreddit) for p in posts]

    compare_payload = {
        "subreddit": subreddit,
        "users": users_map,
        "urls": urls,
        "topics": topics_light,
        "posts": posts_light,
    }

    # Build Plotly bar chart for top 20 topics
    top20 = valid.nlargest(20, "Count").copy()
    fig_bar = px.bar(
        top20,
        x="Name",
        y="Count",
        color="Count",
        title=f"Top Topics in r/{subreddit}",
        custom_data=["Topic"],  # used for click navigation
    )
    fig_bar.update_layout(
        xaxis_title="Topic",
        yaxis_title="Count",
        template="plotly_dark",
        title_x=0.5,
    )
    fig_bar.update_traces(
        hovertemplate="Topic %{customdata}: %{x}<br>Count=%{y}<extra></extra>"
    )

    # Save chart HTML to static folder so frontend can iframe it
    out_path = OUT_DIR / f"{subreddit}_top20.html"
    fig_bar.write_html(out_path, full_html=True, include_plotlyjs=True)

    # Save PNG snapshot for PDF export (kaleido required)
    png_path = OUT_DIR / f"{subreddit}_top20.png"
    try:
        fig_bar.write_image(png_path, width=900, height=500, scale=2)
    except Exception as e:
        print("Could not write PNG chart image:", e)

    # Add click handler: clicking a bar navigates to /topic/<subreddit>/<topicId>
    extra_js = f"""
    <script>
    window.addEventListener('load', function() {{
      var plot = document.querySelector('div.plotly-graph-div');
      if (!plot || !plot.on) return;

      plot.on('plotly_click', function(data) {{
        if (!data || !data.points || !data.points.length) return;
        var p = data.points[0];
        var topicId = p.customdata;
        if (topicId === undefined || topicId === null) return;

        var sr = "{subreddit}";
        var url = "/topic/" + encodeURIComponent(sr) + "/" + topicId;

        // handle iframe embedding
        if (window.parent && window.parent !== window) {{
          window.parent.location.href = url;
        }} else {{
          window.location.href = url;
        }}
      }});
    }});
    </script>
    """
    with out_path.open('a', encoding='utf-8') as f:
        f.write(extra_js)

    return out_path, analytics, compare_payload

# -----------------------------
# Cross-subreddit comparison helpers
# -----------------------------

def jaccard(a: set, b: set) -> float:
    """Standard Jaccard similarity."""
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

def topic_keyword_similarity(t1: dict, t2: dict) -> float:
    """Keyword-level similarity between two topics via Jaccard."""
    k1 = set([x.lower() for x in (t1.get("keywords") or [])])
    k2 = set([x.lower() for x in (t2.get("keywords") or [])])
    return jaccard(k1, k2)

def compare_subreddits(compare_payloads: list[dict]) -> dict:
    """
    Cross-subreddit comparison:
      - overlapping users (intersection + top 25)
      - shared links (intersection of normalized URLs)
      - topic similarity (best-match by keyword Jaccard)
      - duplicates (exact + near clusters ranked by coordination_score)
    """
    subs = [p["subreddit"] for p in compare_payloads]

    # Overlapping users: intersection and union for Jaccard
    user_sets = [set(p.get("users", {}).keys()) for p in compare_payloads]
    inter_users = set.intersection(*user_sets) if user_sets else set()
    union_users = set.union(*user_sets) if user_sets else set()

    # Rank overlapping users by total activity across selected subreddits
    overlap_user_rows = []
    for u in inter_users:
        per_sub_counts = {}
        total = 0
        for p in compare_payloads:
            c = int(p.get("users", {}).get(u, 0) or 0)
            per_sub_counts[p["subreddit"]] = c
            total += c
        overlap_user_rows.append({
            "user": u,
            "total_posts_across_selected": total,
            "by_subreddit": per_sub_counts,
            "url": f"https://www.reddit.com/user/{u}",
        })
    overlap_user_rows.sort(key=lambda r: r["total_posts_across_selected"], reverse=True)
    top_overlapping_users = overlap_user_rows[:25]

    # Shared links: intersection of normalized URL sets
    url_sets = [set([u for u in p.get("urls", []) if u]) for p in compare_payloads]
    shared_urls = set.intersection(*url_sets) if url_sets else set()

    # Sort shared URLs by total mentions across all subs (frequency)
    url_freq = Counter()
    for p in compare_payloads:
        url_freq.update([u for u in p.get("urls", []) if u])

    shared_urls_ranked = sorted(
        [{"url": u, "total_mentions": int(url_freq.get(u, 0))} for u in shared_urls],
        key=lambda x: x["total_mentions"],
        reverse=True
    )[:50]

    # Topic similarity: for each topic in A, find best match in each other subreddit
    topic_matches = []
    if len(compare_payloads) >= 2:
        A = compare_payloads[0]
        for B in compare_payloads[1:]:
            for tA in A.get("topics", []):
                best = None
                best_score = -1.0
                for tB in B.get("topics", []):
                    s = topic_keyword_similarity(tA, tB)
                    if s > best_score:
                        best_score = s
                        best = tB
                if best and best_score > 0:
                    topic_matches.append({
                        "sub_a": A["subreddit"],
                        "topic_a": tA.get("topic_name"),
                        "topic_a_id": tA.get("topic_id"),
                        "sub_b": B["subreddit"],
                        "topic_b": best.get("topic_name"),
                        "topic_b_id": best.get("topic_id"),
                        "keyword_jaccard": round(best_score, 3),
                        "keywords_a": tA.get("keywords", [])[:10],
                        "keywords_b": best.get("keywords", [])[:10],
                        "example_post_a": tA.get("example_post"),
                        "example_post_b": best.get("example_post"),
                    })
    topic_matches.sort(key=lambda r: r["keyword_jaccard"], reverse=True)
    topic_matches = topic_matches[:50]

    # Gather all lightweight posts for duplicate detection
    all_posts = []
    for p in compare_payloads:
        all_posts.extend(p.get("posts", []) or [])

    # Exact duplicate clusters
    exact_clusters = compute_exact_duplicate_clusters(all_posts, min_subreddits=2)
    for c in exact_clusters:
        add_user_time_gap_metrics(c)
        c["score"] = coordination_score(c)
    exact_clusters.sort(key=lambda c: c.get("score", 0), reverse=True)
    exact_clusters = exact_clusters[:50]

    # Near duplicate clusters
    near_pairs = compute_near_duplicate_pairs(all_posts, threshold=0.80, ngram_n=5)
    near_clusters = pairs_to_clusters(near_pairs, max_clusters=50)
    for c in near_clusters:
        add_user_time_gap_metrics(c)
        c["score"] = coordination_score(c)
    near_clusters.sort(key=lambda c: c.get("score", 0), reverse=True)
    near_clusters = near_clusters[:50]

    return {
        "subreddits": subs,
        "users": {
            "intersection_count": len(inter_users),
            "union_count": len(union_users),
            "jaccard": round(jaccard(inter_users, union_users), 4),
            "top_overlapping_users": top_overlapping_users,
        },
        "shared_links": {
            "shared_url_count": len(shared_urls),
            "top_shared_urls": shared_urls_ranked,
        },
        "topic_similarity": {
            "top_matches": topic_matches,
        },
        "duplicates": {
            "exact_clusters": exact_clusters,
            "near_clusters": near_clusters,
        }
    }

# -----------------------------
# Page routes / API routes
# -----------------------------

@app.route('/settings', methods=['GET'])
def settings():
    """Convenience redirect so /settings loads the static settings page."""
    return redirect(url_for('settings_page'))

@app.route('/settings/page', methods=['GET'])
def settings_page():
    """Serves the static settings.html UI page."""
    return app.send_static_file('settings.html')

@app.route('/settings/config', methods=['GET'])
def settings_config():
    """Frontend calls this to pre-fill the Settings UI with CURRENT_CONFIG."""
    return jsonify({"success": True, "config": CURRENT_CONFIG})

@app.route('/')
def index():
    """Main landing page (demo UI)."""
    return app.send_static_file('demo.html')

@app.route('/analyze', methods=['POST'])
def analyze_route():
    """
    Primary analysis endpoint.
    Accepts either:
      {subreddit: "python"}  OR  {subreddits: ["python","learnpython"]}

    Also accepts:
      filter_bots: bool
      with_images: bool
      bot_threshold: float (override CURRENT_CONFIG default)
    """
    try:
        data = request.get_json() or {}

        # Accept either a single subreddit string or a list
        subreddits = data.get("subreddits", None)
        if subreddits is None:
            single = sanitize_subreddit(data.get("subreddit", ""))
            if not single:
                return jsonify({"success": False, "error": "No subreddit provided"}), 400
            subreddits = [single]
        else:
            if not isinstance(subreddits, list) or not subreddits:
                return jsonify({"success": False, "error": "subreddits must be a non-empty list"}), 400
            subreddits = [sanitize_subreddit(s) for s in subreddits]
            subreddits = [s for s in subreddits if s]
            if not subreddits:
                return jsonify({"success": False, "error": "No valid subreddits provided"}), 400

        # Per-request knobs (do not persist unless user uses /settings)
        filter_bots = bool(data.get('filter_bots', False))
        with_images = bool(data.get('with_images', False))

        # Allow client to override bot_threshold for this run
        default_bt = float(CURRENT_CONFIG.get("bot_threshold", 0.5))
        bot_threshold = float(data.get("bot_threshold", default_bt))
        bot_threshold = max(0.0, min(1.0, bot_threshold))

        per_sub = []
        compare_payloads = []

        # Run analysis for each subreddit independently (then compare if 2+)
        for sr in subreddits:
            try:
                out_path, analytics, compare_payload = analyze_subreddit(
                    sr,
                    limit=LIMIT,
                    filter_bots=filter_bots,
                    with_images=with_images,
                    bot_threshold=bot_threshold,
                )
                rel_url = f"/static/plots/{out_path.name}"

                per_sub.append({"subreddit": sr, "url": rel_url, "analytics": analytics, "error": None})
                compare_payloads.append(compare_payload)
            except Exception as e:
                traceback.print_exc()
                per_sub.append({"subreddit": sr, "url": None, "analytics": None, "error": str(e)})

        # Cross-subreddit comparison only if 2+ succeeded payloads
        comparison = None
        if len(compare_payloads) >= 2:
            comparison = compare_subreddits(compare_payloads)

        # Backward compatible: single-sub response includes top-level url/analytics
        if len(per_sub) == 1:
            return jsonify({
                "success": True,
                "url": per_sub[0]["url"],
                "analytics": per_sub[0]["analytics"],
                "per_sub": per_sub,
                "comparison": comparison,
            })

        return jsonify({"success": True, "per_sub": per_sub, "comparison": comparison})

    except requests.HTTPError as e:
        return jsonify({"success": False, "error": f"Reddit API error: {str(e)}"}), 500
    except ValueError as e:
        print("ANALYZE 400:", str(e))
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"success": False, "error": "Internal server error"}), 500

@app.route('/topic/<subreddit>/<int:topic_id>')
def topic_details(subreddit, topic_id):
    """
    Topic details page:
    - reads {subreddit}_topics.json saved during analysis
    - generates a short LLM summary for the selected topic
    - renders topic_details.html
    """
    meta_path = OUT_DIR / f"{subreddit}_topics.json"
    if not meta_path.exists():
        abort(404)

    with meta_path.open('r', encoding='utf-8') as f:
        meta = json.load(f)

    topics = meta.get("topics", [])
    topic = next((t for t in topics if int(t["topic_id"]) == int(topic_id)), None)
    if topic is None:
        abort(404)

    keywords = topic.get("keywords", [])
    rep_docs = topic.get("rep_docs", [])
    rep_top_score = topic.get("rep_top_score", [])
    rep_top_comments = topic.get("rep_top_comments", [])
    avg_sentiment = topic.get("avg_sentiment")

    try:
        topic_summary = generate_topic_summary(keywords=keywords, rep_docs=rep_docs, subreddit=subreddit)
    except Exception as e:
        print("summary error:", e)
        topic_summary = "Summary unavailable (generation error)."

    return render_template(
        "topic_details.html",
        subreddit=subreddit,
        topic_id=topic_id,
        topic_name=topic.get("topic_name"),
        keywords=keywords,
        summary=topic_summary,
        avg_sentiment=avg_sentiment,
        rep_docs=rep_docs,
        rep_top_score=rep_top_score,
        rep_top_comments=rep_top_comments,
    )

@app.route("/export", methods=["GET"])
def export_report():
    """
    PDF export endpoint. Runs analysis and renders ONE export template.
    Supports:
      /export?subs=python
      /export?subs=python,learnpython
      /export?subs=python&subs=learnpython
    """
    # Parse 'subs' query parameter(s) into a list
    subs_params = request.args.getlist("subs")
    subs = []

    if subs_params:
        for raw in subs_params:
            parts = re.split(r"[\s,]+", (raw or "").strip())
            subs.extend([p for p in parts if p])
    else:
        raw = request.args.get("subs", "")
        parts = re.split(r"[\s,]+", (raw or "").strip())
        subs.extend([p for p in parts if p])

    # Sanitize and de-dupe
    cleaned = []
    seen = set()
    for s in subs:
        sr = sanitize_subreddit(s)
        if sr and sr not in seen:
            cleaned.append(sr)
            seen.add(sr)

    if not cleaned:
        abort(400, description="No subreddits provided. Use /export?subs=python,learnpython")

    per_sub = []
    compare_payloads = []

    # Run analysis for each subreddit and build export payload objects
    for sr in cleaned:
        try:
            out_path, analytics, compare_payload = analyze_subreddit(sr, limit=LIMIT)
            compare_payloads.append(compare_payload)

            # Add pretty time range strings for the PDF
            if analytics.get("time_range"):
                tr = analytics["time_range"]
                if tr.get("min"):
                    tr["min_pretty"] = pretty_utc(tr["min"])
                if tr.get("max"):
                    tr["max_pretty"] = pretty_utc(tr["max"])

            # Load saved topic metadata (created by analyze_subreddit)
            meta_path = OUT_DIR / f"{sr}_topics.json"
            topics = []
            subreddit_summary = None
            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                topics = meta.get("topics", [])
                subreddit_summary = meta.get("subreddit_summary")

            # Pick top topics for export (simple heuristic: most rep_docs)
            def topic_size(t):
                return len(t.get("rep_docs", []))

            top_topics = sorted(topics, key=topic_size, reverse=True)[:10]

            # PNG snapshot path for chart if available
            png_name = f"{sr}_top20.png"
            png_path = OUT_DIR / png_name
            chart_image_url = f"/static/plots/{png_name}" if png_path.exists() else None

            per_sub.append({
                "subreddit": sr,
                "analytics": analytics,
                "top_topics": top_topics,
                "subreddit_summary": subreddit_summary,
                "chart_image_url": chart_image_url,
                "error": None,
            })
        except Exception as e:
            traceback.print_exc()
            per_sub.append({
                "subreddit": sr,
                "analytics": None,
                "top_topics": [],
                "subreddit_summary": None,
                "chart_image_url": None,
                "error": str(e),
            })

    # Cross-subreddit comparison only if 2+ subs
    comparison = compare_subreddits(compare_payloads) if len(compare_payloads) >= 2 else None
    config = CURRENT_CONFIG.copy()

    # Render HTML template and convert to PDF via WeasyPrint
    html = render_template(
        "export_report.html",
        subreddits=cleaned,
        comparison=comparison,
        per_sub=per_sub,
        config=config,
    )

    pdf = HTML(string=html, base_url=request.url_root).write_pdf()
    filename = f"socialpulse_report_{'_'.join(cleaned[:5])}.pdf"

    response = make_response(pdf)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response

@app.route('/_ping', methods=['GET'])
def ping():
    """Health check endpoint for deployments."""
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # use_reloader=False avoids running analysis twice due to Flask reloader forking
    app.run(debug=True, port=5000, use_reloader=False)
