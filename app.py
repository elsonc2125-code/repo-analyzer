import os
import re
import json
import base64
import sqlite3
import hashlib
import io
import csv
import logging
import webbrowser
from threading import Timer
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, render_template, request, jsonify, Response
import requests
import markdown
import bleach
import google.generativeai as genai

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_FILE = "database.db"
_analysis_cache = {}

REQUEST_TIMEOUT = 10  # seconds, for GitHub API calls

# ==========================================
# README sanitization
# ==========================================
ALLOWED_TAGS = ["p", "a", "code", "pre", "ul", "ol", "li", "strong", "em",
                "h1", "h2", "h3", "br", "blockquote"]
ALLOWED_ATTRS = {"a": ["href", "title"]}


def render_readme_html(readme_text):
    raw_html = markdown.markdown(readme_text, extensions=["fenced_code", "tables"])
    return bleach.clean(raw_html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)


# Gemini API config
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-3.6-flash")
else:
    model = None
    logger.warning("GEMINI_API_KEY is not set. AI analysis will be unavailable.")

# ==========================================
# Database Setup
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Bookmarks Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_full_name TEXT UNIQUE NOT NULL,
            stars INTEGER,
            relevance_score INTEGER,
            summary TEXT,
            tech_stack TEXT,
            setup_difficulty TEXT,
            notes TEXT,
            url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Search History Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            goal TEXT,
            language TEXT,
            min_stars INTEGER,
            result_count INTEGER,
            top_repo TEXT,
            searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


init_db()

# ==========================================
# Helper Functions
# ==========================================
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def get_cache_key(repo_name, goal):
    return hashlib.md5(f"{repo_name}:{goal}".encode()).hexdigest()


def extract_goal_terms(goal, query):
    combined = f'{goal} {query}'.lower()
    words = set(re.findall(r'\b\w{3,}\b', combined))
    EXPAND = {
        'control': ['adjust', 'setting', 'param', 'config', 'option', 'customize', 'tune', 'modify'],
        'detect': ['recogni', 'find', 'extract', 'identify', 'discover'],
        'convert': ['transform', 'translat', 'map', 'turn'],
        'generate': ['create', 'produce', 'build', 'make', 'output'],
        'train': ['learn', 'fine.tune', 'finetune', 'fit'],
        'real.time': ['realtime', 'live', 'streaming', 'online'],
        'gui': ['graphical', 'interface', 'tkinter', 'qt', 'gradio', 'streamlit', 'webui'],
        'test': ['pytest', 'unittest'],
        'deploy': ['docker', 'container', 'cloud'],
    }
    expanded = set(words)
    for w in list(words):
        for key, syns in EXPAND.items():
            all_forms = [key] + syns
            if any(w in form or form in w for form in all_forms if len(form) >= 4):
                expanded.update(syns)
    file_terms = sorted([t for t in expanded if len(t) >= 3], key=len, reverse=True)
    return {'raw_words': words, 'expanded': expanded, 'file_terms': file_terms}


def get_github_headers():
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_readme(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    try:
        resp = requests.get(url, headers=get_github_headers(), timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        logger.error(f"Failed to fetch README for {owner}/{repo}: {e}")
        return "No README available or repository is private."

    if resp.status_code == 200:
        data = resp.json()
        content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="ignore")
        return content[:6000]
    return "No README available or repository is private."


# Files worth inspecting for tech stack, dependencies, and setup complexity.
# Kept short and specific so we don't spam the GitHub API per repo.
DEPENDENCY_FILES = [
    "requirements.txt", "package.json", "pyproject.toml", "Pipfile",
    "Dockerfile", "docker-compose.yml", ".env.example", "setup.py"
]


def fetch_file_tree(owner, repo, default_branch):
    """Return a list of file paths at the repo root only. We deliberately avoid
    the recursive tree API here: for large repos it can return thousands of
    entries, which is slow and memory-heavy for no benefit, since we only
    match against known root-level filenames (requirements.txt, Dockerfile, etc)."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/"
    try:
        resp = requests.get(url, headers=get_github_headers(), timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        logger.error(f"Failed to fetch file tree for {owner}/{repo}: {e}")
        return []

    if resp.status_code != 200:
        return []

    try:
        items = resp.json()
        if not isinstance(items, list):
            return []
        return [item["name"] for item in items if item.get("type") == "file"]
    except Exception as e:
        logger.error(f"Failed to parse file tree for {owner}/{repo}: {e}")
        return []


def fetch_file_content(owner, repo, path, max_chars=2000):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    try:
        resp = requests.get(url, headers=get_github_headers(), timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {path} for {owner}/{repo}: {e}")
        return None

    if resp.status_code != 200:
        return None

    data = resp.json()
    if data.get("encoding") != "base64":
        return None

    try:
        content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="ignore")
    except Exception:
        return None

    return content[:max_chars]


def fetch_dependency_files(owner, repo, file_names):
    """Given the repo's root-level file names, fetch the contents of any
    recognized dependency/config files found."""
    found = {}
    file_set = set(file_names)
    for fname in DEPENDENCY_FILES:
        if fname in file_set:
            content = fetch_file_content(owner, repo, fname)
            if content:
                found[fname] = content
    return found


def fetch_repo_metadata(owner, repo):
    """Pull license, contributor count, open issue count, and latest release.
    Each call is independent and best-effort: a failure in one shouldn't block
    the others or crash the whole analysis."""
    metadata = {
        "license": None,
        "contributor_count": None,
        "open_issues": None,
        "latest_release": None,
        "latest_release_date": None,
    }

    try:
        resp = requests.get(f"https://api.github.com/repos/{owner}/{repo}",
                             headers=get_github_headers(), timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            license_info = data.get("license") or {}
            metadata["license"] = license_info.get("spdx_id") or license_info.get("name")
            metadata["open_issues"] = data.get("open_issues_count")
    except requests.RequestException as e:
        logger.error(f"Failed to fetch repo metadata for {owner}/{repo}: {e}")

    try:
        resp = requests.get(f"https://api.github.com/repos/{owner}/{repo}/contributors",
                             headers=get_github_headers(), params={"per_page": 1, "anon": "true"},
                             timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            # GitHub returns pagination info in the Link header; last page number
            # approximates total contributor count without fetching every page.
            link_header = resp.headers.get("Link", "")
            match = re.search(r'page=(\d+)>; rel="last"', link_header)
            if match:
                metadata["contributor_count"] = int(match.group(1))
            else:
                metadata["contributor_count"] = len(resp.json())
    except requests.RequestException as e:
        logger.error(f"Failed to fetch contributors for {owner}/{repo}: {e}")

    try:
        resp = requests.get(f"https://api.github.com/repos/{owner}/{repo}/releases/latest",
                             headers=get_github_headers(), timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            metadata["latest_release"] = data.get("tag_name")
            metadata["latest_release_date"] = data.get("published_at")
    except requests.RequestException as e:
        logger.error(f"Failed to fetch latest release for {owner}/{repo}: {e}")

    return metadata


def build_repo_snapshot(readme_text, dependency_files, metadata):
    """Turn raw dependency file contents + metadata into a compact, structured
    text block for the AI prompt. Keeps each file short so the whole snapshot
    stays within a reasonable prompt size even for repos with several files."""
    lines = []

    lines.append("--- GITHUB METADATA ---")
    lines.append(f"License: {metadata.get('license') or 'None detected'}")
    lines.append(f"Contributors: {metadata.get('contributor_count') if metadata.get('contributor_count') is not None else 'Unknown'}")
    lines.append(f"Open issues: {metadata.get('open_issues') if metadata.get('open_issues') is not None else 'Unknown'}")
    if metadata.get("latest_release"):
        lines.append(f"Latest release: {metadata['latest_release']} ({metadata.get('latest_release_date', 'date unknown')})")
    else:
        lines.append("Latest release: None found")

    if dependency_files:
        lines.append("\n--- DEPENDENCY / CONFIG FILES FOUND ---")
        for fname, content in dependency_files.items():
            lines.append(f"\n[{fname}]")
            lines.append(content[:1200])
    else:
        lines.append("\n--- DEPENDENCY / CONFIG FILES FOUND ---")
        lines.append("None detected (no requirements.txt, package.json, pyproject.toml, Dockerfile, etc. at repo root).")

    lines.append("\n--- README (truncated) ---")
    lines.append(readme_text[:3000])

    return "\n".join(lines)


def analyze_with_llm(repo_name, description, snapshot, user_goal, query):
    cache_key = get_cache_key(repo_name, user_goal)
    if cache_key in _analysis_cache:
        return _analysis_cache[cache_key]

    # Handle missing Gemini key cleanly, before attempting any call
    if model is None:
        fallback_result = {
            "goal_match_score": 0,
            "match_summary": "AI analysis unavailable: GEMINI_API_KEY is not set.",
            "key_features": [],
            "tech_stack": [],
            "setup_difficulty": "Unknown",
            "pros": [],
            "cons": [],
            "recommendation": "Set GEMINI_API_KEY on the server and try again.",
            "use_if": "",
            "avoid_if": ""
        }
        _analysis_cache[cache_key] = fallback_result
        return fallback_result

    terms = extract_goal_terms(user_goal, query)
    key_terms = ", ".join(terms['file_terms'][:12])

    prompt = f"""Evaluate this GitHub repository against the user's goal. You are looking at
more than just the README: you also have the repository's actual dependency
files, license, contributor count, issue count, and release history. Use ALL
of this to judge whether the repo is genuinely a good fit, not just whether
its description sounds relevant. A repo with a great README but no releases,
no dependency files, and one contributor should be treated with more caution
than the summary alone would suggest.

User Goal: "{user_goal}"
Key terms of interest: {key_terms}
Repository Name: "{repo_name}"
Description: "{description}"

{snapshot}

Respond ONLY with valid JSON in this exact structure:
{{
"goal_match_score": 8,
"match_summary": "Short explanation of why it matches or fails the goal, referencing concrete evidence (dependencies, license, activity) where relevant",
"key_features": ["Feature 1", "Feature 2"],
"tech_stack": ["Python", "Flask"],
"setup_difficulty": "Easy",
"pros": ["Pro 1", "Pro 2"],
"cons": ["Con 1", "Con 2"],
"recommendation": "Short 1-sentence verdict on whether they should use it",
"use_if": "One short sentence: use this repo if...",
"avoid_if": "One short sentence: avoid this repo if..."
}}"""

    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        parsed = json.loads(response.text)
        _analysis_cache[cache_key] = parsed
        return parsed
    except Exception as e:
        logger.error(f"Gemini API failed for {repo_name}: {e}")

    # Fallback if the API call fails
    fallback_result = {
        "goal_match_score": 0,
        "match_summary": "AI analysis failed. Please try again shortly.",
        "key_features": [],
        "tech_stack": [],
        "setup_difficulty": "Unknown",
        "pros": [],
        "cons": [],
        "recommendation": "Manual review required.",
        "use_if": "",
        "avoid_if": ""
    }
    _analysis_cache[cache_key] = fallback_result
    return fallback_result


def compute_maintenance_score(item, metadata=None):
    """Blends last-push recency (still the strongest signal) with release
    recency and open-issue count so a repo that was updated once and then
    abandoned doesn't score the same as one under active development."""
    pushed_at = item.get("pushed_at") or item.get("updated_at")
    try:
        pushed_dt = datetime.strptime(pushed_at, "%Y-%m-%dT%H:%M:%SZ")
        days_since_push = (datetime.utcnow() - pushed_dt).days
    except Exception:
        days_since_push = None

    if days_since_push is None:
        push_score = 0
    elif days_since_push <= 30: push_score = 10
    elif days_since_push <= 90: push_score = 8
    elif days_since_push <= 180: push_score = 6
    elif days_since_push <= 365: push_score = 4
    elif days_since_push <= 730: push_score = 2
    else: push_score = 0

    if not metadata:
        return push_score

    # Release recency: a repo that has never released anything, or hasn't in
    # years, is weaker evidence of active maintenance even if commits are recent.
    release_date = metadata.get("latest_release_date")
    if release_date:
        try:
            release_dt = datetime.strptime(release_date, "%Y-%m-%dT%H:%M:%SZ")
            days_since_release = (datetime.utcnow() - release_dt).days
            if days_since_release <= 180: release_score = 10
            elif days_since_release <= 365: release_score = 7
            elif days_since_release <= 730: release_score = 4
            else: release_score = 2
        except Exception:
            release_score = 3
    else:
        release_score = 3  # no releases at all isn't disqualifying, just weaker evidence

    # Open issue count as a rough signal of unaddressed backlog. This is a blunt
    # instrument (a popular repo naturally has more issues) so it's weighted lightly.
    open_issues = metadata.get("open_issues")
    if open_issues is None:
        issue_score = 5
    elif open_issues <= 20: issue_score = 10
    elif open_issues <= 75: issue_score = 7
    elif open_issues <= 200: issue_score = 5
    else: issue_score = 3

    blended = 0.6 * push_score + 0.25 * release_score + 0.15 * issue_score
    return round(blended, 1)


def compute_community_score(item):
    import math
    stars = item.get("stargazers_count", 0) or 0
    forks = item.get("forks_count", 0) or 0
    score = min(10, math.log10(stars + forks + 1) * 3.3)
    return round(score, 1)


def process_repo(item, goal, query):
    owner = item["owner"]["login"]
    name = item["name"]
    default_branch = item.get("default_branch", "main")

    readme_text = fetch_readme(owner, name)
    file_paths = fetch_file_tree(owner, name, default_branch)
    dependency_files = fetch_dependency_files(owner, name, file_paths)
    metadata = fetch_repo_metadata(owner, name)

    snapshot = build_repo_snapshot(readme_text, dependency_files, metadata)
    analysis = analyze_with_llm(item["full_name"], item.get("description") or "", snapshot, goal, query)

    try:
        last_updated = datetime.strptime(item["updated_at"], "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
    except Exception:
        last_updated = item.get("updated_at", "")

    goal_score = analysis.get("goal_match_score", analysis.get("relevance_score", 0)) or 0
    maintenance_score = compute_maintenance_score(item, metadata)
    community_score = compute_community_score(item)
    composite_score = round(0.6 * goal_score + 0.25 * maintenance_score + 0.15 * community_score, 1)

    analysis["goal_match_score"] = goal_score
    analysis["maintenance_score"] = maintenance_score
    analysis["community_score"] = community_score
    analysis["relevance_score"] = composite_score

    return {
        "full_name": item["full_name"],
        "description": item.get("description") or "No description provided.",
        "url": item["html_url"],
        "stars": item.get("stargazers_count", 0),
        "last_updated": last_updated,
        "readme_html": render_readme_html(readme_text),
        "license": metadata.get("license"),
        "contributor_count": metadata.get("contributor_count"),
        "open_issues": metadata.get("open_issues"),
        "latest_release": metadata.get("latest_release"),
        "dependency_files_found": list(dependency_files.keys()),
        "analysis": analysis
    }


# ==========================================
# Flask Routes
# ==========================================
@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception:
        return "<h1>Flask is running!</h1>"


@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query", "").strip()
    goal = request.form.get("goal", "").strip()
    language = request.form.get("language", "").strip()
    min_stars = request.form.get("min_stars", "0").strip() or "0"
    max_results = request.form.get("max_results", "5").strip() or "5"

    if not query:
        return jsonify({"error": "Search query is required."}), 400

    try:
        min_stars_int = int(min_stars)
    except ValueError:
        min_stars_int = 0

    try:
        max_results_int = min(int(max_results), 30)
    except ValueError:
        max_results_int = 20

    gh_query_parts = [query]
    if language:
        gh_query_parts.append(f"language:{language}")
    if min_stars_int > 0:
        gh_query_parts.append(f"stars:>={min_stars_int}")
    gh_query = " ".join(gh_query_parts)

    try:
        resp = requests.get(
            "https://api.github.com/search/repositories",
            headers=get_github_headers(),
            params={"q": gh_query, "sort": "stars", "order": "desc", "per_page": max_results_int},
            timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException as e:
        logger.error(f"Failed to reach GitHub search API: {e}")
        return jsonify({"error": "Failed to reach GitHub. Please try again."}), 502

    # Handle GitHub rate limiting with a friendly message
    if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
        return jsonify({"error": "GitHub search limit reached. Please try again in a few minutes."}), 429

    if resp.status_code != 200:
        logger.error(f"GitHub API error ({resp.status_code}): {resp.text[:200]}")
        return jsonify({"error": "GitHub API returned an error. Please try again."}), resp.status_code

    items = resp.json().get("items", [])[:max_results_int]

    results = []
    if items:
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(process_repo, item, goal, query) for item in items]
            for f in futures:
                try:
                    results.append(f.result())
                except Exception as e:
                    logger.error(f"Failed to process repo: {e}")

    results.sort(key=lambda r: r["analysis"].get("relevance_score", 0), reverse=True)

    try:
        conn = get_db()
        cursor = conn.cursor()
        top_repo = results[0]["full_name"] if results else None
        cursor.execute(
            "INSERT INTO search_history (query, goal, language, min_stars, result_count, top_repo) VALUES (?, ?, ?, ?, ?, ?)",
            (query, goal, language, min_stars_int, len(results), top_repo)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to save search history: {e}")

    return jsonify({"results": results})


@app.route("/compare", methods=["POST"])
def compare():
    repos_data = request.form.get("repos_data", "[]")
    try:
        repos = json.loads(repos_data)
    except json.JSONDecodeError:
        repos = []
    return render_template("compare.html", repos=repos)


@app.route("/export", methods=["POST"])
def export():
    data = request.get_json(silent=True) or {}
    results = data.get("results", [])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Repository", "URL", "Stars", "Relevance Score", "Setup Difficulty",
                      "Tech Stack", "Last Updated", "Match Summary", "Recommendation"])
    for repo in results:
        analysis = repo.get("analysis", {})
        writer.writerow([
            repo.get("full_name", ""), repo.get("url", ""), repo.get("stars", ""),
            analysis.get("relevance_score", ""), analysis.get("setup_difficulty", ""),
            ", ".join(analysis.get("tech_stack", []) or []), repo.get("last_updated", ""),
            analysis.get("match_summary", ""), analysis.get("recommendation", "")
        ])
    csv_data = output.getvalue()
    output.close()

    return Response(csv_data, mimetype="text/csv",
                     headers={"Content-Disposition": "attachment; filename=github_repos_analysis.csv"})


@app.route("/api/bookmarks", methods=["GET", "POST"])
def bookmarks():
    conn = get_db()
    cursor = conn.cursor()

    if request.method == "GET":
        cursor.execute("SELECT * FROM bookmarks ORDER BY created_at DESC")
        rows = [dict(row) for row in cursor.fetchall()]
        for row in rows:
            try:
                row["tech_stack"] = json.loads(row["tech_stack"]) if row["tech_stack"] else []
            except (TypeError, json.JSONDecodeError):
                row["tech_stack"] = []
        conn.close()
        return jsonify(rows)

    data = request.get_json(silent=True) or {}
    full_name = data.get("full_name")
    if not full_name:
        conn.close()
        return jsonify({"error": "full_name is required"}), 400

    try:
        cursor.execute(
            """INSERT INTO bookmarks (repo_full_name, stars, relevance_score, summary, tech_stack,
               setup_difficulty, notes, url) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (full_name, data.get("stars", 0), data.get("relevance_score", 0), data.get("summary", ""),
             json.dumps(data.get("tech_stack", [])), data.get("setup_difficulty", ""),
             data.get("notes", ""), data.get("url", ""))
        )
        conn.commit()
        conn.close()
        return jsonify({"status": "bookmarked"}), 201
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "Repository is already bookmarked."}), 409


@app.route("/api/bookmarks/<path:full_name>/notes", methods=["PUT"])
def update_bookmark_notes(full_name):
    data = request.get_json(silent=True) or {}
    notes = data.get("notes", "")
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE bookmarks SET notes = ? WHERE repo_full_name = ?", (notes, full_name))
    conn.commit()
    conn.close()
    return jsonify({"status": "updated"})


@app.route("/api/bookmarks/<path:full_name>", methods=["DELETE"])
def delete_bookmark(full_name):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM bookmarks WHERE repo_full_name = ?", (full_name,))
    conn.commit()
    conn.close()
    return jsonify({"status": "deleted"})


@app.route("/api/history", methods=["GET"])
def history():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM search_history ORDER BY searched_at DESC LIMIT 10")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(rows)


# ==========================================
# Global error handler: never leak raw exceptions to the client
# ==========================================
@app.errorhandler(Exception)
def handle_unexpected_error(e):
    logger.error(f"Unhandled exception: {e}")
    return jsonify({"error": "Something went wrong. Please try again."}), 500


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    url = f"http://127.0.0.1:{port}/"

    def open_browser():
        try:
            brave_paths = [
                r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
                r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe"
            ]
            for p in brave_paths:
                if os.path.exists(p):
                    webbrowser.register("brave", None, webbrowser.BackgroundBrowser(p))
                    webbrowser.get("brave").open(url)
                    return
        except Exception:
            pass
        webbrowser.open(url)

    Timer(1.5, open_browser).start()
    app.run(host="127.0.0.1", port=port, debug=True)
