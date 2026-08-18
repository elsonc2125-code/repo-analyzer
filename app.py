import os
import re
import json
import base64
import sqlite3
import hashlib
import io
import csv
import webbrowser
from threading import Timer
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify, Response
import requests
import markdown
from google import genai

app = Flask(__name__)
DB_FILE = "database.db"
_analysis_cache = {}

# Gemini API config
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

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
        'control':   ['adjust', 'setting', 'param', 'config', 'option', 'customize', 'tune', 'modify'],
        'detect':    ['recogni', 'find', 'extract', 'identify', 'discover'],
        'convert':   ['transform', 'translat', 'map', 'turn'],
        'generate':  ['create', 'produce', 'build', 'make', 'output'],
        'train':     ['learn', 'fine.tune', 'finetune', 'fit'],
        'real.time': ['realtime', 'live', 'streaming', 'online'],
        'gui':       ['graphical', 'interface', 'tkinter', 'qt', 'gradio', 'streamlit', 'webui'],
        'test':      ['pytest', 'unittest'],
        'deploy':    ['docker', 'container', 'cloud'],
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
    resp = requests.get(url, headers=get_github_headers())
    if resp.status_code == 200:
        data = resp.json()
        content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="ignore")
        return content[:6000]
    return "No README available or repository is private."

def analyze_with_llm(repo_name, description, readme_text, user_goal, query):
    cache_key = get_cache_key(repo_name, user_goal)
    if cache_key in _analysis_cache:
        return _analysis_cache[cache_key]

    terms = extract_goal_terms(user_goal, query)
    key_terms = ", ".join(terms['file_terms'][:12])

    prompt = f"""Evaluate this GitHub repository against the user's goal.
User Goal: "{user_goal}"
Key terms of interest: {key_terms}
Repository Name: "{repo_name}"
Description: "{description}"
README Snippet:
{readme_text[:3000]}

Respond ONLY with valid JSON in this exact structure:
{{
  "goal_match_score": 8,
  "match_summary": "Short explanation of why it matches or fails the goal",
  "key_features": ["Feature 1", "Feature 2"],
  "tech_stack": ["Python", "Flask"],
  "setup_difficulty": "Easy",
  "pros": ["Pro 1", "Pro 2"],
  "cons": ["Con 1", "Con 2"],
  "recommendation": "Short 1-sentence verdict on whether they should use it"
}}"""

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash-latest",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "system_instruction": "You are a software engineer evaluating GitHub repositories. Output raw JSON only. Do not include markdown codeblocks or conversational filler."
            }
        )
        parsed = json.loads(response.text)
        _analysis_cache[cache_key] = parsed
        return parsed
    except Exception as e:
        print(f"Gemini API failed: {e}")

    # Fallback if the API call fails (missing/invalid key, network issue, bad JSON, etc.)
    fallback_result = {
        "goal_match_score": 0,
        "match_summary": "AI analysis failed. Check that GEMINI_API_KEY is set correctly.",
        "key_features": [],
        "tech_stack": [],
        "setup_difficulty": "Unknown",
        "pros": [],
        "cons": [],
        "recommendation": "Manual review required."
    }
    _analysis_cache[cache_key] = fallback_result
    return fallback_result

def compute_maintenance_score(item):
    """0-10 score based on how recently the repo was pushed to. Deterministic, no LLM guessing."""
    pushed_at = item.get("pushed_at") or item.get("updated_at")
    try:
        pushed_dt = datetime.strptime(pushed_at, "%Y-%m-%dT%H:%M:%SZ")
        days = (datetime.utcnow() - pushed_dt).days
    except Exception:
        return 0
    if days <= 30:
        return 10
    if days <= 90:
        return 8
    if days <= 180:
        return 6
    if days <= 365:
        return 4
    if days <= 730:
        return 2
    return 0

def compute_community_score(item):
    """0-10 score on a log scale of stars+forks, so 50k-star repos don't just max out identically to 500-star ones."""
    import math
    stars = item.get("stargazers_count", 0) or 0
    forks = item.get("forks_count", 0) or 0
    score = min(10, math.log10(stars + forks + 1) * 3.3)
    return round(score, 1)

def process_repo(item, goal, query):
    owner = item["owner"]["login"]
    name = item["name"]
    readme_text = fetch_readme(owner, name)
    analysis = analyze_with_llm(item["full_name"], item.get("description") or "", readme_text, goal, query)

    try:
        last_updated = datetime.strptime(item["updated_at"], "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
    except Exception:
        last_updated = item.get("updated_at", "")

    # Composite scoring: blend Gemini's qualitative goal-match with deterministic GitHub signals.
    goal_score = analysis.get("goal_match_score", analysis.get("relevance_score", 0)) or 0
    maintenance_score = compute_maintenance_score(item)
    community_score = compute_community_score(item)
    composite_score = round(0.6 * goal_score + 0.25 * maintenance_score + 0.15 * community_score, 1)

    analysis["goal_match_score"] = goal_score
    analysis["maintenance_score"] = maintenance_score
    analysis["community_score"] = community_score
    analysis["relevance_score"] = composite_score  # used for sorting/badge, kept for frontend compatibility

    return {
        "full_name": item["full_name"],
        "description": item.get("description") or "No description provided.",
        "url": item["html_url"],
        "stars": item.get("stargazers_count", 0),
        "last_updated": last_updated,
        "readme_html": markdown.markdown(readme_text, extensions=["fenced_code", "tables"]),
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
        return "<h1>Flask is running!</h1><p>Your file was chopped in half, but I fixed the startup.</p>"

@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query", "").strip()
    goal = request.form.get("goal", "").strip()
    language = request.form.get("language", "").strip()
    min_stars = request.form.get("min_stars", "0").strip() or "0"
    max_results = request.form.get("max_results", "20").strip() or "20"

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
            params={"q": gh_query, "sort": "stars", "order": "desc", "per_page": max_results_int}
        )
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to reach GitHub: {e}"}), 502

    if resp.status_code != 200:
        return jsonify({"error": f"GitHub API error ({resp.status_code}): {resp.text[:200]}"}), resp.status_code

    items = resp.json().get("items", [])[:max_results_int]

    results = []
    if items:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_repo, item, goal, query) for item in items]
            for f in futures:
                try:
                    results.append(f.result())
                except Exception as e:
                    print(f"Failed to process repo: {e}")

    results.sort(key=lambda r: r["analysis"].get("relevance_score", 0), reverse=True)

    # Save search history
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
        print(f"Failed to save search history: {e}")

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
            repo.get("full_name", ""),
            repo.get("url", ""),
            repo.get("stars", ""),
            analysis.get("relevance_score", ""),
            analysis.get("setup_difficulty", ""),
            ", ".join(analysis.get("tech_stack", []) or []),
            repo.get("last_updated", ""),
            analysis.get("match_summary", ""),
            analysis.get("recommendation", "")
        ])

    csv_data = output.getvalue()
    output.close()

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=github_repos_analysis.csv"}
    )

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

    # POST
    data = request.get_json(silent=True) or {}
    full_name = data.get("full_name")
    if not full_name:
        conn.close()
        return jsonify({"error": "full_name is required"}), 400

    try:
        cursor.execute(
            """INSERT INTO bookmarks (repo_full_name, stars, relevance_score, summary, tech_stack,
               setup_difficulty, notes, url) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                full_name,
                data.get("stars", 0),
                data.get("relevance_score", 0),
                data.get("summary", ""),
                json.dumps(data.get("tech_stack", [])),
                data.get("setup_difficulty", ""),
                data.get("notes", ""),
                data.get("url", "")
            )
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
# Main Execution
# ==========================================

if __name__ == "__main__":
    # Local dev only. In production (Render), gunicorn imports the `app` object directly
    # via the Procfile and never executes this block.
    port = int(os.environ.get("PORT", 5000))
    url = f"http://127.0.0.1:{port}/"

    # Open Brave (or default browser) after Flask boots
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
