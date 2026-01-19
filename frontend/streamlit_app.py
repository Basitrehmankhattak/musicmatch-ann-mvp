import streamlit as st
import requests
import pandas as pd
from datetime import datetime

import os

# Prefer env var, fall back to localhost.
# In deployment (Streamlit Cloud), we will set BACKEND_URL as a secret or env var.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="MusicMatch ANN", layout="wide")

# -------------------------
# Helpers
# -------------------------
def backend_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        r.raise_for_status()
        j = r.json()
        return j.get("ready", False), None
    except Exception as e:
        return False, str(e)

def fetch_songs(q: str, limit: int = 20):
    r = requests.get(f"{BACKEND_URL}/songs", params={"q": q, "limit": limit}, timeout=10)
    r.raise_for_status()
    return r.json().get("items", [])

def run_search(payload: dict):
    r = requests.post(f"{BACKEND_URL}/search", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def top_feature_explain(query_row: dict, hit_row: dict):
    # Uses raw feature columns stored in meta.parquet (same names)
    feat_cols = [
        "danceability", "energy", "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo", "key", "mode", "time_signature"
    ]
    diffs = []
    for c in feat_cols:
        if c in query_row and c in hit_row:
            try:
                diffs.append((c, abs(float(hit_row[c]) - float(query_row[c]))))
            except Exception:
                pass
    diffs.sort(key=lambda x: x[1])
    return [d[0] for d in diffs[:3]]

# -------------------------
# Header
# -------------------------
st.markdown("## 🎵 MusicMatch ANN")
st.caption("AI-powered music similarity search using ANN (HNSW & FAISS).")

ready, err = backend_health()
colA, colB = st.columns([1, 3])
with colA:
    if ready:
        st.success("Backend: READY")
    else:
        st.error("Backend: OFFLINE")
with colB:
    if not ready:
        st.warning(f"Start backend first: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`\n\nDetails: {err}")

st.divider()

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Search Settings")

mode = st.sidebar.selectbox(
    "Search Mode",
    options=["hnsw", "pq", "ivfpq"],
    help="HNSW = fast & accurate | PQ/IVF-PQ = memory efficient"
)

k = st.sidebar.slider("Top-K results", min_value=5, max_value=50, value=10)

ef_search = None
nprobe = None

if mode == "hnsw":
    ef_search = st.sidebar.slider("HNSW ef_search", 10, 200, 50)
elif mode == "ivfpq":
    nprobe = st.sidebar.slider("IVF-PQ nprobe", 1, 50, 10)

st.sidebar.divider()
st.sidebar.caption("Tip: Higher ef_search/nprobe usually improves recall but increases latency.")

# -------------------------
# Song search
# -------------------------
query = st.text_input("Search for a song", placeholder="Type a song name (e.g., love, hello, night)...")

song_options = {}
selected_song_index = None
selected_label = None

if query and ready:
    try:
        items = fetch_songs(query, limit=25)
        song_options = {
            f"{it['name']} — {it['artists']} ({it['release_date']})": it["row_index"]
            for it in items
        }
    except Exception as e:
        st.error(f"Failed to fetch songs: {e}")

if song_options:
    selected_label = st.selectbox("Select a song", options=list(song_options.keys()))
    selected_song_index = song_options[selected_label]

# session history
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# Search action
# -------------------------
btn_disabled = (selected_song_index is None) or (not ready)

if st.button("🔍 Find Similar Songs", disabled=btn_disabled):
    payload = {
        "song_index": int(selected_song_index),
        "k": int(k),
        "mode": mode
    }
    if ef_search is not None:
        payload["ef_search"] = int(ef_search)
    if nprobe is not None:
        payload["nprobe"] = int(nprobe)

    with st.spinner("Searching..."):
        try:
            data = run_search(payload)

            # store history (keep last 10)
            st.session_state.history.insert(0, {
                "time": datetime.now().strftime("%H:%M:%S"),
                "query": f"{data['query']['name']} — {data['query']['artists']}",
                "mode": data["mode"],
                "latency_ms": round(data["latency_ms"], 3),
            })
            st.session_state.history = st.session_state.history[:10]

            # Query display
            st.subheader("🎧 Query Song")
            st.markdown(f"**{data['query']['name']}**")
            st.caption(f"{data['query']['artists']} | {data['query']['release_date']} | popularity: {data['query']['popularity']}")

            # Cards for Top 5
            st.subheader("✨ Top Matches (Cards)")
            results = data["results"]
            query_row = data["query"]

            top5 = results[:5]
            cols = st.columns(5)
            for i, hit in enumerate(top5):
                with cols[i]:
                    st.markdown(f"**{hit['name']}**")
                    st.caption(f"{hit['artists']}")
                    st.caption(f"Release: {hit['release_date']}")
                    st.caption(f"Distance: `{hit['distance']:.4f}`")
                    why = top_feature_explain(query_row, hit)
                    st.caption(f"Why: {', '.join(why)}")

            # Full table
            st.subheader("📊 Similar Songs (Full List)")
            df = pd.DataFrame(results)

            # Reorder columns nicely if present
            preferred = ["name", "artists", "release_date", "popularity", "duration_ms", "explicit", "distance", "row_index", "id"]
            cols_in = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
            df = df[cols_in]

            st.dataframe(df, use_container_width=True, height=420)

            st.info(f"Mode: `{data['mode']}` | Latency: **{data['latency_ms']:.2f} ms**")

            st.download_button(
                "⬇️ Download results as CSV",
                data=df.to_csv(index=False),
                file_name="musicmatch_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Search failed: {e}")

# -------------------------
# History panel
# -------------------------
with st.expander("🕘 Search History (last 10)", expanded=False):
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history))
    else:
        st.write("No searches yet.")
