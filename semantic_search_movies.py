#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Semantic search for movies (SBERT + FAISS fallback to sklearn NN)

Usage (CLI):
  # 1) build index
  python semantic_search_movies.py build --csv dataset_clean.csv --out-dir ./index

  # 2) query top-10
  python semantic_search_movies.py query --out-dir ./index --q "романтическая комедия в большом городе" -k 10

  # 3) run simple web app (Streamlit)
  streamlit run semantic_search_movies.py -- --csv dataset_clean.csv --out-dir ./index

Notes:
- Default model: paraphrase-multilingual-mpnet-base-v2 (good for RU)
- You can change model via --model
"""

import argparse
import os
import sys
import json
import re
from typing import List, Tuple, Optional

import sentence_transformers
import numpy as np
import pandas as pd

# --------- optional deps handling ----------
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# ------------- text utils ------------------
def norm_text(s: str) -> str:
    """
    Light normalization: lowercase, collapse spaces, strip.
    Keep punctuation (SBERT ok with it), but remove extra.
    """
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def join_fields(row: pd.Series) -> str:
    parts = []
    for col in ["movie_title", "description", "categories", "actors", "directors", "release_date", "reliease_date"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            parts.append(str(row[col]))
    return norm_text(" | ".join(parts))

# ------------- embeddings -------------------
def load_model(name: str) -> SentenceTransformer:
    # Popular options to try:
    # - "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" (default)
    # - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (faster, smaller)
    # - "sentence-transformers/paraphrase-multilingual-MiniLM-L3-v2" (very fast)
    return SentenceTransformer(name)

def encode_corpus(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)

def encode_query(model: SentenceTransformer, q: str) -> np.ndarray:
    return model.encode([norm_text(q)], normalize_embeddings=True)[0]

# --------------- index I/O ------------------
def save_index(out_dir: str, embs: np.ndarray, df: pd.DataFrame, use_faiss: bool, metric: str = "cosine") -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "embeddings.npy"), embs)
    df.to_parquet(os.path.join(out_dir, "meta.parquet"), index=False)
    meta = {"metric": metric, "use_faiss": bool(use_faiss), "dim": int(embs.shape[1])}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if use_faiss:
        dim = embs.shape[1]
        # cosine with normalized vectors == inner product
        index = faiss.IndexFlatIP(dim)
        index.add(embs.astype(np.float32))
        faiss.write_index(index, os.path.join(out_dir, "faiss.index"))

def load_index(out_dir: str):
    with open(os.path.join(out_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    embs = np.load(os.path.join(out_dir, "embeddings.npy"))
    df = pd.read_parquet(os.path.join(out_dir, "meta.parquet"))

    faiss_index = None
    if meta.get("use_faiss", False) and HAS_FAISS:
        faiss_index = faiss.read_index(os.path.join(out_dir, "faiss.index"))
    return embs, df, meta, faiss_index

# --------------- build ----------------------
def cmd_build(args):
    csv_path = args.csv
    out_dir = args.out_dir
    model_name = args.model
    batch = args.batch

    df = pd.read_csv(csv_path, encoding="utf-8")
    if "movie_title" not in df.columns:
        print("ERROR: CSV must contain 'movie_title' column", file=sys.stderr); sys.exit(1)

    # make a search_text column
    df = df.copy()
    df["search_text"] = df.apply(join_fields, axis=1)

    model = load_model(model_name)
    embs = encode_corpus(model, df["search_text"].tolist(), batch_size=batch).astype(np.float32)

    use_faiss = HAS_FAISS
    save_index(out_dir, embs, df, use_faiss=use_faiss, metric="cosine")
    print(f"✔ Index built at: {out_dir}")
    if not HAS_FAISS:
        print("ℹ FAISS not found, will use sklearn NearestNeighbors at query time.", file=sys.stderr)

# --------------- search ---------------------
def _search_faiss(index, q_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    q = q_vec.reshape(1, -1).astype(np.float32)
    sims, idxs = index.search(q, k)   # inner product (cosine on normalized)
    return idxs[0], sims[0]

def _search_knn(embs: np.ndarray, q_vec: np.ndarray, k: int):
    # cosine = 1 - cosine_dist
    nn = NearestNeighbors(n_neighbors=min(k, len(embs)), metric="cosine", algorithm="auto")
    nn.fit(embs)
    dists, idxs = nn.kneighbors(q_vec.reshape(1, -1))
    sims = 1.0 - dists[0]
    return idxs[0], sims

def run_query(out_dir: str, model_name: str, q: str, k: int = 10):
    embs, df, meta, faiss_index = load_index(out_dir)
    # Reuse model to encode query
    model = load_model(model_name)
    q_vec = encode_query(model, q)

    if faiss_index is not None:
        idxs, sims = _search_faiss(faiss_index, q_vec, k)
    else:
        idxs, sims = _search_knn(embs, q_vec, k)

    res = df.iloc[idxs].copy()
    res.insert(0, "score", np.round(sims.astype(float), 4))
    return res

def cmd_query(args):
    res = run_query(args.out_dir, args.model, args.q, k=args.k)
    cols_to_show = [c for c in ["score","movie_title","categories","release_date","description","page_url","image_url"] if c in res.columns]
    print(res[cols_to_show].to_string(index=False, max_colwidth=120))

# --------------- streamlit ------------------
def cmd_streamlit(args):
    # This function will be executed by Streamlit (run: streamlit run semantic_search_movies.py -- --csv ... --out-dir ...)
    import streamlit as st

    st.set_page_config(page_title="Movie Semantic Search (SBERT)", layout="wide")
    st.title("🎬 Семантический поиск фильмов (SBERT)")

    # Sidebar: settings
    model_name = st.sidebar.selectbox(
        "Модель (SBERT)", 
        options=[
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L3-v2",
        ],
        index=0
    )
    out_dir = st.sidebar.text_input("Папка индекса", args.out_dir or "./index")
    csv_path = st.sidebar.text_input("CSV с фильмами", args.csv or "dataset_clean.csv")
    k = st.sidebar.slider("Top-K", 5, 50, 10)

    build_btn = st.sidebar.button("Пересобрать индекс")
    if build_btn:
        with st.spinner("Строим индекс..."):
            # Build (re)index
            _args = argparse.Namespace(csv=csv_path, out_dir=out_dir, model=model_name, batch=64)
            cmd_build(_args)
        st.success("Индекс готов")

    # Query
    q = st.text_input("Ваш запрос (напр. «романтическая комедия в большом городе», «история о кольце власти», «хорор фелм»)")
    if st.button("Искать") and q.strip():
        with st.spinner("Ищем..."):
            res = run_query(out_dir, model_name, q, k=k)
        cols = [c for c in ["score","movie_title","categories","release_date","description","page_url","image_url"] if c in res.columns]
        st.subheader("Результаты")
        st.dataframe(res[cols])

        # маленькая карточная выдача
        for _, row in res.head(k).iterrows():
            with st.container():
                st.markdown(f"**{row.get('movie_title','')}** &nbsp;&nbsp; _(score: {row.get('score',0):.4f})_")
                cols = st.columns([1,3])
                if "image_url" in row and isinstance(row["image_url"], str) and row["image_url"].startswith("http"):
                    try:
                        cols[0].image(row["image_url"])
                    except Exception:
                        pass
                meta_line = " | ".join([str(row.get("categories","")), str(row.get("release_date",""))]).strip(" |")
                if meta_line:
                    cols[1].markdown(meta_line)
                if isinstance(row.get("description",""), str) and row["description"]:
                    cols[1].markdown(row["description"][:500] + ("..." if len(row["description"])>500 else ""))
                if "page_url" in row and isinstance(row["page_url"], str) and row["page_url"].startswith("http"):
                    cols[1].markdown(f"[Открыть страницу]({row['page_url']})")

# --------------- main -----------------------
def main():
    ap = argparse.ArgumentParser(description="Semantic search for movies (SBERT).")
    sub = ap.add_subparsers(dest="cmd")

    ap_build = sub.add_parser("build", help="Build embeddings index")
    ap_build.add_argument("--csv", required=True)
    ap_build.add_argument("--out-dir", default="./index")
    ap_build.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    ap_build.add_argument("--batch", type=int, default=64)
    ap_build.set_defaults(func=cmd_build)

    ap_query = sub.add_parser("query", help="Query the index from CLI")
    ap_query.add_argument("--out-dir", default="./index")
    ap_query.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    ap_query.add_argument("--q", required=True)
    ap_query.add_argument("-k", type=int, default=10)
    ap_query.set_defaults(func=cmd_query)

    # Streamlit entrypoint: `streamlit run semantic_search_movies.py -- --csv ...`
    ap_st = sub.add_parser("app", help="(Used by Streamlit) Run web app")
    ap_st.add_argument("--csv", default="dataset_clean.csv")
    ap_st.add_argument("--out-dir", default="./index")
    ap_st.set_defaults(func=cmd_streamlit)

    args, unknown = ap.parse_known_args()
    if args.cmd is None:
        # If launched by Streamlit, it passes 'run' outside, so we expose app by default
        if any(x.endswith("streamlit") or x == "streamlit" for x in sys.argv[0:2]):
            args = argparse.Namespace(cmd="app", csv="dataset_clean.csv", out_dir="./index")
            cmd_streamlit(args)
            return
        ap.print_help(); sys.exit(0)

    args.func(args)

if __name__ == "__main__":
    main()
