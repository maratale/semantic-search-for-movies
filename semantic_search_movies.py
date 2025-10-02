#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid movie search: E5 embeddings + TF-IDF (title/text/people) + title & people boosts + alias expansion.

Usage:

  # 1) Собрать индекс
  python semantic_search_movies.py build --csv cleaned_dataset.csv --out-dir ./index_hybrid

  # 2) Запрос из CLI
  python semantic_search_movies.py query --out-dir ./index_hybrid --q "Тарантино" -k 10

  # 3) Веб (Streamlit)
  python -m streamlit run semantic_search_movies.py -- app --csv cleaned_dataset.csv --out-dir ./index_hybrid
"""

import argparse, os, sys, json, re, pickle
from typing import List, Tuple
import numpy as np
import pandas as pd

# -------- optional deps ----------
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

# fuzzy: rapidfuzz (если есть) или difflib (фолбэк)
try:
    from rapidfuzz import fuzz
    def fuzz_ratio(a: str, b: str) -> float:
        return fuzz.WRatio(a, b) / 100.0
except Exception:
    import difflib
    def fuzz_ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

# -------- фиксированная модель (E5) --------
# ВАЖНО: для E5 обязательно использовать префиксы "query:" и "passage:"
EMBED_MODEL = "intfloat/multilingual-e5-base"   # dim = 768

# ---------- alias-driven расширение запроса ----------
ALIAS_PATTERNS = [
    # Гарри Поттер
    (r"(?i)\bмальчик\s+котор(ый|ого)\s+выжил\b", "гарри поттер хогвартс волдеморт роулинг potter"),
    (r"(?i)\bгарри\s*поттер\b",                   "гарри поттер хогвартс волдеморт роулинг potter"),
    # ВК
    (r"(?i)\bкольц[оа]\s+власти\b",               "властелин колец толкин саурон кольцо братство кольца"),
    (r"(?i)\bвластелин\s+колец\b",                "властелин колец толкин саурон кольцо братство кольца"),
    # Marvel
    (r"(?i)\bmarvel\b|\bмарвел\b",                "marvel мстители железный человек тони старк капитан америка тор локи"),
    # Тарантино
    (r"(?i)\bтарантино\b|\btarantino\b",          "квентин тарантино quentin tarantino"),
]

def expand_query(q: str) -> str:
    base = q or ""
    add = []
    for pat, extra in ALIAS_PATTERNS:
        if re.search(pat, base):
            add.append(extra)
    return (base + " " + " ".join(add)).strip() if add else base

# ------------- text utils ------------------
def norm_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def join_fields(row: pd.Series) -> str:
    parts = []
    for col in ["movie_title", "description", "categories", "actors", "directors", "release_date"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            parts.append(str(row[col]))
    return norm_text(" | ".join(parts))

def looks_like_person(q: str) -> bool:
    qn = re.sub(r"\s+", " ", (q or "").lower()).strip()
    toks = re.findall(r"[a-zа-яё]+", qn, flags=re.I)
    return 1 <= len(toks) <= 3

# ------------- embeddings (E5) --------------
def load_model(name: str = EMBED_MODEL) -> SentenceTransformer:
    # насильно на CPU — стабильнее на macOS/без CUDA
    return SentenceTransformer(name, device="cpu")

def encode_corpus(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    texts = [f"passage: {t}" for t in texts]
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)

def encode_query(model: SentenceTransformer, q: str) -> np.ndarray:
    return model.encode([f"query: {norm_text(q)}"], normalize_embeddings=True)[0]

# --------------- index I/O ------------------
def save_index(out_dir: str, embs: np.ndarray, df: pd.DataFrame, use_faiss: bool,
               tfidf_data: dict, metric: str = "cosine", model_name: str = EMBED_MODEL) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "embeddings.npy"), embs)
    df.to_parquet(os.path.join(out_dir, "meta.parquet"), index=False)
    meta = {"metric": metric, "use_faiss": bool(use_faiss), "dim": int(embs.shape[1]), "model_name": model_name}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if use_faiss:
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs.astype(np.float32))
        faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    with open(os.path.join(out_dir, "tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf_data, f)

def load_index(out_dir: str):
    with open(os.path.join(out_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    embs = np.load(os.path.join(out_dir, "embeddings.npy"))
    df = pd.read_parquet(os.path.join(out_dir, "meta.parquet"))
    faiss_index = None
    if meta.get("use_faiss", False) and HAS_FAISS:
        faiss_index = faiss.read_index(os.path.join(out_dir, "faiss.index"))
    with open(os.path.join(out_dir, "tfidf.pkl"), "rb") as f:
        tfidf_data = pickle.load(f)
    return embs, df, meta, faiss_index, tfidf_data

# --------------- build ----------------------
def cmd_build(args):
    csv_path = args.csv
    out_dir = args.out_dir
    batch = args.batch

    df = pd.read_csv(csv_path, encoding="utf-8")
    # нормализуем/фиксируем возможные опечатки колонок
    df.columns = df.columns.str.strip()
    if "reliease_date" in df.columns and "release_date" not in df.columns:
        df = df.rename(columns={"reliease_date": "release_date"})
    if "movie title" in df.columns and "movie_title" not in df.columns:
        df = df.rename(columns={"movie title": "movie_title"})

    if "movie_title" not in df.columns:
        print("ERROR: CSV must contain 'movie_title' column", file=sys.stderr); sys.exit(1)

    df = df.copy()
    df["title_norm"]  = df["movie_title"].map(norm_text)
    df["search_text"] = df.apply(join_fields, axis=1)
    df["people_text"] = (df.get("directors","").fillna("").astype(str) + " ; " +
                         df.get("actors","").fillna("").astype(str)).map(norm_text)

    # E5 embeddings
    model = load_model(EMBED_MODEL)
    embs = encode_corpus(model, df["search_text"].tolist(), batch_size=batch).astype(np.float32)

    # TF-IDF: title, full text, people
    vt_title = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=200_000)
    X_title  = vt_title.fit_transform(df["title_norm"].tolist())

    vt_text  = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=300_000)
    X_text   = vt_text.fit_transform(df["search_text"].tolist())

    vt_people = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=200_000)
    X_people  = vt_people.fit_transform(df["people_text"].tolist())

    tfidf_data = {
        "vt_title": vt_title, "X_title": X_title,
        "vt_text":  vt_text,  "X_text":  X_text,
        "vt_people": vt_people, "X_people": X_people,
    }

    use_faiss = HAS_FAISS
    save_index(out_dir, embs, df, use_faiss=use_faiss, tfidf_data=tfidf_data,
               metric="cosine", model_name=EMBED_MODEL)

    print(f"✔ Index built at: {out_dir}  (model={EMBED_MODEL}, dim={embs.shape[1]})")
    if not HAS_FAISS:
        print("ℹ FAISS not found, will use sklearn NearestNeighbors.", file=sys.stderr)

# --------------- base vector search ----------
def _search_faiss(index, q_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    q = q_vec.reshape(1, -1).astype(np.float32)
    sims, idxs = index.search(q, k)  # inner product on normalized vectors
    return idxs[0], sims[0]

def _search_knn(embs: np.ndarray, q_vec: np.ndarray, k: int):
    nn = NearestNeighbors(n_neighbors=min(k, len(embs)), metric="cosine")
    nn.fit(embs)
    dists, idxs = nn.kneighbors(q_vec.reshape(1, -1))
    sims = 1.0 - dists[0]
    return idxs[0], sims

# --------------- helpers for hybrid ----------
def cosine_sparse_rowvec(X, v):
    v = sk_normalize(v)
    Xn = sk_normalize(X)
    sims = (Xn @ v.T).toarray().ravel()
    return sims

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().strip())

def _token_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", _norm(a)))
    tb = set(re.findall(r"\w+", _norm(b)))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(tb))

def title_boost(row_title: str, q: str) -> float:
    tn = _norm(row_title); qn = _norm(q)
    exact = 1.0 if qn and qn in tn else 0.0
    starts = 1.0 if qn and tn.startswith(qn) else 0.0
    overlap = _token_overlap(tn, qn)
    fuzzy = fuzz_ratio(qn, tn)
    score = 0.55*exact + 0.20*starts + 0.20*overlap + 0.05*fuzzy
    if exact == 1.0:
        score += 0.10
    return min(1.0, max(0.0, score))

def people_direct_hit(row, q_raw: str) -> float:
    qn = _norm(q_raw)
    D = _norm(row.get("directors",""))
    A = _norm(row.get("actors",""))
    hit = 0.0
    if qn and (qn in D or qn in A):
        hit += 0.6
    hit += 0.4 * max(fuzz_ratio(qn, D), fuzz_ratio(qn, A))
    return min(1.0, hit)

def hybrid_combine(sbert, tf_title, tf_text, titleb, tf_people, direct_hits, person_query: bool):
    def nz(x):
        x = np.asarray(x, dtype=float)
        if x.size == 0: return x
        m, M = float(np.min(x)), float(np.max(x))
        return np.zeros_like(x) if M - m < 1e-9 else (x - m) / (M - m)

    sbert_n  = nz(sbert)
    title_n  = nz(tf_title)
    text_n   = nz(tf_text)
    titleb_n = nz(titleb)
    people_n = nz(tf_people)
    dhit_n   = nz(direct_hits)

    if person_query:
        # запрос похож на имя/фамилию → сильно бустим people
        return 0.35*sbert_n + 0.20*title_n + 0.05*text_n + 0.25*people_n + 0.15*dhit_n
    else:
        return 0.45*sbert_n + 0.25*title_n + 0.10*text_n + 0.10*people_n + 0.10*titleb_n

# --------------- run query ------------------
def run_query(out_dir: str, q: str, k: int = 10):
    embs, df, meta, faiss_index, tfidf = load_index(out_dir)

    model_name = meta.get("model_name") or EMBED_MODEL
    model = load_model(model_name)

    q_exp = expand_query(q)
    q_vec = encode_query(model, q_exp)

    index_dim = int(meta.get("dim", embs.shape[1]))
    if q_vec.shape[0] != index_dim:
        raise ValueError(f"Embedding dim mismatch: query {q_vec.shape[0]} vs index {index_dim}.")

    # первичный ретрив по эмбеддингам с расширенным топом
    k0 = min(max(100, k * 10), len(embs))
    if faiss_index is not None:
        idxs, sims = _search_faiss(faiss_index, q_vec, k0)
    else:
        idxs, sims = _search_knn(embs, q_vec, k0)

    hits = df.iloc[idxs].copy()
    hits["sbert_score"] = sims.astype(float)

    # TF-IDF похожести
    vt_title, X_title = tfidf["vt_title"], tfidf["X_title"]
    vt_text,  X_text  = tfidf["vt_text"],  tfidf["X_text"]
    vt_people, X_people = tfidf["vt_people"], tfidf["X_people"]

    q_title_vec  = vt_title.transform([_norm(q_exp)])
    q_text_vec   = vt_text.transform([_norm(q_exp)])
    q_people_vec = vt_people.transform([_norm(q_exp)])

    sims_title_all  = cosine_sparse_rowvec(X_title,  q_title_vec)
    sims_text_all   = cosine_sparse_rowvec(X_text,   q_text_vec)
    sims_people_all = cosine_sparse_rowvec(X_people, q_people_vec)

    sims_title  = sims_title_all[idxs]
    sims_text   = sims_text_all[idxs]
    sims_people = sims_people_all[idxs]

    # ручные бусты
    titleb = np.array([title_boost(t, q_exp) for t in hits["movie_title"].tolist()], dtype=float)
    direct_hits = np.array([people_direct_hit(r, q_exp) for _, r in hits.iterrows()], dtype=float)

    person_q = looks_like_person(q)

    final = hybrid_combine(
        hits["sbert_score"].to_numpy(),
        sims_title, sims_text,
        titleb, sims_people, direct_hits,
        person_q
    )

    hits.insert(0, "final_score", np.round(final, 4))
    hits.insert(1, "title_score", np.round(sims_title, 4))
    hits.insert(2, "people_score", np.round(sims_people, 4))
    hits.sort_values(by=["final_score","sbert_score","title_score","people_score"], ascending=False, inplace=True)
    return hits.head(k).copy()

# --------------- CLI -----------------------
def cmd_query(args):
    res = run_query(args.out_dir, args.q, k=args.k)
    cols = [c for c in [
        "final_score","sbert_score","title_score","people_score",
        "movie_title","categories","release_date",
        "actors","directors","description","page_url"
    ] if c in res.columns]
    print(res[cols].to_string(index=False, max_colwidth=120))

# --------------- Streamlit -----------------
def cmd_streamlit(args):
    import streamlit as st
    st.set_page_config(page_title="Hybrid Movie Search (E5 + TF-IDF + People Boost)", layout="wide")
    st.title("🎬 Гибридный поиск фильмов (E5 + TF-IDF + People Boost)")

    st.sidebar.markdown(f"**Embedding model:** `{EMBED_MODEL}`")
    out_dir = st.sidebar.text_input("Папка индекса", args.out_dir or "./index_hybrid")
    csv_path = st.sidebar.text_input("CSV (для пересборки)", args.csv or "cleaned_dataset.csv")
    k = st.sidebar.slider("Top-K", 5, 50, 10)

    if st.sidebar.button("Пересобрать индекс"):
        with st.spinner("Строим индекс..."):
            _args = argparse.Namespace(csv=csv_path, out_dir=out_dir, batch=64)
            cmd_build(_args)
        st.success("Индекс готов")

    q = st.text_input("Запрос", "Тарантино")
    if st.button("Искать") and q.strip():
        with st.spinner("Ищем..."):
            res = run_query(out_dir, q, k=k)

        cols = [c for c in [
            "final_score","sbert_score","title_score","people_score",
            "movie_title","categories","release_date",
            "actors","directors","description","page_url"
        ] if c in res.columns]
        st.subheader("Результаты")
        st.dataframe(res[cols], use_container_width=True)

        for _, row in res.iterrows():
            with st.expander(str(row.get("movie_title","(без названия)"))):
                meta_line = " | ".join(filter(None, [
                    str(row.get("categories","")).strip(),
                    str(row.get("release_date","")).strip(),
                ]))
                if meta_line:
                    st.markdown(meta_line)

                people = []
                if (row.get("actors") or "").strip():
                    people.append(f"**Актёры:** {row['actors']}")
                if (row.get("directors") or "").strip():
                    people.append(f"**Режиссёры:** {row['directors']}")
                if people:
                    st.markdown("  \n".join(people))

                desc = (row.get("description") or "").strip()
                if desc:
                    st.markdown(desc[:800] + ("…" if len(desc) > 800 else ""))

                url = str(row.get("page_url",""))
                if url.startswith("http"):
                    st.markdown(f"[Открыть страницу]({url})")

# --------------- main ----------------------
def main():
    ap = argparse.ArgumentParser(description="Hybrid semantic search for movies (E5 + TF-IDF + PeopleBoost)")
    sub = ap.add_subparsers(dest="cmd")

    ap_b = sub.add_parser("build", help="Build embeddings + TF-IDF")
    ap_b.add_argument("--csv", required=True)
    ap_b.add_argument("--out-dir", default="./index_hybrid")
    ap_b.add_argument("--batch", type=int, default=64)
    ap_b.set_defaults(func=cmd_build)

    ap_q = sub.add_parser("query", help="Query the index")
    ap_q.add_argument("--out-dir", default="./index_hybrid")
    ap_q.add_argument("--q", required=True)
    ap_q.add_argument("-k", type=int, default=10)
    ap_q.set_defaults(func=cmd_query)

    ap_a = sub.add_parser("app", help="Run Streamlit app")
    ap_a.add_argument("--csv", default="cleaned_dataset.csv")
    ap_a.add_argument("--out-dir", default="./index_hybrid")
    ap_a.set_defaults(func=cmd_streamlit)

    args, _ = ap.parse_known_args()
    if args.cmd is None:
        if any(x.endswith("streamlit") or x == "streamlit" for x in sys.argv[0:2]):
            args = argparse.Namespace(cmd="app", csv="cleaned_dataset.csv", out_dir="./index_hybrid")
            cmd_streamlit(args); return
        ap.print_help(); sys.exit(0)
    args.func(args)

if __name__ == "__main__":
    main()
