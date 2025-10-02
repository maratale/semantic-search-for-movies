#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG over hybrid movie search.
- Берёт кандидатов из semantic_search_movies.run_query (ваш гибридный индекс).
- Собирает компактный контекст (title, year, genres, cast, plot, url).
- Вызывает LLM для связного ответа: резюме, рекомендации, связи.

Запуск:
  # CLI
  python rag_movies.py answer --index ./index_hybrid --q "ромком в большом городе" --k 8

  # Streamlit
  python -m streamlit run rag_movies.py -- app --index ./index_hybrid

Переменные окружения:
  OPENAI_API_KEY — для OpenAI backend (в Streamlit Cloud положите в Secrets).
"""

print("RAG build stamp: 2025-10-02T18:45Z")

import os, sys, argparse
import pandas as pd

# ---------- API key & client ----------
def _get_api_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        # В Streamlit Cloud ключ можно хранить в Secrets
        try:
            import streamlit as st
            key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
        except Exception:
            pass
    return key

def _make_client():
    key = _get_api_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set (env or Streamlit secrets).")
    # Импорт внутри функции, чтобы модуль грузился даже без openai
    from openai import OpenAI
    return OpenAI(api_key=key)

# ---------- Импорт ретривера ----------
try:
    import semantic_search_movies as retr  # ожидается функция run_query(out_dir, query, k)
except Exception:
    retr = None

# ---------- Вспомогательные ----------
def _shorten(s: str, n: int = 550) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"

def _as_year(x) -> str:
    s = str(x or "").strip()
    for ch in "[](){}":
        s = s.replace(ch, "")
    y = "".join(c for c in s if c.isdigit())
    return y[:4] if len(y) >= 4 else s

def _build_context(df: pd.DataFrame, k: int) -> str:
    blocks = []
    for _, row in df.head(k).iterrows():
        title = str(row.get("movie_title", "")).strip()
        year  = _as_year(row.get("release_date", ""))
        cats  = str(row.get("categories", "")).strip()
        actors = str(row.get("actors", "")).strip()
        directors = str(row.get("directors", "")).strip()
        desc = _shorten(str(row.get("description", "")))
        url  = str(row.get("page_url", "")).strip()

        b = []
        b.append(f"Title: {title}" + (f" ({year})" if year else ""))
        if cats:      b.append(f"Genres: {cats}")
        if directors: b.append(f"Directors: {directors}")
        if actors:    b.append(f"Cast: {actors}")
        if desc:      b.append(f"Plot: {desc}")
        if url.startswith("http"): b.append(f"URL: {url}")
        blocks.append("\n".join(b))
    return "\n\n---\n\n".join(blocks)

SYS_PROMPT = (
    "Ты — кинокритик и рекомендательная система. Отвечай по-русски, структурировано.\n"
    "Используй только предоставленный контекст (список фильмов), не выдумывай фактов.\n"
    "Формат ответа:\n"
    "1) Коротко ответь на запрос пользователя (1–2 предложения)\n"
    "2) Подборка 5–10 фильмов с причинами (по 1–2 предложения на фильм)\n"
    "3) Если уместно — предложи альтернативы/подборки\n"
    "Всегда указывай год, жанры, можно актёров/режиссёров; добавляй ссылки, если есть.\n"
)

USER_PROMPT_TMPL = """Запрос пользователя:
{query}

Контекст (кандидаты фильмов):
{context}

Твоя задача: на основе контекста выбрать максимально релевантные фильмы, объяснить выбор и связать их с запросом.
Если контекст не покрывает запрос, честно скажи это и предложи ближайшие варианты из контекста.
"""

# ---------- Вызов LLM ----------
def call_openai(prompt: str, system: str = SYS_PROMPT, model: str = "gpt-4o-mini") -> str:
    # основной путь: новый SDK (openai>=1.x)
    try:
        client = _make_client()
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=900,
        )
        return r.choices[0].message.content.strip()
    except Exception as e_new:
        # фолбэк: старый SDK (openai 0.x), если он внезапно в окружении
        try:
            import openai
            key = _get_api_key()
            if not key:
                return f"[LLM disabled] {e_new}\n\n" + prompt
            openai.api_key = key
            r = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=900,
            )
            return r["choices"][0]["message"]["content"].strip()
        except Exception as e_old:
            return f"[OpenAI error] {e_old}\n\n" + prompt

# ---------- Ретривер-обёртка ----------
def retrieve(index_dir: str, query: str, k: int = 12) -> pd.DataFrame:
    if retr and hasattr(retr, "run_query"):
        # ожидается сигнатура run_query(out_dir, query, k=...)
        return retr.run_query(index_dir, query, k=k)
    # fallback: очень простой поиск по подстроке, если индекс не собран/нет зависимостей
    meta_path = os.path.join(index_dir, "meta.parquet")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No meta.parquet in {index_dir}; build your index first.")
    df = pd.read_parquet(meta_path)
    q = query.lower()
    mask = (
        df.get("movie_title", pd.Series(dtype=str)).str.lower().str.contains(q, na=False) |
        df.get("directors", pd.Series(dtype=str)).str.lower().str.contains(q, na=False) |
        df.get("actors", pd.Series(dtype=str)).str.lower().str.contains(q, na=False) |
        df.get("description", pd.Series(dtype=str)).str.lower().str.contains(q, na=False)
    )
    out = df[mask].copy()
    if out.empty:
        out = df.sample(min(k, len(df)), random_state=42).copy()
    out.insert(0, "final_score", 0.0)
    return out.head(k)

# ---------- RAG ----------
def rag_answer(index_dir: str, query: str, k: int = 10,
               llm_backend: str = "openai", llm_model: str = "gpt-4o-mini") -> tuple[str, pd.DataFrame]:
    hits = retrieve(index_dir, query, k=max(k, 12))
    ctx = _build_context(hits, k=max(k, 12))
    prompt = USER_PROMPT_TMPL.format(query=query, context=ctx)
    if llm_backend == "openai":
        answer = call_openai(prompt, model=llm_model)
    else:
        answer = "[No LLM backend configured]\n\n" + prompt
    return answer, hits

# ---------- CLI ----------
def cmd_answer(args):
    ans, hits = rag_answer(args.index, args.q, k=args.k, llm_backend=args.backend, llm_model=args.model)
    print("\n=== ANSWER ===\n")
    print(ans)
    print("\n=== SOURCES ===\n")
    cols = [c for c in ["movie_title","release_date","categories","actors","directors","page_url"] if c in hits.columns]
    print(hits.head(args.k)[cols].to_string(index=False, max_colwidth=120))

# ---------- Streamlit UI ----------
def cmd_app(args):
    import streamlit as st
    st.set_page_config(page_title="RAG over Movies", layout="wide")
    st.title("🧠 RAG по фильмам (поверх гибридного поиска)")

    index_dir = st.sidebar.text_input("Папка индекса", args.index or "./index_hybrid")
    k = st.sidebar.slider("Top-K документов", 5, 20, 10)
    llm_model = st.sidebar.selectbox("LLM модель", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"], index=0)

    has_key = bool(_get_api_key())
    if not has_key:
        st.warning("OPENAI_API_KEY не найден — интерфейс загрузится, но генерация отключена. "
                   "В Streamlit Cloud добавьте секрет OPENAI_API_KEY в Settings → Secrets.")

    q = st.text_input("Ваш запрос", "романтическая комедия в большом городе")
    if st.button("Спросить", disabled=not has_key) and q.strip():
        with st.spinner("Готовим ответ…"):
            ans, hits = rag_answer(index_dir, q, k=k, llm_backend="openai", llm_model=llm_model)
        st.markdown("### Ответ")
        st.write(ans)

        st.markdown("### Использованные документы")
        cols = [c for c in ["movie_title","release_date","categories","actors","directors","description","page_url"] if c in hits.columns]
        st.dataframe(hits[cols], use_container_width=True)

    st.caption("build: 2025-10-02T18:45Z")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="RAG over movie search")
    sub = ap.add_subparsers(dest="cmd")

    ap_a = sub.add_parser("answer", help="Ask a question and get a synthesized answer")
    ap_a.add_argument("--index", required=True)
    ap_a.add_argument("--q", required=True)
    ap_a.add_argument("-k", type=int, default=10)
    ap_a.add_argument("--backend", default="openai")
    ap_a.add_argument("--model", default="gpt-4o-mini")
    ap_a.set_defaults(func=cmd_answer)

    ap_s = sub.add_parser("app", help="Run Streamlit UI")
    ap_s.add_argument("--index", default="./index_hybrid")
    ap_s.set_defaults(func=cmd_app)

    args, _ = ap.parse_known_args()
    if args.cmd is None:
        # Если запущено через streamlit run, передаём управление UI
        if any(x.endswith("streamlit") or x == "streamlit" for x in sys.argv[0:2]):
            args = argparse.Namespace(cmd="app", index="./index_hybrid")
            cmd_app(args)
            return
        ap.print_help(); sys.exit(0)
    args.func(args)

if __name__ == "__main__":
    main()
