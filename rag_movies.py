#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG over hybrid movie search, with poster cards in Streamlit.
- Берёт кандидатов из semantic_search_movies.run_query (ваш гибридный индекс).
- Собирает компактный контекст (title, year, genres, cast, plot, url).
- Вызывает LLM для связного ответа: резюме, рекомендации, связи.
- В Streamlit рендерит карточки фильмов с постером (из poster_url или og:image по page_url).

Запуск:
  # CLI
  python rag_movies.py answer --index ./index_hybrid --q "ромком в большом городе" --k 8

  # Streamlit
  python -m streamlit run rag_movies.py -- app --index ./index_hybrid

Переменные окружения:
  OPENAI_API_KEY — для OpenAI backend (в Streamlit Cloud положите в Secrets).
"""

print("RAG build stamp: 2025-10-03T12:00Z")

import os, sys, argparse
import pandas as pd
from typing import Optional

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

# ---------- Постеры: утилиты ----------
import requests
from urllib.parse import urlparse
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except Exception:
    HAS_BS4 = False

def _is_http_url(s: str) -> bool:
    try:
        u = urlparse(str(s).strip())
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False

def _clean_url(s: str) -> str:
    return str(s or "").strip()

def _safe_get(url: str, timeout: float = 5.0) -> Optional[requests.Response]:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; RAGMoviesBot/1.0)"}
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code == 200 and r.content:
            return r
    except Exception:
        pass
    return None

def _extract_og_image(html: str) -> Optional[str]:
    if not HAS_BS4:
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        for prop in ("og:image", "twitter:image", "og:image:url"):
            tag = soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"name": prop})
            if tag:
                content = tag.get("content") or tag.get("value")
                if content and _is_http_url(content):
                    return content
    except Exception:
        pass
    return None

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

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Постеры**")
    img_w = st.sidebar.slider("Ширина постера, px", 120, 360, 200, step=10)
    allow_fetch = st.sidebar.checkbox("Пробовать og:image из page_url", value=False,
                                      help="Если в данных нет poster_url, попробуем вытащить изображение со страницы")
    show_plot = st.sidebar.checkbox("Показывать описание", value=True)

    has_key = bool(_get_api_key())
    if not has_key:
        st.warning("OPENAI_API_KEY не найден — интерфейс загрузится, но генерация отключена.\n"
                   "В Streamlit Cloud добавьте секрет OPENAI_API_KEY в Settings → Secrets.")

    @st.cache_data(show_spinner=False)
    def cached_poster_from_page(url: str) -> Optional[str]:
        resp = _safe_get(url)
        if resp is None:
            return None
        og = _extract_og_image(resp.text)
        return og if (og and _is_http_url(og)) else None

    q = st.text_input("Ваш запрос", "романтическая комедия в большом городе")
    if st.button("Спросить", disabled=not has_key) and q.strip():
        with st.spinner("Готовим ответ…"):
            ans, hits = rag_answer(index_dir, q, k=k, llm_backend="openai", llm_model=llm_model)
        st.markdown("### Ответ")
        st.write(ans)

        st.markdown("### Использованные документы (таблица)")
        cols = [c for c in ["movie_title","release_date","categories","actors","directors","description","page_url","poster_url"] if c in hits.columns]
        st.dataframe(hits[cols], use_container_width=True)

        st.markdown("### Карточки фильмов")
        for _, row in hits.iterrows():
            title = str(row.get("movie_title","(без названия)")).strip() or "(без названия)"
            meta_line = " | ".join(filter(None, [
                str(row.get("categories","")) .strip(),
                str(row.get("release_date","")) .strip(),
            ]))

            # определить постер
            poster = None
            for key in ("poster_url","image_url","poster","thumbnail"):
                val = _clean_url(row.get(key, ""))
                if _is_http_url(val):
                    poster = val
                    break
            if poster is None and allow_fetch:
                page_url = _clean_url(row.get("page_url",""))
                if _is_http_url(page_url):
                    poster = cached_poster_from_page(page_url)

            box = st.container()
            col_img, col_txt = box.columns([1,3], vertical_alignment="center")

            with col_img:
                if poster:
                    st.image(poster, width=img_w)
                else:
                    st.markdown(
                        f"<div style='width:{img_w}px;height:{int(img_w*1.48)}px;background:#f3f3f3;"
                        f"border:1px dashed #ccc;display:flex;align-items:center;justify-content:center;color:#888;'>"
                        f"нет постера</div>", unsafe_allow_html=True)

            with col_txt:
                st.markdown(f"**{title}**")
                if meta_line:
                    st.markdown(meta_line)

                people = []
                if (row.get("actors") or "").strip():
                    people.append(f"**Актёры:** {row['actors']}")
                if (row.get("directors") or "").strip():
                    people.append(f"**Режиссёры:** {row['directors']}")
                if people:
                    st.markdown("  \n".join(people))

                if show_plot:
                    desc = (row.get("description") or "").strip()
                    if desc:
                        st.markdown(desc[:600] + ("…" if len(desc) > 600 else ""))

                url = str(row.get("page_url","")) .strip()
                if _is_http_url(url):
                    st.markdown(f"[Открыть страницу]({url})")

            st.divider()

    st.caption("build: 2025-10-03T12:00Z")

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

    # Если команд нет — запускаем UI с дефолтами.
    if args.cmd is None:
        args = argparse.Namespace(cmd="app", index="./index_hybrid")
        cmd_app(args)
        return

    args.func(args)

if __name__ == "__main__":
    main()
