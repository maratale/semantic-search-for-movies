#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG over hybrid movie search — минимальный UI + вкладка «О модели».
- Вкладка «Поиск»: поле запроса, ответ LLM и карточки фильмов с постерами.
- Вкладка «О модели»: описание эмбеддингов, TF-IDF, гибридной смеси, RAG и обогащения.

Запуск: python -m streamlit run rag_movies.py -- app
"""

import os
import argparse
from typing import Optional
import pandas as pd
import requests
from urllib.parse import urlparse

# ====== НЕОТОБРАЖАЕМЫЕ НАСТРОЙКИ ============================================
INDEX_DIR   = "./index_hybrid"
TOP_K       = 10
LLM_MODEL   = "gpt-4o-mini"
IMG_WIDTH   = 200         # px
ALLOW_OGIMG = False       # True — пытаться брать og:image с page_url
SHOW_PLOT   = True        # показывать описание
# ============================================================================

# ---------- API key & client ----------
def _get_api_key() -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
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
    from openai import OpenAI
    return OpenAI(api_key=key)

# ---------- Импорт ретривера ----------
try:
    import semantic_search_movies as retr  # ожидается run_query(out_dir, query, k)
except Exception:
    retr = None

# ---------- Контекст для LLM ----------
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
        b = [f"Title: {title}" + (f" ({year})" if year else "")]
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

USER_PROMPT_TMPL = (
    "Запрос пользователя:\n{query}\n\n"
    "Контекст (кандидаты фильмов):\n{context}\n\n"
    "Твоя задача: на основе контекста выбрать максимально релевантные фильмы, "
    "объяснить выбор и связать их с запросом. Если контекст не покрывает запрос, "
    "честно скажи это и предложи ближайшие варианты из контекста.\n"
)

# ---------- LLM ----------
def call_openai(prompt: str, system: str = SYS_PROMPT, model: str = LLM_MODEL) -> str:
    try:
        client = _make_client()
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": prompt}],
            temperature=0.4,
            max_tokens=900,
        )
        return r.choices[0].message.content.strip()
    except Exception as e_new:
        try:
            import openai
            key = _get_api_key()
            if not key:
                return f"[LLM disabled] {e_new}\n\n" + prompt
            openai.api_key = key
            r = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": prompt}],
                temperature=0.4,
                max_tokens=900,
            )
            return r["choices"][0]["message"]["content"].strip()
        except Exception as e_old:
            return f"[OpenAI error] {e_old}\n\n" + prompt

# ---------- Ретривер-обёртка ----------
def retrieve(index_dir: str, query: str, k: int = TOP_K) -> pd.DataFrame:
    if retr and hasattr(retr, "run_query"):
        return retr.run_query(index_dir, query, k=k)
    meta_path = os.path.join(index_dir, "meta.parquet")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No meta.parquet in {index_dir}; build your index first.")
    df = pd.read_parquet(meta_path)
    q = query.lower()
    mask = (
        df.get("movie_title", pd.Series(dtype=str)).str.lower().str.contains(q, na=False) |
        df.get("directors",   pd.Series(dtype=str)).str.lower().str.contains(q, na=False) |
        df.get("actors",      pd.Series(dtype=str)).str.lower().str.contains(q, na=False) |
        df.get("description", pd.Series(dtype=str)).str.lower().str.contains(q, na=False)
    )
    out = df[mask].copy()
    if out.empty:
        out = df.sample(min(k, len(df)), random_state=42).copy()
    out.insert(0, "final_score", 0.0)
    return out.head(k)

def rag_answer(index_dir: str, query: str, k: int = TOP_K):
    hits = retrieve(index_dir, query, k=max(k, 12))
    ctx  = _build_context(hits, k=max(k, 12))
    prompt = USER_PROMPT_TMPL.format(query=query, context=ctx)
    return call_openai(prompt, model=LLM_MODEL), hits

# ---------- Постеры ----------
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
    ans, hits = rag_answer(INDEX_DIR, args.q, k=args.k)
    print("\n=== ANSWER ===\n")
    print(ans)
    print("\n=== SOURCES ===\n")
    cols = [c for c in ["movie_title","release_date","categories","actors","directors","page_url"] if c in hits.columns]
    print(hits.head(args.k)[cols].to_string(index=False, max_colwidth=120))

# ---------- Streamlit (две вкладки, без сайдбара) ----------
def cmd_app(_args):
    import streamlit as st
    st.set_page_config(page_title="RAG over Movies", layout="wide")
    st.title("🎬 Поиск фильмов по запросу пользователя")

    tabs = st.tabs(["Поиск", "О модели"])

    # --- вкладка «Поиск» ---
    with tabs[0]:
        if not _get_api_key():
            st.warning("OPENAI_API_KEY не найден — генерация ответа отключена.")

        q = st.text_input("Поиск")
        go = st.button("Найти")

        @st.cache_data(show_spinner=False)
        def cached_poster_from_page(url: str) -> Optional[str]:
            resp = _safe_get(url)
            if resp is None:
                return None
            og = _extract_og_image(resp.text)
            return og if (og and _is_http_url(og)) else None

        if go and q.strip():
            with st.spinner("Ищем ваш фильм"):
                ans, hits = rag_answer(INDEX_DIR, q, k=TOP_K)

            st.markdown("## Ответ")
            st.write(ans)

            st.markdown("## Подборка фильмов")
            for _, row in hits.iterrows():
                title = str(row.get("movie_title","(без названия)")).strip() or "(без названия)"
                meta_line = " | ".join(filter(None, [
                    str(row.get("categories","")).strip(),
                    str(row.get("release_date","")).strip(),
                ]))

                # постер
                poster = None
                for key in ("poster_url","image_url","poster","thumbnail"):
                    val = _clean_url(row.get(key, ""))
                    if _is_http_url(val):
                        poster = val
                        break
                if poster is None and ALLOW_OGIMG:
                    page_url = _clean_url(row.get("page_url",""))
                    if _is_http_url(page_url):
                        poster = cached_poster_from_page(page_url)

                box = st.container()
                col_img, col_txt = box.columns([1,3], vertical_alignment="center")

                with col_img:
                    if poster:
                        st.image(poster, width=IMG_WIDTH)
                    else:
                        st.markdown(
                            f"<div style='width:{IMG_WIDTH}px;height:{int(IMG_WIDTH*1.48)}px;"
                            f"background:#f3f3f3;border:1px dashed #ccc;display:flex;"
                            f"align-items:center;justify-content:center;color:#888;'>нет постера</div>",
                            unsafe_allow_html=True)

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

                    if SHOW_PLOT:
                        desc = (row.get("description") or "").strip()
                        if desc:
                            st.markdown(desc[:600] + ("…" if len(desc) > 600 else ""))

                    url = str(row.get("page_url","")).strip()
                    if _is_http_url(url):
                        st.markdown(f"[Открыть страницу]({url})")

                st.divider()

    # --- вкладка «О модели» ---
    with tabs[1]:
        st.markdown("## Как это работает")
        st.markdown(
            """
**Стек:**
- Эмбеддинги: `intfloat/multilingual-e5-base` (768-D). Префиксы: `passage:` для документов, `query:` для запросов; L2-норма → cosine = inner product.
- Поиск: первичный ANN через FAISS (или sklearn fallback), широкий top-*k0*.
- TF-IDF: три канала — `title`, `text` (название|описание|жанры|люди|дата), `people` (actors+directors).
- Эвристики: `title_boost` (точное/префиксное/overlap/fuzzy), `people_direct_hit`.
- Гибридная смесь сигналов (min–max по k0):
  - **персональные запросы (имя/фамилия):** `0.35*sbert + 0.20*title + 0.05*text + 0.25*people + 0.15*direct`
  - **общие запросы:** `0.45*sbert + 0.25*title + 0.10*text + 0.10*people + 0.10*title_boost`
- RAG: топ-кандидаты → компактный контекст → GPT-4o-mini формирует ответ и объяснения.

**Пайплайн:**
1. CSV → нормализация полей → `search_text`, `people_text`.
2. E5(`passage:`) → `embeddings.npy`; TF-IDF матрицы по каналам.
3. Запрос → alias-расширение → E5(`query:`) → ANN top-k0.
4. На k0 считаем TF-IDF и бусты → нормализация → смесь → финальный top-k.
5. RAG собирает контекст (title/year/genres/cast/plot/url) и генерирует структурированный ответ.

**Алиасы:** триггеры на популярные франшизы/персоны (HP/LOTR/Marvel/Тарантино); расширяемо.

**Обогащение:**
- Оффлайн-скрипт через GPT-4o-mini извлекает `themes`, `tropes`, `mood`, `setting`, `keywords_ru/en`, `alt_titles`, `entities` (строгий JSON).
- Эти поля можно добавить в `search_text`/`topics_text` + новый TF-IDF канал; для эмбеддингов — сложить `E5(base)+E5(enriched)` и перенормировать.

**Производительность:**
- До 100–300k фильмов — FAISS Flat на CPU.
"""
)
# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="RAG over movie search (two tabs)")
    sub = ap.add_subparsers(dest="cmd")

    ap_a = sub.add_parser("answer", help="Ask via CLI and print answer")
    ap_a.add_argument("--q", required=True)
    ap_a.add_argument("-k", type=int, default=TOP_K)
    ap_a.set_defaults(func=cmd_answer)

    ap_s = sub.add_parser("app", help="Run Streamlit UI")
    ap_s.set_defaults(func=cmd_app)

    args, _ = ap.parse_known_args()
    if args.cmd is None:
        args = argparse.Namespace(cmd="app")
        cmd_app(args)
        return
    args.func(args)

if __name__ == "__main__":
    main()
