import os
import json
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import requests
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# =========================================================
# CONFIG GERAL DA PÁGINA
# =========================================================

st.set_page_config(
    page_title="HypeQuest – Instagram Engagement & Sentiment Prediction",
    page_icon="🎮",
    layout="wide",
)

# ------------------------------
# CSS ESTILO HYPE QUEST (fontes padronizadas + sidebar centralizada + logo animado)
# ------------------------------
st.markdown("""
<style>
    /* Fonte base menor e uniforme */
    html, body {
        font-size: 14px;
    }
    body, p, span, div, section {
        font-size: 14px;
    }

    .main {
        background-color: #B7E3FF;
    }

    .block-container {
        padding-top: 3rem;              /* mais espaço no topo para o título não cortar */
        max-width: 1100px;
        margin: 0 auto;
    }

    .hype-title {
        font-family: monospace;
        font-size: 30px;
        font-weight: 900;
        letter-spacing: 4px;
        color: #1C0A4A;
        text-align: center;
        text-transform: uppercase;
        text-shadow: 3px 3px 0px #8FD0FF;
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
    }

    .hype-subtitle {
        font-family: monospace;
        font-size: 14px;
        color: #1C0A4A;
        text-align: center;
        margin-bottom: 8px;
    }

    .hype-status {
        font-size: 12px;
        color: #6b7280;
        text-align: center;
        margin-bottom: 6px;
    }

    hr {
        border: none;
        border-top: 2px solid #8FD0FF;
        margin: 0.6rem 0 1rem 0;
    }

    .stButton>button {
        border-radius: 6px;
        border: 3px solid #1C0A4A;
        background: linear-gradient(180deg, #FFDD55 0%, #FFB800 70%);
        color: #1C0A4A;
        font-weight: bold;
        font-size: 15px;
        padding: 6px 22px;
        box-shadow: 0px 4px 0px #D98F00;
    }

    .stButton>button:hover {
        background: linear-gradient(180deg, #FFE680 0%, #FFC933 70%);
        box-shadow: 0px 4px 0px #C57D00;
    }

    /* Sidebar com fundo azul e fontes menores */
    section[data-testid="stSidebar"] {
        background-color: #8FD0FF !important;
        font-size: 13px;
        padding-top: 1.5rem;
    }

    /* Animação suave do logo (flutuar) */
    @keyframes hypeFloat {
        0%   { transform: translateY(0px); }
        50%  { transform: translateY(-4px); }
        100% { transform: translateY(0px); }
    }

    /* Container do logo na sidebar centralizado */
    section[data-testid="stSidebar"] div[data-testid="stImage"] {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }

    /* Remover botão de fullscreen e clique no logo */
    section[data-testid="stSidebar"] div[data-testid="stImage"] button {
        display: none !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stImage"] img {
        width: 130px;
        height: auto;
        pointer-events: none;
        animation: hypeFloat 2.5s ease-in-out infinite;
    }

    /* Centralizar títulos e textos “infos” da sidebar */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .stMarkdown p {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# STATE INICIAL
# =========================================================

if "caption_input" not in st.session_state:
    st.session_state["caption_input"] = ""

# =========================================================
# LOGO NA SIDEBAR
# =========================================================

LOGO_PATH = "HypeLogo(1).png"  # ideal: PNG com fundo transparente

with st.sidebar:
    try:
        st.image(LOGO_PATH)  # tamanho controlado via CSS (width: 130px)
    except Exception:
        pass

# =========================================================
# HEADER PRINCIPAL
# =========================================================

st.markdown("<div class='hype-title'>HYPE QUEST</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hype-subtitle'>Instagram Engagement & Sentiment Prediction</div>",
    unsafe_allow_html=True,
)

# =========================================================
# API Credentials and Client Setup
# =========================================================

META_TOKEN = st.secrets.get("META_ACCESS_TOKEN", None)
INSTAGRAM_ID = st.secrets.get("INSTAGRAM_ACCOUNT_ID", None)  # PUBG MENA


def get_openai_api_key() -> Optional[str]:
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    if "openai_api_key" in st.secrets:
        return st.secrets["openai_api_key"]

    if "openai" in st.secrets:
        nested = st.secrets["openai"]
        if isinstance(nested, dict) and "api_key" in nested:
            return nested["api_key"]

    return None


OPENAI_API_KEY = get_openai_api_key()
openai_client = None
openai_import_error = None

if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        openai_import_error = str(e)
        openai_client = None

if OPENAI_API_KEY and openai_client:
    IA_STATUS = "✅ Generative AI (GPT-4.1-mini) enabled"
elif not OPENAI_API_KEY:
    IA_STATUS = "⚠️ Generative AI disabled (no OPENAI_API_KEY found in env or secrets)"
else:
    IA_STATUS = f"⚠️ Generative AI disabled (OpenAI SDK error: {openai_import_error})"

st.markdown(
    f"<div class='hype-status'>{IA_STATUS}</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# =========================================================
# Simple keyword sentiment fallback
# =========================================================

POSITIVE_WORDS = {
    "win", "victory", "amazing", "awesome", "love", "great", "hype",
    "excited", "epic", "gg", "insane", "crazy", "wow", "🔥", "🥳",
}
NEGATIVE_WORDS = {
    "bug", "issue", "error", "lag", "toxic", "hate", "angry", "sad",
    "broken", "crash", "fuck", "shit", "bad", "😭",
}


def basic_sentiment(caption: str) -> str:
    if not caption or not caption.strip():
        return "NEUTRAL"

    text = caption.lower()
    pos_hits = sum(w in text for w in POSITIVE_WORDS)
    neg_hits = sum(w in text for w in NEGATIVE_WORDS)

    if pos_hits > neg_hits and pos_hits > 0:
        return "POSITIVE"
    elif neg_hits > pos_hits and neg_hits > 0:
        return "NEGATIVE"
    else:
        return "NEUTRAL"


# =========================================================
# Generative AI function (GPT-4.1-mini + fallback)
# =========================================================


def gpt_caption_analysis(caption: str, context: dict) -> dict:
    if openai_client is None:
        sentiment = basic_sentiment(caption)
        explanation = (
            "Sentiment estimated using keyword heuristics. "
            "Connect a valid OPENAI_API_KEY for smarter AI analysis."
        )
        suggestions = [
            "Add more context or a clear benefit for players.",
            "Include a strong call to action (e.g., 'Tell us what you think in the comments').",
            "Use hype words or emojis to match the energy of the post.",
        ]
        improved_caption = (
            caption or "This is going to be huge! Tell us what you think in the comments."
        )
        return {
            "sentiment": sentiment,
            "explanation": explanation,
            "suggestions": suggestions,
            "improved_caption": improved_caption,
        }

    system_prompt = """
You are a social media strategist specialized in gaming content.
Always output ONLY a valid JSON object (no markdown, no extra text).
Sentiment must be exactly one of: POSITIVE, NEUTRAL, NEGATIVE.
"""

    user_prompt = f"""
Analyze the following Instagram caption and posting context.

Caption:
{caption}

Context:
- Post type: {context.get("post_type")}
- Weekday: {context.get("weekday")}
- Posting hour (UTC): {context.get("hour")}
- Number of hashtags: {context.get("hashtags")}
- Profile: {context.get("profile")}

Tasks:
1. Classify sentiment (POSITIVE, NEUTRAL, or NEGATIVE).
2. Give a short explanation (1–2 sentences).
3. Provide 2–3 concrete suggestions to improve this post (writing, timing, clarity, CTA).
4. Suggest a new improved caption in English (hype + CTA + concise)
