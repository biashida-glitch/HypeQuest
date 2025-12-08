import os
import json
from datetime import datetime

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
# CSS ESTILO HYPE QUEST
# ------------------------------
st.markdown("""
<style>
    /* Fundo geral - céu */
    .main {
        background-color: #B7E3FF;
    }

    /* Centralizar container principal e limitar largura */
    .block-container {
        padding-top: 1rem;
        max-width: 1100px;
        margin: 0 auto;
    }

    /* Título principal estilo pixel */
    .hype-title {
        font-family: monospace;
        font-size: 40px;
        font-weight: 900;
        letter-spacing: 4px;
        color: #1C0A4A;
        text-align: center;
        text-transform: uppercase;
        text-shadow: 3px 3px 0px #8FD0FF;
        margin-top: 8px;
        margin-bottom: 4px;
    }

    /* Subtítulo */
    .hype-subtitle {
        font-family: monospace;
        font-size: 16px;
        color: #1C0A4A;
        text-align: center;
        margin-bottom: 8px;
    }

    /* Badge de status IA */
    .hype-status {
        font-size: 12px;
        color: #6b7280;
        text-align: center;
        margin-bottom: 6px;
    }

    /* Linha divisória mais suave */
    hr {
        border: none;
        border-top: 2px solid #8FD0FF;
        margin: 0.6rem 0 1rem 0;
    }

    /* Botão estilo START */
    .stButton>button {
        border-radius: 6px;
        border: 3px solid #1C0A4A;
        background: linear-gradient(180deg, #FFDD55 0%, #FFB800 70%);
        color: #1C0A4A;
        font-weight: bold;
        font-size: 18px;
        padding: 6px 22px;
        box-shadow: 0px 4px 0px #D98F00;
    }

    .stButton>button:hover {
        background: linear-gradient(180deg, #FFE680 0%, #FFC933 70%);
        box-shadow: 0px 4px 0px #C57D00;
    }

    /* Sidebar com azul intermediário */
    section[data-testid="stSidebar"] {
        background-color: #8FD0FF !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# API Credentials and Client Setup
# =========================================================

# Load Meta API Credentials from secrets
META_TOKEN = st.secrets.get("META_ACCESS_TOKEN", None)
INSTAGRAM_ID = st.secrets.get("INSTAGRAM_ACCOUNT_ID", None)

# OpenAI client (optional) – works with new SDK (>=1.0)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
openai_client = None

try:
    if OPENAI_API_KEY:
        from openai import OpenAI

        openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception:
    openai_client = None

if OPENAI_API_KEY and openai_client:
    IA_STATUS = "✅ Generative AI (GPT-4.1-mini) enabled"
else:
    IA_STATUS = "⚠️ Generative AI disabled (no valid OPENAI_API_KEY or OpenAI SDK)"

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
