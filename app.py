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
# CSS ESTILO HYPE QUEST
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
        padding-top: 3rem;              /* mais espaço no topo para não cortar o título */
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

    /* Container do logo na sidebar centralizado (vídeo ou imagem) */
    section[data-testid="stSidebar"] .hype-logo-wrapper {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }

    section[data-testid="stSidebar"] .hype-logo-wrapper video,
    section[data-testid="stSidebar"] div[data-testid="stImage"] img {
        width: 130px;
        height: auto;
        pointer-events: none;              /* não clicável */
        animation: hypeFloat 2.5s ease-in-out infinite;
    }

    /* Remover botão de fullscreen do componente padrão de imagem, caso usado como fallback */
    section[data-testid="stSidebar"] div[data-testid="stImage"] button {
        display: none !important;
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
# LOGO NA SIDEBAR (VÍDEO MP4 + FALLBACK PNG)
# =========================================================

VIDEO_LOGO_PATH = "HypeLogo(1).mp4"
STATIC_LOGO_PATH = "HypeLogo(1).png"  # fallback opcional (PNG com fundo transparente)

with st.sidebar:
    if os.path.exists(VIDEO_LOGO_PATH):
        # Usando HTML para ter autoplay/loop/muted sem controles e sem clique
        st.markdown(
            f"""
            <div class="hype-logo-wrapper">
                <video src="{VIDEO_LOGO_PATH}" autoplay loop muted playsinline></video>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Fallback para imagem estática, se o vídeo não estiver disponível
        try:
            st.image(STATIC_LOGO_PATH)
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
4. Suggest a new improved caption in English (hype + CTA + concise).

Return ONLY valid JSON with the keys:
- sentiment
- explanation
- suggestions (list of strings)
- improved_caption
"""

    try:
        chat = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=400,
        )

        raw_content = chat.choices[0].message.content

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            start = raw_content.find("{")
            end = raw_content.rfind("}")
            parsed = json.loads(raw_content[start: end + 1])

        parsed.setdefault("sentiment", "NEUTRAL")
        parsed.setdefault("explanation", "")
        parsed.setdefault("suggestions", [])
        parsed.setdefault("improved_caption", caption)

        return parsed

    except Exception as e:
        sentiment = basic_sentiment(caption)
        return {
            "sentiment": sentiment,
            "explanation": f"Fallback sentiment (API error: {e}).",
            "suggestions": [
                "Try clarifying what players should do next.",
                "Use more descriptive, emotional language.",
            ],
            "improved_caption": caption,
        }


# =========================================================
# META API FUNCTIONS
# =========================================================

@st.cache_data(ttl=600, show_spinner=False)
def fetch_follower_count(instagram_account_id: str, token: str) -> int:
    if not token or not instagram_account_id:
        return 150000

    BASE_URL = f"https://graph.facebook.com/v19.0/{instagram_account_id}"

    params = {
        "fields": "followers_count",
        "access_token": token,
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        raw_count = data.get("followers_count")
        if isinstance(raw_count, dict):
            count = raw_count.get("count")
        else:
            count = raw_count

        return int(count) if count is not None else 150000

    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Follower API Error: {e}. Using estimated followers.")
        return 150000


@st.cache_data(ttl=600, show_spinner=False)
def fetch_historical_data(instagram_account_id: str, token: str) -> pd.DataFrame:
    BASE_URL = f"https://graph.facebook.com/v19.0/{instagram_account_id}/media"

    fields = [
        "id",
        "caption",
        "timestamp",
        "media_type",
        "like_count",
        "comments_count",
    ]

    params = {
        "fields": ",".join(fields),
        "access_token": token,
        "limit": 100000,
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"API Request Error (media metrics): {e}")
        return pd.DataFrame()

    post_list = []

    for post in data.get("data", []):
        likes = post.get("like_count")
        comments = post.get("comments_count")

        if likes is None or comments is None:
            continue

        engagement = (likes or 0) + (comments or 0)

        media_type = post.get("media_type", "IMAGE").lower()
        if media_type == 'carousel_album':
            media_type = 'carousel'

        try:
            post_time = datetime.fromisoformat(post["timestamp"].replace("+0000", "+00:00"))
        except Exception:
            continue

        caption = post.get("caption", "") or ""

        post_list.append({
            "post_type": media_type,
            "weekday": post_time.strftime("%A"),
            "hour_utc": post_time.hour,
            "hashtags": caption.count("#"),
            "caption_length": len(caption),
            "likes": likes,
            "comments": comments,
            "shares": 0,
            "engagement": engagement,
            "engagement_rate": engagement / 100000,  # placeholder
            "id": post["id"],
            "timestamp": post_time,
            "caption": caption,
        })

    return pd.DataFrame(post_list)


# =========================================================
# FAKE API DATA
# =========================================================

def generate_fake_api_data(profile_handle: str, n_posts: int = 160, followers: int = 150000) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(profile_handle)) % (2**32))

    post_types = ["image", "video", "reels", "carousel"]
    weekdays = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]

    rows = []
    for _ in range(n_posts):
        pt = rng.choice(post_types)
        wd = rng.choice(weekdays)
        hour = int(rng.integers(0, 24))
        tags = int(rng.integers(0, 10))

        caption_len = max(20, int(rng.normal(200, 40)))

        base_eng = {
            "image": 900,
            "video": 1200,
            "reels": 1500,
            "carousel": 1100,
        }[pt]

        prime_time_bonus = 1.0
        if wd in ["Thursday", "Friday", "Saturday"] and 18 <= hour <= 22:
            prime_time_bonus = 1.3
        elif 0 <= hour <= 6:
            prime_time_bonus = 0.7

        hashtag_bonus = 1.0 + min(tags, 6) * 0.03

        noise = rng.normal(1.0, 0.2)

        engagement = max(
            50,
            int(base_eng * prime_time_bonus * hashtag_bonus * noise),
        )

        likes = int(engagement * rng.uniform(0.6, 0.8))
        comments = int(engagement * rng.uniform(0.15, 0.25))
        shares = engagement - likes - comments

        engagement_rate = engagement / followers

        rows.append(
            dict(
                profile=profile_handle,
                post_type=pt,
                weekday=wd,
                hour_utc=hour,
                hashtags=tags,
                caption_length=caption_len,
                likes=likes,
                comments=comments,
                shares=shares,
                followers=followers,
                engagement=engagement,
                engagement_rate=engagement_rate,
            )
        )

    return pd.DataFrame(rows)


# =========================================================
# Engagement-level helpers
# =========================================================

def compute_engagement_thresholds(df: pd.DataFrame) -> dict:
    if df.empty or "engagement_rate" not in df.columns:
        return {"low": 0.0, "high": 0.0}

    low = float(df["engagement_rate"].quantile(0.33))
    high = float(df["engagement_rate"].quantile(0.66))
    return {"low": low, "high": high}


def classify_engagement_level(er: float, thresholds: dict):
    low = thresholds.get("low", 0.0)
    high = thresholds.get("high", 0.0)

    if high <= 0:
        return (
            "UNKNOWN",
            "Not enough data to classify engagement for this profile yet.",
            "#6b7280",
        )

    if er < low:
        return (
            "LOW",
            "Below the typical range for this profile (bottom ~1/3 of historical posts).",
            "#ef4444",
        )
    elif er < high:
        return (
            "MEDIUM",
            "Within the usual range for this profile (middle ~1/3 of historical posts).",
            "#eab308",
        )
    else:
        return (
            "HIGH",
            "Above the usual range for this profile (top ~1/3 of historical posts).",
            "#22c55e",
        )


# =========================================================
# Model Training
# =========================================================

def train_engagement_model(df: pd.DataFrame):
    feature_cols = [
        "post_type",
        "weekday",
        "hour_utc",
        "hashtags",
        "caption_length",
    ]

    X = pd.get_dummies(df[feature_cols], drop_first=True)
    y = df["engagement_rate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeRegressor(max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns


# =========================================================
# Streamlit UI & Data Loading Logic
# =========================================================

st.sidebar.header("Profile & data")

PROFILE_CONFIG = {
    "@pubgbattlegrounds_mena": {
        "use_real_api": True,
        "instagram_id": INSTAGRAM_ID,
        "default_followers": 150_000,
    },
    "@yourprofile": {
        "use_real_api": False,
        "instagram_id": None,
        "default_followers": 10_000,
    },
}

available_profiles = list(PROFILE_CONFIG.keys())
profile = st.sidebar.selectbox("Instagram profile", available_profiles)
cfg = PROFILE_CONFIG[profile]

historical_df = pd.DataFrame()

# Followers
if cfg["use_real_api"] and META_TOKEN and cfg["instagram_id"]:
    current_followers = fetch_follower_count(cfg["instagram_id"], META_TOKEN)
else:
    current_followers = cfg["default_followers"]

st.sidebar.info(f"Followers for {profile}: {current_followers:,}")

# Histórico
if cfg["use_real_api"] and META_TOKEN and cfg["instagram_id"]:
    with st.spinner("Attempting to load real data from Meta API..."):
        historical_df = fetch_historical_data(cfg["instagram_id"], META_TOKEN)

if historical_df.empty or historical_df.shape[0] < 5:
    if cfg["use_real_api"]:
        st.sidebar.error(
            "API failed or returned insufficient data. Generating synthetic data for this profile."
        )
    else:
        st.sidebar.info("Using synthetic data for this profile.")
    historical_df = generate_fake_api_data(profile, n_posts=160, followers=current_followers)
    st.sidebar.success(f"Synthetic posts available for {profile}: {len(historical_df)}")
else:
    historical_df["followers"] = current_followers
    historical_df["engagement_rate"] = historical_df["engagement"] / current_followers
    st.sidebar.success(f"Loaded real posts for {profile}: {len(historical_df)}")

# Thresholds para LOW/MEDIUM/HIGH
engagement_thresholds = compute_engagement_thresholds(historical_df)

# Botão para refresh
if cfg["use_real_api"] and META_TOKEN and cfg["instagram_id"]:
    if st.sidebar.button("🔄 Refresh Instagram data"):
        fetch_follower_count.clear()
        fetch_historical_data.clear()
        st.rerun()

# Modelo
eng_model, eng_feature_columns = train_engagement_model(historical_df)

st.sidebar.markdown("---")
st.sidebar.subheader("Load post dataset (optional)")
uploaded = st.sidebar.file_uploader(
    "Upload CSV, Excel, JSON or Parquet file",
    type=["csv", "xlsx", "xls", "json", "parquet"],
)

if uploaded is not None:
    fname = uploaded.name.lower()
    try:
        if fname.endswith(".csv"):
            extra_df = pd.read_csv(uploaded)
        elif fname.endswith((".xlsx", ".xls")):
            extra_df = pd.read_excel(uploaded)
        elif fname.endswith(".json"):
            extra_df = pd.read_json(uploaded)
        elif fname.endswith(".parquet"):
            extra_df = pd.read_parquet(uploaded)
        st.sidebar.success(f"Loaded extra dataset: {uploaded.name}")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

# =========================================================
# Plan new post
# =========================================================

st.subheader("🧩 Plan a new Instagram post")

col_left, col_right = st.columns(2)

with col_left:
    post_type = st.selectbox("Post type", ["image", "video", "reels", "carousel"])
    hour = st.slider("Posting hour (UTC)", 0, 23, 20)
    weekday = st.selectbox(
        "Weekday",
        [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ],
    )
    hashtags = st.slider("Number of hashtags", 0, 15, 3)

st.markdown("### ✍️ Caption")

caption_text = st.text_area(
    "Caption text",
    key="caption_input",
    placeholder="Write your caption here...",
    height=120,
)

caption_length = len(caption_text or "")
st.caption(f"Caption length: {caption_length} characters (used as a feature).")

# Snapshot de inputs para invalidar resultado quando mudar algo
current_inputs = dict(
    post_type=post_type,
    hour=hour,
    weekday=weekday,
    hashtags=hashtags,
    caption=caption_text,
    profile=profile,
)

if "last_result" in st.session_state:
    last_inputs = st.session_state["last_result"].get("inputs", {})
    if last_inputs != current_inputs:
        del st.session_state["last_result"]

# =========================================================
# Button – Evaluate
# =========================================================

if st.button("✨ Evaluate caption & predict"):
    with st.spinner("Consulting the HypeQuest crystal ball... 🔮"):

        # Flag: legenda vazia ou não
        caption_was_empty = (caption_length == 0)

        if caption_was_empty:
            # NÃO faz análise de sentimento quando não há legenda
            sentiment_label = None
            sentiment_explanation = (
                "No caption text was provided, so sentiment was not evaluated. "
                "The prediction below uses only your posting context "
                "(post type, weekday, hour and hashtags)."
            )
            suggestions = [
                "Add at least a short caption so the AI can evaluate sentiment and optimize the copy.",
                "Include a clear call-to-action (CTA) to encourage interaction.",
            ]
            improved_caption = caption_text
        else:
            # Fluxo normal: usa GPT / fallback para analisar a legenda
            context = {
                "post_type": post_type,
                "weekday": weekday,
                "hour": hour,
                "hashtags": hashtags,
                "profile": profile,
            }
            gpt_result = gpt_caption_analysis(caption_text, context)
            sentiment_label = gpt_result["sentiment"].upper()
            sentiment_explanation = gpt_result.get("explanation", "")
            suggestions = gpt_result.get("suggestions", [])
            improved_caption = gpt_result.get("improved_caption", caption_text)

        # Predição de engajamento (sempre usa contexto, com ou sem legenda)
        new_row = pd.DataFrame(
            [
                dict(
                    post_type=post_type,
                    weekday=weekday,
                    hour_utc=hour,
                    hashtags=hashtags,
                    caption_length=caption_length if caption_length > 0 else 40,
                )
            ]
        )
        new_X = pd.get_dummies(new_row, drop_first=True).reindex(
            columns=eng_feature_columns, fill_value=0
        )
        predicted_eng_rate = float(eng_model.predict(new_X)[0])
        predicted_interactions = int(predicted_eng_rate * current_followers)

        st.session_state["last_result"] = dict(
            sentiment_label=sentiment_label,
            sentiment_explanation=sentiment_explanation,
            suggestions=suggestions,
            improved_caption=improved_caption,
            predicted_interactions=predicted_interactions,
            predicted_eng_rate=predicted_eng_rate,
            followers=current_followers,
            inputs=current_inputs,
            caption_was_empty=caption_was_empty,
        )

# =========================================================
# Results – only appear after Evaluate
# =========================================================

if "last_result" in st.session_state:
    res = st.session_state["last_result"]

    st.markdown("## 🔍 Prediction results")

    caption_was_empty = res.get("caption_was_empty", False)

    if caption_was_empty:
        # NÃO mostra POSITIVE/NEUTRAL/NEGATIVE; só informa que não há legenda
        st.markdown(
            """
            <div style="display:inline-flex;align-items:center;
                        padding:4px 12px;border-radius:999px;
                        background-color:#e5e7eb;
                        border:1px solid #d1d5db;
                        margin-bottom:8px;">
                <span style="width:8px;height:8px;border-radius:999px;
                            background-color:#9ca3af;
                            display:inline-block;margin-right:8px;"></span>
                <span style="font-weight:600;color:#4b5563;">
                    No caption added
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(
            "No caption text was provided, so sentiment was not evaluated. "
            "The engagement prediction below uses only your posting context "
            "(post type, weekday, hour and hashtags)."
        )
    else:
        sentiment_color = {
            "POSITIVE": "#22c55e",
            "NEGATIVE": "#ef4444",
            "NEUTRAL": "#eab308",
        }.get(res["sentiment_label"], "#6b7280")

        st.markdown(
            f"""
            <div style="display:inline-flex;align-items:center;
                        padding:4px 12px;border-radius:999px;
                        background-color:{sentiment_color}22;
                        border:1px solid {sentiment_color};
                        margin-bottom:8px;">
                <span style="width:8px;height:8px;border-radius:999px;
                              background-color:{sentiment_color};
                              display:inline-block;margin-right:8px;"></span>
                <span style="font-weight:600;color:{sentiment_color};">
                    {res['sentiment_label']}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write(res["sentiment_explanation"])

    eng_level, eng_level_msg, eng_level_color = classify_engagement_level(
        res["predicted_eng_rate"], engagement_thresholds
    )

    st.markdown("")
    st.markdown(
        f"""
        <div style="border-radius:12px;padding:14px 18px;
                    background-color:#e0f2fe;border:1px solid #bae6fd;">
            <div style="font-size:13px;color:#0369a1;
                         font-weight:600;margin-bottom:4px;">
                Predicted engagement
            </div>
            <div style="font-size:20px;font-weight:700;color:#0f172a;">
                {res['predicted_interactions']:,} interactions
            </div>
            <div style="font-size:12px;color:#0369a1;margin-top:4px;">
                Engagement rate: {res['predicted_eng_rate']*100:.2f}% for ~{res['followers']:,} followers.
            </div>
            <div style="margin-top:8px;">
                <div style="display:inline-flex;align-items:center;
                            padding:4px 10px;border-radius:999px;
                            background-color:{eng_level_color}22;
                            border:1px solid {eng_level_color};
                            font-size:12px;font-weight:600;
                            color:{eng_level_color};">
                    <span style="width:8px;height:8px;border-radius:999px;
                                 background-color:{eng_level_color};
                                 display:inline-block;margin-right:8px;"></span>
                    <span>Engagement level: {eng_level}</span>
                </div>
                <div style="font-size:11px;color:#0369a1;margin-top:4px;">
                    {eng_level_msg}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Sentiment is estimated using caption text and context (when provided); "
        "engagement is predicted using a Decision Tree model trained on this "
        "profile's historical posts (real or synthetic)."
    )

    st.markdown("### 💡 Suggestions to improve this post")

    tips = []

    if not (18 <= hour <= 22 and weekday in ["Thursday", "Friday", "Saturday"]):
        tips.append(
            "Historical data shows stronger engagement between 18–22 UTC on Thursday–Saturday. "
            "Consider testing this post closer to prime time."
        )

    if caption_length < 40:
        tips.append(
            "Your caption is very short. Consider adding a bit more context, a hook, or a clear benefit for players."
        )
    elif caption_length > 350:
        tips.append(
            "Your caption is quite long. Check if you can tighten it while keeping the key message and CTA."
        )

    tips.extend(res["suggestions"])

    for t in tips:
        st.markdown(f"- {t}")

    st.markdown("### ✨ Suggested improved caption")

    st.info(
        "This suggestion is generated from your original caption. "
        "You can copy & paste it into the caption editor above and tweak it as you like."
    )

    st.text_area(
        "Suggested version (you can copy/paste):",
        value=res["improved_caption"],
        height=90,
        key="suggested_caption_display",
    )

st.markdown("---")
st.caption(
    "Data used for training is either fetched live from the Meta API or generated synthetically as a fallback."
)
