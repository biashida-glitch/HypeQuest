import os
import random
from typing import Tuple, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =====================================================
# Sentiment configuration
# =====================================================

analyzer = SentimentIntensityAnalyzer()

CUSTOM_NEGATIVE_WORDS = {
    "fuck", "fucking", "shit", "trash", "noob", "hate", "garbage",
}

POSITIVE_KEYWORDS = {
    "hype", "amazing", "awesome", "big news", "update", "free", "gift",
    "event", "special", "collab", "collaboration", "season", "launch",
}
NEGATIVE_KEYWORDS = {
    "bug", "issue", "error", "problem", "maintenance", "delay", "cheater",
}

FEATURE_COLS = ["post_type", "hour", "weekday", "topic", "caption_len", "hashtags"]


# =====================================================
# Fake API – histórico por profile
# =====================================================

def load_fake_api_profile(handle: str) -> pd.DataFrame:
    """
    Simula dados vindos de uma API de Instagram para cada profile.
    AQUI é onde o modelo aprende padrões de engajamento.
    """
    seed = 7 if handle == "@pubg" else (21 if handle == "@playinzoi" else 99)
    rng = np.random.default_rng(seed=seed)

    n = 300
    post_types = ["image", "video", "reels", "carousel"]
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    topics = ["update", "maintenance", "trailer", "collaboration", "gameplay", "community"]

    df = pd.DataFrame({
        "post_type": rng.choice(post_types, size=n),
        "hour": rng.integers(0, 24, size=n),
        "weekday": rng.choice(weekdays, size=n),
        "topic": rng.choice(topics, size=n),
        "caption_len": rng.integers(40, 260, size=n),
        "hashtags": rng.integers(0, 15, size=n),
    })

    # --- Engajamento com padrões bem fortes (pra ficar visível) ---
    base = rng.normal(500, 80, size=n)

    # Tipo de post
    base += np.where(df["post_type"] == "reels", 500, 0)
    base += np.where(df["post_type"] == "video", 300, 0)
    base += np.where(df["post_type"] == "image", 150, 0)
    base += np.where(df["post_type"] == "carousel", 200, 0)

    # Horário (UTC)
    hour = df["hour"]
    base += np.where(hour.between(18, 22), 400, 0)       # prime time
    base += np.where(hour.between(12, 15), 150, 0)       # almoço
    base += np.where(hour.between(0, 6), -200, 0)        # madrugada

    # Dia da semana
    base += np.where(df["weekday"].isin(["Friday", "Saturday"]), 350, 0)
    base += np.where(df["weekday"] == "Sunday", 200, 0)
    base += np.where(df["weekday"].isin(["Monday", "Tuesday"]), -80, 0)

    # Tópico
    base += np.where(df["topic"].isin(["update", "trailer", "collaboration"]), 250, 0)
    base += np.where(df["topic"] == "maintenance", -100, 0)

    # Legenda: ideal entre 80 e 220 caracteres
    len_center = 150
    base += -0.8 * np.abs(df["caption_len"] - len_center) + 120

    # Hashtags: até 8 ajuda, depois satura
    base += np.minimum(df["hashtags"], 8) * 40

    noise = rng.normal(0, 80, size=n)
    df["engagement"] = np.maximum(50, base + noise).astype(int)

    return df


# =====================================================
# Sentiment helpers
# =====================================================

def classify_sentiment(text: str) -> Tuple[str, Dict[str, Any]]:
    if not text or not text.strip():
        return "NEUTRAL", {"compound": 0.0}

    low = text.lower()
    scores = analyzer.polarity_scores(low)
    comp = scores["compound"]

    # Força negativo se tiver palavrão e não for claramente positivo
    if any(bad in low for bad in CUSTOM_NEGATIVE_WORDS) and comp <= 0.3:
        return "NEGATIVE", scores

    if comp >= 0.15:
        return "POSITIVE", scores
    elif comp <= -0.15:
        return "NEGATIVE", scores
    else:
        return "NEUTRAL", scores


def render_sentiment_badge(label: str) -> str:
    label = label.upper()
    color = {
        "POSITIVE": "#16a34a",
        "NEGATIVE": "#dc2626",
        "NEUTRAL":  "#f97316",
    }.get(label, "#6b7280")

    return f"""
    <span style="
        display:inline-flex;
        align-items:center;
        padding:4px 12px;
        border-radius:999px;
        font-size:12px;
        font-weight:600;
        color:white;
        background:{color};
        letter-spacing:0.03em;
    ">{label}</span>
    """


# =====================================================
# Caption suggestion (pseudo-generativa)
# =====================================================

def generate_caption_suggestion(
    original: str,
    sentiment_label: str,
    context: Dict[str, Any]
) -> str:
    """
    Gera legenda sugerida. Aqui é onde você plugaria um LLM real no futuro.
    """
    text = (original or "").strip()
    if not text:
        text = "Big news is coming to the Battlegrounds"

    post_type = context.get("post_type", "post")
    topic = context.get("topic", "update")
    handle = context.get("handle", "")

    core = text.strip().rstrip(".!").capitalize()

    hype_phrases = [
        "This is going to be huge!",
        "You don't want to miss this.",
        "Get ready for the next drop.",
        "Your squad is going to love this.",
    ]
    cta_variants = [
        "Drop your thoughts in the comments.",
        "Tell us if you're ready.",
        "Tag your squad and share this.",
        "Let us know what you think below.",
    ]

    if sentiment_label == "NEGATIVE":
        tone_intro = "We know things can be tough sometimes, but we're working on it."
    elif sentiment_label == "NEUTRAL":
        tone_intro = "Here's what's coming up next."
    else:
        tone_intro = random.choice([
            "We're hyped for this!",
            "This is just the beginning!",
            "The next drop is almost here!",
        ])

    if handle:
        handle_part = f" for {handle}"
    else:
        handle_part = ""

    suggestion = (
        f"{core}. "
        f"{tone_intro} "
        f"This {post_type} about {topic}{handle_part} is one you won't want to miss. "
        f"{random.choice(hype_phrases)} "
        f"{random.choice(cta_variants)}"
    )

    suggestion = suggestion.strip()
    if not suggestion.endswith((".", "!", "?")):
        suggestion += "."

    return suggestion


# =====================================================
# Engagement model helpers
# =====================================================

def train_engagement_model(df: pd.DataFrame) -> Tuple[DecisionTreeRegressor, pd.DataFrame]:
    # Garante que temos apenas as colunas necessárias
    df = df[[c for c in FEATURE_COLS + ["engagement"] if c in df.columns]].dropna()

    X = pd.get_dummies(df[FEATURE_COLS], drop_first=False)
    y = df["engagement"]

    model = DecisionTreeRegressor(max_depth=6, random_state=42)
    model.fit(X, y)
    return model, X


def predict_engagement(model: DecisionTreeRegressor, X_train: pd.DataFrame, feat_dict: Dict[str, Any]) -> float:
    new_df = pd.DataFrame([feat_dict])
    new_X = pd.get_dummies(new_df).reindex(columns=X_train.columns, fill_value=0)
    return float(model.predict(new_X)[0])


def baseline_engagement_estimate(features: Dict[str, Any]) -> float:
    base = 400
    hour = features.get("hour", 12)
    weekday = features.get("weekday", "Wednesday")
    post_type = features.get("post_type", "image")
    topic = features.get("topic", "update")
    caption_len = features.get("caption_len", 100)
    hashtags = features.get("hashtags", 3)

    if post_type == "reels":
        base += 300
    elif post_type == "video":
        base += 180
    elif post_type == "carousel":
        base += 120

    if weekday in ["Friday", "Saturday"]:
        base += 250
    if 18 <= hour <= 22:
        base += 250

    if topic in ["update", "trailer", "collaboration"]:
        base += 200

    base += np.minimum(hashtags, 8) * 35
    base += -0.6 * np.abs(caption_len - 150) + 100

    return max(50, base)


# =====================================================
# Streamlit layout
# =====================================================

st.set_page_config(
    page_title="HypeQuest – Instagram Engagement & Sentiment",
    layout="wide",
)

st.title("🔥 HypeQuest – Instagram Engagement & Sentiment Prediction")
st.caption(
    "Prototype that predicts engagement and sentiment using a Decision Tree trained on simulated Instagram API data. "
    "It can later be connected to real APIs per profile."
)
st.markdown("---")

# Sidebar – profile & data
st.sidebar.header("👤 Profile & data")

profiles = ["@pubg", "@playinzoi", "@other"]
selected_profile = st.sidebar.selectbox("Instagram profile", profiles, index=0)

api_df = load_fake_api_profile(selected_profile)
st.sidebar.success(f"Historical posts available for {selected_profile}: {len(api_df)}")

st.sidebar.markdown("---")
st.sidebar.header("📁 Optional dataset upload")
uploaded = st.sidebar.file_uploader(
    "Upload extra history (CSV / Excel / JSON / Parquet)",
    type=["csv", "xlsx", "xls", "json", "parquet"],
)

extra_df = None
if uploaded is not None:
    try:
        name = uploaded.name.lower()
        if name.endswith(".csv"):
            extra_df = pd.read_csv(uploaded)
        elif name.endswith((".xlsx", ".xls")):
            extra_df = pd.read_excel(uploaded)
        elif name.endswith(".json"):
            extra_df = pd.read_json(uploaded)
        elif name.endswith(".parquet"):
            extra_df = pd.read_parquet(uploaded)
        st.sidebar.success(f"Loaded {uploaded.name}")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

train_df = api_df.copy()
if extra_df is not None:
    common_cols = [c for c in FEATURE_COLS + ["engagement"] if c in extra_df.columns]
    if common_cols:
        train_df = pd.concat(
            [train_df, extra_df[common_cols].copy()],
            ignore_index=True,
        )
        st.sidebar.info("Merged uploaded dataset with API history.")

# Treina modelo
engagement_model = None
X_train = None
if not train_df.empty:
    engagement_model, X_train = train_engagement_model(train_df)

# =====================================================
# Plan new post
# =====================================================

st.markdown("## 🧩 Plan a new Instagram post")

col_left, col_right = st.columns([2, 2])

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
post_types = ["image", "video", "reels", "carousel"]
topics = ["update", "maintenance", "trailer", "collaboration", "gameplay", "community"]
months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

with col_left:
    post_type = st.selectbox("Post type", post_types, index=1)
    hour = st.slider("Posting hour (UTC)", 0, 23, 20)
    weekday = st.selectbox("Weekday", weekdays, index=6)
    hashtags = st.slider("Number of hashtags", 0, 15, 3)

with col_right:
    topic = st.selectbox("Post topic", topics, index=0)
    month_label = st.selectbox("Month", months, index=0)
    st.caption("Month is used only for context in your presentation – the model focuses on patterns in hour/day/type/topic.")

st.markdown("### ✍️ Caption")

if "caption_text" not in st.session_state:
    st.session_state["caption_text"] = ""

caption_text = st.text_area(
    "Caption text",
    value=st.session_state["caption_text"],
    height=120,
    key="caption_text",
    placeholder="Write your caption here...",
)

caption_len = len(caption_text or "")
st.caption(f"Caption length: **{caption_len} characters** (feature used by the model).")

run_eval = st.button("✨ Evaluate caption & predict", type="primary")

if run_eval:
    with st.spinner("Thinking like a HypeQuest strategist..."):
        # Sentiment
        sentiment_label, sentiment_scores = classify_sentiment(caption_text)

        feat_dict = {
            "post_type": post_type,
            "hour": hour,
            "weekday": weekday,
            "topic": topic,
            "caption_len": caption_len,
            "hashtags": hashtags,
        }

        if engagement_model is not None and X_train is not None:
            pred_engagement = predict_engagement(engagement_model, X_train, feat_dict)
            source_msg = "Predicted using a Decision Tree model trained on this profile's historical posts."
        else:
            pred_engagement = baseline_engagement_estimate(feat_dict)
            source_msg = "Estimated using a heuristic baseline (no training data available)."

        suggestion_context = {
            "handle": selected_profile,
            "post_type": post_type,
            "topic": topic,
            "weekday": weekday,
            "hour": hour,
        }
        suggested_caption = generate_caption_suggestion(
            original=caption_text,
            sentiment_label=sentiment_label,
            context=suggestion_context,
        )

    # =================================================
    # Visual results
    # =================================================
    st.markdown("## 🔍 Prediction results")

    badge_html = render_sentiment_badge(sentiment_label)
    st.markdown(
        f"""
        <div style="margin-bottom:0.75rem;">
            <div style="font-size:14px;color:#4b5563;margin-bottom:4px;">
                Predicted sentiment for this post (based on caption + context):
            </div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="
            background:#e0edff;
            border-radius:12px;
            padding:16px 20px;
            margin-top:8px;
            margin-bottom:4px;
            border:1px solid #bfdbfe;
        ">
            <div style="font-size:13px;font-weight:600;color:#1d4ed8;margin-bottom:4px;">
                Predicted engagement
            </div>
            <div style="font-size:20px;font-weight:700;color:#111827;">
                {int(pred_engagement):,} interactions
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(source_msg)

    # Sugestões
    st.markdown("### 💡 Suggestions to improve this post")
    suggestions = []

    if caption_len < 40:
        suggestions.append("Caption is very short. Try adding more context, a hook, or a clear benefit for players.")
    if caption_len > 260:
        suggestions.append("Caption is quite long. Consider tightening the message to keep it punchy.")
    if not (18 <= hour <= 22 and weekday in ["Friday", "Saturday", "Sunday"]):
        suggestions.append("You are outside a typical prime-time window (18–22 on Fri–Sun). Check profile audience insights.")
    if hashtags == 0:
        suggestions.append("No hashtags detected. Adding a few relevant tags can help discoverability.")
    if sentiment_label == "NEUTRAL":
        suggestions.append("Caption feels neutral. Try adding hype words, emojis, or a stronger emotion.")
    if sentiment_label == "NEGATIVE":
        suggestions.append("Tone feels negative. Make sure this matches the message (e.g., maintenance/issues). Otherwise, soften the language.")

    if not suggestions:
        suggestions.append("This setup looks solid based on the learned patterns for this profile.")

    for s_ in suggestions:
        st.markdown(f"- {s_}")

    # Caption sugerida
    st.markdown("### ✨ Suggested improved caption")

    st.markdown(
        """
        <div style="
            background:#eff6ff;
            border-radius:12px;
            padding:12px 16px;
            font-size:13px;
            color:#1f2937;
            border:1px solid #bfdbfe;
            margin-bottom:8px;
        ">
            This suggestion is generated from your original caption and the context of this post.
            Click the button below to copy it into the editor above and tweak it as you like.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.text_area(
        "Suggested version (you can copy/paste):",
        value=suggested_caption,
        height=100,
        key="suggested_caption_box",
    )

    if st.button("📋 Use this suggested caption"):
        st.session_state["caption_text"] = suggested_caption
        st.success("Suggested caption copied to the editor above. You can tweak it before posting!")

else:
    st.info("Set your post parameters and caption, then click **✨ Evaluate caption & predict** to see results.")
