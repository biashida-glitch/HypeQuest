import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
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

    /* Card pixelado (pode ser usado depois se quiser) */
    .pixel-card {
        background: #8FD0FF;
        border: 4px solid #1C0A4A;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0px 5px 0px #1C0A4A;
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
# OpenAI client (optional) – works with new SDK (>=1.0)
# =========================================================

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
    "win",
    "victory",
    "amazing",
    "awesome",
    "love",
    "great",
    "hype",
    "excited",
    "epic",
    "gg",
    "insane",
    "crazy",
    "wow",
    "🔥",
    "🥳",
}
NEGATIVE_WORDS = {
    "bug",
    "issue",
    "error",
    "lag",
    "toxic",
    "hate",
    "angry",
    "sad",
    "broken",
    "crash",
    "fuck",
    "shit",
    "bad",
    "😭",
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
    """
    Uses GPT (if available) to:
      - classify sentiment
      - explain
      - give suggestions
      - suggest an improved caption

    If no API / error → falls back to basic heuristic.
    """

    # -------- Fallback (no OpenAI) --------
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

    # -------- Real GPT call (chat.completions) --------
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
- Topic: {context.get("topic")}
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

        import json

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            # If the model adds extra text, try to extract JSON portion
            start = raw_content.find("{")
            end = raw_content.rfind("}")
            parsed = json.loads(raw_content[start : end + 1])

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
# Fake API – historical data per profile
# =========================================================


def generate_fake_api_data(profile_handle: str, n_posts: int = 160) -> pd.DataFrame:
    """
    Creates a synthetic dataset for a given Instagram profile.
    """
    rng = np.random.default_rng(abs(hash(profile_handle)) % (2**32))

    post_types = ["image", "video", "reels", "carousel"]
    weekdays = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    topics = ["update", "trailer", "gameplay", "community", "collab", "maintenance"]

    followers = int(rng.integers(50_000, 200_000))

    rows = []
    for _ in range(n_posts):
        pt = rng.choice(post_types)
        wd = rng.choice(weekdays)
        hour = int(rng.integers(0, 24))
        tags = int(rng.integers(0, 10))
        topic = rng.choice(topics)
        month = int(rng.integers(1, 13))

        base_len = {
            "update": 220,
            "trailer": 160,
            "gameplay": 180,
            "community": 190,
            "collab": 210,
            "maintenance": 170,
        }[topic]
        caption_len = max(20, int(rng.normal(base_len, 40)))

        base_eng = {
            "image": 800,
            "video": 1100,
            "reels": 1400,
            "carousel": 1000,
        }[pt]

        prime_time_bonus = 1.0
        if wd in ["Thursday", "Friday", "Saturday"] and 18 <= hour <= 22:
            prime_time_bonus = 1.3
        elif 0 <= hour <= 6:
            prime_time_bonus = 0.7

        hashtag_bonus = 1.0 + min(tags, 6) * 0.03
        topic_bonus = {
            "update": 1.1,
            "trailer": 1.05,
            "gameplay": 1.08,
            "community": 0.95,
            "collab": 1.12,
            "maintenance": 0.85,
        }[topic]

        noise = rng.normal(1.0, 0.2)

        engagement = max(
            50,
            int(base_eng * prime_time_bonus * hashtag_bonus * topic_bonus * noise),
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
                topic=topic,
                caption_length=caption_len,
                month=month,
                likes=likes,
                comments=comments,
                shares=shares,
                followers=followers,
                engagement=engagement,
                engagement_rate=engagement_rate,
            )
        )

    return pd.DataFrame(rows)


def train_engagement_model(df: pd.DataFrame):
    feature_cols = [
        "post_type",
        "weekday",
        "hour_utc",
        "hashtags",
        "topic",
        "caption_length",
        "month",
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
# HEADER HYPE QUEST (KV + TÍTULO)
# =========================================================

# TODO: ajuste o caminho/URL da imagem para o seu KV
KV_IMAGE_PATH = "hypequest_kv.png"  # pode ser também uma URL

try:
    st.image(KV_IMAGE_PATH, use_column_width=True)
except Exception:
    # se a imagem não existir, só ignora
    pass

st.markdown("<div class='hype-title'>HYPE QUEST</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hype-subtitle'>Instagram Engagement & Sentiment Prediction</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div class='hype-status'>{IA_STATUS}</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# =========================================================
# Streamlit UI
# =========================================================

# Initialise caption input in session_state
if "caption_input" not in st.session_state:
    st.session_state["caption_input"] = ""

# Sidebar – profile & upload
st.sidebar.header("Profile & data")

available_profiles = ["@pubg", "@playinzoi", "@hypequest"]
profile = st.sidebar.selectbox("Instagram profile", available_profiles)

historical_df = generate_fake_api_data(profile, n_posts=160)
st.sidebar.success(f"Historical posts available for {profile}: {len(historical_df)}")

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

# Train engagement model
eng_model, eng_feature_columns = train_engagement_model(historical_df)

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
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
    )
    hashtags = st.slider("Number of hashtags", 0, 15, 3)

with col_right:
    topic = st.selectbox(
        "Post topic",
        ["update", "trailer", "gameplay", "community", "collab", "maintenance"],
    )
    month_name = st.selectbox(
        "Month (display only)",
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
    )
    month = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ].index(month_name) + 1

st.markdown("### ✍️ Caption")

caption_text = st.text_area(
    "Caption text",
    key="caption_input",
    placeholder="Write your caption here...",
    height=120,
)

caption_length = len(caption_text or "")
st.caption(f"Caption length: {caption_length} characters (used as a feature).")

# ---------------------------------------------------------
# Build current input snapshot (for invalidating results)
# ---------------------------------------------------------
current_inputs = dict(
    post_type=post_type,
    hour=hour,
    weekday=weekday,
    hashtags=hashtags,
    topic=topic,
    month=month,
    caption=caption_text,
    profile=profile,
)

# If we have previous result and inputs changed → drop result
if "last_result" in st.session_state:
    last_inputs = st.session_state["last_result"].get("inputs", {})
    if last_inputs != current_inputs:
        del st.session_state["last_result"]

# =========================================================
# Button – Evaluate
# =========================================================

if st.button("✨ Evaluate caption & predict"):
    with st.spinner("Consulting the HypeQuest crystal ball... 🔮"):
        # Generative AI or fallback
        context = {
            "post_type": post_type,
            "weekday": weekday,
            "hour": hour,
            "hashtags": hashtags,
            "topic": topic,
            "profile": profile,
        }
        gpt_result = gpt_caption_analysis(caption_text, context)
        sentiment_label = gpt_result["sentiment"].upper()
        sentiment_explanation = gpt_result.get("explanation", "")
        suggestions = gpt_result.get("suggestions", [])
        improved_caption = gpt_result.get("improved_caption", caption_text)

        # Engagement prediction
        new_row = pd.DataFrame(
            [
                dict(
                    post_type=post_type,
                    weekday=weekday,
                    hour_utc=hour,
                    hashtags=hashtags,
                    topic=topic,
                    caption_length=caption_length if caption_length > 0 else 40,
                    month=month,
                )
            ]
        )
        new_X = pd.get_dummies(new_row, drop_first=True).reindex(
            columns=eng_feature_columns, fill_value=0
        )
        predicted_eng_rate = float(eng_model.predict(new_X)[0])

        followers = int(historical_df["followers"].iloc[0])
        predicted_interactions = int(predicted_eng_rate * followers)

        st.session_state["last_result"] = dict(
            sentiment_label=sentiment_label,
            sentiment_explanation=sentiment_explanation,
            suggestions=suggestions,
            improved_caption=improved_caption,
            predicted_interactions=predicted_interactions,
            predicted_eng_rate=predicted_eng_rate,
            followers=followers,
            inputs=current_inputs,
        )

# =========================================================
# Results – only appear after Evaluate and if inputs unchanged
# =========================================================

if "last_result" in st.session_state:
    res = st.session_state["last_result"]

    st.markdown("## 🔍 Prediction results")

    # Sentiment pill
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

    # Engagement card
    st.markdown("")
    st.markdown(
        f"""
        <div style="border-radius:12px;padding:16px 20px;
                    background-color:#e0f2fe;border:1px solid #bae6fd;">
            <div style="font-size:13px;color:#0369a1;
                        font-weight:600;margin-bottom:4px;">
                Predicted engagement
            </div>
            <div style="font-size:22px;font-weight:700;color:#0f172a;">
                {res['predicted_interactions']:,} interactions
            </div>
            <div style="font-size:12px;color:#0369a1;margin-top:4px;">
                Engagement rate: {res['predicted_eng_rate']*100:.2f}% for ~{res['followers']:,} followers.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Sentiment is estimated using caption text and context; engagement is "
        "predicted using a Decision Tree model trained on this profile's synthetic historical posts."
    )

    # Suggestions
    st.markdown("### 💡 Suggestions to improve this post")

    tips = []

    if not (18 <= hour <= 22 and weekday in ["Thursday", "Friday", "Saturday"]):
        tips.append(
            "Historical data shows stronger engagement between **18–22 UTC** on Thursday–Saturday. "
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

    # Suggested caption
    st.markdown("### ✨ Suggested improved caption")

    st.info(
        "This suggestion is generated from your original caption. "
        "You can apply it to the editor above and tweak it as you like."
    )

    st.text_area(
        "Suggested version (you can copy/paste):",
        value=res["improved_caption"],
        height=90,
        key="suggested_caption_display",
    )

    if st.button("📋 Use this suggested caption"):
        try:
            st.session_state["caption_input"] = res["improved_caption"]
        except Exception:
            pass
        st.success("Suggested caption applied to the editor above. You can edit it before posting.")
        st.experimental_rerun()

st.markdown("---")
st.caption(
    "Tips are based on patterns learned from a simulated API dataset. "
    "In production, this would be connected to real Instagram insights."
)

