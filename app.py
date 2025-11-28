import os
import random
import string
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# Optional: OpenAI client for generative caption help
# ---------------------------------------------------------
try:
    from openai import OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    OPENAI_API_KEY = None
    openai_client = None

# ---------------------------------------------------------
# Basic keyword-based sentiment (fallback when no API)
# ---------------------------------------------------------

POSITIVE_WORDS = {
    "win", "victory", "amazing", "awesome", "love", "great", "hype",
    "excited", "epic", "gg", "insane", "crazy", "wow", "🔥", "🥳"
}
NEGATIVE_WORDS = {
    "bug", "issue", "error", "lag", "toxic", "hate", "angry",
    "sad", "broken", "crash", "fuck", "shit", "bad", "😭"
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


# ---------------------------------------------------------
# Generative caption analysis with OpenAI (if API available)
# ---------------------------------------------------------

def gpt_caption_analysis(caption: str, context: dict) -> dict:
    """
    Uses GPT (if available) to classify sentiment, explain, suggest tips
    and propose an improved caption. If API is not available, falls back
    to rule-based sentiment and simple tips.
    """
    # Fallback if no API key
    if openai_client is None:
        sentiment = basic_sentiment(caption)
        explanation = (
            "Sentiment estimated using keyword heuristics. "
            "Connect a real API key later for smarter analysis."
        )
        suggestions = [
            "Add more context or a clear benefit for players.",
            "Include a strong call to action (e.g., 'Drop your thoughts in the comments').",
            "Use hype words or emojis to match the energy of your post."
        ]
        improved_caption = caption or "This is going to be huge! Tell us what you think in the comments."
        return {
            "sentiment": sentiment,
            "explanation": explanation,
            "suggestions": suggestions,
            "improved_caption": improved_caption,
        }

    # If we DO have an API key, call GPT
    system_prompt = """
You are a social media strategist specialized in gaming content.
Always output a pure JSON object (no markdown, no commentary).
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
        response = openai_client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_output_tokens=400,
        )
        content = response.output[0].content[0].text
        import json
        parsed = json.loads(content)
        # sanity defaults
        parsed.setdefault("sentiment", "NEUTRAL")
        parsed.setdefault("explanation", "")
        parsed.setdefault("suggestions", [])
        parsed.setdefault("improved_caption", caption)
        return parsed
    except Exception:
        # robust fallback
        sentiment = basic_sentiment(caption)
        return {
            "sentiment": sentiment,
            "explanation": "Fallback sentiment (API error).",
            "suggestions": [
                "Try clarifying what players should do next.",
                "Use more descriptive, emotional language.",
            ],
            "improved_caption": caption,
        }


# ---------------------------------------------------------
# Fake "API" historical data (per profile) for the hackathon
# ---------------------------------------------------------

def generate_fake_api_data(profile_handle: str, n_posts: int = 160) -> pd.DataFrame:
    """
    Creates a realistic synthetic dataset for a given Instagram profile.
    Columns: post_type, weekday, hour_utc, hashtags, topic, caption_length,
             month, likes, comments, shares, followers, engagement,
             engagement_rate.
    """
    rng = np.random.default_rng(abs(hash(profile_handle)) % (2**32))

    post_types = ["image", "video", "reels", "carousel"]
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    topics = ["update", "trailer", "gameplay", "community", "collab", "maintenance"]

    followers = rng.integers(50000, 200000)  # each profile has a fixed follower base

    rows = []
    for _ in range(n_posts):
        pt = rng.choice(post_types)
        wd = rng.choice(weekdays)
        hour = int(rng.integers(0, 24))
        tags = int(rng.integers(0, 10))
        topic = rng.choice(topics)
        month = int(rng.integers(1, 13))

        # caption length: different distributions by topic
        base_len = {
            "update": 220,
            "trailer": 160,
            "gameplay": 180,
            "community": 190,
            "collab": 210,
            "maintenance": 170,
        }[topic]
        caption_len = max(20, int(rng.normal(base_len, 40)))

        # engagement baseline by post type
        base_eng = {
            "image": 800,
            "video": 1100,
            "reels": 1400,
            "carousel": 1000,
        }[pt]

        # weekday/hora effect
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

        # split engagement into likes/comments/shares in rough proportions
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


def train_engagement_model(df: pd.DataFrame) -> DecisionTreeRegressor:
    """
    Trains a DecisionTreeRegressor to predict engagement rate based on features.
    """
    feature_cols = ["post_type", "weekday", "hour_utc", "hashtags", "topic", "caption_length", "month"]

    X = pd.get_dummies(df[feature_cols], drop_first=True)
    y = df["engagement_rate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeRegressor(max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns


# ---------------------------------------------------------
# Streamlit UI config
# ---------------------------------------------------------

st.set_page_config(
    page_title="HypeQuest – Instagram Engagement & Sentiment Prediction",
    layout="wide",
)

st.title("🔥 HypeQuest – Instagram Engagement & Sentiment Prediction")
st.caption(
    "Prototype that predicts post engagement and sentiment using Machine Learning. "
    "Works with CSV, Excel, JSON, Parquet, or manual input, and learns from historical posts."
)
st.markdown("---")

# ---------------------------------------------------------
# Sidebar – profile & optional dataset upload
# ---------------------------------------------------------

st.sidebar.header("Profile & data")

available_profiles = ["@pubg", "@playinzoi", "@hypequest"]
profile = st.sidebar.selectbox("Instagram profile", available_profiles)

# Fake “API” data for the selected profile
historical_df = generate_fake_api_data(profile, n_posts=160)
st.sidebar.success(f"Historical posts available for {profile}: {len(historical_df)}")

st.sidebar.markdown("---")
st.sidebar.subheader("Load post dataset (optional)")
uploaded = st.sidebar.file_uploader(
    "Upload CSV, Excel, JSON or Parquet file",
    type=["csv", "xlsx", "xls", "json", "parquet"],
)

extra_df = None
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

# Train engagement model on fake API data (could be merged with extra_df in future)
eng_model, eng_feature_columns = train_engagement_model(historical_df)

# ---------------------------------------------------------
# 1. Plan a new Instagram post (MAIN LAYOUT)
# ---------------------------------------------------------

st.subheader("🧩 Plan a new Instagram post")

col_left, col_right = st.columns(2)

with col_left:
    post_type = st.selectbox("Post type", ["image", "video", "reels", "carousel"])
    hour = st.slider("Posting hour (UTC)", 0, 23, 20)
    weekday = st.selectbox(
        "Weekday",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
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
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
    )
    month = (
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ].index(month_name)
        + 1
    )

st.markdown("### ✍️ Caption")

# Use session_state so "Use suggested caption" can overwrite this
if "caption_input" not in st.session_state:
    st.session_state["caption_input"] = ""

caption_text = st.text_area(
    "Caption text",
    key="caption_input",
    placeholder="Write your caption here...",
    height=120,
)

caption_length = len(caption_text or "")
st.caption(f"Caption length: {caption_length} characters (used as a feature in the model).")

st.markdown("")

# ---------------------------------------------------------
# Button – Evaluate
# ---------------------------------------------------------

if st.button("✨ Evaluate caption & predict", type="primary"):
    with st.spinner("Consulting the HypeQuest crystal ball... 🔮"):

        # 1) Sentiment + tips (GPT when available, otherwise fallback)
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

        # 2) Engagement prediction (Decision Tree on synthetic “API” data)
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

        # Use followers from fake data to convert rate → interactions
        followers = int(historical_df["followers"].iloc[0])
        predicted_interactions = int(predicted_eng_rate * followers)

        # store results to session_state to render below
        st.session_state["last_result"] = dict(
            sentiment_label=sentiment_label,
            sentiment_explanation=sentiment_explanation,
            suggestions=suggestions,
            improved_caption=improved_caption,
            predicted_interactions=predicted_interactions,
            predicted_eng_rate=predicted_eng_rate,
            followers=followers,
        )

# ---------------------------------------------------------
# 2. Prediction results (only if we have something)
# ---------------------------------------------------------

if "last_result" in st.session_state:
    res = st.session_state["last_result"]

    st.markdown("## 🔍 Prediction results")

    # Sentiment pill
    sentiment_color = {
        "POSITIVE": "#22c55e",  # green
        "NEGATIVE": "#ef4444",  # red
        "NEUTRAL": "#eab308",   # yellow
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

    # -----------------------------------------------------
    # 3. Suggestions to improve this post
    # -----------------------------------------------------
    st.markdown("### 💡 Suggestions to improve this post")

    tips = []

    # Time-based tip (learned from synthetic data design)
    if not (18 <= hour <= 22 and weekday in ["Thursday", "Friday", "Saturday"]):
        tips.append(
            "Historical data shows stronger engagement between **18–22 UTC** on Thursday–Saturday. "
            "Consider testing this post closer to prime time."
        )

    # Caption length tip
    if caption_length < 40:
        tips.append(
            "Your caption is very short. Consider adding a bit more context, a hook, or a clear benefit for players."
        )
    elif caption_length > 350:
        tips.append(
            "Your caption is quite long. Check if you can tighten it while keeping the key message and CTA."
        )

    # From GPT suggestions (or fallback)
    tips.extend(res["suggestions"])

    # Show as bullets
    for t in tips:
        st.markdown(f"- {t}")

    # -----------------------------------------------------
    # 4. Suggested improved caption (with copy/use button)
    # -----------------------------------------------------
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
        # Overwrite the main caption input
        st.session_state["caption_input"] = res["improved_caption"]
        st.success("Suggested caption applied to the editor above. You can edit it before posting.")


# small footer
st.markdown("---")
st.caption(
    "Tips are based on patterns learned from a simulated API dataset. "
    "In production, this would be connected to real Instagram insights."
)


