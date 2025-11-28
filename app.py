import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import io
from datetime import datetime, timedelta

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="HypeQuest – Instagram Engagement Prediction",
    layout="wide"
)

st.title("🔥 HypeQuest – Instagram Engagement Prediction")
st.caption(
    "Demo dashboard that simulates an Instagram insights API, "
    "predicting post sentiment and engagement based on historical data."
)
st.markdown("---")


# =========================================================
# 1. FAKE API – SIMULATED INSTAGRAM DATA
# =========================================================
def fake_instagram_api(profile_filter: str | None = None, n_rows: int = 200) -> pd.DataFrame:
    """
    Simulates an Instagram insights API response and returns
    a tabular dataset with one row per post.

    Columns:
      - date_post, post_type, post_hour, weekday
      - caption_text, caption_length, hashtag_count
      - topic, profile
      - engagement_total
    """
    rng = np.random.default_rng(42)

    base_date = datetime(2024, 1, 1)
    post_types = ["image", "video", "reels", "carousel"]
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    topics = ["update", "maintenance", "trailer", "collaboration", "gameplay", "esports", "community"]
    profiles = ["@playinzoi", "@pubg", "@other_channel"]

    # Some keyword pools for captions
    positive_keywords = ["hype", "amazing", "new", "exclusive", "win", "victory", "update", "launch", "free"]
    negative_keywords = ["bug", "issue", "problem", "delay", "downtime", "maintenance", "broken"]

    rows = []
    for i in range(n_rows):
        date = base_date + timedelta(days=int(rng.integers(0, 120)))
        post_type = rng.choice(post_types)
        topic = rng.choice(topics)
        profile = rng.choice(profiles)

        hour = int(rng.integers(8, 23))
        weekday = weekdays[date.weekday()]
        hashtag_count = int(rng.integers(0, 12))

        # Build a simple fake caption
        caption_len = int(rng.integers(50, 280))
        caption_words = []

        # Bias keywords by topic
        if topic in ["update", "trailer", "collaboration", "esports"]:
            # more positive words
            for _ in range(3):
                caption_words.append(rng.choice(positive_keywords))
        elif topic == "maintenance":
            caption_words.append("maintenance")
            if rng.random() < 0.5:
                caption_words.append(rng.choice(negative_keywords))
        else:
            # neutral mix
            if rng.random() < 0.5:
                caption_words.append(rng.choice(positive_keywords))
            if rng.random() < 0.3:
                caption_words.append(rng.choice(negative_keywords))

        caption = " ".join(caption_words)
        # pad caption to caption_len with filler text
        if len(caption) < caption_len:
            filler = " gameplay" * 100
            caption = (caption + filler)[:caption_len]
        else:
            caption = caption[:caption_len]

        # Simple rule to define engagement for demo
        base_eng = 500
        if post_type == "reels":
            base_eng += 200
        if weekday in ["Thursday", "Friday"]:
            base_eng += 100
        if topic in ["update", "trailer", "collaboration"]:
            base_eng += 100
        if "bug" in caption or "delay" in caption or "problem" in caption:
            base_eng -= 150

        base_eng += hashtag_count * 20
        base_eng += rng.normal(0, 100)

        engagement_total = max(50, int(base_eng))

        rows.append(
            dict(
                date_post=date,
                post_type=post_type,
                post_hour=hour,
                weekday=weekday,
                caption_text=caption,
                caption_length=len(caption),
                hashtag_count=hashtag_count,
                topic=topic,
                profile=profile,
                engagement_total=engagement_total,
            )
        )

    df = pd.DataFrame(rows)

    if profile_filter and profile_filter in df["profile"].unique():
        df = df[df["profile"] == profile_filter].reset_index(drop=True)

    return df


# =========================================================
# 2. SIMPLE SENTIMENT ANALYSIS (KEYWORD + CONTEXT)
# =========================================================
POSITIVE_WORDS = [
    "hype", "amazing", "awesome", "new", "exclusive",
    "win", "victory", "gg", "update", "launch", "free",
    "reward", "bonus", "buff"
]
NEGATIVE_WORDS = [
    "bug", "issue", "problem", "delay", "downtime",
    "maintenance", "broken", "crash", "lag"
]


def estimate_sentiment(
    caption: str,
    topic: str,
    hour: int,
    weekday: str,
    caption_length: int,
    hashtag_count: int
) -> str:
    """
    Rule-based sentiment estimation combining:
      - caption text (keyword hits)
      - context (topic, hour, weekdays, length, hashtags)
    Returns: "positive", "neutral", or "negative"
    """
    if caption is None:
        caption = ""
    text = caption.lower()

    score = 0

    # 1) Keywords in caption
    pos_hits = sum(word in text for word in POSITIVE_WORDS)
    neg_hits = sum(word in text for word in NEGATIVE_WORDS)
    score += pos_hits * 2
    score -= neg_hits * 2

    # 2) Topic bias
    if topic in ["update", "trailer", "collaboration", "esports"]:
        score += 1
    if topic == "maintenance":
        score -= 1

    # 3) Time-of-day bias (example)
    if hour >= 18 and hour <= 22:
        score += 0.5  # prime time
    if weekday in ["Saturday", "Sunday"]:
        score += 0.5

    # 4) Caption length / hashtag usage
    if caption_length > 300:
        score -= 0.5  # too long may be tiring
    if hashtag_count >= 8:
        score += 0.5  # more reach potential

    # Final decision
    if score >= 1.5:
        return "positive"
    elif score <= -1.0:
        return "negative"
    else:
        return "neutral"


# =========================================================
# 3. PREPARE DATA & TRAIN ENGAGEMENT MODEL
# =========================================================
def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Uses fake API data to train a simple engagement prediction model.
    Sentiment is derived using the same rule-based function.
    """
    # Derive sentiment for historical posts
    df = df.copy()
    df["sentiment"] = df.apply(
        lambda row: estimate_sentiment(
            caption=row["caption_text"],
            topic=row["topic"],
            hour=row["post_hour"],
            weekday=row["weekday"],
            caption_length=row["caption_length"],
            hashtag_count=row["hashtag_count"],
        ),
        axis=1,
    )

    feature_cols = [
        "post_type",
        "post_hour",
        "weekday",
        "caption_length",
        "hashtag_count",
        "topic",
        "profile",
        "sentiment",
    ]

    X = pd.get_dummies(df[feature_cols], drop_first=False)
    y = df["engagement_total"]

    return X, y


def train_engagement_model(X: pd.DataFrame, y: pd.Series) -> DecisionTreeRegressor:
    model = DecisionTreeRegressor(max_depth=6, random_state=42)
    model.fit(X, y)
    return model


# =========================================================
# 4. LOAD "API" DATA (FAKE) AND TRAIN MODEL
# =========================================================
st.sidebar.header("📡 Data source")

profile_options = ["All profiles", "@playinzoi", "@pubg", "@other_channel"]
selected_profile_filter = st.sidebar.selectbox("Simulated Instagram profile", profile_options)

if selected_profile_filter == "All profiles":
    api_profile = None
else:
    api_profile = selected_profile_filter

# Fetch from fake API
df_api = fake_instagram_api(profile_filter=api_profile)

st.sidebar.success("Data source: Fake Instagram Insights API (demo)")
st.sidebar.write(f"Posts loaded from API: **{len(df_api)}**")

# Train model
X_train, y_train = prepare_training_data(df_api)
engagement_model = train_engagement_model(X_train, y_train)


# =========================================================
# 5. WHAT-IF SIMULATOR (USER INPUT)
# =========================================================
st.subheader("🎛 Plan a new Instagram post")

col1, col2, col3 = st.columns(3)

with col1:
    post_type = st.selectbox("Post type", ["image", "video", "reels", "carousel"])
    post_hour = st.slider("Posting hour", 0, 23, 18)
    weekday = st.selectbox(
        "Weekday",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    )

with col2:
    hashtag_count = st.slider("Number of hashtags", 0, 20, 5)
    topic = st.selectbox(
        "Post topic",
        [
            "update", "maintenance", "trailer", "collaboration",
            "gameplay", "esports", "community"
        ],
    )

with col3:
    profile_ui = st.selectbox(
        "Instagram profile",
        ["@playinzoi", "@pubg", "@other_channel", "Other"],
    )
    if profile_ui == "Other":
        profile = st.text_input("Custom profile handle", "@yourbrand")
    else:
        profile = profile_ui

st.markdown("#### ✏️ Caption")
caption_text = st.text_area(
    "Caption text",
    "",
    height=140,
    placeholder="Type your Instagram caption here...",
)
caption_length = len(caption_text)

st.markdown(
    f"<span style='color:#64748b;'>Caption length: <b>{caption_length}</b> characters</span>",
    unsafe_allow_html=True,
)


# =========================================================
# 6. SENTIMENT ESTIMATION + ENGAGEMENT PREDICTION
# =========================================================
st.markdown("---")
st.markdown("### 🔍 Prediction results")

# 1) Estimate sentiment based on user input
predicted_sentiment = estimate_sentiment(
    caption=caption_text,
    topic=topic,
    hour=post_hour,
    weekday=weekday,
    caption_length=caption_length,
    hashtag_count=hashtag_count,
)

# Sentiment badge color
sent_color_map = {
    "positive": "#22c55e",
    "neutral": "#eab308",
    "negative": "#ef4444",
}
sent_color = sent_color_map.get(predicted_sentiment, "#6b7280")

sentiment_badge = f"""
<div style="margin-top:0.5rem;margin-bottom:0.75rem;">
  <span style="
    background-color:{sent_color};
    color:white;
    padding:0.35rem 0.9rem;
    border-radius:999px;
    font-weight:600;
    font-size:16px;">
    {predicted_sentiment.upper()}
  </span>
</div>
"""

st.markdown("**Predicted sentiment for this post (based on caption + context):**", unsafe_allow_html=True)
st.markdown(sentiment_badge, unsafe_allow_html=True)


# 2) Predict engagement using the same feature engineering as training
new_post = pd.DataFrame(
    [{
        "post_type": post_type,
        "post_hour": post_hour,
        "weekday": weekday,
        "caption_length": caption_length,
        "hashtag_count": hashtag_count,
        "topic": topic,
        "profile": profile,
        "sentiment": predicted_sentiment,
    }]
)

new_post_dummies = pd.get_dummies(new_post).reindex(columns=X_train.columns, fill_value=0)
pred_engagement = engagement_model.predict(new_post_dummies)[0]

st.markdown(
    f"""
    <div style="
        margin-top:0.5rem;
        padding:0.9rem 1rem;
        border-radius:0.75rem;
        background-color:#dbeafe;
        border:1px solid #bfdbfe;
    ">
        <span style="font-size:14px;color:#1d4ed8;">Predicted engagement</span><br/>
        <span style="font-size:22px;font-weight:700;color:#1e3a8a;">
            {int(pred_engagement):,} interactions
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Sentiment is estimated using caption keywords and posting context. "
    "Engagement prediction is trained on a simulated Instagram API dataset."
)


# =========================================================
# 7. HISTORICAL DATA VIEW (FROM FAKE API)
# =========================================================
with st.expander("📊 See historical data used for training (from fake API)"):
    st.write("Sample of posts fetched from the simulated Instagram API:")
    st.dataframe(df_api.head())

    st.write("Engagement distribution:")
    st.write(df_api["engagement_total"].describe())

    colA, colB = st.columns(2)
    with colA:
        eng_type = df_api.groupby("post_type")["engagement_total"].mean()
        fig, ax = plt.subplots()
        ax.bar(eng_type.index, eng_type.values)
        ax.set_title("Average engagement by post type")
        plt.xticks(rotation=15)
        st.pyplot(fig)

    with colB:
        eng_wday = df_api.groupby("weekday")["engagement_total"].mean()
        fig, ax = plt.subplots()
        ax.bar(eng_wday.index, eng_wday.values)
        ax.set_title("Average engagement by weekday")
        plt.xticks(rotation=15)
        st.pyplot(fig)
