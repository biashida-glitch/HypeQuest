import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
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
    "Dashboard that uses historical Instagram-style data per profile to "
    "predict post sentiment and engagement based on caption and posting context."
)
st.markdown("---")


# =========================================================
# 1. INTERNAL DATA GENERATION (PER PROFILE)
#    (This mimics an internal API, but is NEVER shown in the UI)
# =========================================================

POSITIVE_WORDS = [
    "hype", "amazing", "awesome", "new", "exclusive",
    "win", "victory", "gg", "update", "launch", "free",
    "reward", "bonus", "buff", "fun", "drop", "event"
]

NEGATIVE_WORDS = [
    "bug", "issue", "problem", "delay", "downtime",
    "maintenance", "broken", "crash", "lag", "nerf"
]


def build_profile_history(profile_handle: str, n_rows: int = 160) -> pd.DataFrame:
    """
    Creates an internal historical dataset for a specific Instagram profile.
    One row per post.

    Columns:
      - date_post, post_type, post_hour, weekday
      - caption_text, caption_length, hashtag_count
      - topic, profile
      - engagement_total
    """
    rng = np.random.default_rng(hash(profile_handle) % (2**32))

    base_date = datetime(2024, 1, 1)
    post_types = ["image", "video", "reels", "carousel"]
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    topics = ["update", "maintenance", "trailer", "collaboration", "gameplay", "esports", "community"]

    rows = []
    for i in range(n_rows):
        date = base_date + timedelta(days=int(rng.integers(0, 120)))
        post_type = rng.choice(post_types)
        topic = rng.choice(topics)
        post_hour = int(rng.integers(8, 23))
        weekday = weekdays[date.weekday()]
        hashtag_count = int(rng.integers(0, 12))

        # Base caption keywords: we want these to match what we'll detect later
        caption_keywords = []

        # Slightly different behavior per profile
        if profile_handle == "@pubg":
            if topic in ["update", "trailer", "esports"]:
                caption_keywords += ["update", "hype", "event"]
            elif topic == "maintenance":
                caption_keywords += ["maintenance", "delay"]
            else:
                caption_keywords += ["gameplay", "fun"]
        elif profile_handle == "@playinzoi":
            if topic in ["collaboration", "update"]:
                caption_keywords += ["amazing", "new", "exclusive"]
            elif topic == "maintenance":
                caption_keywords += ["maintenance", "issue"]
            else:
                caption_keywords += ["event", "fun"]
        else:
            caption_keywords += ["gameplay"]

        # Add random positive / negative words
        if rng.random() < 0.5:
            caption_keywords.append(rng.choice(POSITIVE_WORDS))
        if rng.random() < 0.3:
            caption_keywords.append(rng.choice(NEGATIVE_WORDS))

        base_caption = " ".join(dict.fromkeys(caption_keywords))  # unique, keep order

        # Desired length and padding
        target_len = int(rng.integers(80, 260))
        filler = " gameplay" * 50
        caption = (base_caption + filler)[:target_len]

        caption_length = len(caption)

        # Engagement rule: depends on type, weekday, topic, sentiment-ish
        base_eng = 500

        if post_type == "reels":
            base_eng += 220
        if weekday in ["Thursday", "Friday"]:
            base_eng += 130
        if topic in ["update", "trailer", "collaboration"]:
            base_eng += 140
        if "bug" in caption or "delay" in caption or "problem" in caption:
            base_eng -= 180

        base_eng += hashtag_count * 22
        base_eng += rng.normal(0, 110)

        engagement_total = max(50, int(base_eng))

        rows.append(
            dict(
                date_post=date,
                post_type=post_type,
                post_hour=post_hour,
                weekday=weekday,
                caption_text=caption,
                caption_length=caption_length,
                hashtag_count=hashtag_count,
                topic=topic,
                profile=profile_handle,
                engagement_total=engagement_total,
            )
        )

    return pd.DataFrame(rows)


# =========================================================
# 2. SENTIMENT ESTIMATION (CAPTION + CONTEXT)
# =========================================================

def estimate_sentiment(
    caption: str,
    topic: str,
    hour: int,
    weekday: str,
    caption_length: int,
    hashtag_count: int,
) -> tuple[str, list[str], list[str]]:
    """
    Rule-based sentiment estimation combining:
      - caption text (keyword hits)
      - context (topic, hour, weekday, caption length, hashtags)

    Returns:
      (sentiment_label, matched_positive_keywords, matched_negative_keywords)
    """
    if caption is None:
        caption = ""
    text = caption.lower()

    score = 0.0

    # 1) Keywords in caption
    matched_pos = [w for w in POSITIVE_WORDS if w in text]
    matched_neg = [w for w in NEGATIVE_WORDS if w in text]

    score += len(matched_pos) * 2.0
    score -= len(matched_neg) * 2.0

    # 2) Topic bias
    if topic in ["update", "trailer", "collaboration", "esports"]:
        score += 1.0
    if topic == "maintenance":
        score -= 1.0

    # 3) Time-of-day / weekday
    if 18 <= hour <= 22:
        score += 0.5  # prime time
    if weekday in ["Saturday", "Sunday"]:
        score += 0.5

    # 4) Caption length / hashtag usage
    if caption_length > 320:
        score -= 0.7  # too long
    if hashtag_count >= 8:
        score += 0.5

    if score >= 1.5:
        label = "positive"
    elif score <= -1.0:
        label = "negative"
        # neutral in the middle
    else:
        label = "neutral"

    return label, matched_pos, matched_neg


def generate_suggestions(
    sentiment: str,
    matched_pos: list[str],
    matched_neg: list[str],
    caption_length: int,
    hashtag_count: int,
    hour: int,
    weekday: str,
    topic: str,
) -> list[str]:
    """
    Generate human-readable suggestions to improve the caption/post setup.
    """
    suggestions = []

    # Sentiment-driven suggestions
    if sentiment == "negative":
        suggestions.append(
            "The caption feels negative. Consider removing or rephrasing words like "
            f"{', '.join(matched_neg)} and adding more hopeful or community-focused tone."
            if matched_neg else
            "The caption is skewed negative. Consider softening the tone or adding a more positive angle."
        )
    elif sentiment == "neutral":
        suggestions.append(
            "The caption looks neutral. You could add a stronger hook or emotional word "
            "to make it more exciting (e.g., hype, amazing, new, exclusive)."
        )

    # Keywords balance
    if not matched_pos:
        suggestions.append(
            "No positive keywords detected. Consider adding words like "
            "hype, amazing, new, exclusive, event or reward."
        )

    if matched_neg:
        suggestions.append(
            "Negative keywords detected: "
            + ", ".join(matched_neg)
            + ". Make sure this is intentional (e.g., addressing issues transparently) "
              "or reduce them if the goal is hype."
        )

    # Caption length
    if caption_length > 320:
        suggestions.append(
            "The caption is quite long. Try shortening or using bullet-style structure "
            "to keep it more scannable."
        )
    elif caption_length < 60:
        suggestions.append(
            "The caption is very short. Consider adding a bit more context or a clear call to action."
        )

    # Hashtags
    if hashtag_count == 0:
        suggestions.append(
            "You are not using hashtags. Adding a few relevant hashtags can help reach new audiences."
        )
    elif hashtag_count > 12:
        suggestions.append(
            "You are using many hashtags. Consider focusing on the most relevant ones "
            "to avoid looking like spam."
        )

    # Timing
    if not (18 <= hour <= 22):
        suggestions.append(
            "Consider testing this post closer to prime time (18–22) if your audience is active then."
        )
    if weekday in ["Monday", "Tuesday"]:
        suggestions.append(
            "Early-week posts can work, but you might compare performance with Thursday/Friday "
            "for hype / announcement content."
        )

    # Topic-specific
    if topic == "maintenance" and sentiment == "positive":
        suggestions.append(
            "Maintenance posts are usually sensitive. Make sure the tone still sounds transparent "
            "and respectful, not overly ‘marketing’."
        )

    # Fallback suggestion
    if not suggestions:
        suggestions.append("The setup looks solid. You can A/B test small variations in caption and timing.")

    return suggestions


# =========================================================
# 3. TRAIN ENGAGEMENT MODEL PER PROFILE
# =========================================================

def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Builds training features for engagement model, using
    sentiment derived from caption + context.
    """
    df = df.copy()

    df["sentiment"] = df.apply(
        lambda row: estimate_sentiment(
            caption=row["caption_text"],
            topic=row["topic"],
            hour=row["post_hour"],
            weekday=row["weekday"],
            caption_length=row["caption_length"],
            hashtag_count=row["hashtag_count"],
        )[0],
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
# 4. SELECT PROFILE & LOAD ITS HISTORY (INTERNALLY)
# =========================================================

st.sidebar.header("Profile & data")

profile_choice = st.sidebar.selectbox(
    "Instagram profile",
    ["@playinzoi", "@pubg", "Other"],
)

if profile_choice == "Other":
    profile_handle = st.sidebar.text_input(
        "Custom profile handle",
        "@yourbrand",
        help="For new profiles without history, only a simple estimate is available.",
    )
else:
    profile_handle = profile_choice

if profile_handle in ["@playinzoi", "@pubg"]:
    df_history = build_profile_history(profile_handle)
    X_train, y_train = prepare_training_data(df_history)
    engagement_model = train_engagement_model(X_train, y_train)
    st.sidebar.success(
        f"Historical posts available for {profile_handle}: {len(df_history)}"
    )
    model_available = True
else:
    df_history = None
    X_train, y_train = None, None
    engagement_model = None
    st.sidebar.warning(
        "No historical data available for this profile yet. "
        "Predictions will use a simple baseline only."
    )
    model_available = False


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

with col3:
    topic = st.selectbox(
        "Post topic",
        [
            "update", "maintenance", "trailer", "collaboration",
            "gameplay", "esports", "community"
        ],
    )

st.markdown("#### ✏️ Caption")

caption_text = st.text_area(
    "Caption text",
    "",
    height=140,
    placeholder="Type your Instagram caption here...",
)
caption_length = len(caption_text)

st.markdown(
    f"<span style='color:#64748b;'>Caption length: "
    f"<b>{caption_length}</b> characters "
    f"(used as a feature in the model).</span>",
    unsafe_allow_html=True,
)

evaluate_clicked = st.button("✨ Evaluate caption & get suggestions")


# =========================================================
# 6. SENTIMENT ESTIMATION + VISUALIZATION + SUGGESTIONS
# =========================================================

st.markdown("---")
st.markdown("### 🔍 Predicted sentiment and engagement")

if evaluate_clicked:
    predicted_sentiment, matched_pos, matched_neg = estimate_sentiment(
        caption=caption_text,
        topic=topic,
        hour=post_hour,
        weekday=weekday,
        caption_length=caption_length,
        hashtag_count=hashtag_count,
    )

    # Sentiment badge
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

    st.markdown(
        "**Predicted sentiment for this post "
        "(based on caption keywords and posting context):**",
        unsafe_allow_html=True,
    )
    st.markdown(sentiment_badge, unsafe_allow_html=True)

    # Show matched keywords so it’s clear the caption is being used
    kw_cols = st.columns(2)
    with kw_cols[0]:
        st.markdown("**Matched positive keywords in caption:**")
        if matched_pos:
            st.markdown(
                " ".join(
                    f"<span style='background-color:#dcfce7;"
                    f"color:#166534;padding:0.1rem 0.5rem;margin:0.1rem;"
                    f"border-radius:999px;font-size:12px;'>{w}</span>"
                    for w in matched_pos
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span style='color:#94a3b8;font-size:13px;'>No positive keywords detected.</span>",
                unsafe_allow_html=True,
            )

    with kw_cols[1]:
        st.markdown("**Matched negative keywords in caption:**")
        if matched_neg:
            st.markdown(
                " ".join(
                    f"<span style='background-color:#fee2e2;"
                    f"color:#991b1b;padding:0.1rem 0.5rem;margin:0.1rem;"
                    f"border-radius:999px;font-size:12px;'>{w}</span>"
                    for w in matched_neg
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span style='color:#94a3b8;font-size:13px;'>No negative keywords detected.</span>",
                unsafe_allow_html=True,
            )

    # Suggestions block
    suggestions = generate_suggestions(
        sentiment=predicted_sentiment,
        matched_pos=matched_pos,
        matched_neg=matched_neg,
        caption_length=caption_length,
        hashtag_count=hashtag_count,
        hour=post_hour,
        weekday=weekday,
        topic=topic,
    )

    st.markdown("#### 💡 Suggestions to improve this caption")
    suggestion_html = "<ul>"
    for s in suggestions:
        suggestion_html += f"<li style='margin-bottom:0.3rem;'>{s}</li>"
    suggestion_html += "</ul>"

    st.markdown(
        f"""
        <div style="
            margin-top:0.5rem;
            padding:0.85rem 1rem;
            border-radius:0.75rem;
            background-color:#f8fafc;
            border:1px solid #e2e8f0;
            font-size:14px;
            color:#0f172a;
        ">
            {suggestion_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # 7. ENGAGEMENT PREDICTION (MODEL OR BASELINE)
    # =====================================================
    if model_available and X_train is not None:
        new_post = pd.DataFrame(
            [{
                "post_type": post_type,
                "post_hour": post_hour,
                "weekday": weekday,
                "caption_length": caption_length,
                "hashtag_count": hashtag_count,
                "topic": topic,
                "profile": profile_handle,
                "sentiment": predicted_sentiment,
            }]
        )

        new_post_dummies = pd.get_dummies(new_post).reindex(
            columns=X_train.columns, fill_value=0
        )
        pred_engagement = engagement_model.predict(new_post_dummies)[0]

        st.markdown(
            f"""
            <div style="
                margin-top:0.75rem;
                padding:0.9rem 1rem;
                border-radius:0.75rem;
                background-color:#dbeafe;
                border:1px solid #bfdbfe;
            ">
                <span style="font-size:14px;color:#1d4ed8;">
                    Predicted engagement for <b>{profile_handle}</b>
                </span><br/>
                <span style="font-size:22px;font-weight:700;color:#1e3a8a;">
                    {int(pred_engagement):,} interactions
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Sem histórico (perfil novo) → baseline simples
        baseline = (caption_length + hashtag_count * 40 + post_hour * 5) / 10
        st.markdown(
            f"""
            <div style="
                margin-top:0.75rem;
                padding:0.9rem 1rem;
                border-radius:0.75rem;
                background-color:#f1f5f9;
                border:1px solid #e2e8f0;
            ">
                <span style="font-size:14px;color:#475569;">
                    Estimated engagement for <b>{profile_handle}</b> (baseline only)
                </span><br/>
                <span style="font-size:22px;font-weight:700;color:#0f172a;">
                    {int(baseline)} interactions
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Sentiment is estimated using caption keywords and posting context. "
        "Engagement prediction is trained on historical posts per profile "
        "(when history exists); new profiles fall back to a simple baseline."
    )

else:
    st.markdown(
        "<span style='color:#94a3b8;'>Click the button above to evaluate the caption and see sentiment, suggestions and predicted engagement.</span>",
        unsafe_allow_html=True,
    )


# =========================================================
# 8. OPTIONAL: INTERNAL HISTORY VIEW (ONLY IF PROFILE HAS DATA)
# =========================================================

if 'df_history' in locals() and df_history is not None:
    with st.expander(f"📊 Historical posts used for {profile_handle} (internal view)"):
        st.write("Sample of historical posts:")
        st.dataframe(df_history.head())

        st.write("Engagement distribution:")
        st.write(df_history["engagement_total"].describe())

        colA, colB = st.columns(2)
        with colA:
            eng_type = df_history.groupby("post_type")["engagement_total"].mean()
            fig, ax = plt.subplots()
            ax.bar(eng_type.index, eng_type.values)
            ax.set_title("Average engagement by post type")
            plt.xticks(rotation=15)
            st.pyplot(fig)

        with colB:
            eng_wday = df_history.groupby("weekday")["engagement_total"].mean()
            fig, ax = plt.subplots()
            ax.bar(eng_wday.index, eng_wday.values)
            ax.set_title("Average engagement by weekday")
            plt.xticks(rotation=15)
            st.pyplot(fig)
