import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import io

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="HypeQuest – Instagram Engagement Prediction",
    layout="wide"
)

st.title("🔥 HypeQuest – Instagram Engagement Prediction")
st.caption(
    "Prototype that predicts post engagement and sentiment using Machine Learning. "
    "Works with CSV, Excel, JSON, Parquet, or manual input, and learns from historical posts."
)
st.markdown("---")


# ---------------------------------------------------------
# LOAD MULTI-FORMAT FILE
# ---------------------------------------------------------
def load_file(uploaded):
    if uploaded is None:
        return None

    filename = uploaded.name.lower()

    try:
        if filename.endswith(".csv"):
            return pd.read_csv(uploaded)
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            return pd.read_excel(uploaded)
        elif filename.endswith(".json"):
            return pd.read_json(uploaded)
        elif filename.endswith(".parquet"):
            return pd.read_parquet(uploaded)
        else:
            st.error("Unsupported file type. Upload CSV, Excel, JSON, or Parquet.")
            return None

    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


# ---------------------------------------------------------
# PREPARE DATASET FOR TWO MODELS (SENTIMENT + ENGAGEMENT)
# ---------------------------------------------------------
def prepare_dataframe(df: pd.DataFrame):
    """
    Expected columns in English:
        date_post, post_type, post_hour, weekday,
        caption_length, hashtag_count,
        sentiment, engagement_total

    Optional:
        topic, profile
    """

    required = [
        "date_post",
        "post_type",
        "post_hour",
        "weekday",
        "caption_length",
        "hashtag_count",
        "sentiment",          # positive / neutral / negative
        "engagement_total",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(
            f"Dataset missing required columns: {missing}. "
            "Models cannot be trained; using baseline only."
        )
        return None, None, None, None, None

    # Ensure datetime
    df["date_post"] = pd.to_datetime(df["date_post"], errors="coerce")

    # Month / Year
    df["month"] = df["date_post"].dt.month.fillna(1).astype(int)
    df["year"] = df["date_post"].dt.year.fillna(2024).astype(int)

    # Base features used for SENTIMENT PREDICTION
    base_features = [
        "post_type",
        "post_hour",
        "weekday",
        "caption_length",
        "hashtag_count",
        "month",
        "year",
    ]

    if "topic" in df.columns:
        base_features.append("topic")

    if "profile" in df.columns:
        base_features.append("profile")

    X_base = pd.get_dummies(df[base_features], drop_first=False)

    # Targets
    y_sent = df["sentiment"]
    y_eng = df["engagement_total"]

    # ENGAGEMENT MODEL uses all factors + the predicted sentiment
    X_eng = df[base_features + ["sentiment"]]
    X_eng = pd.get_dummies(X_eng, drop_first=False)

    return df, X_base, y_sent, X_eng, y_eng


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------
def train_sentiment_model(X_base, y_sent):
    clf = DecisionTreeClassifier(max_depth=6, random_state=42)
    clf.fit(X_base, y_sent)
    return clf


def train_engagement_model(X_eng, y_eng):
    reg = DecisionTreeRegressor(max_depth=6, random_state=42)
    reg.fit(X_eng, y_eng)
    return reg


# ---------------------------------------------------------
# SIDEBAR LOAD
# ---------------------------------------------------------
st.sidebar.header("📁 Load post dataset (optional)")

uploaded = st.sidebar.file_uploader(
    "Upload CSV, Excel, JSON or Parquet",
    type=["csv", "xlsx", "xls", "json", "parquet"],
)

df_loaded = load_file(uploaded)

if df_loaded is not None:
    st.sidebar.success(f"Loaded dataset: {uploaded.name}")
else:
    st.sidebar.info("No dataset uploaded – baseline mode only.")


# ---------------------------------------------------------
# TRAIN MODELS ONLY IF THERE IS HISTORY
# ---------------------------------------------------------
MIN_HISTORY = 30

if df_loaded is not None:
    df, X_base, y_sent, X_eng, y_eng = prepare_dataframe(df_loaded)

    if X_base is not None and len(df) >= MIN_HISTORY:
        sentiment_model = train_sentiment_model(X_base, y_sent)
        engagement_model = train_engagement_model(X_eng, y_eng)
    else:
        sentiment_model = None
        engagement_model = None
        if X_base is not None and len(df) < MIN_HISTORY:
            st.warning(
                f"Not enough historical posts to train ML models "
                f"({len(df)} < {MIN_HISTORY}). Using baseline only."
            )
else:
    df, X_base, y_sent, X_eng, y_eng = None, None, None, None, None
    sentiment_model = None
    engagement_model = None


# ---------------------------------------------------------
# WHAT-IF SIMULATOR
# ---------------------------------------------------------
st.subheader("🎛 Plan a new post and get predicted sentiment + engagement")

col1, col2, col3 = st.columns(3)

with col1:
    post_type = st.selectbox("Post type", ["image", "video", "reels", "carousel"])
    hour = st.slider("Posting hour", 0, 23, 12)
    weekday = st.selectbox(
        "Weekday",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    )

with col2:
    hashtag_ct = st.slider("Number of hashtags", 0, 20, 3)

with col3:
    profile = st.text_input(
        "Instagram profile (handle)",
        "@yourbrand",
        help="This is only used as a categorical feature if it exists in the historical dataset.",
    )
    topic = st.selectbox(
        "Post topic",
        ["update", "maintenance", "trailer", "collaboration", "gameplay",
         "esports", "community", "other"],
    )
    month = st.selectbox("Month", list(range(1, 13)))
    year = st.selectbox("Year", [2023, 2024, 2025])

caption_text = st.text_area(
    "Caption text",
    "",
    height=140,
    placeholder="Type your Instagram caption here...",
)
caption_len = len(caption_text)

st.markdown(
    f"<span style='color:#64748b;'>Caption length: <b>{caption_len}</b> characters</span>",
    unsafe_allow_html=True,
)

# Build new post (BASE FEATURES)
new_base = pd.DataFrame(
    [{
        "post_type": post_type,
        "post_hour": hour,
        "weekday": weekday,
        "caption_length": caption_len,
        "hashtag_count": hashtag_ct,
        "month": month,
        "year": year,
        "topic": topic,
        "profile": profile,
    }]
)


# ---------------------------------------------------------
# PREDICT (SENTIMENT + ENGAGEMENT)
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 🔍 Prediction results")

if sentiment_model is not None and engagement_model is not None:

    # 1) Predict sentiment
    new_base_dummies = pd.get_dummies(new_base).reindex(
        columns=X_base.columns, fill_value=0
    )
    pred_sentiment = sentiment_model.predict(new_base_dummies)[0]

    # Color pill for sentiment
    color_map = {
        "positive": "#22c55e",  # green
        "neutral": "#eab308",   # yellow
        "negative": "#ef4444",  # red
    }
    color = color_map.get(str(pred_sentiment).lower(), "#6b7280")

    sentiment_badge = f"""
    <div style="margin-top:0.5rem;margin-bottom:0.75rem;">
      <span style="
        background-color:{color};
        color:white;
        padding:0.35rem 0.9rem;
        border-radius:999px;
        font-weight:600;
        font-size:16px;">
        {str(pred_sentiment).upper()}
      </span>
    </div>
    """

    st.markdown("**Predicted sentiment for this post:**", unsafe_allow_html=True)
    st.markdown(sentiment_badge, unsafe_allow_html=True)

    # 2) Predict engagement using predicted sentiment
    new_eng = new_base.copy()
    new_eng["sentiment"] = pred_sentiment

    new_eng_dummies = pd.get_dummies(new_eng).reindex(
        columns=X_eng.columns, fill_value=0
    )

    pred_engagement = engagement_model.predict(new_eng_dummies)[0]

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

else:
    # Baseline fallback
    st.warning("No ML model available – using baseline estimation.")
    baseline = (caption_len + hashtag_ct * 40 + hour * 5) / 10

    st.markdown(
        f"""
        <div style="
            margin-top:0.5rem;
            padding:0.9rem 1rem;
            border-radius:0.75rem;
            background-color:#f1f5f9;
            border:1px solid #e2e8f0;
        ">
            <span style="font-size:14px;color:#475569;">Estimated engagement (baseline)</span><br/>
            <span style="font-size:22px;font-weight:700;color:#0f172a;">
                {int(baseline)} interactions
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.caption(
    "This prediction engine is history-driven: it learns from past posts. "
    "New profiles without historical data will only have baseline estimates."
)


# ---------------------------------------------------------
# DATA EXPLORATION IF AVAILABLE
# ---------------------------------------------------------
if df is not None:
    st.markdown("---")
    st.subheader("📊 Data overview")
    st.dataframe(df.head())

    st.write(df["engagement_total"].describe())

    st.markdown("---")
    st.subheader("📈 Engagement patterns")

    colA, colB = st.columns(2)

    with colA:
        if "post_type" in df.columns:
            eng = df.groupby("post_type")["engagement_total"].mean()
            fig, ax = plt.subplots()
            ax.bar(eng.index, eng.values)
            ax.set_title("Avg engagement by post type")
            plt.xticks(rotation=15)
            st.pyplot(fig)

    with colB:
        if "weekday" in df.columns:
            eng = df.groupby("weekday")["engagement_total"].mean()
            fig, ax = plt.subplots()
            ax.bar(eng.index, eng.values)
            ax.set_title("Avg engagement by weekday")
            plt.xticks(rotation=15)
            st.pyplot(fig)

