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
    "Prototype that predicts post engagement using Machine Learning. "
    "Works with CSV, Excel, JSON, Parquet, or manual input."
)
st.markdown("---")


# ---------------------------------------------------------
# DATA LOADER (Supports multiple formats or no file)
# ---------------------------------------------------------
def load_file(uploaded):
    """Load multiple file types."""
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
# DATA PREPARATION FOR TWO MODELS
# ---------------------------------------------------------
def prepare_dataframe(df):
    """
    Prepare features for:
      - sentiment model (classification)
      - engagement model (regression)
    """

    required = [
        "data_post",
        "tipo_post",
        "hora_post",
        "dia_semana",
        "tam_legenda",
        "hashtag_count",
        "sentimento_legenda",
        "engajamento_total",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(
            f"Dataset missing required columns: {missing}. "
            "Models cannot be trained; baseline mode only."
        )
        return None, None, None, None, None

    # Ensure datetime
    df["data_post"] = pd.to_datetime(df["data_post"], errors="coerce")

    # Month / Year
    df["month"] = df["data_post"].dt.month.fillna(1).astype(int)
    df["year"] = df["data_post"].dt.year.fillna(2024).astype(int)

    # Base features (without sentiment) used for the sentiment model
    base_feature_cols = [
        "tipo_post",
        "hora_post",
        "dia_semana",
        "tam_legenda",
        "hashtag_count",
        "month",
        "year",
    ]

    # Optional topic column
    if "topic" in df.columns:
        base_feature_cols.append("topic")

    X_base = pd.get_dummies(df[base_feature_cols], drop_first=False)

    # Targets
    y_sent = df["sentimento_legenda"]
    y_eng = df["engajamento_total"]

    # For engagement model we use same base features + sentiment as feature
    X_eng = df[base_feature_cols + ["sentimento_legenda"]]
    X_eng = pd.get_dummies(X_eng, drop_first=False)

    return df, X_base, y_sent, X_eng, y_eng


# ---------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------
def train_sentiment_model(X_base, y_sent):
    """Decision Tree to predict caption sentiment."""
    clf = DecisionTreeClassifier(max_depth=6, random_state=42)
    clf.fit(X_base, y_sent)
    return clf


def train_engagement_model(X_eng, y_eng):
    """Decision Tree regressor to predict engagement."""
    reg = DecisionTreeRegressor(max_depth=6, random_state=42)
    reg.fit(X_eng, y_eng)
    return reg


# ---------------------------------------------------------
# SIDEBAR – Load data
# ---------------------------------------------------------
st.sidebar.header("📁 Load post dataset (optional)")

uploaded = st.sidebar.file_uploader(
    "Upload CSV, Excel, JSON or Parquet file",
    type=["csv", "xlsx", "xls", "json", "parquet"],
)

df_loaded = load_file(uploaded)

if df_loaded is not None:
    st.sidebar.success(f"Loaded dataset: {uploaded.name}")
else:
    st.sidebar.info("No dataset uploaded. App will run in baseline-only mode.")

# Prepare data & train models (only if dataset exists)
if df_loaded is not None:
    df, X_base, y_sent, X_eng, y_eng = prepare_dataframe(df_loaded)

    if X_base is not None:
        sentiment_model = train_sentiment_model(X_base, y_sent)
        engagement_model = train_engagement_model(X_eng, y_eng)
    else:
        sentiment_model = None
        engagement_model = None
else:
    df, X_base, y_sent, X_eng, y_eng = None, None, None, None, None
    sentiment_model = None
    engagement_model = None


# ---------------------------------------------------------
# WHAT-IF SIMULATOR (Always works – model or baseline)
# ---------------------------------------------------------
st.subheader("🎛 Plan a new post and get predicted sentiment + engagement")

col1, col2, col3 = st.columns(3)

with col1:
    post_type = st.selectbox("Post type", ["image", "video", "reels", "carousel"])
    hour = st.slider("Posting hour", 0, 23, 12)
    weekday = st.selectbox(
        "Day of week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    )

with col2:
    caption_len = st.slider("Caption length (characters)", 0, 500, 120)
    hashtag_ct = st.slider("Number of hashtags", 0, 20, 3)

with col3:
    topic = st.selectbox(
        "Post topic",
        [
            "update",
            "maintenance",
            "trailer",
            "collaboration",
            "gameplay",
            "esports",
            "community",
            "other",
        ],
    )
    month = st.selectbox("Month", list(range(1, 13)))
    year = st.selectbox("Year", [2023, 2024, 2025])

# Build base feature DF for prediction (the same structure used in training)
new_base = pd.DataFrame(
    [
        {
            "tipo_post": post_type,
            "hora_post": hour,
            "dia_semana": weekday,
            "tam_legenda": caption_len,
            "hashtag_count": hashtag_ct,
            "month": month,
            "year": year,
            "topic": topic,
        }
    ]
)

if sentiment_model is not None and engagement_model is not None:
    # 1) Predict SENTIMENT using all factors
    new_base_dummies = pd.get_dummies(new_base).reindex(
        columns=X_base.columns, fill_value=0
    )
    pred_sentiment_pt = sentiment_model.predict(new_base_dummies)[0]  # 'positivo', etc.

    # Map PT → EN only for display
    pt_to_en = {"positivo": "positive", "neutro": "neutral", "negativo": "negative"}
    pred_sentiment_en = pt_to_en.get(str(pred_sentiment_pt), str(pred_sentiment_pt))

    st.success(f"🧠 Predicted sentiment for this post: **{pred_sentiment_en}**")

    # 2) Use predicted sentiment as feature for ENGAGEMENT model
    new_eng = new_base.copy()
    new_eng["sentimento_legenda"] = pred_sentiment_pt

    new_eng_dummies = pd.get_dummies(new_eng).reindex(
        columns=X_eng.columns, fill_value=0
    )

    pred_engagement = engagement_model.predict(new_eng_dummies)[0]
    st.info(
        f"⭐ Predicted total engagement for this post: "
        f"**{int(pred_engagement):,} interactions**"
    )

else:
    # No dataset → fallback baseline
    st.warning(
        "No trained models available (no dataset with the required columns). "
        "Using a simple baseline estimation instead."
    )
    baseline = (caption_len + hashtag_ct * 40 + hour * 5) / 10
    st.info(f"Estimated engagement (baseline): **{int(baseline):,} interactions**")

st.caption("This prediction engine can be connected to an API to use live post data.")


# ---------------------------------------------------------
# DATA EXPLORATION (only if dataset available)
# ---------------------------------------------------------
if df is not None:
    st.markdown("---")
    st.subheader("👀 Data overview")
    st.dataframe(df.head())

    st.write(df["engajamento_total"].describe())

    # Charts
    st.markdown("---")
    st.subheader("📊 Engagement patterns")

    cg1, cg2 = st.columns(2)

    with cg1:
        if "tipo_post" in df.columns:
            eng = df.groupby("tipo_post")["engajamento_total"].mean()
            fig, ax = plt.subplots()
            ax.bar(eng.index, eng.values)
            ax.set_title("Avg engagement by post type")
            plt.xticks(rotation=15)
            st.pyplot(fig)

    with cg2:
        if "dia_semana" in df.columns:
            eng = df.groupby("dia_semana")["engajamento_total"].mean()
            fig, ax = plt.subplots()
            ax.bar(eng.index, eng.values)
            ax.set_title("Avg engagement by weekday")
            plt.xticks(rotation=15)
            st.pyplot(fig)
