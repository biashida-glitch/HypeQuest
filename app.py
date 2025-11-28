import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import io
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
# -------------------------------------------------------
# SENTIMENT PREDICTION FUNCTION (ADD THIS RIGHT AFTER IMPORTS)
# -------------------------------------------------------

def predict_sentiment_from_text(text: str) -> str:
    """
    Predict sentiment (positive / neutral / negative) from caption text
    using VADER sentiment analysis.
    """
    if not text or not text.strip():
        return "neutral"

    scores = analyzer.polarity_scores(text)
    comp = scores["compound"]

    if comp >= 0.05:
        return "positive"
    elif comp <= -0.05:
        return "negative"
    else:
        return "neutral"

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="HypeQuest – Instagram Engagement Prediction",
    layout="wide"
)

st.title("🔥 HypeQuest – Instagram Engagement Prediction")
st.caption("Prototype that predicts post engagement using Machine Learning. Works with CSV, Excel, JSON, Parquet, or manual input.")
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
# DATA NORMALIZATION (Handles missing columns)
# ---------------------------------------------------------
def prepare_dataframe(df):
    expected = [
        "data_post","tipo_post","hora_post","dia_semana",
        "tam_legenda","hashtag_count","emoji_count",
        "sentimento_legenda","engajamento_total"
    ]

    # If key columns missing, the app goes into "simulation only" mode
    missing = [c for c in expected if c not in df.columns]
    if missing:
        st.warning("Dataset missing required columns. Simulation mode only.")
        return None, None, None
    
    # Process date
    df["data_post"] = pd.to_datetime(df["data_post"], errors="coerce")

    # Add month / year if missing
    df["month"] = df["data_post"].dt.month.fillna(1).astype(int)
    df["year"] = df["data_post"].dt.year.fillna(2024).astype(int)

    # Prepare features + target
    feature_cols = [
        "tipo_post","hora_post","dia_semana","tam_legenda",
        "hashtag_count","emoji_count","sentimento_legenda",
        "month","year"
    ]
    X = pd.get_dummies(df[feature_cols], drop_first=False)
    y = df["engajamento_total"]

    return df, X, y


# ---------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------
def train_model(X, y):
    model = DecisionTreeRegressor(max_depth=6, random_state=42)
    model.fit(X, y)
    return model


# Sidebar – Load data
st.sidebar.header("📁 Load post dataset (optional)")

uploaded = st.sidebar.file_uploader(
    "Upload CSV, Excel, JSON or Parquet file",
    type=["csv","xlsx","xls","json","parquet"]
)

df_loaded = load_file(uploaded)

if df_loaded is not None:
    st.sidebar.success(f"Loaded dataset: {uploaded.name}")
else:
    st.sidebar.info("No dataset uploaded. App will run in simulation-only mode.")

# Prepare data & train model (only if dataset exists)
if df_loaded is not None:
    df, X, y = prepare_dataframe(df_loaded)

    if X is not None:
        model = train_model(X, y)
    else:
        model = None
else:
    df = None
    X = None
    model = None


# ---------------------------------------------------------
# WHAT-IF SIMULATOR (Works ALWAYS, even without dataset)
# ---------------------------------------------------------
st.subheader("🎛 Plan a new post and get the predicted engagement")

col1, col2, col3 = st.columns(3)

with col1:
    post_type = st.selectbox("Post type", ["image","video","reels","carousel"])
    hour = st.slider("Posting hour", 0, 23, 12)
    weekday = st.selectbox(
        "Day of week",
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    )

with col2:
    caption_len = st.slider("Caption length (characters)", 0, 500, 120)
    hashtag_ct = st.slider("Number of hashtags", 0, 20, 3)
    emoji_ct = st.slider("Number of emojis", 0, 10, 2)

with col3:
    sentiment = st.selectbox("Caption sentiment", ["positive","neutral","negative"])
    month = st.selectbox("Month", list(range(1,13)))
    year = st.selectbox("Year", [2023, 2024, 2025])

# Build DF for prediction
new_post = pd.DataFrame([{
    "tipo_post": post_type,
    "hora_post": hour,
    "dia_semana": weekday,
    "tam_legenda": caption_len,
    "hashtag_count": hashtag_ct,
    "emoji_count": emoji_ct,
    "sentimento_legenda": sentiment,
    "month": month,
    "year": year
}])

if X is not None:
    new_X = pd.get_dummies(new_post).reindex(columns=X.columns, fill_value=0)
    pred = model.predict(new_X)[0]
    st.success(f"⭐ Predicted engagement for this post: **{int(pred):,} interactions**")
else:
    st.warning("No dataset available → prediction uses placeholder baseline.")
    baseline = (caption_len + hashtag_ct*50 + emoji_ct*30 + hour*10) / 10
    st.info(f"Estimated engagement (baseline): **{int(baseline)} interactions**")

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
