import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# ---------------- BASIC CONFIG ----------------
st.set_page_config(
    page_title="HypeQuest – Instagram Engagement Prediction",
    layout="wide"
)

# ------------- DATA LOADING LAYER -------------
@st.cache_data
def load_data(uploaded_file=None):
    """
    For now: load from CSV.
    Future: replace this function to pull data from an API
    and return a pandas DataFrame with the same columns.
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        source = "Uploaded CSV"
    else:
        df = pd.read_csv("hypequest_dataset_ficticio.csv")
        source = "Default demo dataset (hypequest_dataset_ficticio.csv)"

    # Ensure date column is datetime
    if "data_post" in df.columns:
        df["data_post"] = pd.to_datetime(df["data_post"], errors="coerce")
    else:
        # In case real data comes without this column, just create a dummy
        df["data_post"] = pd.NaT

    # Create month/year if missing
    if "mes" in df.columns:
        df["month"] = df["mes"]
    else:
        df["month"] = df["data_post"].dt.month.fillna(1).astype(int)

    if "ano" in df.columns:
        df["year"] = df["ano"]
    else:
        df["year"] = df["data_post"].dt.year.fillna(2024).astype(int)

    return df, source

# ------------- MODEL TRAINING LAYER -----------
def train_model(df):
    # Expected feature columns
    feature_cols = [
        "tipo_post",
        "hora_post",
        "dia_semana",
        "tam_legenda",
        "hashtag_count",
        "emoji_count",
        "sentimento_legenda",
        "month",
        "year",
    ]
    target_col = "engajamento_total"

    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        st.error(f"Missing columns in dataset: {missing}")
        return None, None, None

    X = pd.get_dummies(df[feature_cols], drop_first=False)
    y = df[target_col]

    model = DecisionTreeRegressor(max_depth=6, random_state=42)
    model.fit(X, y)

    return model, X, y


# ------------------- UI -----------------------
st.title("🔥 HypeQuest – Instagram Engagement Prediction")
st.caption("Hackathon prototype using simulated data + a Machine Learning model.")

# Sidebar – data source
st.sidebar.header("📁 Input data")
uploaded_file = st.sidebar.file_uploader(
    "Upload posts data (CSV)", type=["csv"],
    help="In the future this will be replaced by an API connection."
)

df, source = load_data(uploaded_file)
st.sidebar.success(f"Data loaded from: {source}")

# Train model
st.subheader("🤖 Training prediction model (Decision Tree)")
model, X, y = train_model(df)

if model is None:
    st.stop()

# ---------------- WHAT-IF SCENARIO (TOP) ----------------
st.markdown("---")
st.subheader("🔮 Plan a new post and see the predicted engagement")

c1, c2, c3 = st.columns(3)

with c1:
    post_type = st.selectbox("Post type", ["imagem", "video", "reels", "carrossel"])
    hour = st.slider("Posting hour", 0, 23, 12)
    weekday = st.selectbox(
        "Day of week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    )

with c2:
    caption_len = st.slider("Caption length (characters)", 0, 400, 120)
    hashtag_ct = st.slider("Number of hashtags", 0, 15, 3)
    emoji_ct = st.slider("Number of emojis", 0, 10, 2)

with c3:
    sentiment = st.selectbox(
        "Caption sentiment", ["positivo", "neutro", "negativo"]
    )
    month = st.selectbox("Month", list(range(1, 13)))
    year = st.selectbox("Year", [2023, 2024, 2025])

new_post = pd.DataFrame(
    [
        {
            "tipo_post": post_type,
            "hora_post": hour,
            "dia_semana": weekday,
            "tam_legenda": caption_len,
            "hashtag_count": hashtag_ct,
            "emoji_count": emoji_ct,
            "sentimento_legenda": sentiment,
            "month": month,
            "year": year,
        }
    ]
)

new_proc = pd.get_dummies(new_post)
new_proc = new_proc.reindex(columns=X.columns, fill_value=0)

pred = model.predict(new_proc)[0]

st.success(
    f"⭐ Predicted total engagement for this post: **{int(pred):,} interactions**"
)

st.caption(
    "This is a demo model trained on simulated data – for production we would plug in live data via API."
)

# ---------------- DATA OVERVIEW & EXPLORATION --------------
st.markdown("---")
st.subheader("👀 Data overview")

st.write("First rows of the dataset:")
st.dataframe(df.head())

st.write("Target variable statistics (`engajamento_total`):")
st.write(df["engajamento_total"].describe())

# ---------------- QUICK EDA CHARTS ----------------
st.markdown("---")
st.subheader("📊 Quick engagement analysis")

col_g1, col_g2 = st.columns(2)

with col_g1:
    if "tipo_post" in df.columns:
        eng_tipo = (
            df.groupby("tipo_post")["engajamento_total"]
            .mean()
            .sort_values(ascending=False)
        )
        fig1, ax1 = plt.subplots()
        ax1.bar(eng_tipo.index, eng_tipo.values)
        ax1.set_title("Average engagement by post type")
        ax1.set_ylabel("Average engagement")
        ax1.set_xlabel("Post type")
        plt.xticks(rotation=15)
        st.pyplot(fig1)
    else:
        st.warning("Column 'tipo_post' not found for chart.")

with col_g2:
    if "dia_semana" in df.columns:
        eng_dia = (
            df.groupby("dia_semana")["engajamento_total"]
            .mean()
            .sort_values(ascending=False)
        )
        fig2, ax2 = plt.subplots()
        ax2.bar(eng_dia.index, eng_dia.values)
        ax2.set_title("Average engagement by weekday")
        ax2.set_ylabel("Average engagement")
        ax2.set_xlabel("Day of week")
        plt.xticks(rotation=30)
        st.pyplot(fig2)
    else:
        st.warning("Column 'dia_semana' not found for chart.")

# ---------------- FEATURE IMPORTANCE ----------------
st.markdown("---")
st.subheader("🌟 Feature importance")

importances = model.feature_importances_
idx = np.argsort(importances)[-15:]

fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
ax_imp.barh(X.columns[idx], importances[idx])
ax_imp.set_title("Top 15 most important features for the model")
plt.tight_layout()
st.pyplot(fig_imp)
