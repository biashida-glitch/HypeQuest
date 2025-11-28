import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from typing import List
import re
import time

# =========================================================
#  PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="HypeQuest – Instagram Engagement & Sentiment",
    layout="wide",
)


# =========================================================
#  HELPER: NICE INFO CARDS
# =========================================================
def info_card(title: str, body: str, tone: str = "neutral"):
    """
    Small helper to render nice cards.
    tone: 'neutral', 'positive', 'negative'
    """
    colors = {
        "neutral": {"bg": "#f9fafb", "border": "#e5e7eb", "title": "#111827"},
        "positive": {"bg": "#ecfdf5", "border": "#6ee7b7", "title": "#065f46"},
        "negative": {"bg": "#fef2f2", "border": "#fecaca", "title": "#991b1b"},
    }
    c = colors.get(tone, colors["neutral"])

    st.markdown(
        f"""
        <div style="
            margin-top:0.5rem;
            padding:0.85rem 1rem;
            border-radius:0.75rem;
            background-color:{c['bg']};
            border:1px solid {c['border']};
        ">
            <div style="font-weight:600;color:{c['title']};margin-bottom:0.15rem;">
                {title}
            </div>
            <div style="font-size:14px;color:#374151;line-height:1.5;">
                {body}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
#  FAKE "API" PER PROFILE (EASY TO SWAP FOR REAL API LATER)
# =========================================================
def load_history_for_profile(profile_handle: str) -> pd.DataFrame | None:
    """
    Simulates an API returning historical posts for a given profile.
    In production, this function would call the real API and
    return a DataFrame with the same columns.
    """
    # Expected columns (Portuguese names kept to match your TCC)
    cols = [
        "data_post",
        "tipo_post",
        "hora_post",
        "dia_semana",
        "tam_legenda",
        "hashtag_count",
        "emoji_count",
        "sentimento_legenda",
        "engajamento_total",
    ]

    # Small fake datasets per profile
    if profile_handle == "@pubg":
        data = [
            ["2024-02-01", "video", 20, "Thursday", 320, 5, 3, "positivo", 2800],
            ["2024-02-05", "reels", 19, "Monday", 180, 4, 2, "positivo", 2600],
            ["2024-02-10", "image", 14, "Saturday", 140, 3, 1, "neutro", 1900],
            ["2024-02-15", "video", 21, "Thursday", 400, 6, 4, "positivo", 3100],
            ["2024-02-20", "reels", 18, "Tuesday", 210, 5, 3, "positivo", 2950],
        ]
    elif profile_handle == "@playinzoi":
        data = [
            ["2024-02-02", "image", 18, "Friday", 260, 4, 2, "positivo", 1900],
            ["2024-02-07", "video", 21, "Wednesday", 380, 6, 3, "positivo", 2300],
            ["2024-02-12", "reels", 17, "Monday", 200, 5, 2, "neutro", 1700],
            ["2024-02-18", "image", 20, "Sunday", 300, 4, 3, "positivo", 2100],
        ]
    else:
        # No fake API for other profiles → return None
        return None

    df = pd.DataFrame(data, columns=cols)
    return df


# =========================================================
#  FILE LOADER (CSV / EXCEL / JSON / PARQUET)
# =========================================================
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


# =========================================================
#  DATA NORMALIZATION (HISTORICAL DATA)
# =========================================================
def prepare_dataframe(df: pd.DataFrame):
    expected = [
        "data_post",
        "tipo_post",
        "hora_post",
        "dia_semana",
        "tam_legenda",
        "hashtag_count",
        "emoji_count",
        "sentimento_legenda",
        "engajamento_total",
    ]

    missing = [c for c in expected if c not in df.columns]
    if missing:
        st.warning(
            "Dataset is missing required columns: "
            + ", ".join(missing)
            + ". Using baseline instead of ML model."
        )
        return None, None, None

    df = df.copy()
    df["data_post"] = pd.to_datetime(df["data_post"], errors="coerce")

    df["month"] = df["data_post"].dt.month.fillna(1).astype(int)
    df["year"] = df["data_post"].dt.year.fillna(2024).astype(int)

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

    X = pd.get_dummies(df[feature_cols], drop_first=False)
    y = df["engajamento_total"]

    return df, X, y


def train_model(X, y):
    model = DecisionTreeRegressor(max_depth=6, random_state=42)
    model.fit(X, y)
    return model


# =========================================================
#  SIMPLE SENTIMENT ENGINE (KEYWORD-BASED)
# =========================================================
POSITIVE_KEYWORDS = [
    "new",
    "update",
    "collab",
    "collaboration",
    "event",
    "win",
    "victory",
    "legendary",
    "exclusive",
    "drop",
    "season",
    "launch",
    "free",
    "reward",
    "rewarded",
    "celebrate",
    "best",
    "amazing",
    "exciting",
    "fun",
    "hype",
    "gg",
    "buff",
]

NEGATIVE_KEYWORDS = [
    "bug",
    "issue",
    "problem",
    "lag",
    "crash",
    "nerf",
    "toxic",
    "hate",
    "boring",
    "broken",
    "unfair",
    "angry",
    "frustrated",
    "delay",
    "late",
]


def predict_sentiment_from_text(text: str):
    """
    Returns:
        sentiment_label: 'positive' | 'neutral' | 'negative'
        matched_pos: list of positive keywords found
        matched_neg: list of negative keywords found
    """
    if not text or not text.strip():
        return "neutral", [], []

    lower = text.lower()
    matched_pos = [w for w in POSITIVE_KEYWORDS if w in lower]
    matched_neg = [w for w in NEGATIVE_KEYWORDS if w in lower]

    score = len(matched_pos) - len(matched_neg)

    # add small signal based on exclamation marks and emojis
    exclam = lower.count("!")
    if exclam >= 2:
        score += 1

    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    else:
        label = "neutral"

    return label, matched_pos, matched_neg


# =========================================================
#  CAPTION IMPROVEMENT – BASED ON ORIGINAL TEXT
# =========================================================
def generate_new_caption(
    original_caption: str,
    sentiment: str,
    matched_pos: List[str],
    matched_neg: List[str],
    profile_handle: str,
) -> str:
    """
    Gera uma legenda melhorada a partir da legenda original,
    reaproveitando o conteúdo, reforçando o tom positivo e
    adicionando CTA, evitando palavras negativas.
    """

    text = original_caption.strip()

    if not text:
        # fallback se a pessoa não escreveu nada
        if profile_handle == "@pubg":
            return "New drop is coming. Jump in, squad up and share your best plays with us! 🔥"
        elif profile_handle == "@playinzoi":
            return "New creation incoming. Build, customize and show us your world! ✨"
        else:
            return "Something new is coming. Tell us what you think in the comments! 💬"

    # 1) Limpar espaços extras
    text = re.sub(r"\s+", " ", text)

    # 2) Remover palavras claramente negativas, se tiver alguma match
    for w in matched_neg:
        pattern = re.compile(r"\b" + re.escape(w) + r"\b", flags=re.IGNORECASE)
        text = pattern.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 3) Garantir que tem algum "hook" positivo
    if matched_pos:
        pos_word = matched_pos[0]
    else:
        pos_word = "exciting"

    # 4) Normalizar primeira letra (deixar com cara de frase bem escrita)
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]

    # 5) Adicionar algum emoji leve se não tiver nenhum
    if not re.search(r"[🔥✨🎮🚀💥💛⭐️🎥🏆]", text):
        if profile_handle == "@pubg":
            text += " 🔥"
        elif profile_handle == "@playinzoi":
            text += " ✨"
        else:
            text += " ⭐️"

    # 6) Adicionar CTA baseado no perfil
    if profile_handle == "@pubg":
        cta = "Drop into the comments and tell us what you think!"
    elif profile_handle == "@playinzoi":
        cta = "Show us what you would create and share your ideas below!"
    else:
        cta = "Tell us what you think and share your thoughts below!"

    # 7) Montar a legenda final
    new_caption = f"{text} This is going to be {pos_word}! {cta}"
    new_caption = " ".join(new_caption.split())

    return new_caption


# =========================================================
#  HEADER
# =========================================================
st.title("🔥 HypeQuest – Instagram Engagement & Sentiment Prediction")
st.caption(
    "Prototype that predicts post engagement and sentiment using Machine Learning. "
    "Works with CSV, Excel, JSON, Parquet, or manual input, and learns from historical posts."
)
st.markdown("---")


# =========================================================
#  SIDEBAR – PROFILE, DATASET, DATA SOURCE
# =========================================================
st.sidebar.header("Profile & data")

profile_handle = st.sidebar.selectbox(
    "Instagram profile",
    ["@pubg", "@playinzoi", "@other"],
)

# Upload manual dataset
st.sidebar.subheader("Load post dataset (optional)")
uploaded = st.sidebar.file_uploader(
    "Upload CSV / Excel / JSON / Parquet",
    type=["csv", "xlsx", "xls", "json", "parquet"],
)

df_uploaded = load_file(uploaded)

# Fake API data
df_api = load_history_for_profile(profile_handle)

# Decide which data source is used for the model
if df_uploaded is not None:
    df_history = df_uploaded.copy()
    data_source = "Uploaded dataset"
elif df_api is not None:
    df_history = df_api.copy()
    data_source = "Profile API (simulated)"
else:
    df_history = None
    data_source = "No historical data"

st.sidebar.markdown("**Data source used:** " + data_source)

if df_history is not None:
    st.sidebar.success(f"Historical posts available: {len(df_history)}")
else:
    st.sidebar.info("No historical posts available for this profile.")

# Prepare data & model
if df_history is not None:
    df_prepared, X, y = prepare_dataframe(df_history)
    if X is not None:
        model = train_model(X, y)
    else:
        model = None
else:
    df_prepared = None
    X = None
    model = None


# =========================================================
#  MAIN: WHAT-IF SIMULATOR
# =========================================================
st.subheader("🧩 Plan a new Instagram post")

col_left, col_right = st.columns(2)

with col_left:
    post_type = st.selectbox("Post type", ["image", "video", "reels", "carousel"])
    posting_hour = st.slider("Posting hour", 0, 23, 20)
    weekday = st.selectbox(
        "Weekday",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    )
    num_hashtags = st.slider("Number of hashtags", 0, 20, 3)

with col_right:
    # Game/IP removed, using only profile handle as context
    post_topic = st.selectbox(
        "Post topic",
        ["update", "trailer", "collaboration", "gameplay", "community", "other"],
    )
    month = st.selectbox("Month", list(range(1, 13)))
    # Year not exposed as input anymore; fixed as 2025 for context
    year = 2025
    st.text_input("Year (used internally for trends)", value=str(year), disabled=True)

st.markdown("### ✍️ Caption")

caption_text = st.text_area(
    "Caption text",
    placeholder="Write your caption here...",
    height=150,
)

caption_length = len(caption_text.strip())
st.caption(f"Caption length: {caption_length} characters (used as a feature in the model).")

evaluate_button = st.button("✨ Evaluate caption & get suggestions")


# =========================================================
#  EVALUATION LOGIC
# =========================================================
if evaluate_button:

    if not caption_text.strip():
        st.warning("Please write a caption before evaluating.")
    else:
        # nice little "thinking" animation
        with st.spinner("HypeQuest is thinking about this post…"):
            time.sleep(0.7)

            # 1) Sentiment from caption
            predicted_sentiment, matched_pos, matched_neg = predict_sentiment_from_text(
                caption_text
            )

            # 2) Build feature row for engagement prediction
            #    (even if we only have baseline)
            new_row = pd.DataFrame(
                [
                    {
                        "tipo_post": post_type,
                        "hora_post": posting_hour,
                        "dia_semana": weekday,
                        "tam_legenda": caption_length,
                        "hashtag_count": num_hashtags,
                        "emoji_count": 0,  # could be improved by counting emojis later
                        "sentimento_legenda": predicted_sentiment,
                        "month": month,
                        "year": year,
                    }
                ]
            )

            if X is not None and model is not None:
                new_X = pd.get_dummies(new_row).reindex(columns=X.columns, fill_value=0)
                predicted_engagement = float(model.predict(new_X)[0])
                model_used = True
            else:
                # Baseline estimation if no ML model
                baseline = (
                    caption_length * 0.8
                    + num_hashtags * 50
                    + (5 if predicted_sentiment == "positive" else 0)
                )
                predicted_engagement = baseline
                model_used = False

            # 3) Suggestions & improved caption
            suggestions: List[str] = []

            if caption_length < 80:
                suggestions.append(
                    "Caption is very short. Consider adding a bit more context or a clearer call to action."
                )
            elif caption_length > 400:
                suggestions.append(
                    "Caption is quite long. Check if you can simplify it to keep players engaged."
                )

            if num_hashtags == 0:
                suggestions.append(
                    "No hashtags used. Consider adding 2–5 relevant hashtags to increase discoverability."
                )
            elif num_hashtags > 10:
                suggestions.append(
                    "You are using many hashtags. Make sure they are truly relevant to the content."
                )

            if predicted_sentiment == "negative":
                suggestions.append(
                    "Caption sentiment looks negative. Check wording to avoid frustration or toxic tone."
                )
            elif predicted_sentiment == "neutral":
                suggestions.append(
                    "Caption sentiment is neutral. You might add more hype or positive language to excite players."
                )

            # Simple prime-time suggestion
            if posting_hour < 10 or posting_hour > 22:
                suggestions.append(
                    "Consider testing this post closer to your prime time (e.g., 18–22h) if your audience is active then."
                )

            new_caption = generate_new_caption(
                original_caption=caption_text,
                sentiment=predicted_sentiment,
                matched_pos=matched_pos,
                matched_neg=matched_neg,
                profile_handle=profile_handle,
            )

        # =====================================================
        #  RESULTS DISPLAY
        # =====================================================
        st.markdown("### 🔍 Predicted sentiment and engagement")

        # Sentiment card
        if predicted_sentiment == "positive":
            info_card(
                "Predicted sentiment",
                "POSITIVE – players are likely to feel hyped or excited about this copy.",
                tone="positive",
            )
        elif predicted_sentiment == "negative":
            info_card(
                "Predicted sentiment",
                "NEGATIVE – wording may trigger frustration or negative reactions.",
                tone="negative",
            )
        else:
            info_card(
                "Predicted sentiment",
                "NEUTRAL – mostly informative; you can add more hype or emotion if desired.",
                tone="neutral",
            )

        # Keywords
        col_pos, col_neg = st.columns(2)
        with col_pos:
            if matched_pos:
                info_card(
                    "Matched positive keywords in caption",
                    ", ".join(sorted(set(matched_pos))),
                    tone="positive",
                )
            else:
                info_card(
                    "Matched positive keywords in caption",
                    "No clearly positive keywords detected.",
                    tone="neutral",
                )
        with col_neg:
            if matched_neg:
                info_card(
                    "Matched negative keywords in caption",
                    ", ".join(sorted(set(matched_neg))),
                    tone="negative",
                )
            else:
                info_card(
                    "Matched negative keywords in caption",
                    "No negative keywords detected.",
                    tone="positive",
                )

        # Engagement prediction
        tone = "positive" if model_used else "neutral"
        model_text = "ML model (Decision Tree) using historical posts." if model_used else "baseline estimation (no historical data loaded)."
        info_card(
            "Predicted engagement",
            f"Estimated engagement: **{int(predicted_engagement):,} interactions**. "
            f"This value comes from the {model_text}",
            tone=tone,
        )

        # Suggestions list
        st.markdown("### 💡 Suggestions to improve this caption")
        if suggestions:
            for s in suggestions:
                info_card("Suggestion", s, tone="neutral")
        else:
            info_card(
                "Suggestion",
                "This caption already looks solid based on our current rules.",
                tone="positive",
            )

        # Suggested caption
        st.markdown("### ✨ Suggested improved caption")
        info_card(
            "New version (you can copy & paste)",
            new_caption,
            tone="positive",
        )

        # ORIGINAL vs SUGGESTED COMPARISON
        st.markdown("### 📊 Original vs. suggested caption")

        orig_len = len(caption_text.strip())
        new_len = len(new_caption.strip())

        orig_words = len(caption_text.split())
        new_words = len(new_caption.split())

        cta_terms = [
            "tell us",
            "share",
            "show us",
            "drop in",
            "comment",
            "join",
            "watch",
            "check out",
        ]

        def has_cta(text: str) -> bool:
            lower = text.lower()
            return any(t in lower for t in cta_terms)

        orig_has_cta = has_cta(caption_text)
        new_has_cta = has_cta(new_caption)

        col_o, col_n = st.columns(2)

        with col_o:
            st.markdown("**Original caption**")
            info_card(
                "Text",
                caption_text if caption_text.strip() else "<i>No caption provided.</i>",
                tone="neutral",
            )
            info_card(
                "Length",
                f"{orig_len} characters – {orig_words} words",
                tone="neutral",
            )
            info_card(
                "Call to action",
                "✅ Contains a CTA" if orig_has_cta else "⚠️ No clear CTA detected",
                tone="positive" if orig_has_cta else "negative",
            )

        with col_n:
            st.markdown("**Suggested caption**")
            info_card(
                "Text",
                new_caption,
                tone="positive" if predicted_sentiment == "positive" else "neutral",
            )
            info_card(
                "Length",
                f"{new_len} characters – {new_words} words",
                tone="neutral",
            )
            info_card(
                "Call to action",
                "✅ Contains a CTA" if new_has_cta else "⚠️ No clear CTA detected",
                tone="positive" if new_has_cta else "negative",
            )


# =========================================================
#  OPTIONAL: DATA EXPLORATION (IF HISTORY EXISTS)
# =========================================================
if df_prepared is not None:
    st.markdown("---")
    st.subheader("👀 Historical data overview")
    st.dataframe(df_prepared.head())

    st.write("Target distribution (engajamento_total):")
    st.write(df_prepared["engajamento_total"].describe())

    st.markdown("### 📊 Engagement patterns")
    cg1, cg2 = st.columns(2)

    with cg1:
        if "tipo_post" in df_prepared.columns:
            eng = df_prepared.groupby("tipo_post")["engajamento_total"].mean()
            fig, ax = plt.subplots()
            ax.bar(eng.index, eng.values)
            ax.set_title("Average engagement by post type")
            plt.xticks(rotation=15)
            st.pyplot(fig)

    with cg2:
        if "dia_semana" in df_prepared.columns:
            eng = df_prepared.groupby("dia_semana")["engajamento_total"].mean()
            fig, ax = plt.subplots()
            ax.bar(eng.index, eng.values)
            ax.set_title("Average engagement by weekday")
            plt.xticks(rotation=15)
            st.pyplot(fig)
