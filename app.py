import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# ---------------------------------------------------------
# BASIC CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="HypeQuest – Instagram Engagement & Sentiment Prediction",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* Card base */
    .hq-card {
        padding: 0.9rem 1.1rem;
        border-radius: 0.8rem;
        border: 1px solid rgba(0,0,0,0.05);
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    .hq-positive {
        background-color: #ecfdf3;
        border-color: #22c55e33;
    }
    .hq-neutral {
        background-color: #eff6ff;
        border-color: #3b82f633;
    }
    .hq-negative {
        background-color: #fef2f2;
        border-color: #ef444433;
    }
    .hq-pill {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .hq-pill-positive {
        background-color: #22c55e1A;
        color: #15803d;
        border: 1px solid #22c55e55;
    }
    .hq-pill-neutral {
        background-color: #3b82f61A;
        color: #1d4ed8;
        border: 1px solid #3b82f655;
    }
    .hq-pill-negative {
        background-color: #ef44441A;
        color: #b91c1c;
        border: 1px solid #ef444455;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔥 HypeQuest – Instagram Engagement & Sentiment Prediction")
st.caption(
    "Prototype that predicts post engagement and sentiment using Machine Learning. "
    "Works with CSV, Excel, JSON, Parquet, or manual input, and learns from historical posts per profile."
)
st.markdown("---")

# =========================================================
# 1) FAKE API (simulação por perfil) – FUTURAMENTE TROCA POR API REAL
# =========================================================

def fake_api_profile_data(profile_handle: str) -> pd.DataFrame:
    """
    Simula uma API que retorna histórico de posts para um perfil específico.
    No futuro, esta função pode ser substituída por chamadas reais de API.
    Cada perfil tem um 'estilo' diferente de engajamento.
    """
    rng = np.random.default_rng(42)

    n = 160  # histórico por perfil
    base_engagement = {
        "@pubg": 2800,
        "@playinzoi": 1900,
        "@generic": 1200
    }.get(profile_handle, 1000)

    post_types = ["image", "video", "reels", "carousel"]
    topics = ["update", "trailer", "gameplay", "collaboration", "community"]

    data = []
    for i in range(n):
        ptype = rng.choice(post_types)
        topic = rng.choice(topics)
        hour = int(rng.integers(0, 24))
        weekday = rng.choice(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        month = int(rng.integers(1, 13))
        num_hashtags = int(rng.integers(0, 10))

        # caption artificial
        caption_len = int(rng.integers(50, 280))
        caption = f"{topic} news for the community – item {i}"

        # engajamento baseado em alguns fatores
        prime_time_boost = 1.2 if 18 <= hour <= 22 else 1.0
        weekend_boost = 1.15 if weekday in ["Saturday", "Sunday"] else 1.0
        topic_boost = {
            "trailer": 1.3,
            "collaboration": 1.25,
            "update": 1.1,
            "gameplay": 1.0,
            "community": 0.95,
        }.get(topic, 1.0)

        noise = rng.normal(0, base_engagement * 0.15)
        engagement_total = max(
            50,
            base_engagement * prime_time_boost * weekend_boost * topic_boost
            + num_hashtags * 35
            + caption_len * 3
            + noise,
        )

        data.append(
            {
                "post_type": ptype,
                "topic": topic,
                "hour": hour,
                "weekday": weekday,
                "month": month,
                "num_hashtags": num_hashtags,
                "caption": caption,
                "caption_length": caption_len,
                "engagement_total": engagement_total,
            }
        )

    return pd.DataFrame(data)


# =========================================================
# 2) UPLOAD DE DATASET – CSV / EXCEL / JSON / PARQUET
# =========================================================

def load_uploaded_file(uploaded):
    """Carrega múltiplos tipos de arquivo se o usuário enviar algo."""
    if uploaded is None:
        return None

    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded)
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(uploaded)
        elif name.endswith(".json"):
            return pd.read_json(uploaded)
        elif name.endswith(".parquet"):
            return pd.read_parquet(uploaded)
        else:
            st.sidebar.error("Unsupported file type. Use CSV, Excel, JSON, or Parquet.")
            return None
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        return None


# =========================================================
# 3) PREP & TRAIN MODEL
# =========================================================

FEATURE_COLS = [
    "post_type",
    "topic",
    "hour",
    "weekday",
    "month",
    "num_hashtags",
    "caption_length",
]


def prepare_training_data(df: pd.DataFrame):
    """Garante colunas e prepara X, y para treino."""
    required = set(FEATURE_COLS + ["engagement_total"])
    missing = required - set(df.columns)
    if missing:
        st.sidebar.warning(
            f"Dataset for this profile is missing required columns: {missing}. "
            "Model training skipped; baseline estimation will be used."
        )
        return None, None

    X = df[FEATURE_COLS].copy()
    X = pd.get_dummies(X, columns=["post_type", "topic", "weekday"], drop_first=False)
    y = df["engagement_total"].astype(float)
    return X, y


def train_profile_model(df: pd.DataFrame):
    """Treina um modelo DecisionTreeRegressor para o perfil."""
    X, y = prepare_training_data(df)
    if X is None:
        return None, None

    model = DecisionTreeRegressor(max_depth=6, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()


# =========================================================
# 4) SENTIMENT & SUGGESTIONS (regra simples baseada em keywords)
# =========================================================

POSITIVE_KEYWORDS = [
    "amazing",
    "awesome",
    "exciting",
    "insane",
    "win",
    "victory",
    "new",
    "exclusive",
    "epic",
    "legendary",
    "gg",
    "let's go",
    "update",
    "trailer",
]

NEGATIVE_KEYWORDS = [
    "bug",
    "issue",
    "problem",
    "delay",
    "sorry",
    "unfortunately",
    "toxic",
    "angry",
    "rage",
]


def analyze_caption(caption: str, context: dict):
    """
    Avalia legenda:
    - Sentimento: positive / neutral / negative
    - Palavras-chave positivas/negativas
    - Sugestões de melhoria
    - Legenda sugerida (mais hype / CTA)
    """
    text = caption.lower()
    matched_pos = [w for w in POSITIVE_KEYWORDS if w in text]
    matched_neg = [w for w in NEGATIVE_KEYWORDS if w in text]

    score = len(matched_pos) - len(matched_neg)

    if score > 1:
        sentiment = "POSITIVE"
    elif score < -1:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"

    suggestions = []

    # comprimento
    length = len(caption)
    if length < 40:
        suggestions.append(
            "Caption is very short. Try adding more context, a hook, or a clear benefit for players."
        )
    elif length > 250:
        suggestions.append(
            "Caption is very long. Consider trimming to keep only the most important message."
        )

    # hashtags
    if context["num_hashtags"] == 0:
        suggestions.append("Consider adding a few hashtags to increase discoverability.")
    elif context["num_hashtags"] > 10:
        suggestions.append(
            "There are many hashtags. Try focusing on 3–8 highly relevant ones."
        )

    # horário
    if not (18 <= context["hour"] <= 22):
        suggestions.append(
            "You are outside the usual prime time (18–22). Check if your audience is active at this hour."
        )

    # sentimento neutro / negativo
    if sentiment == "NEUTRAL":
        suggestions.append(
            "Caption feels neutral. Try adding hype words, emojis or a stronger emotion."
        )
    if sentiment == "NEGATIVE":
        suggestions.append(
            "Tone is negative. Make sure this matches the intention of the post or soften the wording."
        )

    if not suggestions:
        suggestions.append("This caption already looks strong for your audience.")

    # Legenda sugerida – usa parte do texto original + CTA
    base = caption.strip()
    if not base:
        base = "We have something new for you"

    improved = (
        base.rstrip(".! ")
        + " 🔥 This is going to be huge! Tell us what you think in the comments and tag your squad."
    )

    return sentiment, matched_pos, matched_neg, suggestions, improved


# =========================================================
# 5) SIDEBAR – SELEÇÃO DE PERFIL + DATASET / API
# =========================================================

st.sidebar.header("Profile & data")

profile = st.sidebar.selectbox(
    "Instagram profile",
    ["@pubg", "@playinzoi", "@generic"],
    index=0,
)

st.sidebar.markdown("**Optional:** upload a custom dataset for this profile.")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV, Excel, JSON, or Parquet",
    type=["csv", "xlsx", "xls", "json", "parquet"],
)

df_profile = None
data_source_label = ""

if uploaded_file is not None:
    df_profile = load_uploaded_file(uploaded_file)
    if df_profile is not None:
        data_source_label = f"Using uploaded dataset: `{uploaded_file.name}`"
else:
    # fallback para fake API
    df_profile = fake_api_profile_data(profile)
    data_source_label = "Using internal fake-API data (for demo)."

if df_profile is None or df_profile.empty:
    st.warning(
        "No historical data available for this profile. The app will use a simple baseline "
        "formula instead of a Machine Learning model."
    )
    model = None
    feature_columns = []
else:
    model, feature_columns = train_profile_model(df_profile)

st.sidebar.info(data_source_label)

# =========================================================
# 6) MAIN UI – PLANEJAR NOVO POST
# =========================================================

st.subheader("🧩 Plan a new Instagram post")

col_left, col_right = st.columns(2)

with col_left:
    post_type = st.selectbox("Post type", ["image", "video", "reels", "carousel"])
    posting_hour = st.slider("Posting hour", 0, 23, 18)
    weekday = st.selectbox(
        "Weekday",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        index=4,
    )
    num_hashtags = st.slider("Number of hashtags", 0, 20, 3)

with col_right:
    topic = st.selectbox(
        "Post topic",
        ["update", "trailer", "gameplay", "collaboration", "community"],
        index=0,
    )
    month = st.selectbox("Month", list(range(1, 13)), index=0)

st.markdown("### ✍️ Caption")

if "caption_text" not in st.session_state:
    st.session_state["caption_text"] = ""

caption_text = st.text_area(
    "Caption text",
    value=st.session_state["caption_text"],
    height=120,
    placeholder="Write the caption in English – sentiment and suggestions will be based on this text.",
)

caption_length = len(caption_text)
st.caption(f"Caption length: **{caption_length}** characters (used as a feature in the model).")

# Botão para avaliar
evaluate = st.button("✨ Evaluate caption & predict", type="primary")

# =========================================================
# 7) PREDIÇÃO QUANDO CLICA NO BOTÃO
# =========================================================

if evaluate:
    with st.spinner("Thinking like the HypeQuest brain..."):
        # Sentiment & suggestions
        context = {
            "hour": posting_hour,
            "num_hashtags": num_hashtags,
            "weekday": weekday,
            "topic": topic,
        }
        sentiment, matched_pos, matched_neg, suggestions, improved_caption = analyze_caption(
            caption_text, context
        )

        # Monta DF para predição
        new_post = pd.DataFrame(
            [
                {
                    "post_type": post_type,
                    "topic": topic,
                    "hour": posting_hour,
                    "weekday": weekday,
                    "month": month,
                    "num_hashtags": num_hashtags,
                    "caption_length": caption_length,
                }
            ]
        )

        # One-hot para as mesmas colunas do treino, se existir modelo
        if model is not None and feature_columns:
            new_X = pd.get_dummies(new_post, columns=["post_type", "topic", "weekday"], drop_first=False)
            # alinhar colunas
            new_X = new_X.reindex(columns=feature_columns, fill_value=0)
            pred_engagement = float(model.predict(new_X)[0])
            engagement_note = "Estimated using the ML model trained on this profile's historical posts."
        else:
            # baseline simples caso não tenha modelo
            pred_engagement = (
                0.8 * caption_length + num_hashtags * 40 + posting_hour * 5
            )
            engagement_note = (
                "No ML model available – using a simple baseline estimation instead."
            )

        # ----------------------------------------------------
        # RESULTADO VISUAL – SENTIMENTO + ENGAJAMENTO
        # ----------------------------------------------------
        st.markdown("---")
        st.subheader("🔍 Predicted sentiment and engagement")

        # Badge de sentimento
        if sentiment == "POSITIVE":
            icon = "✅"
            css_class = "hq-card hq-positive"
            pill_class = "hq-pill hq-pill-positive"
        elif sentiment == "NEGATIVE":
            icon = "⚠️"
            css_class = "hq-card hq-negative"
            pill_class = "hq-pill hq-pill-negative"
        else:
            icon = "💬"
            css_class = "hq-card hq-neutral"
            pill_class = "hq-pill hq-pill-neutral"

        st.markdown(
            f"""
            <div class="{css_class}">
                <span class="{pill_class}">{icon} {sentiment}</span><br/>
                <span>Overall tone for this caption based on keywords and context.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="hq-card hq-neutral">
                <strong>Predicted engagement:</strong> ~ <strong>{int(pred_engagement):,} interactions</strong><br/>
                <span style='font-size: 0.85rem; color: #555;'>{engagement_note}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Palavras-chave em dois cards pequenos
        kw_col1, kw_col2 = st.columns(2)
        with kw_col1:
            st.caption("Matched positive keywords in caption:")
            if matched_pos:
                st.markdown(
                    f"<div class='hq-card hq-positive'>{', '.join(matched_pos)}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='hq-card hq-neutral'>No clearly positive keywords detected.</div>",
                    unsafe_allow_html=True,
                )

        with kw_col2:
            st.caption("Matched negative keywords in caption:")
            if matched_neg:
                st.markdown(
                    f"<div class='hq-card hq-negative'>{', '.join(matched_neg)}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='hq-card hq-neutral'>No negative keywords detected.</div>",
                    unsafe_allow_html=True,
                )

        # ----------------------------------------------------
        # SUGESTÕES DE MELHORIA
        # ----------------------------------------------------
        st.markdown("### 💡 Suggestions to improve this caption")
        for s in suggestions:
            st.markdown(f"- {s}")

        # ----------------------------------------------------
        # LEGENDA SUGERIDA + BOTÃO CHAMATIVO DE CÓPIA
        # ----------------------------------------------------
        st.markdown("### ✨ Suggested improved caption")

        st.info(
            "This suggestion is generated from your original caption. "
            "Click the button below to copy it into the editor above and tweak as you like."
        )

        st.text_area(
            "Suggested version (you can copy/paste):",
            value=improved_caption,
            height=120,
            key="suggested_caption_view",
        )

        apply_suggestion = st.button("📋 Use this suggested caption", type="secondary")
        if apply_suggestion:
            st.session_state["caption_text"] = improved_caption
            st.success("Suggested caption applied to the editor above – you can now adjust or post it!")

        st.markdown("---")
        st.caption(
            "This prototype is history-driven: the logic learns from past posts of each profile. "
            "New profiles without historical data will only have baseline estimates until real data is ingested via API or upload."
        )

