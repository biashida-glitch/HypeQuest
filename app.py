import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -------------------------------------------------------------------
st.set_page_config(
    page_title="HypeQuest – Instagram Engagement & Sentiment Prediction",
    layout="wide"
)

# -------------------------------------------------------------------
# PARÂMETROS GERAIS / “FAKE API”
# -------------------------------------------------------------------

# Seguidores aproximados por perfil (só p/ transformar taxa em interações)
FOLLOWERS_BY_PROFILE = {
    "@pubg": 800_000,
    "@playinzoi": 120_000,
    "@other": 50_000,
}

POST_TYPES = ["image", "video", "reels", "carousel"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TOPICS = ["update", "maintenance", "trailer", "collaboration", "gameplay", "community"]

# Palavras-chave para sentimento (simples e explicável)
POSITIVE_KEYWORDS = [
    "win", "victory", "gg", "amazing", "great", "hype", "huge", "awesome",
    "incredible", "excited", "fun", "love", "🔥", "🎉", "😍"
]
NEGATIVE_KEYWORDS = [
    "bug", "lag", "toxic", "cheater", "angry", "hate", "broken",
    "sad", "problem", "issue", "😡", "💀"
]


def generate_fake_profile_data(handle: str, n: int = 350) -> pd.DataFrame:
    """
    Simula dados históricos vindos de uma API para um perfil específico.
    Aqui definimos padrões REALISTAS de engajamento para o modelo aprender.
    """
    if handle == "@pubg":
        seed = 7
    elif handle == "@playinzoi":
        seed = 21
    else:
        seed = 99

    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "post_type": rng.choice(POST_TYPES, size=n),
        "hour": rng.integers(0, 24, size=n),
        "weekday": rng.choice(WEEKDAYS, size=n),
        "topic": rng.choice(TOPICS, size=n),
        "caption_len": rng.integers(40, 260, size=n),
        "hashtags": rng.integers(0, 15, size=n),
    })

    # ------------------------------
    # Engajamento absoluto (base)
    # ------------------------------
    base = rng.normal(500, 90, size=n)

    # Tipo de post
    base += np.where(df["post_type"] == "reels", 550, 0)
    base += np.where(df["post_type"] == "video", 350, 0)
    base += np.where(df["post_type"] == "image", 170, 0)
    base += np.where(df["post_type"] == "carousel", 220, 0)

    # Horário
    hour = df["hour"]
    base += np.where(hour.between(18, 22), 450, 0)   # prime time
    base += np.where(hour.between(12, 15), 160, 0)
    base += np.where(hour.between(0, 6), -220, 0)

    # Dia da semana
    base += np.where(df["weekday"].isin(["Friday", "Saturday"]), 380, 0)
    base += np.where(df["weekday"] == "Sunday", 230, 0)
    base += np.where(df["weekday"].isin(["Monday", "Tuesday"]), -90, 0)

    # Tópico
    base += np.where(df["topic"].isin(["update", "trailer", "collaboration"]), 260, 0)
    base += np.where(df["topic"] == "maintenance", -110, 0)

    # Tamanho da legenda (ideal ~150 chars)
    len_center = 150
    base += -0.9 * np.abs(df["caption_len"] - len_center) + 130

    # Hashtags – benefício decrescente
    base += np.minimum(df["hashtags"], 8) * 45

    noise = rng.normal(0, 80, size=n)
    engagement = np.maximum(50, base + noise).astype(int)
    df["engagement"] = engagement

    # ------------------------------
    # Taxa de engajamento (%)
    # ------------------------------
    followers = FOLLOWERS_BY_PROFILE.get(handle, 100_000)
    df["followers"] = followers
    df["engagement_rate"] = (df["engagement"] / followers) * 100.0

    # ------------------------------
    # Legenda “fictícia” + sentimento
    # ------------------------------
    def build_fake_caption(row):
        words = []
        if row["topic"] == "update":
            words.append("new")
            words.append("update")
            words.append("coming")
        if row["topic"] == "trailer":
            words.append("trailer")
            words.append("watch")
        if row["topic"] == "collaboration":
            words.append("collab")
            words.append("special")
        if row["topic"] == "maintenance":
            words.append("maintenance")
            words.append("sorry")
        if row["post_type"] == "reels":
            words.append("reels")
        if row["hashtags"] > 5:
            words.append("#hype")
        base_caption = " ".join(words) or "new content"
        extra_len = max(0, row["caption_len"] - len(base_caption))
        return (base_caption + " x" * extra_len)[:row["caption_len"]]

    df["caption"] = df.apply(build_fake_caption, axis=1)

    # Cria sentimento “ground truth” combinando keywords + performance
    def label_sentiment(row):
        txt = row["caption"].lower()
        pos_hit = any(k in txt for k in POSITIVE_KEYWORDS)
        neg_hit = any(k in txt for k in NEGATIVE_KEYWORDS)

        if row["engagement_rate"] > df["engagement_rate"].quantile(0.7):
            return "positive"
        if row["engagement_rate"] < df["engagement_rate"].quantile(0.3):
            return "negative"
        if pos_hit and not neg_hit:
            return "positive"
        if neg_hit and not pos_hit:
            return "negative"
        return "neutral"

    df["sentiment"] = df.apply(label_sentiment, axis=1)

    return df


# -------------------------------------------------------------------
# CARREGAR DATASET EXTERNO (CSV / XLSX / JSON / PARQUET)
# -------------------------------------------------------------------

def load_uploaded_file(uploaded):
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded)
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(uploaded)
        if name.endswith(".json"):
            return pd.read_json(uploaded)
        if name.endswith(".parquet"):
            return pd.read_parquet(uploaded)
        st.error("Unsupported file type. Please upload CSV, Excel, JSON or Parquet.")
        return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


# -------------------------------------------------------------------
# TRAINING UTILITIES
# -------------------------------------------------------------------

FEATURE_COLS = ["post_type", "hour", "weekday", "topic", "caption_len", "hashtags"]


def train_models(df: pd.DataFrame):
    """
    Treina:
      - DecisionTreeRegressor para engagement_rate (%)
      - DecisionTreeClassifier para sentiment
    Retorna (regressor, X_reg, classifier, X_clf, label_encoder_sentiment)
    """
    df = df.copy()
    df = df.dropna(subset=["engagement_rate", "sentiment"])

    X = pd.get_dummies(df[FEATURE_COLS], drop_first=False)

    # Modelo de regressão (taxa de engajamento)
    y_reg = df["engagement_rate"]
    reg = DecisionTreeRegressor(max_depth=6, random_state=42)
    reg.fit(X, y_reg)

    # Modelo de classificação (sentimento)
    le = LabelEncoder()
    y_clf = le.fit_transform(df["sentiment"])
    clf = DecisionTreeClassifier(max_depth=6, random_state=42)
    clf.fit(X, y_clf)

    return reg, X, clf, X, le


def predict_rate(reg, X_train, feat_dict):
    df_new = pd.DataFrame([feat_dict])
    X_new = pd.get_dummies(df_new).reindex(columns=X_train.columns, fill_value=0)
    return float(reg.predict(X_new)[0])


def predict_sentiment(clf, X_train, le, feat_dict):
    df_new = pd.DataFrame([feat_dict])
    X_new = pd.get_dummies(df_new).reindex(columns=X_train.columns, fill_value=0)
    pred_idx = clf.predict(X_new)[0]
    return le.inverse_transform([pred_idx])[0]


# Heurística fallback quando não há dados
def baseline_rate(feat_dict):
    base = 3.0
    post_type = feat_dict.get("post_type", "image")
    hour = feat_dict.get("hour", 12)
    weekday = feat_dict.get("weekday", "Wednesday")
    topic = feat_dict.get("topic", "update")
    caption_len = feat_dict.get("caption_len", 100)
    hashtags = feat_dict.get("hashtags", 3)

    if post_type == "reels":
        base += 1.5
    elif post_type == "video":
        base += 1.0
    elif post_type == "carousel":
        base += 0.6

    if weekday in ["Friday", "Saturday"]:
        base += 1.0
    if 18 <= hour <= 22:
        base += 1.1

    if topic in ["update", "trailer", "collaboration"]:
        base += 0.8
    if topic == "maintenance":
        base -= 0.4

    base += np.minimum(hashtags, 8) * 0.12
    base += -0.012 * np.abs(caption_len - 150) + 0.4

    return max(0.2, base)


def keyword_based_sentiment(caption_text: str) -> str:
    if not caption_text or not caption_text.strip():
        return "neutral"
    txt = caption_text.lower()
    has_pos = any(k in txt for k in POSITIVE_KEYWORDS)
    has_neg = any(k in txt for k in NEGATIVE_KEYWORDS)
    if has_pos and not has_neg:
        return "positive"
    if has_neg and not has_pos:
        return "negative"
    return "neutral"


# -------------------------------------------------------------------
# SIDEBAR – PERFIL + DATASETS
# -------------------------------------------------------------------

st.sidebar.header("Profile & data")

profile = st.sidebar.selectbox(
    "Instagram profile",
    ["@pubg", "@playinzoi", "@other"],
    index=0
)

uploaded_file = st.sidebar.file_uploader(
    "Optional: upload historical post dataset (CSV, Excel, JSON, Parquet)",
    type=["csv", "xlsx", "xls", "json", "parquet"]
)

df_uploaded = load_uploaded_file(uploaded_file)

# Carrega dados da “API” fake
df_api = generate_fake_profile_data(profile)

if df_uploaded is not None:
    st.sidebar.success("External dataset loaded. Merged with API-like data.")
    df_all = pd.concat([df_api, df_uploaded], ignore_index=True, sort=False)
else:
    df_all = df_api.copy()

st.sidebar.markdown(
    f"<span style='font-size:13px;color:#6b7280'>Historical posts available for <b>{profile}</b>: {len(df_all)}</span>",
    unsafe_allow_html=True
)

# Treina modelos
reg_model, X_reg, clf_model, X_clf, le_sent = train_models(df_all)

# Estatísticas para dicas
best_hours = df_all.groupby("hour")["engagement_rate"].mean().sort_values(ascending=False)
best_weekdays = df_all.groupby("weekday")["engagement_rate"].mean().sort_values(ascending=False)
best_topics = df_all.groupby("topic")["engagement_rate"].mean().sort_values(ascending=False)

# -------------------------------------------------------------------
# LAYOUT PRINCIPAL – CONTROLES
# -------------------------------------------------------------------

st.title("HypeQuest – Instagram Engagement & Sentiment Prediction")
st.caption(
    "Prototype that predicts post engagement and sentiment using Machine Learning. "
    "Works with CSV, Excel, JSON, Parquet, or manual input, and learns from historical posts."
)
st.markdown("---")

st.subheader("🧩 Plan a new Instagram post")

c1, c2 = st.columns(2)

with c1:
    post_type = st.selectbox("Post type", POST_TYPES, index=1)  # video default
    hour = st.slider("Posting hour (UTC)", 0, 23, 20)
    weekday = st.selectbox("Weekday", WEEKDAYS, index=6)  # Sunday default
    hashtags = st.slider("Number of hashtags", 0, 20, 3)

with c2:
    topic = st.selectbox("Post topic", TOPICS, index=0)
    month = st.selectbox(  # só display, não entra no modelo por enquanto
        "Month (display only)",
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ],
        index=0
    )
    st.markdown(
        "<span style='font-size:12px;color:#6b7280'>Month is used for context only in this prototype.</span>",
        unsafe_allow_html=True
    )

st.markdown("### ✏️ Caption")

caption_text = st.text_area(
    "Caption text",
    value="pubg is coming",
    height=80,
    help="Write the caption as you would post on Instagram."
)

caption_len = len(caption_text.strip())

st.markdown(
    f"<span style='font-size:12px;color:#6b7280'>Caption length: {caption_len} characters.</span>",
    unsafe_allow_html=True
)

run_eval = st.button("✨ Evaluate caption & predict")


# -------------------------------------------------------------------
# RODAR PREDIÇÃO
# -------------------------------------------------------------------

if run_eval:
    with st.spinner("Thinking like a HypeQuest crystal ball... 🔮"):
        feat_dict = {
            "post_type": post_type,
            "hour": hour,
            "weekday": weekday,
            "topic": topic,
            "caption_len": caption_len if caption_len > 0 else 1,
            "hashtags": hashtags,
        }

        # 1) Taxa de engajamento
        if reg_model is not None and X_reg is not None:
            rate = predict_rate(reg_model, X_reg, feat_dict)
            rate_source = (
                "Predicted engagement rate using a Decision Tree model "
                "trained on this profile's historical posts."
            )
        else:
            rate = baseline_rate(feat_dict)
            rate_source = (
                "Estimated engagement rate using a heuristic baseline "
                "(no training data available)."
            )

        followers = FOLLOWERS_BY_PROFILE.get(profile, 100_000)
        interactions = int(followers * (rate / 100.0))

        # 2) Sentimento
        if clf_model is not None and X_clf is not None:
            model_sent = predict_sentiment(clf_model, X_clf, le_sent, feat_dict)
            # combinamos com keywords da legenda para deixar mais sensível
            kw_sent = keyword_based_sentiment(caption_text)
            if kw_sent == "negative":
                final_sentiment = "negative"
            elif kw_sent == "positive" and model_sent != "negative":
                final_sentiment = "positive"
            else:
                final_sentiment = model_sent
        else:
            final_sentiment = keyword_based_sentiment(caption_text)

        # -------------------------------------------------------------------
        # VISUAL – RESULTADOS
        # -------------------------------------------------------------------
        st.markdown("## 🔍 Prediction results")

        # Sentiment badge
        if final_sentiment == "positive":
            sent_color = "#16a34a"
            sent_bg = "#dcfce7"
            sent_icon = "😊"
        elif final_sentiment == "negative":
            sent_color = "#b91c1c"
            sent_bg = "#fee2e2"
            sent_icon = "⚠️"
        else:
            sent_color = "#6b7280"
            sent_bg = "#e5e7eb"
            sent_icon = "😐"

        st.markdown(
            f"""
            <div style="margin-bottom:8px;">
                <span style="
                    display:inline-flex;
                    align-items:center;
                    padding:6px 14px;
                    border-radius:999px;
                    background:{sent_bg};
                    color:{sent_color};
                    font-weight:600;
                    font-size:13px;
                ">
                    {sent_icon}&nbsp; {final_sentiment.upper()}
                </span>
                <span style="font-size:13px;color:#4b5563;margin-left:8px;">
                    Predicted sentiment for this post (based on caption & posting context).
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Engagement card (sem HTML quebrado!)
        st.markdown(
            f"""
            <div style="
                background:#e0edff;
                border-radius:12px;
                padding:16px 20px;
                margin-top:8px;
                margin-bottom:4px;
                border:1px solid #bfdbfe;
            ">
                <div style="font-size:13px;font-weight:600;color:#1d4ed8;margin-bottom:4px;">
                    Predicted engagement
                </div>
                <div style="font-size:18px;font-weight:600;color:#111827;margin-bottom:2px;">
                    Engagement rate: {rate:.2f}% 
                </div>
                <div style="font-size:13px;color:#4b5563;">
                    ≈ {interactions:,} interactions for ~{followers:,} followers
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(rate_source)

        # -------------------------------------------------------------------
        # DICAS (NÃO GERAMOS LEGENDA, APENAS TIPS)
        # -------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 💡 Suggestions to improve this post")

        tips = []

        # 1. Horário vs melhores horários
        top_hours = list(best_hours.head(3).index)
        if hour not in top_hours:
            tips.append(
                f"Your posting hour is **{hour}:00 UTC**. Historical data for this profile "
                f"shows stronger engagement around **{', '.join(str(h) + ':00' for h in top_hours)} UTC**."
            )

        # 2. Dia da semana
        top_days = list(best_weekdays.head(3).index)
        if weekday not in top_days:
            tips.append(
                f"You're posting on **{weekday}**. Engagement is usually higher on "
                f"**{', '.join(top_days)}** for this profile."
            )

        # 3. Tópico
        top_topic = best_topics.index[0]
        if topic != top_topic:
            tips.append(
                f"Posts about **{top_topic}** tend to perform better historically. "
                f"Consider tying this content more to **{top_topic}** if possible."
            )

        # 4. Tamanho da legenda
        if caption_len < 60:
            tips.append(
                "The caption is very short. Consider adding a bit more context, a hook, "
                "or a clear benefit for players."
            )
        elif caption_len > 260:
            tips.append(
                "The caption is quite long. Try tightening the message to keep it punchy "
                "and easy to read on mobile."
            )

        # 5. Hashtags
        if hashtags == 0:
            tips.append(
                "You are not using any hashtags. A few relevant hashtags can help discoverability."
            )
        elif hashtags > 10:
            tips.append(
                "You are using many hashtags. Consider focusing on the most relevant 5–10 "
                "to avoid looking spammy."
            )

        # 6. Sentimento
        if final_sentiment == "negative":
            tips.append(
                "The overall tone feels **negative**. If this is not intentional, consider "
                "softening the language or adding a positive call to action."
            )
        elif final_sentiment == "neutral":
            tips.append(
                "The overall tone feels **neutral**. You might add more hype words, emojis "
                "or a stronger emotion to excite players."
            )

        if not tips:
            tips.append("This caption already aligns well with historical high-performing posts.")

        for t in tips:
            st.markdown(f"- {t}")

        st.markdown(
            "<span style='font-size:11px;color:#9ca3af;'>Tips are based on patterns learned from the simulated API dataset.</span>",
            unsafe_allow_html=True,
        )

else:
    st.info("Set your post details and caption, then click **“✨ Evaluate caption & predict”** to see the results.")

