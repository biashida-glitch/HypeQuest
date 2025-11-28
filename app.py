# Instala o Streamlit para garantir que esteja disponível no ambiente da app
!pip install streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# ---------------- CONFIG GERAL ----------------
st.set_page_config(page_title="HypeQuest – Predição de Engajamento", layout="wide")

st.title("🔥 HypeQuest – Predição de Engajamento em Posts do Instagram")
st.caption("Protótipo para hackathon usando dados simulados + modelo de Machine Learning.")

st.markdown("---")

# ---------------- CARREGAR DADOS ----------------
st.sidebar.header("📁 Dados de entrada")

uploaded_file = st.sidebar.file_uploader(
    "Envie o dataset de posts (CSV)", type=["csv"],
    help="Se não enviar nada, o app usa o arquivo hypequest_dataset_ficticio.csv do Colab."
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    origem = "Arquivo enviado"
else:
    # ajuste o nome do arquivo se o seu CSV tiver outro nome/caminho
    df = pd.read_csv("hypequest_dataset_ficticio.csv")
    origem = "Dataset fictício padrão (hypequest_dataset_ficticio.csv)"

st.sidebar.success(f"Dados carregados de: {origem}")

st.subheader("👀 Visão geral dos dados")
st.write("Primeiras linhas do dataset:")
st.dataframe(df.head())

st.write("Estatísticas descritivas da variável alvo (engajamento_total):")
if "engajamento_total" in df.columns:
    st.write(df["engajamento_total"].describe())
else:
    st.error("A coluna 'engajamento_total' não foi encontrada no CSV. Verifique o arquivo.")
    st.stop()

st.markdown("---")

# ---------------- GRÁFICOS EXPLORATÓRIOS ----------------
st.subheader("📊 Análises rápidas de engajamento")

col_g1, col_g2 = st.columns(2)

# 1) Engajamento médio por tipo de post
with col_g1:
    if "tipo_post" in df.columns:
        eng_tipo = df.groupby("tipo_post")["engajamento_total"].mean().sort_values(ascending=False)
        fig1, ax1 = plt.subplots()
        ax1.bar(eng_tipo.index, eng_tipo.values)
        ax1.set_title("Engajamento médio por tipo de post")
        ax1.set_ylabel("Engajamento médio")
        ax1.set_xlabel("Tipo de post")
        plt.xticks(rotation=15)
        st.pyplot(fig1)
    else:
        st.warning("Coluna 'tipo_post' não encontrada para o gráfico.")

# 2) Engajamento médio por dia da semana
with col_g2:
    if "dia_semana" in df.columns:
        eng_dia = df.groupby("dia_semana")["engajamento_total"].mean().sort_values(ascending=False)
        fig2, ax2 = plt.subplots()
        ax2.bar(eng_dia.index, eng_dia.values)
        ax2.set_title("Engajamento médio por dia da semana")
        ax2.set_ylabel("Engajamento médio")
        ax2.set_xlabel("Dia da semana")
        plt.xticks(rotation=30)
        st.pyplot(fig2)
    else:
        st.warning("Coluna 'dia_semana' não encontrada para o gráfico.")

st.markdown("---")

# ---------------- PREPARAR MODELO ----------------
st.subheader("🤖 Treinando modelo de previsão (Árvore de Decisão)")

# features usadas no seu notebook
features = ['tipo_post','hora_post','dia_semana','tam_legenda',
            'hashtag_count','emoji_count','sentimento_legenda','mes','ano']
target = 'engajamento_total'

# verificar se todas existem
missing = [c for c in features if c not in df.columns]
if missing:
    st.error(f"As seguintes colunas esperadas não foram encontradas no CSV: {missing}")
    st.stop()

X = pd.get_dummies(df[features], drop_first=False)
y = df[target]

modelo = DecisionTreeRegressor(max_depth=6, random_state=42)
modelo.fit(X, y)

st.write("Modelo treinado com sucesso em cima do dataset atual.")
st.write(f"Número de observações: {X.shape[0]} | Número de features após one-hot: {X.shape[1]}")

# ---------------- IMPORTÂNCIA DAS VARIÁVEIS ----------------
st.subheader("🌟 Importância das variáveis para o modelo")

importances = modelo.feature_importances_
idx = np.argsort(importances)[-15:]  # top 15

fig_imp, ax_imp = plt.subplots(figsize=(8,4))
ax_imp.barh(X.columns[idx], importances[idx])
ax_imp.set_title("Top 15 features mais importantes")
plt.tight_layout()
st.pyplot(fig_imp)

st.markdown("---")

# ---------------- SIMULAÇÃO (WHAT-IF) ----------------
st.subheader("🔮 Simular um novo post e prever engajamento")

c1, c2, c3 = st.columns(3)

with c1:
    tipo = st.selectbox("Tipo do post", ["imagem", "video", "reels", "carrossel"])
    hora = st.slider("Hora do post", 0, 23, 12)
    dia_sem = st.selectbox(
        "Dia da semana",
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    )

with c2:
    tam_leg = st.slider("Tamanho da legenda (caracteres)", 0, 400, 120)
    hashtag_ct = st.slider("Quantidade de hashtags", 0, 15, 3)
    emoji_ct = st.slider("Quantidade de emojis", 0, 10, 2)

with c3:
    sent = st.selectbox("Sentimento da legenda", ["positivo","neutro","negativo"])
    mes = st.selectbox("Mês", list(range(1,13)))
    ano = st.selectbox("Ano", [2023, 2024, 2025])

novo = pd.DataFrame([{
    "tipo_post": tipo,
    "hora_post": hora,
    "dia_semana": dia_sem,
    "tam_legenda": tam_leg,
    "hashtag_count": hashtag_ct,
    "emoji_count": emoji_ct,
    "sentimento_legenda": sent,
    "mes": mes,
    "ano": ano
}])

# aplicar mesmo processamento que X
novo_proc = pd.get_dummies(novo)
novo_proc = novo_proc.reindex(columns=X.columns, fill_value=0)

pred = modelo.predict(novo_proc)[0]

st.success(f"⭐ Previsão de engajamento total para esse post: **{int(pred):,}**")

st.caption("Obs.: Modelo demonstrativo, treinado em dados simulados, apenas para fins de protótipo.")
