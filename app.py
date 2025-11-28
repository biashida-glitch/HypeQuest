# ============================================
# 1. IMPORTS
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Para o dashboard simples (modo "linha de comando" no Colab)
from datetime import datetime

# ============================================
# 2. CARREGAR DADOS
# ============================================
# Suba seu CSV no Colab (ícone de pastinha à esquerda) e ajuste o nome aqui:
caminho_arquivo = "dados_instagram.csv"  # <-- troque para o nome do seu arquivo
df = pd.read_csv(caminho_arquivo)

print("Primeiras linhas do dataset:")
display(df.head())

print("Colunas disponíveis:")
print(df.columns.tolist())

# ============================================
# 3. PRÉ-PROCESSAMENTO BÁSICO
# ============================================
# Exemplo de colunas esperadas (ajuste os nomes conforme seu CSV):
# - 'data_postagem' (string de data/hora)
# - 'tipo_post'     (imagem, video, carrossel...)
# - 'horario'       (opcional, se já vier separado)
# - 'likes'
# - 'comentarios'
# - 'alcance' ou 'impressões' (alvo que queremos prever)

# ---- Converter data/hora em recursos numéricos ----
if 'data_postagem' in df.columns:
    df['data_postagem'] = pd.to_datetime(df['data_postagem'], errors='coerce')
    df['dia_semana'] = df['data_postagem'].dt.weekday  # 0=segunda, 6=domingo
    df['hora'] = df['data_postagem'].dt.hour
else:
    # Se já tem 'hora' em coluna separada e não tem data, não faz mal
    if 'hora' not in df.columns:
        raise ValueError("O dataset precisa ter pelo menos uma coluna de hora: 'data_postagem' ou 'hora'.")

# ---- Tratar tipo de post (categórico) ----
if 'tipo_post' in df.columns:
    df['tipo_post'] = df['tipo_post'].astype(str)
    dummies_tipo = pd.get_dummies(df['tipo_post'], prefix='tipo')
    df = pd.concat([df, dummies_tipo], axis=1)

# ============================================
# 4. DEFINIR ALVO E FEATURES
# ============================================
# Escolha o alvo a ser previsto:
# Exemplo: 'alcance' (troque para 'impressões', 'engajamento', etc, se preferir)
alvo = 'alcance'  # <-- AJUSTE para o nome correto no seu CSV

if alvo not in df.columns:
    raise ValueError(f"Não encontrei a coluna '{alvo}' no dataset. Ajuste o nome da variável 'alvo'.")

# Features numéricas básicas
features_numericas = []

if 'hora' in df.columns:
    features_numericas.append('hora')
if 'dia_semana' in df.columns:
    features_numericas.append('dia_semana')

# Exemplo de variáveis passadas de desempenho histórico (se existirem)
for col in ['likes', 'comentarios']:
    if col in df.columns:
        features_numericas.append(col)

# Features dummies de tipo de post
features_dummies = [c for c in df.columns if c.startswith('tipo_')]

features = features_numericas + features_dummies

print("Features usadas no modelo:")
print(features)

X = df[features].copy()
y = df[alvo].copy()

# Remove linhas com NaN
mask_validos = X.notnull().all(axis=1) & y.notnull()
X = X[mask_validos]
y = y[mask_validos]

# ============================================
# 5. TREINAR MODELO
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

modelo.fit(X_train, y_train)

# Avaliação simples
y_pred = modelo.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R² do modelo: {r2:.3f}")
print(f"MAE do modelo: {mae:.1f}")

# ============================================
# 6. "DASHBOARD" INTERATIVO VIA INPUT
#    (rodando no próprio Colab)
# ============================================

def prever_post(hora, dia_semana, tipo_post, likes_est, comentarios_est):
    """
    Função que gera uma previsão de alcance para
    um post com as características informadas.
    """
    # Criar linha "vazia"
    linha = {col: 0 for col in features}
    
    # Preencher numéricos
    if 'hora' in features:
        linha['hora'] = hora
    if 'dia_semana' in features:
        linha['dia_semana'] = dia_semana
    if 'likes' in features:
        linha['likes'] = likes_est
    if 'comentarios' in features:
        linha['comentarios'] = comentarios_est
    
    # Tratar tipo de post
    col_tipo = f"tipo_{tipo_post}"
    if col_tipo in features:
        linha[col_tipo] = 1
    else:
        # Se for um tipo novo, tudo zero nas dummies (modelo extrapola)
        pass
    
    df_linha = pd.DataFrame([linha])
    pred = modelo.predict(df_linha)[0]
    return pred

print("\n=== SIMULADOR DE POST PARA INSTAGRAM ===")
print("Preencha as infos abaixo para ver a previsão de alcance.\n")

# Você pode rodar esta célula várias vezes para testar diferentes cenários
try:
    # Inputs
    tipo_post_input = input("Tipo de post (ex: imagem, video, carrossel): ").strip().lower()
    hora_input = int(input("Hora de postagem (0-23): "))
    dia_semana_input = int(input("Dia da semana (0=Seg, 6=Dom): "))
    likes_input = float(input("Likes esperados ou médios em posts parecidos: "))
    comentarios_input = float(input("Comentários esperados ou médios em posts parecidos: "))

    previsao = prever_post(
        hora=hora_input,
        dia_semana=dia_semana_input,
        tipo_post=tipo_post_input,
        likes_est=likes_input,
        comentarios_est=comentarios_input
    )

    print(f"\n>>> Previsão de {alvo} para esse post: {previsao:,.0f}")
except Exception as e:
    print("Erro ao rodar simulador:", e)

