import os
import pandas as pd
import numpy as np
from langdetect import detect
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go

# Crear carpeta "reports" si no existe
reports_dir = os.path.join(os.path.dirname(__file__), '../reports')
os.makedirs(reports_dir, exist_ok=True)

# Cargar datos
df = pd.read_csv('../data/raw/amazon.csv')

# Eliminar duplicados
df = df.drop_duplicates()
print("Initial rows:", df.shape)

# Ver columnas
columns = df.columns
print(columns)

# Limpiar contenido
df["review_content"] = df["review_content"].str.strip(",🔸*-_\t\n+")

# Separar categorías
df["category"] = df["category"].apply(lambda x: x.split("|"))

# Limpiar ratings
for x in df.index:
    if df.loc[x, "rating"] == '|':
        df.drop(x, inplace=True)

# Formatear ratings
df["rating"] = df["rating"].apply(lambda x: float(x))

# Formatear rating_count
df["rating_count"] = df["rating_count"].fillna(0)
df["rating_count"] = df["rating_count"].astype(str).str.replace(",", "")
df["rating_count"] = df["rating_count"].apply(lambda x: float(x))

# Detectar idioma
df["language"] = df["review_content"].apply(lambda x: detect(x))

# Filtrar por idioma inglés
for x in df.index:
    if df.loc[x, "language"] != "en":
        df.drop(x, inplace=True)

df.reset_index(drop=True, inplace=True)
print("Cleaned rows:", df.shape)

# Análisis de sentimiento
def get_sentiment(review):
    if review >= 0.05:
        return 'Positivo'
    elif review < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

nltk.download("vader_lexicon")
sid = SentimentIntensityAnalyzer()

df[['negative', 'neutral', 'positive', 'compound']] = df['review_content'].apply(
    lambda x: pd.Series(sid.polarity_scores(x))
)

df["compound"] = df["compound"].apply(lambda x: get_sentiment(x))

# Expandir categorías
df_exploded = df.explode('category')
df_exploded[['negative', 'neutral', 'positive']] = df_exploded[['negative', 'neutral', 'positive']] * 100

# Top 20 productos
df_exploded = df_exploded.sort_values(by=['positive', 'neutral', 'negative'], ascending=False).head(20)
print(df_exploded[["category", "negative", "neutral", "positive"]].head(5))

# === DIAGRAMAS ===

# 1. Sentimiento por categoría
df_sentiment_by_category = df_exploded.groupby('category')[['negative', 'neutral', 'positive']].mean().reset_index()
fig = px.bar(
    df_sentiment_by_category,
    x='category',
    y=['negative', 'neutral', 'positive'],
    barmode='group',
    title='Distribución de Sentimientos por Categoría (%)'
)
fig.update_layout(xaxis_tickangle=-45)
fig.write_html(os.path.join(reports_dir, 'sentimientos_por_categoria.html'))

# 2. Distribución de sentimientos por rating
df_melted = df_exploded.melt(
    id_vars=['rating'], 
    value_vars=['negative', 'neutral', 'positive'], 
    var_name='sentiment', 
    value_name='score'
)
fig = px.box(
    df_melted,
    x='rating',
    y='score',
    color='sentiment',
    title='Distribución de Sentimientos por Rating',
    labels={'rating': 'Rating (1-5 estrellas)', 'score': 'Valor del Sentimiento'}
)
fig.write_html(os.path.join(reports_dir, 'sentimientos_por_rating.html'))

# 3. Densidad del sentimiento (histograma)
df_melt = df_exploded.melt(
    value_vars=['negative', 'positive'],
    var_name='Sentiment',
    value_name='Score'
)
fig = px.histogram(
    df_melt,
    x='Score',
    color='Sentiment',
    marginal='box',
    barmode='overlay',
    histnorm='density',
    opacity=0.6,
    color_discrete_map={
        'positive': '#00CC96',
        'negative': '#EF553B',
    },
    title='Densidad del Sentimiento (Positivo vs Negativo)'
)
fig.update_layout(
    title_font_size=24,
    title_x=0.5,
    xaxis_title='Puntaje de Sentimiento',
    yaxis_title='Densidad',
    bargap=0.1,
    template='plotly_white',
    legend_title='Tipo de Sentimiento',
    font=dict(size=14)
)
fig.write_html(os.path.join(reports_dir, 'densidad_sentimientos.html'))

# 4. Longitud del review vs sentimiento
df_exploded['review_length'] = df_exploded['review_content'].apply(lambda x: len(str(x)))
fig = px.scatter(
    df_exploded,
    x='review_length',
    y='compound',
    color='compound',
    color_continuous_scale='RdYlGn',
    title='Relación entre Longitud del Review y Sentimiento',
    labels={
        'review_length': 'Longitud del Review (número de caracteres)',
        'compound': 'Sentimiento (Compound Score)'
    }
)
fig.update_layout(
    title_font_size=24,
    title_x=0.5,
    template='plotly_white',
    font=dict(size=14)
)
fig.write_html(os.path.join(reports_dir, 'longitud_vs_sentimiento.html'))

# 5. Reviews positivos/negativos por categoría
fig = go.Figure()

fig.add_trace(go.Bar(
    x=df_exploded['category'],
    y=df_exploded['positive'],
    name='Positivo',
    marker_color='green'
))

fig.add_trace(go.Bar(
    x=df_exploded['category'],
    y=df_exploded['negative'],
    name='Negativo',
    marker_color='red'
))

fig.update_layout(
    barmode='stack',
    title='Reviews Positivos y Negativos por Categoría',
    xaxis_title='Categoría',
    yaxis_title='Proporción de Sentimiento',
    title_x=0.5,
    template='plotly_white',
    font=dict(size=14)
)
fig.write_html(os.path.join(reports_dir, 'reviews_categoria_stack.html'))

# 6. Proporción de sentimiento por producto
df_exploded['product_name'] = df_exploded['product_name'].apply(lambda x: x[:10] + '...' if len(x) > 10 else x)

fig = go.Figure()

fig.add_trace(go.Bar(
    x=df_exploded['product_name'],
    y=df_exploded['positive'],
    name='Positivo',
    marker_color='green'
))

fig.add_trace(go.Bar(
    x=df_exploded['product_name'],
    y=df_exploded['neutral'],
    name='Neutral',
    marker_color='gray'
))

fig.add_trace(go.Bar(
    x=df_exploded['product_name'],
    y=df_exploded['negative'],
    name='Negativo',
    marker_color='red'
))

fig.update_layout(
    barmode='stack',
    title='Proporción de Sentimiento por Producto',
    xaxis_title='Producto',
    yaxis_title='Proporción',
    title_x=0.5,
    template='plotly_white',
    font=dict(size=14),
    xaxis_tickangle=-45
)
fig.write_html(os.path.join(reports_dir, 'sentimientos_por_producto.html'))
