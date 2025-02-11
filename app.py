import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import random
import numpy as np
import networkx as nx


# ========================
# Cargar los resultados 
# ========================

# URL de descarga directa
url = 'https://drive.google.com/uc?id=1ll_Ml0pS3RTNLP087dv5xEHen-LbcBIQ'
# Cargar el DataFrame desde Google Drive
df_filtered = pd.read_csv(url)

# ========================
# Visualización con Streamlit
# ========================

# Título principal
st.markdown("<h1 style='color: #36445D; font-size: 40px;text-align: center;'>Análisis de Tópicos y Sentimientos de una base de datos de Tweets</h1>", unsafe_allow_html=True)
# Título de encabezado
st.markdown("<h2 style='color: #666666 ; font-size: 20px;text-align: center;'>Aplicacion interactiva para explorar el análisis de Sentimientos de Tweets vinculados a la depresión</h2>", unsafe_allow_html=True)
# Título de subencabezado
st.markdown("<h3 style='color: #666666; font-size: 18px;text-align: center;'>Explora los resultados y relaciones seleccionando diferentes tópicos</h3>", unsafe_allow_html=True)

# ========================
# Paleta de colores 
pastel_colors = ["#03627E", "#DC8079", "#36445D", "#03627E", "#096266" , '#36445D', '#EDDEE0', '#F7F7FF', '#E3B3A9', '#739899']
# Función para asignar colores aleatorios de la paleta pastel_colors
def random_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return random.choice(pastel_colors)
#==========================

# ========================
# Graficos Estáticos
# ========================

st.markdown("<br><br>", unsafe_allow_html=True)  # Añade espacio en blanco

# Crear un gráfico boxplot de sentimientos por tópico
plt.figure(figsize=(10, 6))
sns.boxplot(x='topic', y='sentiment_score', data=df_filtered, palette=pastel_colors)
plt.xlabel('Tópico')
plt.ylabel('Sentimiento')
st.markdown("<h4 style='color: #262626; font-size: 20px;text-align: center;'>Distribución de Sentimientos por Tópicos</h4>", unsafe_allow_html=True)
st.pyplot(plt)


# ========================
# Gráficos interactivos
# ========================

# Sidebar para seleccionar tópicos
topic_selected = st.selectbox('Selecciona un Tópico:', df_filtered['topic'].unique())
filtered_data = df_filtered[df_filtered['topic'] == topic_selected]

st.markdown("<br><br>", unsafe_allow_html=True)  # Añade espacio en blanco

# Boxplot de sentimientos por tópico
st.markdown("<h4 style='color: #262626; font-size: 20px;text-align: center;'>Boxplot de la distribución del análisis de sentimientos</h4>", unsafe_allow_html=True)
plt.figure(figsize=(10, 6))
sns.boxplot(x='topic', y='sentiment_score', data=filtered_data, palette=pastel_colors)
plt.xlabel('Tópico')
plt.ylabel('Sentimiento')
plt.title(f'Distribución de Sentimientos para el Tópico {topic_selected}')
st.pyplot(plt)

st.markdown("<br><br>", unsafe_allow_html=True)  # Añade espacio en blanco

# Histograma de Sentimientos por Tópico
st.markdown("<h4 style='color: #262626; font-size: 20px;text-align: center;'>Distribución de los sentimientos según tópico</h4>", unsafe_allow_html=True)
fig = px.histogram(filtered_data, x="topic", color="sentiment_label", barmode="group",
                   color_discrete_sequence=pastel_colors)
st.plotly_chart(fig)

st.markdown("<br><br>", unsafe_allow_html=True)  # Añade espacio en blanco

# Heatmap de Sentimiento Promedio por Tópico
st.markdown("<h4 style='color: #262626; font-size: 20px;text-align: center;'>Heatmap de sentimiento promedio por tópico</h4>", unsafe_allow_html=True)
topic_sentiment_avg = filtered_data.groupby("topic")["sentiment_score"].mean().reset_index()
fig = px.imshow(topic_sentiment_avg.pivot(index="topic", columns=[], values="sentiment_score")
                .to_numpy().reshape(-1, 1),
                labels=dict(color="Sentimiento Promedio"),
                color_continuous_scale=px.colors.sequential.Peach)
st.plotly_chart(fig)

st.markdown("<br><br>", unsafe_allow_html=True)  # Añade espacio en blanco

# Gráfico de Dispersión: Relación entre Polaridad y Tópico
st.markdown("<h4 style='color: #262626; font-size: 20px;text-align: center;'>Análisis de polaridad por tópico: gráfico de dispersión</h4>", unsafe_allow_html=True)
fig = px.strip(filtered_data, x="topic", y="sentiment_score", color="sentiment_label", 
               color_discrete_sequence=pastel_colors)
st.plotly_chart(fig)

# Nota de pie
st.markdown("<p style='color: #666666; font-size: 16px;>El gráfico de dispersión ilustra cómo se distribuyen las polaridades de los sentimientos en función de los tópicos analizados. Cada punto representa una observación dentro de un tópico, donde el eje X representa el topico seleccionado y el eje Y la intensidad de la polaridad. La dispersión de los puntos sugiere el grado de relación emocional de cada tema.</p>", unsafe_allow_html=True)
st.markdown("""
<style>
    .explicacion {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #ccc;
    }
    .negativo { color: gray; font-weight: bold; }
    .neutro { color: gray; font-weight: bold; }
    .positivo { color: gray; font-weight: bold; }
</style>

<div class='explicacion'>
    <p><b>Interpretación de la Polaridad:</b></p>
    <p><span class='negativo'>0 - Negativo:</span> Comentarios críticos o desfavorables.</p>
    <p><span class='neutro'>1 - Neutro:</span> Opiniones objetivas o sin emoción clara.</p>
    <p><span class='positivo'>2 - Positivo:</span> Comentarios de satisfacción o elogios.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)  # Añade espacio en blanco

# Nube de palabras
words = " ".join(filtered_data[filtered_data['topic'] == topic_selected]['tweet'])
wordcloud = WordCloud(width=800, height=400, background_color='white', color_func=random_color_func).generate(words)
st.markdown(f"<h4 style='font-size: 20px;text-align: center;'>Nube de Palabras - Tópico {topic_selected}</h4>", unsafe_allow_html=True)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

st.markdown("<br><br>", unsafe_allow_html=True)  # Añade espacio en blanco

# Frecuencia de Tweets por Tópico NO VA
#st.markdown(f"<h4 style='font-size: 20px;text-align: center;'>Frecuencia de tweets de tópico {topic_selected}</h4>", unsafe_allow_html=True)
plt.figure(figsize=(10, 5))
sns.countplot(x='topic', data=df_filtered, palette=pastel_colors)
plt.xlabel('Tópico')
plt.ylabel('Número de Tweets')
#st.pyplot(plt)



# Matrix de correlación, para ver la relación entre tópicos y sentimientos

st.markdown(f"<h4 style='font-size: 20px;text-align: center;'>Matrix de correlación: para ver la relación entre tópicos y sentimientos</h4>", unsafe_allow_html=True)
df_corr = df_filtered.pivot_table(index='topic', values='sentiment_score', aggfunc=np.mean)
plt.figure(figsize=(10, 6))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.xlabel('Tópico')
plt.ylabel('Sentimiento Promedio')
st.pyplot(plt)

# Nota de pie
st.markdown("<p style='color: #666666; font-size: 16px;text-align: center;'>Un heatmap de correlación entre sentimientos y tópicos muestra cómo varían los sentimientos promedio en cada tópico.</p>", unsafe_allow_html=True)
# Agregar referencia de lo que significa cada valor en el heatmap
st.markdown("""
<p style='text-align: center;'>
    Rojo intenso: Relación fuerte y positiva. | 
    Azul intenso: Relación inversa.
</p>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)  # Añade espacio en blanco







##############################################################  VER DESPUES #######################################

###########SIMILITUD COSENO EN SIMILITARITY.PY######################



# Red de Conexiones entre Tópicos
#st.markdown("<h4 style='color: #262626; font-size: 20px;text-align: center;'>Red de Conexiones entre Tópicos</h4>", unsafe_allow_html=True)

# Crear un grafo
#G = nx.Graph()

# Añadir nodos (tópicos)
#for topic in df_filtered['topic'].unique():
 #   G.add_node(topic)


# Calcular la similitud coseno entre los embeddings de los tópicos
# Aquí se usa una matriz de similitud ficticia para ilustrar
# Reemplaza esto con tus embeddings reales
#embeddings = np.random.rand(len(df_filtered['topic'].unique()), 10)  # Ejemplo de embeddings aleatorios
#similarity_matrix = cosine_similarity(embeddings)

# Añadir aristas (conexiones) basadas en la similitud coseno
#for i, topic1 in enumerate(df_filtered['topic'].unique()):
    #for j, topic2 in enumerate(df_filtered['topic'].unique()):
        #if i < j:
         #   similarity = similarity_matrix[i, j]
          #  if similarity > 0.5:  # Umbral para añadir una conexión
           #     G.add_edge(topic1, topic2, weight=similarity)

# Dibujar el grafo
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 10))
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, edge_color='gray', linewidths=1, font_size=15)
plt.title('Red de Conexiones entre Tópicos')
#st.pyplot(plt)