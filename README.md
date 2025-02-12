# Análisis de Tópicos y Sentimientos de una base de datos de tweets

Este proyecto tiene como objetivo explorar el análisis de tópicos y sentimientos en una base de datos de tweets relacionados con la depresión. Se llevó a cabo un análisis exploratorio sobre un conjunto de datos descargado desde la plataforma *Kaggle* (Sentimental Analysis for Tweets), en el que se examinan los principales temas tratados en los tweets. Para este estudio, se utilizaron dos modelos de procesamiento de lenguaje natural: **BERTopic**, para identificar los tópicos predominantes, y **RoBERTa**, para clasificar los sentimientos asociados a los tweets.

La base de datos se encuentra [aqui](https://www.kaggle.com/datasets/gargmanas/sentimental-analysis-for-tweets)


## Características y Funcionalidades:

### Primer paso: preprocesamiento y limpieza de datos.
#### Limpieza y preprocesamiento de los tweets:
- Creación de funcion tex_processing para limpieza de datos . La funcion se encuentra [aqui](./src/utils/text_processing.py)
- Eliminación de ruido (lista de stopwords, URLs, menciones, hashtags, caracteres especiales, emoji, normalizacion de texto).[aqui](./src/01_clean_data.py)
#### Verificación y revisión de la limpieza de los datos:
- Uso de gráficos para visualizar la calidad de los datos después de la limpieza.[aqui](./src/01_clean_data_verification.py)
[aqui](./results/distribucion%20de%20longitud%20de%20tweets.png)
[aqui](./results/nube%20de%20palabras%20d%20elos%20principales%20topicos.png)
- Guardado de los datos limpios para análisis posteriores.[aqui](./data/CleanAndEmbeddins/cleaned_tweets.pkl)

### Segundo paso: generación de embeddings.
#### Creación de embeddings para tópicos:
- Uso de un modelo de *sentence-transformers* integrado dentro de **BERTopic**. Guardado de embeddings generados para análisis de tópicos.[aqui](./src/02_generateTopicEmbeddings.py)
#### Generación de embeddings sentimentales:
- Aplicación de **RoBERTa** para clasificar los sentimientos de los tweets. Guardado de los embeddings sentimentales generados.[aqui](./src/02_generate_SentimentalEmbeddings.py)

### Tercer paso: modelado y análisis de tópicos.
#### Entrenamiento del modelo de tópicos:
- Entrenamiento del modelo BERTopic con los embeddings generados.
[aqui](./src/03_train_topic_model.py)
#### Análisis de tópicos:
- Visualización de los tópicos usando **UMAP** para reducción de dimensionalidad. [aqui](./src/04_analyze_topics.py)
Grafico UMAP [aqui](./results/similitudes%20entre%20los%20topicos%20UMAP.png)
- Exploración de los tópicos identificados y análisis de su coherencia. [aqui](./src/04_analyze_topics2.py)
- Análisis de similitudes entre tweets utilizando la función de similitud coseno.  [aqui](./src/similarity.py)

### Cuarto paso: modelado y análisis de sentimientos.
#### Entrenamiento de los modelos de clasificación de sentimiento: 
- Uso de modelos de clasificación de sentimiento basados en RoBERTa para etiquetar los tweets.[aqui](./src/03_train_Sentimental_model.py)
#### Análisis de resultados de sentimiento:
- Evaluación de la distribución de sentimientos (positivos, negativos, neutros) y creación de DataFrame uniendo tópicos con sentimiento.
[aqui](./src/04_analyze_sentiment.py)
- Guardado de DataFrame [aqui](./data/sentiment_topic_analysis.csv)

### Visualización
#### Visualización gráfica del análisis de tópicos:
- Presentación interactiva de los tópicos utilizando gráficos y mapas de calor.
[aqui](./results/palabras%20por%20topicos.png)
[aqui](./results/nube%20de%20palabras%20d%20elos%20principales%20topicos.png)
#### Visualización gráfica del análisis de sentimientos:
- Representación gráfica de las distribuciones sentimentales a lo largo de los tweets.
[aqui](./results/distribucion%20de%20polaridad%20de%20los%209%20topicos%20imp.png)
[aqui](./results/sentimiento%20promedio%20por%209%20topicos%20mas%20importantes.png)


### App en streamlit 
#### link de la app [aqui](https://cnfef5nejtagivzyb6xmpi.streamlit.app/)
La aplicación interactiva fue desarrollada con **Streamlit** y se encuentra en [aqui](./app.py)
#### Cargar los resultados 
- Cargar la base de datos desde Google Drive para conexión online en la aplicación.
#### Visualización con Streamlit
- Generación de gráficos estáticos y dinámicos.




## Tecnologías utilizadas

- **Lenguaje**: Python
- **Modelos**: BERTopic y RoBERTa
- **Framework para aplicaciones web**: Streamlit
- **Análisis de Datos**: pandas, numpy, scikit-learn
- **Reducción de dimensionalidad**: UMAP
- **Clustering**: HDBSCAN
- **Procesamiento de lenguaje natural (NLP)**: nltk, transformers, PyTorch
- **Visualización de datos**: matplotlib, seaborn, plotly, altair
- **Serialización de Modelos**: joblib



## Contacto:
Camila Amado: [Correo Electrónico](mailto:amadocamilaines@gmail.com)









