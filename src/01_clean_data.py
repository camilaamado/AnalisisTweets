############################### Limpieza y preprocesamiento de tweets ###############################

import pandas as pd
import joblib
from utils.text_processing import preprocess_text

# ========================
#  Carga de datos
# ========================

def load_data(file_path):
    """Carga un archivo CSV y selecciona la columna de texto."""
    data = pd.read_csv(file_path, encoding='latin-1', header=None, usecols=[1])
    data.columns = ['text']
    return data

archivo_csv = "/home/mario/Documents/camiApp/data/raw/sentiment_tweets3.csv"
data = load_data(archivo_csv)

# Aplicar limpieza llamando a la función preprocess_text
data["cleaned_tweets"] = data["text"].apply(preprocess_text)

# ========================
# Guardado de datos
# ========================

def save_data(data, file_path):
    """Guarda un DataFrame en un archivo .pkl."""
    joblib.dump(data, file_path)

output_path = "/home/mario/Documents/camiApp/data/CleanAndEmbeddins/cleaned_tweets.pkl"
save_data(data, output_path)

print("Tweets limpiados y guardados correctamente.")
