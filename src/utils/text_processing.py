## Funciones de limpieza de texto

import re
import unicodedata
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


import nltk
nltk.download("stopwords")
nltk.download("punkt")

# Lista de stopwords en español e inglés
STOPWORDS = set(stopwords.words("spanish") + stopwords.words("english"))

def remove_mentions(text):
    """Elimina menciones (@usuario) de un tweet"""
    return re.sub(r"@\w+", "", text)

def remove_hashtags(text):
    """Elimina hashtags de un tweet"""
    return re.sub(r"#\w+", "", text)

def remove_urls(text):
    """Elimina URLs de un tweet"""
    return re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

def remove_emojis(text):
    """Elimina emojis del texto"""
    return emoji.replace_emoji(text, replace="")

def remove_special_characters(text):
    """Elimina caracteres especiales, dejando solo letras y números"""
    return re.sub(r"[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ ]", "", text)

def normalize_text(text):
    """Convierte texto a minúsculas y elimina tildes"""
    text = text.lower()
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    return text

def remove_stopwords(text):
    """Elimina stopwords del texto"""
    words = word_tokenize(text)
    words_filtered = [word for word in words if word not in STOPWORDS]
    return " ".join(words_filtered)

def preprocess_text(text):
    """Aplica todas las funciones de limpieza a un tweet"""
    text = remove_mentions(text)
    text = remove_hashtags(text)
    text = remove_urls(text)
    text = remove_emojis(text)
    text = remove_special_characters(text)
    text = normalize_text(text)
    text = remove_stopwords(text)
    return text.strip()

