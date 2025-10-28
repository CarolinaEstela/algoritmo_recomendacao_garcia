import pandas as pd
import re
import unicodedata
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
import nltk
from nltk.corpus import stopwords

# Baixar stopwords do português
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))

# === Função de pré-processamento ===
def preprocess_title(title):
    if pd.isna(title):
        return ""

    # Remover colchetes, aspas e pontuação
    title = re.sub(r"[\[\]']", "", title)
    title = re.sub(r"[^\w\s]", "", title)
    title = title.lower()

    # Remover acentuação
    title = ''.join(
        char for char in unicodedata.normalize('NFD', title)
        if unicodedata.category(char) != 'Mn'
    )

    # Remover stopwords
    words = title.split()
    filtered_words = [word for word in words if word not in stop_words]

    return " ".join(filtered_words)

# === Função de recomendação ===
def recommend_titles(user_input, books_df):
    user_input = preprocess_title(user_input)
    user_vector = np.zeros(len(unique_words))

    for word in user_input.split():
        if word in word_index:
            user_vector[word_index[word]] = 1

    similarities = cosine_similarity([user_vector], word_bow.values)[0]
    angles = [round(math.degrees(math.acos(sim)), 1) if -1 <= sim <= 1 else None for sim in similarities]

    books_df['similarity'] = similarities
    books_df['angle_degrees'] = angles

    # Ordenar e exibir top 5 (ignorando resultados idênticos)
    recommended_books = (
        books_df.sort_values(by='similarity', ascending=False)
        .query("similarity < 0.999")  # evita mostrar o mesmo título
        .head(5)
    )
    return recommended_books[['title', 'similarity', 'angle_degrees']]

# === Carregar dataset ===
data_path = 'C:/Users/Cicero/OneDrive/fatec.1/3_semestre/Análise de Algoritmos - Códigos/books.csv'

# Tenta ler com vírgula, depois com ponto e vírgula se falhar
try:
    books_df = pd.read_csv(data_path, on_bad_lines='skip')
except Exception:
    books_df = pd.read_csv(data_path, sep=';', on_bad_lines='skip')

# === Limpeza dos títulos ===
books_df['cleaned_title'] = books_df['title'].apply(preprocess_title)

# === Criar vocabulário e matriz Bag of Words ===
unique_words = sorted(set(" ".join(books_df['cleaned_title']).split()))
word_index = {word: idx for idx, word in enumerate(unique_words)}

word_bow = pd.DataFrame(0, index=books_df.index, columns=unique_words)
for idx, title in enumerate(books_df['cleaned_title']):
    for word in title.split():
        if word in word_bow.columns:
            word_bow.loc[idx, word] = 1

# === Entrada do usuário ===
user_input = input("Digite o título do livro para receber recomendações: ")
recommendations = recommend_titles(user_input, books_df)

# === Exibir resultados ===
print("\nLivros recomendados:")
for idx, row in recommendations.iterrows():
    print(f"{row['title']} (similaridade: {row['similarity']:.3f})")
