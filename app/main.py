import streamlit as st
import string
import nltk
from gensim.models import Word2Vec
import numpy as np
import requests

nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
stop_words = set(stop_words)

EMBEDDING_DIM = 100
WORD2VEC_PATH = "models/word2vec.model"
API_URL = "http://127.0.0.1:5001/invocations"
HEADERS = {"Content-Type": "application/json"}
word2vec_model = Word2Vec.load(WORD2VEC_PATH)


# remove punctuation marks and lowercase a word
def clean_word(word: str) -> str:

    word = word.lower()    #convierto todas las palabras a minusculas, para que sean iguales
    word = word.strip()    #elimino espacios para quedarme solo con las letras

    for letter in word:    #elimino simbolos de puntuación
        if letter in string.punctuation:
            word = word.replace(letter,'')
    
    return word

# remove stop words and puctuation from a whole text
def clean_text(text: str) -> list[str]:

    clean_text_list = []
    for word in text.split():
        cleaned_word = clean_word(word)
        if cleaned_word not in stop_words:
            clean_text_list.append(cleaned_word)

    return clean_text_list


def vectorize_text(text: list[str]) -> np.ndarray:      # vectoriza un texto promediando todos los vectores de palabras en el texto
    text_vector = np.zeros(EMBEDDING_DIM)  # vector vacio (longitud embedding_dim)
    for word in text:                      # por cada palabra, le decimos al modelo que nos de ese vector
        try:
            word_vector = word2vec_model.wv[word]
        except KeyError:
            st.warning(f"Word {word} not in vocabulary")
            continue
        text_vector += word_vector         # text_vector = text_vector + word_vector 

    return text_vector


def classify_embedding(embedding: np.ndarray) -> bool:
    """ Classify a text by using ML model
    Args -> embedding (np.ndarray): the vectorized text. Shape (100, )
    Returns -> bool: True if text is real, false otherwise. 
    """
    
    embedding = np.expand_dims(embedding, axis=0) #changes from (100,) to (1, 100)

    data = {
    "input": embedding.tolist(),
    }

    response = requests.post(API_URL, json=data, headers=HEADERS)
    response_json = response.json()
    is_real = bool(response_json["predictions"][0])
    return is_real


st.title("Fake News Detector")
st.subheader("Detecting fake news with machine learning")

text_to_predict = st.text_area("Enter the new to check if is fake or not.")
button = st.button("Analyze")


if button:

    st.info("Cleaning text...")
    text_to_predict_clean = clean_text(text_to_predict)
    st.info("Vectorizing text...")
    text_to_predict_vectorized = vectorize_text(text_to_predict_clean)
    st.info("Classifying text...")
    is_real = classify_embedding(text_to_predict_vectorized)

    if is_real:
        st.success("The new is real!!!")
    else:
        st.error("The new is fake!!!")
