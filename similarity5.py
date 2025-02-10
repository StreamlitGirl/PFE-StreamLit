import pandas as pd
import streamlit as st
import joblib
import nltk
from nltk.tokenize import word_tokenize
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from translate import Translator
from Monchat2 import model, vectorizer
from deep_translator import GoogleTranslator


# Load the CSV files
data = pd.read_csv("data/cleaned_dataset.csv")

# Extract questions and answers from the DataFrames
questions = data['question'].tolist()
answers = data['answer'].tolist()


# Initialize the TF-IDF Vectorizers
vectorizer = TfidfVectorizer()

# Fit and transform the questions to get the TF-IDF matrices
question_vectors = vectorizer.fit_transform(questions)


# Initialize the translators

translator_fr = Translator(to_lang="fr")
translator_en = Translator(to_lang="en")
translator_ar = Translator(to_lang="ar")

# Function to preprocess and match query
def preprocess_and_match_query(user_query, vectorizer, question_vectors, answers, threshold=0.3):
    # Tokenize and preprocess the user query
    nltk.download('punkt')
    tokens = word_tokenize(user_query.lower())
    preprocessed_query = " ".join(tokens)

    # Transform the user query into the TF-IDF vector
    user_query_vector = vectorizer.transform([preprocessed_query])
    
    # Compute the cosine similarity between the user query vector and the question vectors
    similarities = cosine_similarity(user_query_vector, question_vectors).flatten()
    
    # Get the index of the most similar question
    most_similar_idx = similarities.argmax()
    
    # Check if the similarity is above the threshold
    if similarities[most_similar_idx] > threshold:
        answer = answers[most_similar_idx]
    else:
        answer = None
    
    return answer

# Function to predict the answer
def predict_answer(user_input):
    # Detect the language of the user input
    lang = detect(user_input)
    
    translatuser = Translator (to_lang = lang )
    language = GoogleTranslator(source="en", target=lang).translate(" You are speaking in : ")
    st.write(language +"  " + lang)
  
    
    # Translate the user input to English, French, and Arabic
    user_input_en = GoogleTranslator(source=lang, target="en").translate(user_input)

    user_input_fr = GoogleTranslator(source=lang, target="fr").translate(user_input)
    #st.write("\n Vous avez dit :  " + user_input_fr)
    user_input_ar = GoogleTranslator(source=lang, target="ar").translate(user_input)
    
    # Tokenize the user input
    tokens_en = word_tokenize(user_input_en)
    tokens_fr = word_tokenize(user_input_fr)
    tokens_ar = word_tokenize(user_input_ar)
    input_text_en = ' '.join(tokens_en)
    input_text_fr = ' '.join(tokens_fr)
    input_text_ar = ' '.join(tokens_ar)
    
    # Search for the question in all three languages
    answer = preprocess_and_match_query(input_text_en, vectorizer, question_vectors, answers)
    answerlang = "en"
    if not answer:
        answer = preprocess_and_match_query(input_text_fr, vectorizer, question_vectors, answers)
        answerlang = "fr"
    if not answer:
        answer = preprocess_and_match_query(input_text_ar, vectorizer, question_vectors, answers)
        answerlang = "ar"
    if not answer :
        answer = "Sorry , i have no idea"
    answer = GoogleTranslator(source = answerlang, target = lang).translate(answer)
    
    # Translate the response back to the original language
  
    
    return answer

# Streamlit UI setup
st.title("Hello , what can i help you with ?")


# User input
user_input = st.text_input("Your question:")
if "chat_history"not in st.session_state:
    st.session_state.chat_history = []

if user_input:
    answer = predict_answer(user_input)
    #st.write(f"Answer: {answer}")
    st.session_state.chat_history.append([user_input, answer])
    for user, answer in st.session_state.chat_history:
        #st.write("You :")
        st.write(f"You '{user}'")
        st.write(f"Answer : '{answer}'")
