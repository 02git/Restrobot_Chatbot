import streamlit as st
import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

lemmatizer = WordNetLemmatizer()

# Load model and assets
model = tf.keras.models.load_model('models/chatbot_model.keras')
words = pickle.load(open('models/words.pkl', 'rb'))
classes = pickle.load(open('models/classes.pkl', 'rb'))

with open('data/Chatbot_Dataset.json') as file:
    intents = json.load(file)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def get_response(predictions):
    if not predictions:
        return "I don't understand."
    tag = classes[predictions[0][0]]
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

st.title("🤖 RestroBot Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    predictions = predict_class(user_input)
    response = get_response(predictions)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)