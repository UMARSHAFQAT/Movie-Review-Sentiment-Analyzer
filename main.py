import numpy as np
import tensorflow as tf 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load word index and reverse it
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load your trained model
model = load_model('simple_rnn_imdb.h5')

# Decode encoded reviews (optional)
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Preprocess the input review
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences(
        [encoded_review], maxlen=500, dtype='int32', padding='pre', truncating='pre'
    )
    return padded_review

# Predict sentiment
def predict_sentiment(review):
    processed = preprocess_text(review)
    prediction = model.predict(processed, verbose=0)
    sentiment = 'Positive 😊' if prediction[0][0] > 0.5 else 'Negative 😞'
    return sentiment, prediction[0][0]

# --- Streamlit App ---

st.set_page_config(page_title="IMDB Sentiment Classifier", page_icon="🎬")
st.title('🎬 Sentiment Analysis of IMDB Movie Reviews')
st.write('Enter a movie review and classify it as **Positive** or **Negative**.')

# User input
user_input = st.text_area('✍️ Movie Review:')

# Button to classify
if st.button('Classify'):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a valid movie review.")
    else:
        try:
            sentiment, score = predict_sentiment(user_input)
            st.success(f'**Sentiment:** {sentiment}')
            st.info(f'**Prediction Score:** {score:.4f}')
        except Exception as e:
            st.error("🚨 An error occurred during prediction.")
            st.exception(e)
