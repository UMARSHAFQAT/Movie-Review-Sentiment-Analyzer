import numpy as np
import tensorflow as tf 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.models import load_model

word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}
model=load_model('simple_rnn_imdb.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?')for i in encoded_review])

def preprocess_text(text):
    words=text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocess_input=preprocess_text(review)
    prediction=model.predict(preprocess_input)
    sentiment='positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]

import streamlit as st

st.title=('Sentiment Analysis of IMDB Movie Reviews')
st.write('Enter A Movie Review To Classify It As Positive or Negative')

user_input= st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)
    
    prediction=model.predict(preprocess_input)
    sentiment='positive' if prediction[0][0] > 0.5 else 'Negative'
    
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('please enter a movie review.')
    
    