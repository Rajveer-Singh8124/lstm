import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle


#load model
model = load_model("lstm_model.h5")

# load tokerizer
with open("tokerizer.pkl","rb") as f:
    token = pickle.load(f)

def predict_next_word(model,tokenizer,text,max_seq_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_seq_len:
        token_list = token_list[-(max_seq_len-1):]
    token_list = pad_sequences([token_list],maxlen=max_seq_len-1)
    predicted = model.predict(token_list,verbose=0)
    predicted_word_index = np.argmax(predicted,axis=1)

    for word,index in tokenizer.word_index.items():
        if index== predicted_word_index:
            return word
    return None


st.title("Next Word Prediction")

input_text = st.text_input("Enter the sequence of words")
max_sequence_len=model.input_shape[1]+1
next_word = predict_next_word(model,token,input_text,max_seq_len=14)

if (st.button("Predict Next Word")):
    st.write(next_word)