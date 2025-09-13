import os
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# load artifacts saved with tf.keras
MODEL_PATH = "next_word_lstm.keras"        # use the new file
TOKENIZER_PATH = "tokenizer.pickle"

model = load_model(MODEL_PATH, compile=False)
with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)

max_sequence_len = model.input_shape[1] + 1

def predict_next_word(model, tokenizer, text, max_sequence_len):
    seq = tokenizer.texts_to_sequences([text.lower()])[0]
    if not seq:
        return None
    if len(seq) >= max_sequence_len:
        seq = seq[-(max_sequence_len - 1):]
    padded = pad_sequences([seq], maxlen=max_sequence_len - 1, padding="pre")
    predicted = model.predict(padded, verbose=0)
    idx = int(np.argmax(predicted, axis=1)[0])
    return tokenizer.index_word.get(idx)

st.title("📝 Next Word Prediction App")
st.write("Enter a phrase and let the model predict the next word!")

user_input = st.text_input("Enter text:", "Long live the King")

if st.button("Predict Next Word"):
    next_word = predict_next_word(model, tokenizer, user_input, max_sequence_len)
    if next_word:
        st.success(f"👉 Predicted Next Word: *{next_word}*")
    else:
        st.error("❌ Could not predict the next word. Try a different input.")
