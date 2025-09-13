import nltk
import pickle
import numpy as np
import tensorflow as tf

from nltk.corpus import gutenberg
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


def predict_next_word(model, tokenizer, text, max_sequence_len):
    seq = tokenizer.texts_to_sequences([text.lower()])[0]

    if len(seq) >= max_sequence_len:
        seq = seq[-(max_sequence_len - 1):]

    padded = pad_sequences([seq], maxlen=max_sequence_len - 1, padding="pre")
    predicted = model.predict(padded, verbose=0)
    predicted_word_index = int(np.argmax(predicted, axis=1)[0])

    return tokenizer.index_word.get(predicted_word_index)


# 1 download data before using it
nltk.download("gutenberg")

# 2 load text
text = gutenberg.raw("shakespeare-hamlet.txt").lower()

# 3 fit one tokenizer on full text
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# 4 build n-gram sequences correctly
input_sequences = []
for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram = token_list[: i + 1]
        input_sequences.append(n_gram)

# guard: if text lines were empty
if not input_sequences:
    raise RuntimeError("No training sequences were generated")

# 5 pad
max_sequence_len = max(len(s) for s in input_sequences)
input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
)

# 6 features and labels
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# 7 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# 8 model
model = Sequential(
    [
        Embedding(total_words, 100, input_length=max_sequence_len - 1),
        LSTM(150, return_sequences=True),
        Dropout(0.2),
        LSTM(100),
        Dense(total_words, activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# 9 train with early stopping
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1,
)

# 10 quick test prediction
seed_text = "long liue the king"
next_word = predict_next_word(model, tokenizer, seed_text, max_sequence_len)
print(f'Input: "{seed_text}"')
print(f"Next word prediction: {next_word}")

# 11 save artifacts
model.save("next_word_lstm.h5")
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# convert h5 to keras
m = load_model("next_word_lstm.h5", compile=False)
m.save("next_word_lstm.keras")
print("converted to next_word_lstm.keras")
