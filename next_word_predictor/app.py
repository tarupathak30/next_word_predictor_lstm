import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import pickle 
import numpy as np
import streamlit as st 


#Load model once at startup
@st.cache_resource
def load_model():
    import tensorflow as tf
    model = tf.keras.models.load_model("models/next_word_model.keras", compile=False)
    return model

model = load_model()

# tokenizer
with open("models\\token.pkl", "rb") as f: 
    tokenizer = pickle.load(f)
    


# streamlit ui 
st.title("Next Word Predictor")
st.write("Type a phrase or choose a starter sentence and see what the model predicts next:")


starter_sentences = [
    "I entered the room",
    "The detective stared",
    "A tapping sound came",
    "Gas lamps flickered",
    "Footsteps echoed",
    "The clock struck midnight",
    "A chill ran down",
    "Smoke rose from the fireplace",
    "I paused at the door", 
    "Something was hidden",
    "The window rattled",
    "The floorboards groaned",
]



# checkbox for custom input
use_custom = st.checkbox("Or type your own sentence")

# radio buttons only if custom input is not selected
input_text = ""
if not use_custom:
    input_text = st.radio("Pick a starter sentence:", starter_sentences)

# text input only if custom is selected
if use_custom:
    input_text = st.text_input("Enter your text here:")






# reverse mapping for decoding
reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}


if st.button("predict"): 
    if input_text.strip() == "": 
        st.write("Please enter some text.")
    else: 
        # preprocess input
        sequence = tokenizer.texts_to_sequences([input_text])
        x = pad_sequences(sequence, maxlen=10)
        
        # model prediction
        y = model.predict(x)
        predicted_id = y.argmax(axis=-1)[0]
        
        # decode prediction 
        output = reverse_word_index.get(predicted_id, "<UNK>")
        
        next_text = f"{input_text} {output}"
        
        # display results 
        st.success(f"Prediction : {output}")
        st.info(f"Next Text : {next_text}")


