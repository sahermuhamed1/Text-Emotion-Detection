import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import TextProcessor, data_preprocessing
from sklearn.pipeline import Pipeline
import pickle

#make a project title
st.title("Emotion Detection App!🤖")

model = pickle.load(open("model.pkl", "rb"))
text_processing_odj = TextProcessor(lower=True, stem=False)
process_pip = Pipeline(
    [
        ("text_processing", text_processing_odj),
    ]
)
text = st.text_input("## Enter your text here")

if text is not None:

    data = {"text": [text]}

    df = pd.DataFrame(data)
    process_pip.fit(df["text"])
    df["text"] = process_pip.transform(df["text"])
    model.predict(df["text"])
    st.write("### Emotion Detected :")

   # make the emotion detected by default is blank until the user enter a text
    if text == "":
        st.markdown(f"<h1 style='text-align: center; color: white; border: 2px solid white; border-radius: 10px;'> </h1>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h1 style='text-align: center; color: white; border: 2px solid white; border-radius: 10px;'>{model.predict(df['text'])[0]}</h1>", unsafe_allow_html=True)
    
        
    

    
    