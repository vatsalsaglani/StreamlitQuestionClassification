import streamlit as st
import time
from streamlitinf import load_model, inf

st.set_page_config(
    page_title = 'Question Classification',
    page_icon = '❓',
)

st.title("Question Classification")
questionText = st.text_input("Input Question?")

classPlaceHolder = st.empty()
subclassPlaceHolder = st.empty()

if questionText:
    with st.spinner(text = 'Inference in Progress...'):
        model = load_model()
        op = inf(questionText, model)
    st.balloons()
    classPlaceHolder.header(f"Class: {op['class']}")
    subclassPlaceHolder.header(f"Subclass: {op['subclass']}")