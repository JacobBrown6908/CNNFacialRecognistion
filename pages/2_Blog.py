import streamlit as st

st.title("My Blog in Streamlit")

blog_url = "https://jacobbrown6908.github.io/CNN-Facial_Recognistion-JacobBrown/"
st.components.v1.iframe(blog_url, height=800, scrolling=True)

