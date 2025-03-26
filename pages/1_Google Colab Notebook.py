import streamlit as st

st.title("My Google Colab Notebook")

st.write("As I've been diving into Convolutional Neural Networks, I've also been exploring powerful tools to enhance this Facial Recognition Software."
"While mastering Convolutional Neural Networks, I've been uncovering cutting-edge tools to supercharge this Facial Recognition Software, while working with limited data.")

st.write("I have built out my code using my google colab notebook, Feel free to check it out! and look at some of the code that I have written!")

st.link_button(f"My Google Colab Notebook","https://colab.research.google.com/drive/1G-7KXo_Uv_hdWE3ru8r8dvqVEJFCfctC")



blog_url = "https://colab.research.google.com/drive/1G-7KXo_Uv_hdWE3ru8r8dvqVEJFCfctC"
st.components.v1.iframe(blog_url, height=800, scrolling=True)