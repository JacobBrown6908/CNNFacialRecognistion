import streamlit as st

# Show title and description.
st.title("Convolutional Neural Network Facial Recognition Application")

st.write("This project involves the development of an advanced facial recognition system leveraging Convolutional Neural Networks (CNNs)"
" to accurately identify individuals across a diverse range of images. The model is trained on a comprehensive dataset containing"
" multiple images of the same person captured from different angles, under varying lighting conditions, and displaying a range of emotions"
" and facial expressions over time. By extracting and analyzing intricate facial features, the system enhances recognition accuracy"
" even in challenging real-world scenarios.")
st.write(" ")
st.write(" ")
st.write("This technology has significant applications in security, enabling rapid identification of individuals in large-scale"
" image databases or crowded environments. Additionally, it can be integrated with smart device cameras to reinforce authentication protocols,"
" enhancing security and privacy when accessing sensitive information. The system’s adaptability makes it a powerful tool for biometric "
"verification and identity protection in both public and private sectors.") 


st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")

st.write("Example Output:")

st.image("Example.png", caption="Example Output of the Facial Recognition Model", use_container_width=True)
