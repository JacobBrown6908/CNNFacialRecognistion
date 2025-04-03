import streamlit as st

st.title("Challenges and Learnings")

st.subheader("challenges:")

st.write("1. **Data Quantity**: One of the main challenges I faced was the limited amount of data available—1700 images for training the model. "
         "This made it difficult to achieve high accuracy and generalization performance. "
         "To overcome this, I used data augmentation techniques to artificially increase the size of the "
         "dataset by applying random transformations to the images. This helped increase "
         "image diversity and improve the model's performance. ")

st.write("2. **Multiclass Classification**: Another challenge was the multiclass classification problem, where the model had "
         "to distinguish between 17 classes of images. "
         "This required careful tuning of the model architecture and hyperparameters to ensure that the model could learn to differentiate between the classes effectively. "
         "To address this, I used a convolutional neural network (CNN) architecture, which is well-suited for image classification tasks. ")

st.write("3. **Overfitting**: Overfitting was a significant challenge, especially given the limited amount of data. "
         "To mitigate this, I used techniques such as dropout and weight decay to regularize the model and prevent it from memorizing the training data. "
         "Additionally, I employed early stopping to halt training when the validation loss stopped improving, further reducing the risk of overfitting. ")

st.subheader("Learnings:")

st.write("1. **Data Augmentation**: I learned what parameters to change within a facial image identification model to increase the model's accuracy. "
         "Using data augmentation techniques such as random rotation, flipping, and brightness adjustments helped improve the model's performance. "
         "More importantly, I learned how much to change these parameters to increase accuracy without overfitting.")

st.write("2. **Model Architecture**: I experimented with different model architectures, some larger than others, which led to overfitting. "
         "I then adjusted to a smaller model architecture. The main takeaway was that each model is optimal for different datasets and sizes.")

st.write("3. **Hyperparameter Tuning / Adaptive Learning**: I learned the importance of hyperparameter tuning in improving model performance. "
         "I experimented with different learning rates, batch sizes, and dropout rates to find the optimal combination for my model. "
         "Additionally, I discovered adaptive learning structures. I built a model that can dynamically adjust the learning rate, dropout, and weight decay during training based on validation loss performance. ")
