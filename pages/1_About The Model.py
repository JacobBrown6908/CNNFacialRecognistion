import streamlit as st

st.title("How the Model Works")


st.subheader("Classification of Images")

st.write("Within this code, it finds the data path to the files named for the celebrity they have images of. "
    "Using the names of the folders, it creates the target names. I designed this to be able to add more data and more profiles, "
    "making the model dynamic and scalable.")

code = """
path = "/content/drive/My Drive/Senior_Project/Celebrity Faces Dataset/"
target_names = os.listdir(path)
print(target_names)

num_classes = len(target_names)
"""

st.code(code, language="python")

st.markdown('<hr style="border:2px dashed #FFFFFF;">', unsafe_allow_html=True)


st.subheader("Data Augmentation")

st.write( "This code is used to load the images and apply different, randomly generated image augmentations. "
    "Some of the augmentations include: RandomBrightness, RandomZoom, and RandomCrop. "
    "These augmentations are applied to each picture at random to create diversity within the dataset, reducing overfitting and increasing the accuracy of the model. "
    "All of these ranges can be changed within a space from 0 to 1, representing a percentage of the range. "
    "For example, a brightness setting of 0.2 would indicate a random brightness adjustment of ±20%.")

code = """
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomCrop(height=200, width=200),
    tf.keras.layers.Resizing(height=224, width=224),
])
"""

st.code(code, language="python")

st.markdown('<hr style="border:2px dashed #FFFFFF;">', unsafe_allow_html=True)


st.subheader("Dynamic Classification")

st.write("This will split each folder of image classes so that each person has an 80% training data split and 20% validation data split. "
    "This ensures that the model does not overfit to the training data and can generalize well to new data. "
    "The goal is for the model to correctly identify a person in a new image that it has never seen before.")

code = """
train_dataset, val_dataset = load_train_val_data(
    path= path,
    img_size=(224, 224),
    batch_size=64,
    validation_split=0.2,
    seed=432
)
"""

st.code(code, language="python")

st.markdown('<hr style="border:2px dashed #FFFFFF;">', unsafe_allow_html=True)


import streamlit as st

st.subheader("The Model")

st.write("This is the model architecture, meaning this is how the images are read and interpreted "
"so the computer can understand and later make predictions on the data.")

st.write("   ")

st.write("##### First: Model Initialization")
st.write("We build the model by setting parameters such as the input shape and the number of classes, "
"which corresponds to the number of different people we are trying to identify.")

code_init = """
def create_simplified_face_recognition_model(
    input_shape=(224, 224, 3),
    num_classes=num_classes,
    base_filters=32,
    dropout_rate=0.5,
    decay_rate=0.01
):
"""
st.code(code_init, language="python")

st.markdown('<hr style="border:1px dashed #FFFFFF;">', unsafe_allow_html=True)

st.write("##### Second: Convolutional Layers")
st.write("We define the convolutional layers, which are responsible for reading the images. "
"We set the filter size to 3x3 pixels, meaning the image is processed in 3x3 blocks and then filtered. "
"This is how the computer 'sees' the image. "
"To help prevent overfitting, I have added dropout layers, which randomly drop out 30% of the data during training.")

code_conv = """
    model = Sequential([
        # First Convolutional Block
        Conv2D(base_filters, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(decay_rate)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Second Convolutional Block
        Conv2D(base_filters * 2, (3, 3), activation='relu', kernel_regularizer=l2(decay_rate)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
"""
st.code(code_conv, language="python")

st.markdown('<hr style="border:1px dashed #FFFFFF;">', unsafe_allow_html=True)

st.write("##### Third: Flattening the Data")
st.write("After the convolutional layers process the images, the data is flattened, "
"meaning the 2D image data is converted into a single long list of numbers. "
"This allows the model to analyze the information more effectively.")

code_flatten = """
        # Flatten and Dense Layers
        Dropout(0.5),
        Flatten(),
"""
st.code(code_flatten, language="python")

st.markdown('<hr style="border:1px dashed #FFFFFF;">', unsafe_allow_html=True)

st.write("##### Fourth: Dense Layers for Pattern Recognition")
st.write("The dense layers help the model learn patterns in the images and make final predictions. "
"The first dense layer reduces the data into a more manageable form, while the final dense layer determines which person is in the image. "
"The activation function used in the last layer is called 'softmax,' which helps the model decide the most likely match.")

code_dense = """
        # Simplified Dense Layer
        Dense(128, activation='relu', kernel_regularizer=l2(decay_rate)),  

        # Output Layer
        Dense(num_classes, activation='softmax', kernel_regularizer=l2(decay_rate))
    ])
"""
st.code(code_dense, language="python")

st.markdown('<hr style="border:1px dashed #FFFFFF;">', unsafe_allow_html=True)

st.write("##### Finally: Model Compilation")
st.write("Finally, the model is compiled with an optimizer and a loss function. "
"The optimizer helps the model adjust its learning based on errors, and the loss function measures how well the model is performing. "
"The model is now ready to be trained with the dataset!")

code_compile = """
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
"""
st.code(code_compile, language="python")

st.markdown('<hr style="border:2px dashed #FFFFFF;">', unsafe_allow_html=True)

