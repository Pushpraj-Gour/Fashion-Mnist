import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/Trained_fashion_mnist_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Class labels for Fashion MNIST dataset
class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"]


# Function to preprocess the input image

def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28,28))  # Because we have trained the model with this dimension of image
    img = img.convert("L") # Converting it into grayscale(L--> Luminous)
    img = np.array(img)/255
    img = img.reshape((1,28,28,1))
    return img



#Streamlit App
st.title("Fashion Item Classifier")

uploaded_image = st.file_uploader("Upload an image....",type=["jpg","jpeg","png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    col1 , col2 = st.columns(2)  #Creating 2 Columns of the page for different-different purpose
    
    
    with col1:  # To represent the image
        resized_img = image.resize((100,100))
        st.image(resized_img)
        
    with col2:
        if st.button("Classify"):
            # Preprocess the uploaded image
            
            img_array = preprocess_image(uploaded_image)
            
            # Make a prediction using the pre-trained model
            
            result = model.predict(img_array)
            
            # st.write(str(result))
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]
            
            st.success(f"Prediction: {prediction}")
            
    