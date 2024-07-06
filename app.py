import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('skin_cancer_detection_model.h5')

# Function to predict images
def predict(image):
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize image to match model's expected sizing
    img = np.asarray(img)  # Convert image to numpy array
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    return prediction

# Streamlit code
st.title('Skin Cancer Detection')
st.write('Upload a skin image for detection')

uploaded_file = st.file_uploader("Choose a skin image ...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make predictions
    prediction = predict(uploaded_file)
    if prediction > 0.5:
        st.write("Prediction: Melanoma")
    else:
        st.write("Prediction: Non-Melanoma")

