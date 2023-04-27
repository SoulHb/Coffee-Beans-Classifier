import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import io

# Add some content to the app
st.title('Coffee Beans Classifier')
st.write('Upload an image of a coffee bean and click the "Classify" button to predict its type.')
uploaded_file = st.file_uploader("Choose file...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    bytes_image = BytesIO()
    image.save(bytes_image, format="JPEG")
    bytes_image = bytes_image.getvalue()
    st.image(image, caption='Uploaded Image', use_column_width=True)
    if st.button('Classify'):
        predicted = requests.post("http://127.0.0.1:5000/predict", files={'file': bytes_image})
        predicted = predicted.json()
        print(predicted['prediction'])
        if predicted['prediction'] == 0:
            st.write('Prediction:', 'Dark')
        elif predicted['prediction'] == 1:
            st.write('Prediction:', 'Green')
        elif predicted['prediction'] == 2:
            st.write('Prediction:', 'Light')
        elif predicted['prediction'] == 3:
            st.write('Prediction:', 'Medium')

