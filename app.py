
# Importing necessary libraries
import streamlit as st
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import requests

# Load the model and the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Function to classify the image
def classify_image(image):
    # Convert the PIL image to RGB format if it's not
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize the image to the size expected by the model (224x224 for ViT)
    image = image.resize((224, 224))

    # Process the image with the feature extractor
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# Creating the Streamlit interface
st.title('Image Classification App')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = classify_image(image)
    st.write(f'Prediction: {label}')
