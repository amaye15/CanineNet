# Importing necessary libraries
import streamlit as st
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
import plotly.express as px

import os

import torch

# Paths for model and feature extractor
model_path = 'model'
feature_extractor_path = 'feature_extractor'

# Check if model and feature extractor exist
if not os.path.exists(model_path) or not os.path.exists(feature_extractor_path):
    # Download the feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('amaye15/ViT-Standford-Dogs')
    feature_extractor.save_pretrained(feature_extractor_path)

    # Download the model
    model = ViTForImageClassification.from_pretrained('amaye15/ViT-Standford-Dogs')

    # Convert model to FP16
    model = model.to(dtype=torch.float16)

    # Save the FP16 model
    model.save_pretrained(model_path)
else:
    # Load locally saved model and feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained(feature_extractor_path)
    model = ViTForImageClassification.from_pretrained(model_path)



# Function to classify the image and get sorted probabilities
def classify_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    #image = image.resize((224, 224))
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    # Sort probabilities and labels
    sorted_indices = torch.argsort(probs, descending=True)
    sorted_probs = probs[sorted_indices].detach().numpy()
    sorted_labels = [model.config.id2label[idx.item()] for idx in sorted_indices]

    return sorted_probs, sorted_labels

# Function to plot the bar chart using Plotly with classes on the y-axis
def plot_bar_chart(sorted_probs, sorted_labels):
    top_n = 9
    fig = px.bar(y=sorted_labels[:top_n], x=sorted_probs[:top_n], labels={'y':'Class', 'x':'Probability'}, title='Class Probabilities')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig)


# Creating the Streamlit interface
st.title('CanineNet')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    sorted_probs, sorted_labels = classify_image(image)
    plot_bar_chart(sorted_probs, sorted_labels)
