# Importing necessary libraries
import streamlit as st
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
import plotly.express as px

# Load the model and the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('ViT-Standford-Dogs-Feature-Extractor')
model = ViTForImageClassification.from_pretrained('ViT-Standford-Dogs')



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
