# Importing necessary libraries
import streamlit as st
from PIL import Image
import requests

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/amaye15/ViT-Standford-Dogs"
headers = {"Authorization": "Bearer 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"}

# Function to query the Hugging Face API
def query(image):
    response = requests.post(API_URL, headers=headers, files={"file": image})
    return response.json()

# Function to classify the image using Hugging Face API
def classify_image(image):
    response = query(image)
    # Assuming the response contains probabilities and labels
    sorted_probs = [item['score'] for item in response]
    sorted_labels = [item['label'] for item in response]
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
    sorted_probs, sorted_labels = classify_image(uploaded_file)
    plot_bar_chart(sorted_probs, sorted_labels)
