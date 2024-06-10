import io
import time
import base64
import requests

from PIL import Image

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(layout="wide")

MODELS = [
    "amaye15/ViT-Standford-Dogs",
    "amaye15/google-vit-base-patch16-224-batch32-lr0.005-standford-dogs",
    "amaye15/microsoft-swinv2-base-patch4-window16-256-batch32-lr0.005-standford-dogs",
    "amaye15/google-siglip-base-patch16-224-batch32-lr0.005-standford-dogs",
    "amaye15/google-vit-base-patch16-224-batch64-lr0.005-standford-dogs",
    "amaye15/google-siglip-base-patch16-224-batch64-lr5e-05-standford-dogs",
    "amaye15/microsoft-swinv2-base-patch4-window16-256-batch32-lr5e-05-standford-dogs",
    "amaye15/google-vit-base-patch16-224-batch32-lr5e-05-standford-dogs",
    "amaye15/google-vit-base-patch16-224-batch32-lr0.0005-standford-dogs",
    "amaye15/microsoft-resnet-50-batch32-lr0.0005-standford-dogs",
    "amaye15/microsoft-resnet-50-batch32-lr0.005-standford-dogs",
]

MODELS = list(set(MODELS))

API_URL = "https://api-inference.huggingface.co/models/{model}"


# Function to plot the bar chart using Plotly with classes on the y-axis
def plot_bar_chart(prediction, model_name):
    # top_n = 9
    df = pd.DataFrame(prediction)
    fig = px.bar(
        df,
        y="label",
        x="score",
        labels={"y": "Class", "x": "Probability"},
        title=model_name,
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig)


# Function to convert image to base64
def image_to_base64(image):
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def query(base64_image, model_name):
    headers = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json",
    }
    data = {"inputs": base64_image}
    while True:
        response = requests.post(
            API_URL.format(model=model_name), headers=headers, json=data
        )

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            # If the model is loading, wait and retry
            print("Model is loading, waiting for it to be ready...")
            time.sleep(5)  # Wait for 5 seconds before retrying
        else:
            # If there is another error, raise an exception or handle it appropriately
            print(f"Error: {response}")
            response.raise_for_status()


_, middle, _ = st.columns([0.2, 0.6, 0.2])

with middle:
    # Creating the Streamlit interface
    st.title("CanineNet Area: Model vs Model")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

model_one_select_column, model_two_select_column = st.columns(2)

with model_one_select_column:
    model_one = st.selectbox("Select Model 1:", MODELS)

with model_two_select_column:
    model_two = st.selectbox("Select Model 2:", MODELS[::-1])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    _, middle_image, _ = st.columns(3)

    (model_one_prediction_column, model_two_prediction_column) = st.columns(2)

    with middle_image:
        st.image(image, caption="Uploaded Image.", use_column_width=True)
    image_base64 = image_to_base64(image)

    with model_one_prediction_column:
        output_one = query(image_base64, model_one)
        plot_bar_chart(output_one, model_one)

    with model_two_prediction_column:
        output_two = query(image_base64, model_two)
        plot_bar_chart(output_two, model_two)

    # sorted_probs, sorted_labels = classify_image(image)
    # plot_bar_chart(sorted_probs, sorted_labels)

    # # Query the model
    # output = query(image_base64)

    # print(type(output))
    # plot_bar_chart(output)

    # Display the output
    # st.write(output)
