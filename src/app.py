import streamlit as st
from PIL import Image
import os
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

corn = ["Corn__healthy", "Corn__blight",
        "Corn__common_rust", "Corn__gray_leaf_spot"]
pepper_bell = ["Pepper__bell___healthy", "Pepper__bell___Bacterial_spot"]
potato = ["Potato___healthy", "Potato___Early_blight", "Potato___Late_blight"]
tomato = ["Tomato_healthy", "Tomato__Target_Spot", "Tomato__Tomato_mosaic_virus",
          "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato_Bacterial_spot"]


def predictDisease(imgPath, model_name):
    corn_df = pd.DataFrame({
        'imgPath': imgPath,
    }, index=[0])

    IMG_SIZE = (150, 150)
    imgs = []
    for imgPath in tqdm(corn_df['imgPath'], total=len(corn_df)):
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        imgs.append(img)

    images = np.array(imgs)
    images = images / 255.0

    model = load_model(f'src/{model_name}.keras')

    result = model.predict(images)
    prediction = [np.argmax(x) for x in result]

    return prediction[0], (result[0][prediction]*100).round(decimals=2)


def display_percentage(label, percentage):

    percentage = max(0, min(percentage, 100))

    st.write(f"**Result**")

    st.progress(int(percentage))

    st.metric(label="Accuracy", value=f"{label} = {percentage:.2f}%")


st.title("Crop Disease Prediction")
st.write("Select a crop type, upload an image of the crop leaf.")

crop_list = ["Corn", "Pepper Bell", "Potato", "Tomato"]
crop_choice = st.selectbox("Choose the crop:", crop_list)

uploaded_file = st.file_uploader(
    "Upload an image of the crop leaf...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image, caption=f"Uploaded {crop_choice} leaf image", use_column_width=True)

    save_path = f"./src/crop_images/{uploaded_file.name}"

    image.save(save_path)

    prediction, [percentage] = predictDisease(
        imgPath=save_path, model_name=crop_choice)

    if crop_choice == "Corn":
        display_percentage(corn[prediction], percentage)
    elif crop_choice == "Pepper Bell":
        display_percentage(pepper_bell[prediction], percentage)
    elif crop_choice == "Tomato":
        display_percentage(tomato[prediction], percentage)
    elif crop_choice == "Potato":
        display_percentage(potato[prediction], percentage)
    # st.write(f"**Result: No Disease Predicted!!!**")
