import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import jaccard_score
import nibabel as nib
import tempfile
import os
import matplotlib.pyplot as plt


def read_nii(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")


model = load_model()


def preprocess_image(image):
    image = Image.fromarray(image)
    image = image.resize((128, 128))
    image = np.array(image)

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    image = image / 255.0
    image = image[np.newaxis, ...]
    return image


def calculate_jaccard(y_true, y_pred):
    y_pred_binary = (y_pred > 0.5).astype(int)
    return jaccard_score(y_true.flatten(), y_pred_binary.flatten(), average='macro')


def display_results(image, y_pred, y_true=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(y_pred.squeeze(), cmap='gray')
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')

    if y_true is not None:
        axes[2].imshow(y_true.squeeze(), cmap='gray')
        axes[2].set_title("True Mask")
        axes[2].axis('off')

    st.pyplot(fig)


st.title("Обработка снимков КТ печени")

uploaded_file = st.file_uploader("Загрузите КТ-скан", type=["nii"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_filename = temp_file.name

    image = read_nii(temp_filename)

    first_slice = image[:, :, image.shape[2] // 2]

    preprocessed_image = preprocess_image(first_slice)

    y_pred = model.predict(preprocessed_image)

    y_true = np.random.randint(0, 2, size=y_pred.shape)

    jaccard = calculate_jaccard(y_true, y_pred)
    st.write(f"Jaccard Index: {jaccard:.4f}")

    display_results(first_slice, y_pred, y_true)

    os.remove(temp_filename)
