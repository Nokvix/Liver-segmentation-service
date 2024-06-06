import streamlit as st
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import nibabel as nib
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


def preprocess_im(im):
    max_val = np.max(im)
    im[im < 0] = 0
    return im / max_val


def read_nii(im):
    return nib.load(im).get_fdata().astype('float32')


def apply_transformations(im):
    transformed = trans(image=im)
    return transformed["image"].unsqueeze(0).float()


trans = A.Compose([A.Resize(256, 256), ToTensorV2()])

# Определение модели
n_cls = 1  # Количество классов для сегментации: 0 - фон, 1 - печень
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Загрузка модели
def load_model():
    model = smp.DeepLabV3Plus(classes=n_cls, in_channels=1)
    model.load_state_dict(torch.load('final_model.pth', map_location=device))
    model.to(device)
    model.eval()
    print("Модель загружена и установлена в режим eval")
    return model


model = load_model()


# Отображение результатов
def display_results(image, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    image = image.squeeze().squeeze()
    y_pred = y_pred.squeeze()

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(y_pred, cmap='gray')
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')

    st.pyplot(fig)


st.title("Обработка снимков КТ печени")

uploaded_file = st.file_uploader("Загрузите КТ-скан", type=["nii"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_filename = temp_file.name

    image_volume = read_nii(temp_filename)
    image_volume = preprocess_im(image_volume)

    # Слайдер для выбора среза
    slice_idx = st.slider("Выберите срез", 0, image_volume.shape[2] - 1, image_volume.shape[2] // 2)

    image = image_volume[:, :, slice_idx]
    image = apply_transformations(image)

    with torch.no_grad():
        y_pred = model(image)
    # y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
    y_pred = torch.sigmoid(y_pred) > 0.5

    display_results(image, y_pred)

    os.remove(temp_filename)

