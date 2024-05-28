import streamlit as st
import numpy as np
import torch
import segmentation_models_pytorch as smp
import nibabel as nib
import tempfile
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2


# Определение модели
n_cls = 2  # Количество классов для сегментации: 0 - фон, 1 - печень
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели
def load_model():
    model = smp.DeepLabV3Plus(classes=n_cls, in_channels=1)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Чтение Nifti файла
def read_nii(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan


# Предобработка изображения
def preprocess_image(image):
    image = Image.fromarray(image)
    image = image.resize((256, 256))
    image = np.array(image)

    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0)
    return image


# Отображение результатов
def display_results(image, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(y_pred.squeeze(), cmap='gray')
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')

    st.pyplot(fig)


st.title("Обработка снимков КТ печени")

uploaded_file = st.file_uploader("Загрузите КТ-скан", type=["nii"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_filename = temp_file.name

    image = read_nii(temp_filename)

    # Добавляем ползунок для выбора слайса
    selected_slice = st.slider("Выберите слайс", 0, image.shape[2] - 1, image.shape[2] // 2)

    selected_slice_image = image[:, :, selected_slice]
    preprocessed_image = preprocess_image(selected_slice_image).to(device)

    with torch.no_grad():
        y_pred = model(preprocessed_image)
    y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()

    display_results(selected_slice_image, y_pred)

    os.remove(temp_filename)

# Добавляем форму для загрузки файла маски
st.subheader("Загрузите правильную маску")
ground_truth_mask = st.file_uploader("Загрузите файл маски", type=["nii"])

if ground_truth_mask is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as mask_temp_file:
        mask_temp_file.write(ground_truth_mask.getbuffer())
        mask_temp_filename = mask_temp_file.name

    gt_image = read_nii(mask_temp_filename)

    # Вычисляем индекс Жаккара
    jaccard_indices = []
    for i in range(image.shape[2]):
        selected_slice_image = image[:, :, i]
        preprocessed_image = preprocess_image(selected_slice_image).to(device)

        with torch.no_grad():
            y_pred = model(preprocessed_image)
        y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()

        gt_slice = gt_image[:, :, i]
        pred_mask = y_pred.squeeze()

        # Изменить размер маски истинных значений, чтобы соответствовать размерам предсказанной маски
        gt_slice_resized = cv2.resize(gt_slice, (256, 256), interpolation=cv2.INTER_NEAREST)
        intersection = np.logical_and(gt_slice_resized, pred_mask).sum()
        union = np.logical_or(gt_slice_resized, pred_mask).sum()

        # Обработка деления на ноль
        if union == 0:
            jaccard_index = 0.0  # Задаем нулевое значение индекса Жаккара в случае деления на ноль
        else:
            jaccard_index = intersection / union

        jaccard_indices.append(jaccard_index)

    # Вычисляем средний индекс Жаккара
    average_jaccard_index = np.nanmean(jaccard_indices)
    st.write(f"Индекс Жаккара: {round(average_jaccard_index, 4)}")

    try:
        mask_temp_file.close()
    except Exception as e:
        print("Ошибка при попытке закрыть файл:", e)

    # Удаление файла
    try:
        os.remove(mask_temp_filename)
    except Exception as e:
        print("Ошибка при удалении файла:", e)

