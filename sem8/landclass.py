import streamlit as st
import pandas as pd
import numpy as np
import rasterio
import earthpy.plot as ep
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Input, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from matplotlib.colors import from_levels_and_colors
from PIL import Image
import tempfile
import os

# Streamlit page configuration
st.set_page_config(page_title="Satellite Image Classification", layout="wide")
st.title("🌍 Satellite Image Classification using CNN")
st.image('./static/img/land.png', width=400)
# Constants

FEATURES = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'EVI', 'NBR', 'NDMI', 'NDWI', 'NDBI', 'NDBaI', 'elevation']
LABEL = ['classvalue']
N_CLASSES = 9
CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
PALETTE = ['#F08080', '#D2B48C', '#87CEFA', '#008080', '#90EE90', '#228B22', '#808000', '#FF8C00', '#006400']
CLASS_NAMES = ['Built-up', 'Bareland', 'Water', 'Wetland', 'Herbaceous', 'Dry shrub', 'Wet shrub', 'Mixed Crops', 'Plantation forest']
# Your data
CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
CLASS_NAMES = ['Built-up', 'Bareland', 'Water', 'Wetland', 'Herbaceous', 'Dry shrub', 'Wet shrub', 'Mixed Crops', 'Plantation forest']
CROPS = [
    ['Wheat', 'Rice', 'Maize'],               # Crops grown in Built-up areas (example)
    ['Cassava', 'Sweet potatoes'],            # Crops grown in Bareland
    ['Water lilies', 'Cattails'],             # Crops in Water regions
    ['Rice', 'Aquatic plants'],              # Crops in Wetland
    ['Grass', 'Clovers'],                    # Crops in Herbaceous
    ['Goji berries', 'Lavender'],            # Crops in Dry Shrubs
    ['Taro', 'Cranberries'],                 # Crops in Wet Shrubs
    ['Oil palm', 'Cocoa'],                   # Crops in Palm oil plantations
    ['Coffee', 'Cocoa', 'Rubber', 'Cinnamon'] # Crops in Plantation forests
]

# Create a DataFrame with specific crops
data = {
    'Class': CLASSES,
    'Class Name': CLASS_NAMES,
}

df = pd.DataFrame(data)

# Model definition
def build_model(input_shape):
    neuron = 64
    drop = 0.2
    kernel = 2
    pool = 2

    model = Sequential([
        Input(input_shape),
        Conv1D(neuron, kernel, activation='relu'),
        Conv1D(neuron, kernel, activation='relu'),
        MaxPooling1D(pool),
        Dropout(drop),
        Conv1D(neuron * 2, kernel, activation='relu'),
        Conv1D(neuron * 2, kernel, activation='relu'),
        MaxPooling1D(pool),
        Dropout(drop),
        GlobalMaxPooling1D(),
        Dense(neuron * 2, activation='relu'),
        Dropout(drop),
        Dense(neuron, activation='relu'),
        Dropout(drop),
        Dense(N_CLASSES + 1, activation='softmax')
    ])

    model.compile(optimizer='Adam', loss='CategoricalCrossentropy', metrics=['accuracy'])
    return model

# Utility functions
def reshape_input(array):
    return array.reshape(array.shape[0], array.shape[1], 1)

def visualize_image(image, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    ep.plot_rgb(image, ax=ax, stretch=True)
    ax.set_title(title)
    st.pyplot(fig)

def visualize_prediction(prediction, shape):
    cmap, norm = from_levels_and_colors(CLASSES, PALETTE, extend='max')
    fig, ax = plt.subplots(figsize=(8, 8))
    ep.plot_bands(prediction, cmap=cmap, norm=norm, ax=ax)
    ax.set_title("Predicted Land Cover")
    st.pyplot(fig)

def save_prediction(prediction, reference_image, output_path):
    with rasterio.open(reference_image) as src:
        new_dataset = rasterio.open(
            output_path, 'w', driver='GTiff',
            height=prediction.shape[0], width=prediction.shape[1],
            count=1, dtype=prediction.dtype,
            crs=src.crs, transform=src.transform
        )
        new_dataset.write(prediction, 1)
        new_dataset.close()

# Sidebar for file uploads
st.sidebar.header("Upload Files")
image_file = st.sidebar.file_uploader("Upload Satellite Image (TIFF)", type=["tif", "tiff"])
sample_file = st.sidebar.file_uploader("Upload Sample CSV (Optional)", type=["csv"])

if image_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_img:
        tmp_img.write(image_file.read())
        image_path = tmp_img.name

    image = rasterio.open(image_path)
    image_vis = np.stack([image.read(band) for band in [6, 5, 4]])

    st.subheader("Input Satellite Image")
    visualize_image(image_vis, "RGB Visualization")

    if sample_file:
        samples = pd.read_csv(sample_file).sample(frac=1)
        train = samples[samples['sample'] == 'train']
        test = samples[samples['sample'] == 'test']

        train_input = reshape_input(train[FEATURES].to_numpy())
        test_input = reshape_input(test[FEATURES].to_numpy())

        train_output = to_categorical(train[LABEL].to_numpy(), N_CLASSES + 1)
        test_output = to_categorical(test[LABEL].to_numpy(), N_CLASSES + 1)

        model = build_model((train_input.shape[1], train_input.shape[2]))

        st.subheader("Training Model")
        with st.spinner('Training in progress...'):
            history = model.fit(
                train_input, train_output,
                validation_data=(test_input, test_output),
                epochs=30, batch_size=512,
                callbacks=[EarlyStopping(monitor='loss', patience=5)],
                verbose=0
            )

        st.success("Training completed!")

        st.subheader("Training Metrics")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history.history['accuracy'], label='Train Accuracy')
        ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        st.pyplot(fig)

        st.subheader("Prediction on Test Set")
        test_pred = np.argmax(model.predict(test_input), axis=1)
        test_true = np.argmax(test_output, axis=1)

        cm = confusion_matrix(test_true, test_pred, normalize='true')
        cm_display = ConfusionMatrixDisplay(cm)
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_display.plot(ax=ax)
        st.pyplot(fig)

        st.text("Classification Report:")
        st.text(classification_report(test_true, test_pred, target_names=CLASS_NAMES))

    st.subheader("Prediction on Uploaded Image")
    image_input = np.stack([image.read(i + 1) for i in range(14)]).reshape(14, -1).T
    image_input = reshape_input(image_input)

    with st.spinner('Predicting image classes...'):
        prediction = model.predict(image_input, batch_size=4096 * 20)
        prediction = np.argmax(prediction, axis=1).reshape(image.height, image.width)

    visualize_prediction(prediction, (image.height, image.width))
    # Display the table in Streamlit
    st.write("### Class to Crops Grown Mapping Table")
    st.table(df)
    csv = df.to_csv(index=False)

# Add a download button
    st.download_button(
        label="Download Table as CSV",
        data=csv,
        file_name='class_to_crops.csv',
        mime='text/csv'
    )
    save_button = st.button("Save Predicted Image")
    if save_button:
        output_path = os.path.join(tempfile.gettempdir(), 'predicted_image.tif')
        save_prediction(prediction, image_path, output_path)
        with open(output_path, "rb") as file:
            st.download_button("Download Predicted Image", file, "predicted_image.tif")
else:
    st.info("Please upload a satellite image to start.")