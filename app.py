import streamlit as st
import torch
import numpy as np
from PIL import Image
from train_cgan_mnist import Generator, load_generator, generate_images, device

# Streamlit app
st.title("Handwritten Digit Generator")
st.write("Select a digit (0-9) to generate 5 handwritten images using a Conditional GAN.")

# Load model
@st.cache_resource
def get_generator():
    return load_generator()

G = get_generator()

# User input
digit = st.selectbox("Select a digit:", list(range(10)))

if st.button("Generate Images"):
    images = generate_images(digit, num_images=5)
    images = (images * 0.5 + 0.5) * 255  # Denormalize from [-1, 1] to [0, 255]
    st.write(f"Generated images for digit {digit}:")
    cols = st.columns(5)
    for i, img in enumerate(images):
        img_pil = Image.fromarray(img.squeeze().numpy().astype(np.uint8))
        cols[i].image(img_pil, caption=f"Image {i+1}", use_column_width=True)

st.write("Note: This app uses a Conditional GAN trained on the MNIST dataset in Google Colab with a T4 GPU.")
