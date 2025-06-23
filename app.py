import streamlit as st
import torch
import numpy as np
from PIL import Image
from train_vae_mnist import VAE, generate_images

# Streamlit app
st.title("Handwritten Digit Generator")
st.write("Select a digit (0-9) to generate 5 handwritten images using a Variational Autoencoder.")

# Load model
@st.cache_resource
def load_model():
    model = VAE().to("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("vae_mnist.pth", map_location=model.device))
    model.eval()
    return model

model = load_model()

# User input
digit = st.selectbox("Select a digit:", list(range(10)))

if st.button("Generate Images"):
    images = generate_images(model, digit, num_images=5)
    st.write(f"Generated images for digit {digit}:")
    cols = st.columns(5)
    for i, img in enumerate(images):
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        cols[i].image(img_pil, caption=f"Image {i+1}", use_column_width=True)

st.write("Note: This app uses a Variational Autoencoder trained on the MNIST dataset to generate images.")
