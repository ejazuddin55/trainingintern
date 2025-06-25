import streamlit as st
import torch
import numpy as np
from PIL import Image
import sys

# Error handling decorator
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.stop()
    return wrapper

@handle_errors
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Sequential()  # Replace with your VAE model
    try:
        model.load_state_dict(torch.load("vae_mnist.pth", map_location=device))
    except:
        st.warning("Couldn't load model, using random generation")
    return model

@handle_errors
def main():
    st.set_page_config(page_title="Digit Generator", layout="centered")
    st.title("✍️ Handwritten Digit Generator")
    
    model = load_model()
    digit = st.selectbox("Select digit (0-9):", options=range(10))
    
    if st.button("Generate Images"):
        with st.spinner("Creating digits..."):
            # Replace with your actual generation code
            images = [np.random.rand(28, 28) for _ in range(5)]
            
        cols = st.columns(5)
        for i, img in enumerate(images):
            cols[i].image(img, caption=f"Digit {digit}-{i+1}")

if __name__ == "__main__":
    main()
