import streamlit as st
import torch
import numpy as np
from PIL import Image
from train_vae_mnist import VAE, generate_images

# App configuration
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="✍️",
    layout="centered"
)

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    model.load_state_dict(torch.load("vae_mnist_trained.pth", map_location=device))
    model.eval()
    return model

def main():
    st.title("✍️ Handwritten Digit Generator")
    st.markdown("""
    Generate MNIST-style handwritten digits using a Variational Autoencoder.
    Select a digit below and click **Generate** to create 5 unique samples!
    """)
    
    model = load_model()
    digit = st.selectbox("Select digit (0-9):", options=range(10))
    
    if st.button("Generate 5 Images"):
        with st.spinner("Generating digits..."):
            images = generate_images(model, digit, num_images=5)
            
        st.success(f"Generated 5 samples of digit {digit}:")
        cols = st.columns(5)
        for i, img in enumerate(images):
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            cols[i].image(img_pil, caption=f"Sample {i+1}", use_column_width=True)

if __name__ == "__main__":
    main()
