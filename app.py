import streamlit as st
import torch
import numpy as np
from PIL import Image
import time
import traceback

# Debugging output
st.write("Starting app initialization...")

# Force CPU
device = torch.device("cpu")
st.write(f"Using device: {device}")

# Simplified Generator (same as training)
class Generator(torch.nn.Module):
    def __init__(self, latent_dim=50, num_classes=10):
        super().__init__()
        self.label_emb = torch.nn.Embedding(num_classes, 20)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + 20, 128 * 7 * 7),
            torch.nn.ReLU(True)
        )
        self.model = torch.nn.Sequential(
            torch.nn.Unflatten(1, (128, 7, 7)),
            torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            torch.nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        x = self.fc(x)
        return self.model(x)

# Load model with error handling
@st.cache_resource
def load_generator():
    try:
        st.write("Loading model...")
        start_time = time.time()
        model = Generator().to(device)
        model.load_state_dict(torch.load("generator_mnist.pth", map_location=device))
        model.eval()
        st.write(f"Model loaded in {time.time() - start_time:.2f} seconds")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.error(traceback.format_exc())
        return None

# Generate images
def generate_images(model, digit, num_images=5, latent_dim=50):
    try:
        st.write("Generating images...")
        z = torch.randn(num_images, latent_dim).to(device)
        labels = torch.full((num_images,), digit, dtype=torch.long).to(device)
        with torch.no_grad():
            images = model(z, labels).cpu()
        return images
    except Exception as e:
        st.error(f"Image generation failed: {str(e)}")
        return None

# Streamlit app
st.title("Handwritten Digit Generator")
st.write("Select a digit (0-9) to generate 5 handwritten images using a Conditional GAN.")

# Load model
G = load_generator()
if G is None:
    st.stop()

# User input
digit = st.selectbox("Select a digit:", list(range(10)))

if st.button("Generate Images"):
    images = generate_images(G, digit)
    if images is not None:
        images = (images * 0.5 + 0.5) * 255  # Denormalize
        st.write(f"Generated images for digit {digit}:")
        cols = st.columns(5)
        for i, img in enumerate(images):
            img_pil = Image.fromarray(img.squeeze().numpy().astype(np.uint8))
            cols[i].image(img_pil, caption=f"Image {i+1}", use_column_width=True)
    else:
        st.error("Failed to generate images. Check logs for details.")

st.write("Note: This app runs on Streamlit Cloud with a CPU and a lightweight CGAN model.")
