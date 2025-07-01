import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import time

# 1. VAE Model
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=200, latent_dim=10, num_classes=10):
        super(ConditionalVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim//2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.num_classes = num_classes

    def encode(self, x, c):
        c_onehot = torch.nn.functional.one_hot(c, self.num_classes).float()
        x = torch.cat([x.view(-1, 784), c_onehot], dim=1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c):
        c_onehot = torch.nn.functional.one_hot(c, self.num_classes).float()
        z = torch.cat([z, c_onehot], dim=1)
        return self.decoder(z)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

# 2. Model Loading
@st.cache_resource
def load_model():
    st.write("Loading model...")
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Device: {device}")
    model = ConditionalVAE().to(device)
    
    if os.path.exists("vae_mnist.pth"):
        model.load_state_dict(torch.load("vae_mnist.pth", map_location=device))
        st.write(f"Model loaded in {time.time() - start_time:.2f} seconds")
    else:
        st.error("Pre-trained model 'vae_mnist.pth' not found. Please upload it.")
        st.stop()
    
    return model

# 3. Generate Images
def generate_digit_images(model, digit, num_images=3):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        z = torch.randn(num_images, 10).to(device)  # Match latent_dim=10
        digits = torch.full((num_images,), digit, dtype=torch.long).to(device)
        samples = model.decode(z, digits).cpu().numpy()
    return samples.reshape(-1, 28, 28)

# 4. Streamlit UI
def main():
    st.set_page_config(page_title="Digit-Specific VAE Generator", page_icon="🔢", layout="centered")
    st.title("🔢 Digit-Specific Generator")
    st.markdown("Generate 3 images of the same handwritten digit (0-9)")
    
    model = load_model()
    digit = st.selectbox("Select digit to generate:", options=range(10), index=5)
    
    if st.button("Generate 3 Images", type="primary"):
        with st.spinner(f"Generating 3 images of digit {digit}..."):
            start_time = time.time()
            images = generate_digit_images(model, digit)
            st.write(f"Image generation time: {time.time() - start_time:.2f} seconds")
        
        st.success(f"Generated 3 images of digit {digit}:")
        cols = st.columns(3)
        for i, img in enumerate(images):
            with cols[i]:
                st.image(img, caption=f"Digit {digit}-{i+1}", use_container_width=True)
        
        with st.expander("Download Options"):
            img_pil = Image.fromarray((images[0]*255).astype(np.uint8))
            st.download_button(
                label="Download First Image",
                data=img_pil.tobytes(),
                file_name=f"digit_{digit}_sample.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
