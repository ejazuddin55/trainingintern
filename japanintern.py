import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os

# Conditional VAE model definition
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, num_classes=10):
        super(ConditionalVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
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
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        c_onehot = torch.nn.functional.one_hot(c, self.num_classes).float()
        z = torch.cat([z, c_onehot], dim=1)
        return self.decoder(z)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

# Load pretrained model from state_dict
@st.cache_resource
def load_model():
    model_path = "vae_mnist_trained.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalVAE().to(device)

    if not os.path.exists(model_path):
        st.error("❌ Model file not found! Please place `vae_mnist_trained.pth` in the app folder.")
        st.stop()

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error("❌ Failed to load model. Make sure it was saved using `model.state_dict()`.")
        st.code("torch.save(model.state_dict(), 'vae_mnist_trained.pth')")
        st.stop()

    return model

# Generate 5 images of the selected digit
def generate_digit_images(model, digit, num_images=5):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        z = torch.randn(num_images, 20).to(device)
        labels = torch.full((num_images,), digit, dtype=torch.long).to(device)
        samples = model.decode(z, labels).cpu().numpy()
    return samples.reshape(-1, 28, 28)

# Streamlit UI
def main():
    st.set_page_config(page_title="Digit Generator", page_icon="🔢", layout="centered")
    st.title("🔢 Digit-Specific VAE Generator")
    st.markdown("Generate 5 synthetic handwritten digits (0–9) using a pretrained Conditional VAE.")

    model = load_model()

    digit = st.selectbox("Choose a digit to generate:", options=list(range(10)), index=5)

    if st.button("Generate Images", type="primary"):
        with st.spinner(f"Generating images for digit {digit}..."):
            images = generate_digit_images(model, digit)

        st.success(f"Here are 5 generated images of digit {digit}:")
        cols = st.columns(5)
        for i, img in enumerate(images):
            with cols[i]:
                st.image(img, caption=f"{digit}-{i+1}", use_container_width=True)

        with st.expander("Download First Image"):
            img_pil = Image.fromarray((images[0] * 255).astype(np.uint8))
            st.download_button(
                label="Download Image",
                data=img_pil.tobytes(),
                file_name=f"digit_{digit}_sample.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
