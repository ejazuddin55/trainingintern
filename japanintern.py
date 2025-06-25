# app.py - Complete Working Version
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# 1. VAE Model Definition
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim//2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//2, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x.view(-1, 784))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 2. Model Loading with Error Handling
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    
    try:
        if os.path.exists("vae_mnist.pth"):
            model.load_state_dict(torch.load("vae_mnist.pth", map_location=device))
        else:
            st.warning("Model file not found, training new model...")
            model = train_model()
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()
    
    return model

# 3. Training Function
def train_model(epochs=5):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader.dataset):.4f}")
    
    torch.save(model.state_dict(), "vae_mnist.pth")
    return model

# 4. Loss Function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 5. Image Generation
def generate_images(model, num_images=5):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_images, 20).to(next(model.parameters()).device)
        samples = model.decode(z).cpu().numpy()
    return samples.reshape(-1, 28, 28)

# 6. Streamlit App Interface
def main():
    st.set_page_config(
        page_title="MNIST VAE Generator",
        page_icon="🎨",
        layout="centered"
    )
    
    st.title("🎨 Handwritten Digit Generator")
    st.markdown("""
    Generate new MNIST-style digits using a Variational Autoencoder
    """)
    
    model = load_model()
    
    col1, col2 = st.columns([1, 3])
    with col1:
        num_images = st.slider("Number of images", 1, 10, 5)
        generate_btn = st.button("✨ Generate", type="primary")
    
    if generate_btn:
        with st.spinner("Generating images..."):
            images = generate_images(model, num_images)
        
        st.success("Generated Images:")
        cols = st.columns(num_images)
        for i, img in enumerate(images):
            with cols[i]:
                st.image(img, caption=f"Image {i+1}", use_column_width=True)
        
        # Download button
        with st.expander("Download Options"):
            st.download_button(
                label="Download as PNG",
                data=Image.fromarray((images[0]*255).astype(np.uint8)).tobytes(),
                file_name="generated_digit.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
