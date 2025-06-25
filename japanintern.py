import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time

# App title and config
st.set_page_config(page_title="MNIST VAE Generator", layout="wide")
st.title("🎨 MNIST Variational Autoencoder")
st.markdown("""
This app trains a VAE on MNIST digits and generates new images.
""")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    epochs = st.slider("Training epochs", 1, 50, 5)
    batch_size = st.slider("Batch size", 32, 256, 128)
    latent_dim = st.slider("Latent dimension", 2, 100, 20)
    train_button = st.button("Train Model")
    generate_button = st.button("Generate Images")

# Set device and random seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# VAE Model Definition
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Training function with progress bar
def train_vae(epochs, batch_size, latent_dim):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_history = []
    
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
        
        avg_loss = train_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)
        
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        time.sleep(0.1)
    
    st.line_chart(loss_history)
    st.success("Training completed!")
    return model

# Image generation function
def generate_images(model, latent_dim):
    model.eval()
    with torch.no_grad():
        z = torch.randn(10, latent_dim).to(device)
        generated = model.decode(z).cpu().numpy()
    
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
    st.pyplot(fig)

# Main app logic
if train_button:
    st.subheader("Training Progress")
    with st.spinner("Training in progress..."):
        model = train_vae(epochs, batch_size, latent_dim)
        st.session_state.model = model

if generate_button and 'model' in st.session_state:
    st.subheader("Generated Images")
    generate_images(st.session_state.model, latent_dim)
elif generate_button:
    st.warning("Please train the model first!")
