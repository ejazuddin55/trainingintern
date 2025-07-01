import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# 1. Enhanced VAE Model with Conditional Generation
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, num_classes=10):
        super(ConditionalVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim//2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//2, latent_dim)
        
        # Decoder
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

# 2. Improved Model Loading with Training Fallback
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalVAE().to(device)
    
    model_path = "conditional_vae_mnist.pth"
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            st.success("Pre-trained model loaded successfully!")
        except Exception as e:
            st.warning(f"Error loading model: {e}. Training a new model...")
            model = train_model()
    else:
        with st.spinner("Training model for the first time (this may take 5-10 minutes)..."):
            model = train_model()
    
    return model

# 3. Enhanced Training Function with Progress Bar
def train_model(epochs=10):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data, labels)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            # Update progress
            progress = (epoch * len(train_loader) + batch_idx) / (epochs * len(train_loader))
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        st.write(f"Epoch {epoch+1} completed. Average loss: {train_loss/len(train_loader.dataset):.4f}")
    
    torch.save(model.state_dict(), "conditional_vae_mnist.pth")
    progress_bar.empty()
    status_text.empty()
    st.success("Training completed successfully!")
    return model

# 4. Loss Function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 5. Enhanced Digit Generation with Visualization Options
def generate_digit_images(model, digit, num_images=5):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        z = torch.randn(num_images, 20).to(device)
        digits = torch.full((num_images,), digit, dtype=torch.long).to(device)
        samples = model.decode(z, digits).cpu().numpy()
    
    return samples.reshape(-1, 28, 28)

# 6. Improved Streamlit UI with More Features
def main():
    st.set_page_config(
        page_title="Digit-Specific VAE Generator",
        page_icon="🔢",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔢 Digit-Specific VAE Generator")
    st.markdown("""
    Generate multiple images of the same handwritten digit (0-9) using a Conditional Variational Autoencoder.
    The model will be automatically trained if no pre-trained weights are found.
    """)
    
    # Sidebar with additional controls
    with st.sidebar:
        st.header("Configuration")
        num_images = st.slider("Number of images to generate", 1, 10, 5)
        show_grid = st.checkbox("Show as grid", True)
        download_all = st.checkbox("Enable download all", False)
    
    # Load or train model
    model = load_model()
    
    # Main interface
    col1, col2 = st.columns([1, 3])
    
    with col1:
        digit = st.selectbox("Select digit to generate:", options=range(10), index=5)
        if st.button("Generate Images", type="primary"):
            with st.spinner(f"Generating {num_images} images of digit {digit}..."):
                images = generate_digit_images(model, digit, num_images)
            
            # Display results
            st.success(f"Generated {num_images} images of digit {digit}:")
            
            if show_grid:
                cols = st.columns(min(5, num_images))
                for i, img in enumerate(images[:num_images]):
                    with cols[i % 5]:
                        st.image(img, caption=f"Digit {digit}-{i+1}", use_container_width=True)
            else:
                for i, img in enumerate(images[:num_images]):
                    st.image(img, caption=f"Digit {digit}-{i+1}", use_container_width=True)
            
            # Download options
            with st.expander("Download Options"):
                if download_all:
                    for i, img in enumerate(images[:num_images]):
                        img_pil = Image.fromarray((img*255).astype(np.uint8))
                        st.download_button(
                            label=f"Download Image {i+1}",
                            data=img_pil.tobytes(),
                            file_name=f"digit_{digit}_sample_{i+1}.png",
                            mime="image/png"
                        )
                else:
                    img_pil = Image.fromarray((images[0]*255).astype(np.uint8))
                    st.download_button(
                        label="Download First Image",
                        data=img_pil.tobytes(),
                        file_name=f"digit_{digit}_sample.png",
                        mime="image/png"
                    )

    with col2:
        st.markdown("### About This App")
        st.markdown("""
        This application uses a Conditional Variational Autoencoder (CVAE) to generate 
        handwritten digits of your chosen number.
        
        - **Model Architecture**: The VAE has an encoder-decoder structure with conditional generation
        - **Training**: The model is trained on MNIST dataset
        - **Generation**: You can generate multiple samples of the same digit
        
        Adjust the settings in the sidebar to customize your experience.
        """)

if __name__ == "__main__":
    main()
