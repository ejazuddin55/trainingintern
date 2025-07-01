
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os

# 1. VAE Model with Conditional Generation
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

# 2. Model Loading with Training Fallback
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalVAE().to(device)
    
    try:
        if os.path.exists("conditional_vae_mnist.pth"):
            model.load_state_dict(torch.load("conditional_vae_mnist.pth", map_location=device))
        else:
            with st.spinner("Training model for the first time (this may take 5-10 minutes)..."):
                model = train_model()
    except Exception as e:
        st.error(f"Model error: {str(e)}")
        st.stop()
    
    return model

# 3. Training Function
def train_model(epochs=10):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data, labels)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        st.write(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader.dataset):.4f}")
    
    torch.save(model.state_dict(), "conditional_vae_mnist.pth")
    return model

# 4. Loss Function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 5. Digit-Specific Generation
def generate_digit_images(model, digit, num_images=5):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        # Create latent vectors + digit labels
        z = torch.randn(num_images, 20).to(device)
        digits = torch.full((num_images,), digit, dtype=torch.long).to(device)
        
        # Generate images
        samples = model.decode(z, digits).cpu().numpy()
    
    return samples.reshape(-1, 28, 28)

# 6. Streamlit UI
def main():
    st.set_page_config(
        page_title="Digit-Specific VAE Generator",
        page_icon="🔢",
        layout="centered"
    )
    
    st.title("🔢 Digit-Specific Generator")
    st.markdown("Generate 5 images of the same handwritten digit (0-9)")
    
    model = load_model()
    
    # Digit selection
    digit = st.selectbox("Select digit to generate:", options=range(10), index=5)
    
    if st.button("Generate 5 Images", type="primary"):
        with st.spinner(f"Generating 5 images of digit {digit}..."):
            images = generate_digit_images(model, digit)
        
        st.success(f"Generated 5 images of digit {digit}:")
        cols = st.columns(5)
        for i, img in enumerate(images):
            with cols[i]:
                st.image(img, caption=f"Digit {digit}-{i+1}", use_column_width=True)
        
        # Download option
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
