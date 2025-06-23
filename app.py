import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator class (must match the trained one)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        return self.model(x).view(-1, 1, 28, 28)

# Load trained generator
@st.cache_resource
def load_model():
    model = Generator().to(device)
    model.load_state_dict(torch.load("generator_mnist.pth", map_location=device))
    model.eval()
    return model

# Streamlit UI
st.title("🧠 Handwritten Digit Generator (MNIST GAN)")
st.write("Select a digit (0–9) and generate 5 images!")

digit = st.selectbox("Choose a digit", list(range(10)))

if st.button("Generate Images"):
    G = load_model()
    z = torch.randn(5, 100).to(device)
    labels = torch.tensor([digit] * 5).to(device)
    with torch.no_grad():
        gen_imgs = G(z, labels).cpu()

    gen_imgs = gen_imgs * 0.5 + 0.5  # Denormalize
    grid = make_grid(gen_imgs, nrow=5).permute(1, 2, 0).numpy()

    st.image(grid, caption=f"Generated Images for Digit {digit}", use_column_width=True)
