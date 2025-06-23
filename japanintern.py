import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Hyperparameters
latent_dim = 50  # Reduced for lower memory
num_classes = 10
image_size = 28
batch_size = 64  # Reduced batch size
epochs = 20  # Reduced for faster training
lr = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Verify T4 GPU in Colab
if device.type == "cuda":
    assert torch.cuda.get_device_name(0).lower().find("t4") != -1, "Must use T4 GPU"

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# Simplified Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 20)  # Smaller embedding
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + 20, 128 * 7 * 7),
            nn.ReLU(True)
        )
        self.model = nn.Sequential(
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        x = self.fc(x)
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 20)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7 + 20, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        c = self.label_emb(labels)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, c], dim=1)
        return self.fc(x)

# Initialize models
G = Generator().to(device)
D = Discriminator().to(device)
loss_fn = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# Training loop
for epoch in range(epochs):
    for real_imgs, labels in loader:
        batch_size = real_imgs.size(0)
        real_imgs, labels = real_imgs.to(device), labels.to(device)

        # Label smoothing
        valid = torch.ones(batch_size, 1).to(device) * 0.9
        fake = torch.zeros(batch_size, 1).to(device)

        # Train Generator
        z = torch.randn(batch_size, latent_dim).to(device)
        gen_labels = torch.randint(0, 10, (batch_size,)).to(device)
        gen_imgs = G(z, gen_labels)
        g_loss = loss_fn(D(gen_imgs, gen_labels), valid)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        # Train Discriminator
        real_loss = loss_fn(D(real_imgs, labels), valid)
        fake_loss = loss_fn(D(gen_imgs.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) / 2

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

    # Save sample images
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            z = torch.randn(5, latent_dim).to(device)
            labels = torch.full((5,), 0, dtype=torch.long).to(device)
            sample_imgs = G(z, labels)
            torchvision.utils.save_image(sample_imgs, f"sample_epoch_{epoch+1}.png", normalize=True)

    print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")

# Save model
torch.save(G.state_dict(), "generator_mnist.pth")
