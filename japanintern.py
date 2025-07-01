import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader.dataset):.4f}")
    torch.save(model.state_dict(), "vae_mnist.pth")
    return model

if __name__ == "__main__":
    train_model()
