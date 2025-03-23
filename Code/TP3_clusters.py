import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data as Data

DOWNLOAD_MNIST = True
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 2) 
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)

EPOCH = 100
losses = []

for epoch in range(EPOCH):
    epoch_loss = 0
    for step, (x, _) in enumerate(train_loader):
        inputX = x.view(-1, 28 * 28)
        _, decoded = autoencoder(inputX)
        loss = criterion(decoded, inputX)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{EPOCH}], Loss: {avg_loss:.4f}')

encoded_data = []
encoded_labels = []

with torch.no_grad():
    for step, (x, y) in enumerate(test_loader):
        inputX = x.view(-1, 28 * 28)
        encoded, _ = autoencoder(inputX)
        encoded_data.append(encoded.numpy())
        encoded_labels.append(y.numpy())

encoded_data = np.concatenate(encoded_data, axis=0)
encoded_labels = np.concatenate(encoded_labels, axis=0)

s
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

scatter = axes[0].scatter(encoded_data[:, 0], encoded_data[:, 1], c=encoded_labels, cmap='tab10', alpha=0.7)
axes[0].set_title("Clusters dans l'Espace Latent")
axes[0].set_xlabel("Dimension 1")
axes[0].set_ylabel("Dimension 2")
axes[0].grid(True)
fig.colorbar(scatter, ax=axes[0], label="Labels")


colors = plt.cm.tab10(np.arange(10)) 
for i in range(10):
    axes[0].scatter([], [], c=[colors[i]], label=str(i)) 
axes[0].legend(title="Chiffres", loc="upper right", bbox_to_anchor=(1.3, 1))


axes[1].plot(range(1, EPOCH + 1), losses, label='Loss', color='blue')
axes[1].set_title("Convergence de la Fonction Perte")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()

