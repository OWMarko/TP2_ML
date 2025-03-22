import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data as Data

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 2)  # Espace latent de dimension 2
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()  # Pour ramener les valeurs entre 0 et 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

BATCH_SIZE = 64
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

EPOCH = 5
LEARNING_RATE = 0.005
autoencoder = AutoEncoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
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
    average_loss = epoch_loss / len(train_loader)
    losses.append(average_loss)
    print(f'Epoch [{epoch+1}/{EPOCH}], Loss: {average_loss:.4f}')

plt.figure(figsize=(8, 6))
plt.plot(range(1, EPOCH + 1), losses, label='MSE Loss', color='blue')
plt.title('Convergence de la fonction coût')
plt.xlabel('Époque')
plt.ylabel('Erreur quadratique moyenne (MSE)')
plt.legend()
plt.grid()
plt.show()

N_TEST_IMG = 5
view_data = train_data.data[:N_TEST_IMG].view(-1, 28 * 28).float() / 255.

fig, axes = plt.subplots(2, N_TEST_IMG, figsize=(10, 4))

for i in range(N_TEST_IMG):
    original_img = view_data[i].view(28, 28).numpy()
    axes[0, i].imshow(original_img, cmap='gray')
    axes[0, i].set_xticks(())
    axes[0, i].set_yticks(())
    axes[0, i].set_title(f"Original {i+1}")

autoencoder.eval()  
for i in range(N_TEST_IMG):
    test_img = view_data[i].unsqueeze(0) 
    decoded_img = autoencoder(test_img)[1].detach().view(28, 28).numpy()
    axes[1, i].imshow(decoded_img, cmap='gray')
    axes[1, i].set_xticks(())
    axes[1, i].set_yticks(())
    axes[1, i].set_title(f"Decoded {i+1}")

plt.tight_layout()
plt.show()
