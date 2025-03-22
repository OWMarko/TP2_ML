import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data as Data

#Exercice1
#a)Acquisition de données

DOWNLOAD_MNIST = True
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

print(train_data.data.size())  # (60000, 28, 28)
print(train_data.targets.size())  # (60000)

#b) Construction du réseau de neurones

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
    
#c) Optimisation

BATCH_SIZE = 64
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

EPOCH = 100
LEARNING_RATE = 0.005

autoencoder = AutoEncoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCH):
    for step, (x, _) in enumerate(train_loader):
        inputX = x.view(-1, 28 * 28) 
        _, decoded = autoencoder(inputX)
        loss = criterion(decoded, inputX)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{EPOCH}], Loss: {loss.item():.4f}')

#d) Visualisation
N_TEST_IMG = 10
view_data = train_data.data[:N_TEST_IMG].view(-1, 28 * 28).float() / 255.
view_labels = train_data.targets[:N_TEST_IMG]

fig, axes = plt.subplots(2, N_TEST_IMG, figsize=(10, 4))

for i in range(N_TEST_IMG):
    original_img = view_data[i].view(28, 28).numpy()
    label = view_labels[i].item()
    axes[0, i].imshow(original_img, cmap='gray')
    axes[0, i].set_xticks(())
    axes[0, i].set_yticks(())
    axes[0, i].set_title(f"Original {label}")

for i in range(N_TEST_IMG):
    test_img = view_data[i].unsqueeze(0)
    decoded_img = autoencoder(test_img)[1].detach().view(28, 28).numpy()
    label = view_labels[i].item()
    axes[1, i].imshow(decoded_img, cmap='gray')
    axes[1, i].set_xticks(())
    axes[1, i].set_yticks(())
    axes[1, i].set_title(f"Decode {label}")

plt.tight_layout()
plt.show()

