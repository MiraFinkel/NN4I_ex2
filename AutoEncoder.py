import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
epochs = 15

transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(3)], [0.5 for _ in range(3)]
        ),
    ]
)
class AE_CELEBA(nn.Module):
    def __init__(self):
        super(AE_CELEBA, self).__init__()
        self.encoder = nn.Sequential(
            # (3, 64, 64)

            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            # (16, 32, 32)


            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.25),
            # (32, 16, 16)


            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.25),
            # (64, 8, 8)
            nn.Flatten(),
            # (4096, 1)
            nn.Linear(4096, 512),
            # (512,1)
            nn.Linear(512, 64),
            # (64,1)
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            # (64,1)
            nn.Conv2d(64, 128, 4, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.25),
            # (128,4,4)
            nn.Conv2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.25),
            # (64, 8, 8)
            nn.Conv2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.25),
            # (32, 16, 16)
            nn.Conv2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.25),
            # (32,16,16)
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            # (3, 64, 64)
            nn.Tanh()
        )

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded.reshape(encoded.shape + (1, 1))

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))


def train(net: AE_CELEBA, dataloader, criterion=nn.MSELoss()):
    optimizer = optim.Adam(net.parameters(), lr=0.0004, betas=(0.5, 0.999))
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            batch = data[0].to(device)
            optimizer.zero_grad()
            latent_vec = net.encoder(batch)
            mean = torch.mean(latent_vec),
            variance = torch.var(latent_vec)
            kurtosis = torch.mean(((latent_vec - mean) / variance) ** 4)
            loss = criterion(net(batch), batch) + (mean ** 2 + (variance - 1) ** 2 + (kurtosis - 3) ** 2)
            loss.backward()
            optimizer.step()

    torch.save(net.state_dict(), './auto_encoder')


if __name__ == '__main__':
    AE = AE_CELEBA().to(device)
    dataset = torchvision.datasets.ImageFolder(root='./celeba', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    train(AE, dataloader=dataloader)


