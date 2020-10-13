import torch
from torch import nn
from jcopdl.layers import linear_block


# Buat claass untuk model Descriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        return self.fc(x)


# Buat class untuk model Generator
class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Sequential(
            linear_block(z_dim, 128, activation="lrelu"),
            linear_block(126, 256, activation="lrelu", batch_norm=True),
            linear_block(256, 512, activation="lrelu", batch_norm=True),
            linear_block(512, 1024, activation="lrelu", batch_norm=True),
            linear_block(1024, 784, activation="tanh")
        )

    def forward(self, x):
        return self.fc(x)

    def generate(self, n, device):
        z = torch.randn((n, self.z_dim), device=device)
        return self.fc(z)


# Buat class untuk model Generator Experiment
class Generator_experiment(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256), ## normalization
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512), ## normalization
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024), ## normalization
            nn.LeakyReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
            )

    def forward(self, x):
        return self.fc(x)

    def generate(self, n, device):
        z = torch.randn((n, self.z_dim), device=device)
        return self.fc(z)