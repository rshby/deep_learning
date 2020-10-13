import torch
from torch import nn
from jcopdl.layers import linear_block


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            linear_block(784, 512, activation="lrelu"),
            linear_block(512, 256, activation="lrelu"),
            linear_block(256, 128, activation="lrelu"),
            linear_block(128, 1, activation='sigmoid')
        )

    def forward(self, x):
        return self.fc(x)


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
