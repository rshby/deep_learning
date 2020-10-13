# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 18:08:44 2020

@author: Reo Sahobby
"""
import torch
from torch import nn

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
    
    
class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
            )
        
    def forward(self, x):
        return self.fc(x)
    
    def generate(self, n, device):
        fake_img = torch.randn((n, self.z_dim), device=device)
        return self.fc(fake_img)