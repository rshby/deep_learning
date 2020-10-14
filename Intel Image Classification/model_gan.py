# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 09:22:57 2020

@author: ROG
"""


import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(22500, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
            )
    
    def forward(self, x):
        return self.fc(x)
    
    
class Generator(nn.Module):
    def __init__(self, z_dimenssion):
        super().__init__()
        self.z_dimenssion = z_dimenssion
        self.fc = nn.Sequential(
            nn.Linear(z_dimenssion, out_features)
            )