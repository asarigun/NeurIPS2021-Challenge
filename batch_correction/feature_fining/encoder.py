# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 01:04:57 2021

@author: Sameitos
"""


import torch 
import torch.nn as nn
import torch.optim as optim



class encoder(nn.Module):
    def __init__(self, in_dim, out_dim, p):
        
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096,1024),
            nn.SiLU(),
            nn.Dropout(p),
            nn.Linear(1024,out_dim)
            
            )
        
        
        # self.decoder = nn.Sequential(
        #     nn.Linear(out_dim, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 4096),
        #     nn.ReLU(),
        #     nn.Dropout(p),
        #     nn.Linear(4096, in_dim),
        #     nn.SiLU()
        #     )
        
    def forward(self,x):
        encoded = self.encoder(x)
        #decoded = self.decoder(encoded)
        return encoded#, decoded
        

        