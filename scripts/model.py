import torch
import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim

#import time
#import argparse
#import numpy as np
#import pandas as pd
#import logging
#from sklearn.decomposition import TruncatedSVD
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error

#import scanpy as sc
#import anndata as ad
#import matplotlib.pyplot as plt

#from tqdm import tqdm
#from copy import deepcopy

#from utils import calculate_rmse, baseline_mean


class Generator(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None,
                 dropout=0.5):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = nn.Linear(in_feat, hid_feat)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_feat, out_feat)
        self.droprateout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.droprateout(x)

"""
class Generator(nn.Module):
    def __init__(self, in_feat, hid_feat, out_feat, dropout=0.):
        super().__init__()

        self.in_feat = in_feat
        self.hid_feat = hid_feat
        self.out_feat = out_feat
        self.drop = dropout
        self.mlp = MLP(self.in_feat, self.hid_feat, self.out_feat, self.drop)
    
    def forward(self, x):
        
        return self.mlp(x)
"""

class Discriminator(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None, num_classes=1, dropout=0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = nn.Linear(in_feat, hid_feat)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_feat, out_feat)
        self.droprateout = nn.Dropout(dropout)
        self.regressor = nn.Linear(out_feat, num_classes)

    def forward(self, x):
        print("x.shape:", x.shape)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.droprateout(x)
        return self.regressor(x)

"""
class Discriminator(nn.Module):
    def __init__(self, in_feat, hid_feat, out_feat, num_classes, dropout=0.):
        super().__init__()

        self.in_feat = in_feat
        self.hid_feat = hid_feat
        self.out_feat = out_feat
        self.drop = dropout
        self.num_classes = num_classes
        self.mlp = MLP(self.in_feat, self.hid_feat, self.out_feat, self.drop)
        self.regressor = nn.Linear(self.out_feat, self.num_classes)
    
    def forward(self, x):
        
        x = self.mlp(x)
        x = self.regressor(x)
        return x
"""


