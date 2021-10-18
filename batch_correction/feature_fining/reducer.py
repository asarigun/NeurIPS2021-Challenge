# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 01:15:02 2021
@author: Sameitos
"""

import torch
import torch.nn as nn
from .encoder import encoder
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from copy import deepcopy
import scipy

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device('cpu')
print("Device:",device)



class reducer():
    '''
    This class to reduce the dimension and correct the batch effects by changing features according to clusters.
    '''
    def __init__(self, epochs = 50, n_clusters = 20, tol = 0.01, lr = 1e-3, weight_decay = 1e-5, out_dim = 128, p = 0.3):
        ''' 
        
        X: sparse matrix 
        epochs: iteration time to train method
        out_dim: # of feature encoder output
        n_clusters: number of cluster to decrease batch size
        lr: learning rate for encoder
        weight_decay: regularization constant
        p: dropout fraction
        tol: tolerance that euclidean between new and old cluster centers
        
        '''
        self.n_clusters = n_clusters
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.out_dim = out_dim
        self.p = p
        self.init_centroids = None
        self.centroids = None
        self.loss = 0
        self.tol = tol
        self.best_features_ = None
        self.exit_tol = float('inf')
    
    def train(self, X):
        
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(device)
        if isinstance(X, scipy.sparse.csc.csc_matrix):
            X = X.todense()
            X = torch.from_numpy(X).to(device)
        
        model = encoder(X.shape[1], self.out_dim, self.p).to(device)
        
        criterion = nn.KLDivLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        
        for epoch in range(self.epochs):
            
            encoded = model(X)
            
            if self.init_centroids is None:
                np_encoded =  encoded.clone().detach().numpy()
                self.init_centroids = KMeans(n_clusters = self.n_clusters).fit(np_encoded).cluster_centers_
        
            else:
                np_encoded =  encoded.clone().detach().numpy()
                gmm = GMM(self.n_clusters).fit(np_encoded)
                self.centroids = np.empty(shape=(gmm.n_components, np_encoded.shape[1]))
                for i in range(gmm.n_components):
                    density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(np_encoded)
                    self.centroids[i, :] = np_encoded[np.argmax(density)]
               
            if self.centroids is not None and self.init_centroids is not None:
                self.exit_tol = np.linalg.norm(self.centroids - self.init_centroids)
                self.init_centroids = self.centroids
                if self.exit_tol <self.tol:
                    self.best_features_ = encoded
                    print(f'out tolerance: {self.exit_tol} at epoch: {epoch +1} with KL loss: {self.loss.item()}')
                    return self.best_features_
                
            pred_dist = self.get_dist(encoded)
            self.loss = criterion(encoded, pred_dist)
            self.loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            
            print(f'Epoch:{1 + epoch}, Loss:{self.loss.item():4f}, exit_tol: {self.exit_tol} ')
            
        return encoded
    
    
    def get_dist(self,embeds):
        
        weight = embeds**2 / embeds.sum(0)
        target_data = (weight.T / weight.sum(1)).T
        return target_data
    