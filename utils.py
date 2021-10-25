import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets

import time
import os
import numpy as np
import pandas as pd
import logging
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

from tqdm import tqdm
from copy import deepcopy

import anndata as ad


class AtacTrainDataset(Dataset):
    def __init__(self, gex_data_dir, atac_data_dir):
        
        """
        :self.gex_data_dir --------------> data direction for GEX
        :self.atac_data_dir -------------> data direction for ATAC

        :self.adata_gex -----------------> loading GEX dataset 
        :self.adata_atac ----------------> loading ATAC 
        
        ######################################################################################################
        #   The GEX data has 20952 (adata_gex.n_obs) observations and 15037 (adata_gex.n_vars) features.     #
        #   The ATAC data has 20952 (adata_atac.n_obs) observations and 119014 (adata_atac.n_vars) features. #
        ######################################################################################################

        :self.adata_gex.var -------------> 15037 rows × 15 columns
        :self.adata_atac.var ------------> 119014 rows × 7 columns

        :self.adata_gex.obs -------------> 20952 rows × 9 columns
        :self.adata_atac.obs ------------> 20952 rows × 8 columns

        :self.train_cells ---------------> training cells
        :self.test_cells ----------------> test cells

        :self.input_train_mod1 ----------> ATAC training cells
        :self.input_train_mod2 ----------> GEX training cells

        :self.input_test_mod1 -----------> ATAC test cells
        :self.true_test_mod2 ------------> GEX test cells

        :mod1_pca -----------------------> PCA for ATAC
        :mod2_pca -----------------------> PCA for GEX

        :self.X_train -------------------> ATAC Training Dataset
        :self.X_test --------------------> ATAC Test Dataset
        :self.y_train -------------------> GEX Training Dataset
        """
        gex_data_dir = "/home/ubuntu/Documents/ahmetr/single-cell/data/explore/multiome/multiome_gex_processed_training.h5ad"
        atac_data_dir = "/home/ubuntu/Documents/ahmetr/single-cell/data/explore/multiome/multiome_atac_processed_training.h5ad"
        
        self.gex_data_dir = gex_data_dir
        self.atac_data_dir = atac_data_dir
        self.adata_gex = ad.read_h5ad(self.gex_data_dir)
        self.adata_atac = ad.read_h5ad(self.atac_data_dir)

        self.train_cells = self.adata_gex.obs_names[self.adata_gex.obs["batch"] != "s2d4"]
        self.test_cells  = self.adata_gex.obs_names[self.adata_gex.obs["batch"] == "s2d4"]
        
        # This will get passed to the method
        self.input_train_mod1 = self.adata_atac[self.train_cells]
        self.input_train_mod2 = self.adata_gex[self.train_cells]
        self.input_test_mod1 =  self.adata_atac[self.test_cells]

        # This will get passed to the metric
        self.true_test_mod2 =  self.adata_gex[self.test_cells]

        input_mod1 = ad.concat(
        {"train": self.input_train_mod1, "test": self.input_test_mod1},
        axis=0,
        join="outer",
        label="group",
        fill_value=0,
        index_unique="-") 
                                
        # Binarize ATAC 
        if self.input_train_mod1.var["feature_types"][0] == "ATAC":
            input_mod1.X[input_mod1.X > 1] = 1
        elif self.input_train_mod2.var["feature_types"][0] == "ATAC":
            self.input_train_mod2.X[input_mod1.X > 1] = 1


        # Do PCA on the input data
        logging.info('Performing dimensionality reduction on modality 1 values...')
        embedder_mod1 = TruncatedSVD(n_components=50)
        mod1_pca = embedder_mod1.fit_transform(input_mod1.X)
        
        logging.info('Performing dimensionality reduction on modality 2 values...')
        embedder_mod2 = TruncatedSVD(n_components=50)
        mod2_pca = embedder_mod2.fit_transform(self.input_train_mod2.layers["log_norm"])
        

        # split dimred mod 1 back up for training
        self.X_train = mod1_pca[input_mod1.obs['group'] == 'train']
        self.X_test = mod1_pca[input_mod1.obs['group'] == 'test']
        self.y_train = mod2_pca
        
        assert len(self.X_train) + len(self.X_test) == len(mod1_pca)

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):

        #X_train = self.X_train.iloc[idx]
        X_train = self.X_train[idx]
        print("self.X_train:", type(self.X_train))
        print("self.X_train:", self.X_train.shape)
        #df = pd.DataFrame(X_train, columns= ['Column_A'])
        X_train = torch.from_numpy(X_train)
        return X_train
        
class AtacTestDataset(Dataset):
    def __init__(self, gex_data_dir, atac_data_dir):
        
        """
        :self.gex_data_dir --------------> data direction for GEX
        :self.atac_data_dir -------------> data direction for ATAC

        :self.adata_gex -----------------> loading GEX dataset 
        :self.adata_atac ----------------> loading ATAC dataset

        ######################################################################################################
        #   The GEX data has 20952 (adata_gex.n_obs) observations and 15037 (adata_gex.n_vars) features.     #
        #   The ATAC data has 20952 (adata_atac.n_obs) observations and 119014 (adata_atac.n_vars) features. #
        ######################################################################################################

        :self.adata_gex.var -------------> 15037 rows × 15 columns
        :self.adata_atac.var ------------> 119014 rows × 7 columns

        :self.adata_gex.obs -------------> 20952 rows × 9 columns
        :self.adata_atac.obs ------------> 20952 rows × 8 columns

        :self.train_cells ---------------> training cells
        :self.test_cells ----------------> test cells

        :self.input_train_mod1 ----------> ATAC training cells
        :self.input_train_mod2 ----------> GEX training cells

        :self.input_test_mod1 -----------> ATAC test cells
        :self.true_test_mod2 ------------> GEX test cells

        :mod1_pca -----------------------> PCA for ATAC
        :mod2_pca -----------------------> PCA for GEX

        :self.X_train -------------------> ATAC Training Dataset
        :self.X_test --------------------> ATAC Test Dataset
        :self.y_train -------------------> GEX Training Dataset
        """
        gex_data_dir = "/home/ubuntu/Documents/ahmetr/single-cell/data/explore/multiome/multiome_gex_processed_training.h5ad"
        atac_data_dir = "/home/ubuntu/Documents/ahmetr/single-cell/data/explore/multiome/multiome_atac_processed_training.h5ad"
        
        self.gex_data_dir = gex_data_dir
        self.atac_data_dir = atac_data_dir
        self.adata_gex = ad.read_h5ad(self.gex_data_dir)
        self.adata_atac = ad.read_h5ad(self.atac_data_dir)

        self.train_cells = self.adata_gex.obs_names[self.adata_gex.obs["batch"] != "s2d4"]
        self.test_cells  = self.adata_gex.obs_names[self.adata_gex.obs["batch"] == "s2d4"]
        
        # This will get passed to the method
        self.input_train_mod1 = self.adata_atac[self.train_cells]
        self.input_train_mod2 = self.adata_gex[self.train_cells]
        self.input_test_mod1 =  self.adata_atac[self.test_cells]

        # This will get passed to the metric
        self.true_test_mod2 =  self.adata_gex[self.test_cells]

        input_mod1 = ad.concat(
        {"train": self.input_train_mod1, "test": self.input_test_mod1},
        axis=0,
        join="outer",
        label="group",
        fill_value=0,
        index_unique="-") 
                                
        # Binarize ATAC 
        if self.input_train_mod1.var["feature_types"][0] == "ATAC":
            input_mod1.X[input_mod1.X > 1] = 1
        elif self.input_train_mod2.var["feature_types"][0] == "ATAC":
            self.input_train_mod2.X[input_mod1.X > 1] = 1


        # Do PCA on the input data
        logging.info('Performing dimensionality reduction on modality 1 values...')
        embedder_mod1 = TruncatedSVD(n_components=50)
        mod1_pca = embedder_mod1.fit_transform(input_mod1.X)
        
        logging.info('Performing dimensionality reduction on modality 2 values...')
        embedder_mod2 = TruncatedSVD(n_components=50)
        mod2_pca = embedder_mod2.fit_transform(self.input_train_mod2.layers["log_norm"])
        

        # split dimred mod 1 back up for training
        self.X_train = mod1_pca[input_mod1.obs['group'] == 'train']
        self.X_test = mod1_pca[input_mod1.obs['group'] == 'test']
        self.y_train = mod2_pca
        
        assert len(self.X_train) + len(self.X_test) == len(mod1_pca)

    def __len__(self):
        return len(self.X_test)

    def __getitem__(self, idx):

        #X_test = self.X_test.iloc[idx]
        X_test = self.X_test[idx]
        X_test = torch.from_numpy(X_test)
        return X_test

class GexTrainDataset(Dataset):
    def __init__(self, gex_data_dir, atac_data_dir):
        
        """
        :self.gex_data_dir --------------> data direction for GEX
        :self.atac_data_dir -------------> data direction for ATAC

        :self.adata_gex -----------------> loading GEX dataset 
        :self.adata_atac ----------------> loading ATAC dataset

        ######################################################################################################
        #   The GEX data has 20952 (adata_gex.n_obs) observations and 15037 (adata_gex.n_vars) features.     #
        #   The ATAC data has 20952 (adata_atac.n_obs) observations and 119014 (adata_atac.n_vars) features. #
        ######################################################################################################

        :self.adata_gex.var -------------> 15037 rows × 15 columns
        :self.adata_atac.var ------------> 119014 rows × 7 columns
        
        :self.adata_gex.obs -------------> 20952 rows × 9 columns
        :self.adata_atac.obs ------------> 20952 rows × 8 columns

        :self.train_cells ---------------> training cells
        :self.test_cells ----------------> test cells

        :self.input_train_mod1 ----------> ATAC training cells
        :self.input_train_mod2 ----------> GEX training cells

        :self.input_test_mod1 -----------> ATAC test cells
        :self.true_test_mod2 ------------> GEX test cells

        :mod1_pca -----------------------> PCA for ATAC
        :mod2_pca -----------------------> PCA for GEX

        :self.X_train -------------------> ATAC Training Dataset
        :self.X_test --------------------> ATAC Test Dataset
        :self.y_train -------------------> GEX Training Dataset
        """
        gex_data_dir = "/home/ubuntu/Documents/ahmetr/single-cell/data/explore/multiome/multiome_gex_processed_training.h5ad"
        atac_data_dir = "/home/ubuntu/Documents/ahmetr/single-cell/data/explore/multiome/multiome_atac_processed_training.h5ad"
        
        self.gex_data_dir = gex_data_dir
        self.atac_data_dir = atac_data_dir
        self.adata_gex = ad.read_h5ad(self.gex_data_dir)
        self.adata_atac = ad.read_h5ad(self.atac_data_dir)

        self.train_cells = self.adata_gex.obs_names[self.adata_gex.obs["batch"] != "s2d4"]
        self.test_cells  = self.adata_gex.obs_names[self.adata_gex.obs["batch"] == "s2d4"]
        
        # This will get passed to the method
        self.input_train_mod1 = self.adata_atac[self.train_cells]
        self.input_train_mod2 = self.adata_gex[self.train_cells]
        self.input_test_mod1 =  self.adata_atac[self.test_cells]

        # This will get passed to the metric
        self.true_test_mod2 =  self.adata_gex[self.test_cells]

        input_mod1 = ad.concat(
        {"train": self.input_train_mod1, "test": self.input_test_mod1},
        axis=0,
        join="outer",
        label="group",
        fill_value=0,
        index_unique="-") 
                                
        # Binarize ATAC 
        if self.input_train_mod1.var["feature_types"][0] == "ATAC":
            input_mod1.X[input_mod1.X > 1] = 1
        elif self.input_train_mod2.var["feature_types"][0] == "ATAC":
            self.input_train_mod2.X[input_mod1.X > 1] = 1


        # Do PCA on the input data
        logging.info('Performing dimensionality reduction on modality 1 values...')
        embedder_mod1 = TruncatedSVD(n_components=50)
        mod1_pca = embedder_mod1.fit_transform(input_mod1.X)
        
        logging.info('Performing dimensionality reduction on modality 2 values...')
        embedder_mod2 = TruncatedSVD(n_components=50)
        mod2_pca = embedder_mod2.fit_transform(self.input_train_mod2.layers["log_norm"])
        

        # split dimred mod 1 back up for training
        self.X_train = mod1_pca[input_mod1.obs['group'] == 'train']
        self.X_test = mod1_pca[input_mod1.obs['group'] == 'test']
        self.y_train = mod2_pca
        
        assert len(self.X_train) + len(self.X_test) == len(mod1_pca)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):

        #y_train = self.y_train.iloc[idx]
        y_train = self.y_train[idx]
        #print("self.y_train:", type(self.y_train))
        #print("self.y_train:", self.y_train.shape)
        print("y_train:", y_train)
        y_train = torch.from_numpy(y_train)
        #print("self.y_train:", type(y_train))
        #print("self.y_train:", self.y_train.shape)
        print("torch y_train:", y_train)
        return y_train

def load_atac_train(gex_data_dir, atac_data_dir):
    
    #gex_data_dir = "/home/ubuntu/Documents/ahmetr/single-cell/data/explore/multiome/multiome_gex_processed_training.h5ad"
    #atac_data_dir = "/home/ubuntu/Documents/ahmetr/single-cell/data/explore/multiome/multiome_atac_processed_training.h5ad"
    atac_train_dataset = AtacTrainDataset(gex_data_dir, atac_data_dir)

    atac_train_dataset = torch.utils.data.DataLoader(dataset=atac_train_dataset, batch_size=32, shuffle=True)
    
    return atac_train_dataset

def load_atac_test(gex_data_dir, atac_data_dir):
    
    #gex_data_dir = "/home/ubuntu/Documents/ahmetr/single-cell/data/explore/multiome/multiome_gex_processed_training.h5ad"
    #atac_data_dir = "/home/ubuntu/Documents/ahmetr/single-cell/data/explore/multiome/multiome_atac_processed_training.h5ad"
    atac_test_dataset = AtacTestDataset(gex_data_dir, atac_data_dir)

    atac_test_dataset = torch.utils.data.DataLoader(dataset=atac_test_dataset, batch_size=32, shuffle=True)
    
    return atac_test_dataset

def load_gex_train(gex_data_dir, atac_data_dir):
    
    #gex_data_dir = "/home/ubuntu/Documents/ahmetr/single-cell/data/explore/multiome/multiome_gex_processed_training.h5ad"
    #atac_data_dir = "/home/ubuntu/Documents/ahmetr/single-cell/data/explore/multiome/multiome_atac_processed_training.h5ad"
    gex_train_dataset = GexTrainDataset(gex_data_dir, atac_data_dir)

    gex_train_dataset = torch.utils.data.DataLoader(dataset=gex_train_dataset, batch_size=32, shuffle=True)
    
    return gex_train_dataset


def calculate_rmse(true_test_mod2, pred_test_mod2):
    if pred_test_mod2.var["feature_types"][0] == "GEX":
        return  mean_squared_error(true_test_mod2.layers["log_norm"].toarray(), pred_test_mod2.X, squared=False)
    else:
        raise NotImplementedError("Only set up to calculate RMSE for GEX data")    


def baseline_mean(input_train_mod1, input_train_mod2, input_test_mod1):
    '''Dummy method that predicts mean(input_train_mod2) for all cells'''
    logging.info('Calculate mean of the training data modality 2...')
    y_pred = np.repeat(input_train_mod2.layers["log_norm"].mean(axis=0).reshape(-1,1).T, input_test_mod1.shape[0], axis=0)
    
    # Prepare the ouput data object
    pred_test_mod2 = ad.AnnData(
        X=y_pred,
        obs=input_test_mod1.obs,
        var=input_train_mod2.var,
    )
    
    pred_test_mod2.uns["method"] = "mean"

    return pred_test_mod2


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr

def inits_weight(m):
        if type(m) == nn.Linear:
                nn.init.xavier_uniform(m.weight.data, 1.)

def save_checkpoint(states,is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))

if __name__ == '__main__':
    gex_data_dir = "/home/ubuntu/Documents/ahmetr/single-cell/data/explore/multiome/multiome_gex_processed_training.h5ad"
    atac_data_dir = "/home/ubuntu/Documents/ahmetr/single-cell/data/explore/multiome/multiome_atac_processed_training.h5ad"
    atac_dataset = load_atac_train(gex_data_dir, atac_data_dir)
    gex_dataset = load_gex_train(gex_data_dir, atac_data_dir)