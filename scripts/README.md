# CellGAN: Generative Adversarial Networks for Single-Cell Modality Prediction

## Overview:

* ```model.py```: Generator and Discriminator
* ```utils.py```: RMSE score and other stuffs
* ```train.py```: Training CellGAN for Modality Prediction

## Usage

```bash
python train.py
```

## Issue

```bash
Device: cuda:0
/home/ubuntu/Documents/ahmetr/single-cell/neurips2021-notebooks/utils.py:69: UserWarning: nn.init.xavier_uniform is now deprecated in favor of nn.init.xavier_uniform_.
  nn.init.xavier_uniform(m.weight.data, 1.)
real_data.shape: torch.Size([15496, 50])
x.shape: torch.Size([15496, 50])
Traceback (most recent call last):
  File "train.py", line 366, in <module>
    epoch, lr_schedulers, n_critic = args.n_critic, device="cuda:0")
  File "train.py", line 256, in train
    real_valid = discriminator(real_data)
  File "/home/ubuntu/miniconda3/envs/new_gpu_env/lib/python3.7/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/ubuntu/Documents/ahmetr/single-cell/neurips2021-notebooks/model.py", line 75, in forward
    x = self.fc1(x)
  File "/home/ubuntu/miniconda3/envs/new_gpu_env/lib/python3.7/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/ubuntu/miniconda3/envs/new_gpu_env/lib/python3.7/site-packages/torch/nn/modules/linear.py", line 94, in forward
    return F.linear(input, self.weight, self.bias)
  File "/home/ubuntu/miniconda3/envs/new_gpu_env/lib/python3.7/site-packages/torch/nn/functional.py", line 1755, in linear
    return torch._C._nn.linear(input, weight, bias)
RuntimeError: mat1 dim 1 must match mat2 dim 0
```