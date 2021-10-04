import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import argparse
import numpy as np

from tqdm import tqdm
from copy import deepcopy


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None,
                 dropout=0.):
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

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=1 , help='Number of classes for discriminator.')
parser.add_argument('--lr_gen', type=float, default=0.0001 , help='Learning rate for generator.')
parser.add_argument('--lr_dis', type=float, default=0.0001 , help='Learning rate for discriminator.')
parser.add_argument('--weight_decay', type=float, default=1e-3 , help='Weight decay.')
parser.add_argument('--gener_batch_size', type=int, default=64 , help='Batch size for generator.')
parser.add_argument('--dis_batch_size', type=int, default=32 , help='Batch size for discriminator.')
parser.add_argument('--epoch', type=int, default=200 , help='Number of epoch.')
parser.add_argument('--optim', type=str, default="Adam" , help='Choose your optimizer')
parser.add_argument('--loss', type=str, default="wgangp_eps" , help='Loss function')
parser.add_argument('--phi', type=int, default="1" , help='phi')
parser.add_argument('--beta1', type=int, default="0" , help='beta1')
parser.add_argument('--n_critic', type=int, default=5 , help='n_critic.')
parser.add_argument('--max_iter', type=int, default=500000 , help='max_iter.')
parser.add_argument('--beta2', type=float, default="0.99" , help='beta2')
parser.add_argument('--lr_decay', type=str, default=True , help='lr_decay')
parser.add_argument('--in_feat', type=int, default=1 , help='in_feat.')
parser.add_argument('--hid_feat', type=int, default=5 , help='hid_feat.')
parser.add_argument('--out_feat', type=int, default=5 , help='out_feat.')
parser.add_argument('--dropout', type=float, default=5 , help='dropout.')

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)
print("Device:",device)

args = parser.parse_args()

generator= Generator(in_feat=args.in_feat, hid_feat=args.hid_feat, out_feat=args.out_feat, dropout=args.dropout)#,device = device)
generator.to(device)

discriminator = Discriminator(in_feat=args.in_feat, hid_feat=args.hid_feat, out_feat=args.out_feat, num_classes=args.num_classes, dropout=args.dropout)
discriminator.to(device)


generator.apply(inits_weight)
discriminator.apply(inits_weight)

if args.optim == 'Adam':
    optim_gen = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=args.lr_gen, betas=(args.beta1, args.beta2))

    optim_dis = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()),lr=args.lr_dis, betas=(args.beta1, args.beta2))
    
elif args.optim == 'SGD':
    optim_gen = optim.SGD(filter(lambda p: p.requires_grad, generator.parameters()),
                lr=args.lr_gen, momentum=0.9)

    optim_dis = optim.SGD(filter(lambda p: p.requires_grad, discriminator.parameters()),
                lr=args.lr_dis, momentum=0.9)

elif args.optim == 'RMSprop':
    optim_gen = optim.RMSprop(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_dis, eps=1e-08, weight_decay=args.weight_decay, momentum=0, centered=False)

    optim_dis = optim.RMSprop(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_dis, eps=1e-08, weight_decay=args.weight_decay, momentum=0, centered=False)

gen_scheduler = LinearLrDecay(optim_gen, args.lr_gen, 0.0, 0, args.max_iter * args.n_critic)
dis_scheduler = LinearLrDecay(optim_dis, args.lr_dis, 0.0, 0, args.max_iter * args.n_critic)

def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.get_device())
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.get_device())
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty

def train(generator, discriminator, optim_gen, optim_dis,
        epoch, schedulers, n_critic = args.n_critic,
        gener_batch_size=args.gener_batch_size, device="cuda:0"):

    gen_step = 0

    generator = generator.train()
    discriminator = discriminator.train()

    train_loader= ##training dataset
    global_steps = 0

    for index, (img, _) in enumerate(train_loader):

        real_imgs = img.type(torch.cuda.FloatTensor)

        optim_dis.zero_grad()
        real_valid=discriminator(real_imgs)
        fake_imgs = generator("real data").detach() #change here

        fake_valid = discriminator(fake_imgs)

        if args.loss == 'hinge':
            loss_dis = torch.mean(nn.ReLU(inplace=True)(1.0 - real_valid)).to(device) + torch.mean(nn.ReLU(inplace=True)(1 + fake_valid)).to(device)
        elif args.loss == 'wgangp':
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs.detach(), args.phi)
            loss_dis = -torch.mean(real_valid) + torch.mean(fake_valid) + gradient_penalty * 10 / (args.phi ** 2)
        elif args.loss == 'wgangp_eps':
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs.detach(), args.phi)
            loss_dis = -torch.mean(real_valid) + torch.mean(fake_valid) + gradient_penalty * 10 / (args.phi ** 2)         
            loss_dis += (torch.mean(real_valid) ** 2) * 1e-3 

        loss_dis.backward()
        optim_dis.step()

        if global_steps % n_critic == 0:

            optim_gen.zero_grad()
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)

            
            generated_samples= generator("real data")
            fake_valid = discriminator(generated_samples)

            gener_loss = -torch.mean(fake_valid).to(device)
            gener_loss.backward()
            optim_gen.step()
        

            gen_step += 1

for epoch in range(args.epoch):

    lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None

    train(generator, discriminator, optim_gen, optim_dis,
        epoch, schedulers, n_critic = args.n_critic,
        gener_batch_size=args.gener_batch_size, device="cuda:0")


