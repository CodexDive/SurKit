#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import time
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="")
 
parser.add_argument('--backend', type=str, default="pytorch")
parser.add_argument('--cuda', type=str, default="3")

args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
# choose backend from pytorch, oneflow, jax
os.environ['SRK_BACKEND'] = args.backend

import surkit
import surkit.backend as bkd
from surkit.nn import afno
from surkit.train import train2d

if bkd.backend_name == 'jax':
    from jax_dataloader import Dataset, DataLoader
elif bkd.backend_name == 'pytorch':
    from torch.utils.data import Dataset, DataLoader
elif bkd.backend_name == 'oneflow':
    from oneflow.utils.data import Dataset, DataLoader

import h5py
data_path = "./dataset/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
np.random.seed(2300)
with h5py.File(data_path, "r") as f:
    data_array= np.array(f['density'], dtype=np.float32)[:10000,:,:,:]
    data = data_array

train_num = 7500
t = 20
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return bkd.np_to_tensor(self.data[index, :20]), bkd.np_to_tensor(self.data[index, 20:])

    def __len__(self):
        return len(self.data)

if bkd.backend_name == 'jax':
    train_loader = DataLoader(MyDataset(data[:train_num]), batch_size=128, shuffle=False, backend='jax')
    val_loader = DataLoader(MyDataset(data[train_num:]), batch_size=128, shuffle=False, backend='jax')
else:
    train_loader = DataLoader(MyDataset(data[:train_num]), batch_size=128, shuffle=False)
    val_loader = DataLoader(MyDataset(data[train_num:]), batch_size=128, shuffle=False)


activation = "Gelu"
loss_function = "MSE"
optimizer = "Adam"
lr = 1e-4

max_iteration = 200
save_path = 'model/cfd2d_%s_afno' % bkd.backend_name
# save_path = None


net = afno.AFNONet(in_channels=t, out_channels=1)


start = time.time()
train2d(input=train_loader, evaluation=val_loader, net=net, iterations=max_iteration, scheduler_step=100, scheduler_gamma=0.5,
        optimizer=optimizer, lr=lr, loss_function=loss_function, path=save_path, report_interval=10, multi_gpu=',' in args.cuda)

print(time.time() - start)