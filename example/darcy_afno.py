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
data_path = "./dataset/2D_DarcyFlow_beta0.1_Train.hdf5"
np.random.seed(2300)
with h5py.File(data_path, "r") as f:

    in_array = np.expand_dims(np.array(f['nu'], dtype=np.float32), 1)
    out_array = np.array(f['tensor'], dtype=np.float32)
    index = np.random.choice(range(len(in_array) - 1), 1000)
    in_t = bkd.np_to_tensor(in_array[index])
    out_t = bkd.np_to_tensor(out_array[index])

train_num = 750

class MyDataset(Dataset):
    def __init__(self, in_t, out_t):
        self.in_t = in_t
        self.out_t = out_t

    def __getitem__(self, index):
        # print(self.data[index].shape)
        return self.in_t[index], self.out_t[index]

    def __len__(self):
        return len(self.in_t)


if bkd.backend_name == 'jax':
    train_loader = DataLoader(MyDataset(in_t[:train_num], out_t[:train_num]), batch_size=8, shuffle=False, backend='jax')
    val_loader = DataLoader(MyDataset(in_t[train_num:], out_t[train_num:]), batch_size=8, shuffle=False, backend='jax')
else:
    train_loader = DataLoader(MyDataset(in_t[:train_num], out_t[:train_num]), batch_size=8, shuffle=False)
    val_loader = DataLoader(MyDataset(in_t[train_num:], out_t[train_num:]), batch_size=8, shuffle=False)

activation = "Gelu"
loss_function = "MSE"
optimizer = "Adam"
lr = 1e-4

max_iteration = 500
save_path = 'model/darcy_%s_afno' % bkd.backend_name
# save_path = None


net = afno.AFNONet(in_channels=1, out_channels=1)

start = time.time()
train2d(input=train_loader, evaluation=val_loader, net=net, iterations=max_iteration, scheduler_step=10, scheduler_gamma=0.5,
    optimizer=optimizer, lr=lr, loss_function=loss_function, path=save_path, report_interval=10)

print(time.time() - start)