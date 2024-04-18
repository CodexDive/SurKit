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
from surkit.nn import fno
from surkit.train import train2d

if bkd.backend_name == 'jax':
    from jax_dataloader import Dataset, DataLoader
elif bkd.backend_name == 'pytorch':
    from torch.utils.data import Dataset, DataLoader
elif bkd.backend_name == 'oneflow':
    from oneflow.utils.data import Dataset, DataLoader

import h5py

data_path = "./dataset/2D_rdb_NA_NA.h5"
np.random.seed(2300)
with h5py.File(data_path, "r") as f:
    keys = list(f.keys())
    data_arrays = [(np.array(f[key]['data'], dtype=np.float32)[0], np.array(f[key]['data'], dtype=np.float32)[-1])  for key in keys]
    x, y = np.array(f['0000']['grid']['x'], dtype=np.float32), np.array(f['0000']['grid']['y'],
                                                                                dtype=np.float32)
    data = bkd.np_to_tensor(np.stack(data_arrays, axis=0))
    grid_x, grid_y = np.meshgrid(x, y, indexing='ij')
    grid_x = np.expand_dims(grid_x, 0).repeat(len(data_arrays), 0)
    grid_y = np.expand_dims(grid_y, 0).repeat(len(data_arrays), 0)
    grid = bkd.cat((bkd.unsqueeze(bkd.np_to_tensor(grid_x), -1), bkd.unsqueeze(bkd.np_to_tensor(grid_y), -1)), -1)


class MyDataset(Dataset):
    def __init__(self, data, grid):
        self.data = data
        self.grid = grid

    def __getitem__(self, index):
        # print(self.data[index].shape)
        # print(self.grid[index].shape)
        if bkd.backend_name == "jax":
            # print(self.grid[index].shape)
            return bkd.cat([self.data[index, 0], self.grid[index]], -1), self.data[index, -1]
        return bkd.cat([self.data[index][0], self.grid[index]], -1), self.data[index][-1]

    def __len__(self):
        return len(self.data)

train_num = 750

if bkd.backend_name == 'jax':
    train_loader = DataLoader(MyDataset(data[:train_num], grid), batch_size=8, shuffle=False, backend='jax')
    val_loader = DataLoader(MyDataset(data[train_num:], grid), batch_size=8, shuffle=False, backend='jax')
else:
    train_loader = DataLoader(MyDataset(data[:train_num], grid), batch_size=8, shuffle=False)
    val_loader = DataLoader(MyDataset(data[train_num:], grid), batch_size=8, shuffle=False)

activation = "Gelu"
loss_function = "MSE"
optimizer = "Adam"
lr = 1e-3

max_iteration = 500
save_path = 'model/swe2d_%s_fno' % bkd.backend_name
# save_path = None


net = fno.FNO2d(modes1=12, modes2=12, width=32, in_channels=3, out_channels=1, activation=activation)

start = time.time()
train2d(input=train_loader, evaluation=val_loader, net=net, iterations=max_iteration,
    optimizer=optimizer, lr=lr, loss_function=loss_function, path=save_path, report_interval=10)

print(time.time() - start)