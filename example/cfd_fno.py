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
data_path = "./dataset/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
np.random.seed(2300)
with h5py.File(data_path, "r") as f:

    data_array = np.zeros([10000, 21, 128, 128, 1])
    data_array[..., 0] = np.array(f['density'], dtype=np.float32)[:10000,:,:,:]
    data_array = data_array.transpose((0, 1, 4, 2, 3))
    x = np.array(f['x-coordinate'], dtype=np.float32)
    y = np.array(f['y-coordinate'], dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y, indexing='ij')
    grid_x = np.expand_dims(np.expand_dims(grid_x, 0), 0).repeat(10000, 0)
    grid_y = np.expand_dims(np.expand_dims(grid_y, 0), 0).repeat(10000, 0)


train_num = 7500
t = 20
class MyDataset(Dataset):
    def __init__(self, data, grid_x, grid_y):
        self.data = data
        self.grid_x = grid_x
        self.grid_y = grid_y

    def __getitem__(self, index):
        # print(self.data[index, :20, 0].shape,
        #         self.grid_x[index].shape,
        #         self.grid_y[index].shape,
        #       self.data[index, 20].shape)
        if bkd.backend_name == 'jax':
            return bkd.np_to_tensor(
                np.concatenate([
                    self.data[index, :20, 0],
                    self.grid_x[index],
                    self.grid_y[index],
                ], 1).transpose((0, 2, 3, 1))), \
                   bkd.np_to_tensor(self.data[index, 20].transpose((0, 2, 3, 1)))

        return bkd.np_to_tensor(
            np.concatenate([
                self.data[index, :20, 0],
                self.grid_x[index],
                self.grid_y[index],
            ], 0).transpose((1, 2, 0))), \
               bkd.np_to_tensor(self.data[index, 20].transpose((1, 2, 0)))

    def __len__(self):
        return len(self.data)


if bkd.backend_name == 'jax':
    train_loader = DataLoader(MyDataset(data_array[:train_num], grid_x, grid_y), batch_size=128, shuffle=False, backend='jax')
    val_loader = DataLoader(MyDataset(data_array[train_num:], grid_x, grid_y), batch_size=128, shuffle=False, backend='jax')
else:
    train_loader = DataLoader(MyDataset(data_array[:train_num], grid_x, grid_y), batch_size=128, shuffle=False)
    val_loader = DataLoader(MyDataset(data_array[train_num:], grid_x, grid_y), batch_size=128, shuffle=False)


activation = "Gelu"
loss_function = "MSE"
optimizer = "Adam"
lr = 1e-3

max_iteration = 500
save_path = 'model/cfd2d_%s_fno' % bkd.backend_name
# save_path = None


net = fno.FNO2d(in_channels=t+2, out_channels=1, modes1=12, modes2=12, width=20, activation=activation)


start = time.time()
train2d(input=train_loader, evaluation=val_loader, net=net, iterations=max_iteration, scheduler_step=100, scheduler_gamma=0.5,
        optimizer=optimizer, lr=lr, loss_function=loss_function, path=save_path, report_interval=10)

print(time.time() - start)