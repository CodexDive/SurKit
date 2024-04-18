#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import time
import os
import argparse
 
parser = argparse.ArgumentParser(description="")
 
parser.add_argument('--backend', type=str, default="pytorch")
parser.add_argument('--cuda', type=str, default="3")
parser.add_argument('--lr', type=str, default="1e-4")
parser.add_argument('--loss', type=str, default="dice and bce")
parser.add_argument('--seed', type=int, default=2300)

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

from dataset_Nuclie import Nuclie_data


train_set = Nuclie_data("./dataset/kaggle-dsbowl-2018-dataset-fixes-master/stage1_train")
eval_set = Nuclie_data("./dataset/kaggle-dsbowl-2018-dataset-fixes-master/stage1_train", False)

batch_size = 2048

if bkd.backend_name == 'jax':
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, backend='jax')
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False, backend='jax')
else:
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)

activation = "sigmoid"
loss_function = args.loss
optimizer = "Adam"
lr = float(args.lr)
max_iteration = 2000
save_path = 'model/image_%s_anfo' % bkd.backend_name
# save_path = None


net = afno.AFNONet(img_size=96, in_channels=3, out_channels=1)

start = time.time()
train2d(input=train_loader, evaluation=eval_loader, net=net, iterations=max_iteration,
    optimizer=optimizer, lr=lr, loss_function=loss_function, path=save_path, report_interval=10, classify=True, seed=args.seed)

print(time.time() - start)
