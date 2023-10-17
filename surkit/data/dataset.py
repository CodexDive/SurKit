#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import numpy as np

def is_number(s):
    try:
        float(s)
        return True
    except:
        return False

def load_dataset(path, key=None):
    import csv

    with open(path, "r", encoding='UTF-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        all_number = all(is_number(x) for x in header)
        if all_number and not key:
            raise "Need to know the match between the data and the variable name."
        if all_number:
            data.append(header)
        # 遍历每一行
        for row in reader:
            data.append(row)

    data = np.array(data, dtype=float)
    output = {}
    if not all_number:
        for i, v in enumerate(header):
            output[v] = data[:, i].reshape(-1, 1)
        return output
    elif key:
        for i, v in enumerate(key):
            output[v] = data[:, i].reshape(-1, 1)
        return output
    else:
        return data