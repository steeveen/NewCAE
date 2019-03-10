# -*- coding: utf-8 -*-
# 统计 E:\pyWorkspace\NewCAE\data\res\highSuvBlock中的正例和反例数量


from natsort import natsorted
from glob import glob
import os
import pickle as pkl
import numpy as np


def sta( mode='train'):
    posCenter = 0
    negCenter = 0
    posSum = 0
    negSum = 0
    for _ in natsorted(glob(os.path.join(r'E:\pyWorkspace\CAE\res\highSuvBlock', mode, '*'))):
        with open(_, 'rb') as f:
            y = pkl.load(f)['y']
            if np.sum(y) > np.size(y)*0.1:
                posSum += 1
            else:
                negSum += 1
            if y[y.shape[0] // 2+1, y.shape[1] // 2+1, y.shape[2] // 2+1] > 0:
                posCenter += 1
            else:
                negCenter += 1
    return posCenter, negCenter, posSum, negSum


if __name__ == '__main__':
    print('train')
    print(sta('train'))
    print('test')
    print(sta('test'))
