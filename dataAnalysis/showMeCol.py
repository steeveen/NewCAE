# -*- coding: utf-8 -*-
'''
        司马懿：“善败能忍，然厚积薄发”
                                    ——李叔说的
code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
          --┃      ☃      ┃--
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗II━II┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
 @Belong = 'NewCAE'  @MadeBy = 'PyCharm'
 @Author = 'steven'   @DateTime = '2019/3/17 19:23'
'''
from natsort import natsorted
from glob import glob
import os
from skimage import io as skio
import numpy as np

ip = r'E:\pyWorkspace\CAE\res\cp250'
op = r'E:\pyWorkspace\CAE\res\cp250Col'
os.mkdir(op) if not os.path.exists(op) else None
for _p in  natsorted(glob(os.path.join(ip, '*')))[50:]:
    patientOp = os.path.join(op, os.path.basename(_p))
    os.mkdir(patientOp) if not os.path.exists(patientOp) else None
    ctOp = os.path.join(patientOp, 'ct')
    os.mkdir(ctOp) if not os.path.exists(ctOp) else None
    gtOp = os.path.join(patientOp, 'labelClear')
    os.mkdir(gtOp) if not os.path.exists(gtOp) else None
    suvOp = os.path.join(patientOp, 'suv')
    os.mkdir(suvOp) if not os.path.exists(suvOp) else None
    cts = np.stack([skio.imread(_) for _ in natsorted(glob(os.path.join(_p, 'ct', '*')))])
    cts = (cts - cts.min()) / (cts.max() - cts.min())
    for i in range(cts.shape[1]):
        skio.imsave(os.path.join(ctOp, str(i) + '.jpg'), cts[:, i, :])
    suvs = np.stack([skio.imread(_) for _ in natsorted(glob(os.path.join(_p, 'suv', '*')))])
    suvs=np.clip(suvs,0,5)
    suvs = (suvs - suvs.min()) / (suvs.max() - suvs.min())
    for i in range(suvs.shape[1]):
        skio.imsave(os.path.join(suvOp, str(i) + '.jpg'), suvs[:, i, :, ])
    gt = np.stack([skio.imread(_) for _ in natsorted(glob(os.path.join(_p, 'labelClear', '*')))])
    gt = (gt - gt.min()) / (gt.max() - gt.min())
    for i in range(gt.shape[1]):
        skio.imsave(os.path.join(gtOp, str(i) + '.jpg'), gt[:, i, :])
