# -*- coding: utf-8 -*-
'''用于处理一些垃圾的东西，比如删除错的生成数据
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
 @Author = 'steven'   @DateTime = '2019/2/23 20:42'
'''
from keras.utils import to_categorical

from FCDenseNet.tiramisu_net import Tiramisu
import os
from natsort import natsorted
from glob import glob
import numpy as np
from keras.callbacks import CSVLogger
from keras.losses import categorical_crossentropy,binary_crossentropy
from skimage import io as skio
from skimage.morphology import label
from skimage.measure import regionprops

dataRootp = r'E:\pyWorkspace\CAE\res\cp250'


# def datagene(mode='train'):
#     thickness = 64
#     if mode == 'train':
#         dataList = natsorted(glob(os.path.join(dataRootp, '*')))[:2]
#     else:
#         dataList = natsorted(glob(os.path.join(dataRootp, '*')))[2:]
#     while True:
#         for _patientRoot in dataList:
#             suvs = np.stack([skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'suv', '*')))])
#             cts = np.stack([skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'ct', '*')))])
#             highAreas = np.stack(
#                 [skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'highAreaInfo', '*')))])
#             labels = np.stack(
#                 [skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'labelClear', '*')))]) / 255
#             suvs = np.pad(suvs, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)
#             cts = np.pad(cts, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)
#             labels = np.pad(labels, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)
#             highAreas = np.pad(highAreas, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)
#
#             x = []
#             y = []
#             for i in range(0, suvs.shape[0], thickness):
#                 if i + thickness < suvs.shape[0]:
#                     suv = suvs[i:i + thickness, :, :]
#                     highArea = highAreas[i:i + thickness, :, :]
#                     ct = cts[i:i + thickness, :, :]
#                     x.append(np.stack([suv, ct, highArea],axis=-1))
#                     y.append(labels[i:i + thickness, :, :])
#                 else:
#
#                     suv = suvs[suvs.shape[0] - thickness:suvs.shape[0], :, :]
#                     highArea = highAreas[suvs.shape[0] - thickness:suvs.shape[0], :, :]
#                     ct = cts[suvs.shape[0] - thickness:suvs.shape[0], :, :]
#                     x.append(np.stack([suv, ct, highArea],axis=-1))
#                     y.append(labels[suvs.shape[0] - thickness:suvs.shape[0], :, :])
#             yield x,y

def datagene(mode='train'):
    thickness = 64
    if mode == 'train':
        dataList = natsorted(glob(os.path.join(dataRootp, '*')))[:2]
    else:
        dataList = natsorted(glob(os.path.join(dataRootp, '*')))[2:]
    while True:
        for _patientRoot in dataList:
            suvs = np.stack([skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'suv', '*')))])
            cts = np.stack([skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'ct', '*')))])
            highAreas = np.stack(
                [skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'highAreaInfo', '*')))])
            labels = np.stack(
                [skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'labelClear', '*')))]) / 255
            suvs = np.pad(suvs, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)
            cts = np.pad(cts, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)
            labels = np.pad(labels, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)
            highAreas = np.pad(highAreas, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)

            x = []
            y = []
            for i in range(0, suvs.shape[0]):
                    suv = suvs[i, :, :]
                    highArea = highAreas[i, :, :]
                    ct = cts[i, :, :]
                    x.append(np.stack([suv, ct, highArea],axis=-1))
                    y.append(labels[i, :, :])
            x=np.array(x)
            y=np.array(y)[:,:,:,np.newaxis]
            # y=to_categorical(np.array(y),num_classes=2)
            sliceNum=1
            for i in range(0,suvs.shape[0],sliceNum):
                yield x[i:i+sliceNum,:,:,:],y[i:i+sliceNum,:,:]

def dice1(y_true,y_pre,smooth=0.001):
    return 2*(y_true*y_pre)/(np.sum(y_true)+np.sum(y_pre)+smooth)
model = Tiramisu(n_classes=1, input_shape=(256,256,3),)
model.compile('adam', binary_crossentropy, metrics=['acc',dice1])
model.fit_generator(datagene('train'), steps_per_epoch=560, epochs=2, validation_data=datagene('test'),
                    validation_steps=2)
