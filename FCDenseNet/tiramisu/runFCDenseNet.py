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

import os
from natsort import natsorted
from glob import glob
import numpy as np
from skimage import io as skio
from Config import Config
from kerasTools import visualLoss, recall, precision
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
import keras.backend as K
import tensorflow as tf

config = Config()
config.dataRootp = r'E:\pyWorkspace\CAE\res\cp250'
config.epochs = 2000
config.lrReduceRate = 0.1
config.lrReducePatience = 20
config.estpPatient = 30
config.estpDelta = 5e-5
config.expRoot = 'experiment'


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
    if mode == 'train':
        dataList = natsorted(glob(os.path.join(config.dataRootp, '*')))[:40]
    else:
        dataList = natsorted(glob(os.path.join(config.dataRootp, '*')))[40:45]

    while True:
        for _patientRoot in dataList:
            x = []
            y = []
            suvs = np.stack([skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'suv', '*')))])
            cts = np.stack([skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'ct', '*')))])
            highAreas = np.stack(
                [skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'highAreaInfo', '*')))])
            labels = np.stack(
                [skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'labelClear', '*')))]) / 255
            cts = (cts - cts.min()) / (cts.max() - cts.min())
            highAreas = np.log(highAreas, where=(highAreas != 0)) / np.log(100)
            suvs = np.pad(suvs, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)
            cts = np.pad(cts, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)
            labels = np.pad(labels, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)
            highAreas = np.pad(highAreas, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)

            for i in range(0, suvs.shape[0]):
                suv = suvs[i, :, :]
                highArea = highAreas[i, :, :]
                ct = cts[i, :, :]
                # x.append(np.stack([suv, ct, highArea], axis=-1))
                # x.append(np.stack([suv,highArea], axis=-1))
                x.append(suv[:, :, np.newaxis])
                y.append(labels[i, :, :])
            x = np.array(x)
            y = np.array(y)[:, :, :, np.newaxis]
            # y=to_categorical(np.array(y),num_classes=2)
            sliceNum = 1
            for i in range(0, suvs.shape[0], sliceNum):
                yield x[i:i + sliceNum, :, :, :], y[i:i + sliceNum, :, :]


def conGene(mode='train'):
    import pickle as pkl
    # 共有9787张切片
    with open(r'E:\pyWorkspace\NewCAE\data\path.pkl', 'rb')  as f:
        allPath = pkl.load(f)
    if mode == 'train':
        allPath = allPath[:int(9787 / 10 * 8)]
    else:
        allPath = allPath[int(9787 / 10 * 8):]
    sliceNum = 1
    allSuvs = []
    allCts = []
    allHighs = []
    allLabels = []
    for i, _p in enumerate(allPath):
        ct =  np.pad((skio.imread(_p[0])+250)/500, [ [3, 3], [3, 3]], mode='constant', constant_values=0)
        suv =  np.clip(np.pad(skio.imread(_p[1]), [[3, 3], [3, 3]], mode='constant', constant_values=0)+1,1,500)

        suv = np.log(suv, where=(suv>0)) / np.log(10)
        label = np.pad(skio.imread(_p[2]), [ [3, 3], [3, 3]], mode='constant', constant_values=0)
        label = (label > 0.5).astype(np.int8)
        high =  np.pad(skio.imread(_p[3]), [ [3, 3], [3, 3]], mode='constant', constant_values=0)
        high[high==0]=1
        high = np.log(high, where=(high > 0)) / np.log(100)
        allSuvs .append(suv)
        allCts.append(ct)
        allHighs.append(high)
        allLabels.append(label)
        # print('-------------------------')
        # print(_p)
        # print(np.max(allSuvs))
        # print(np.max(allCts))
        # print(np.max(allHighs))
        # print(np.max(allLabels))

        if (i+1)%sliceNum==0 and i!=0:

            allSuvs=np.array(allSuvs)
            allCts =np.array(allCts)
            allHighs =np.array(allHighs)
            allLabels =  np.array(allLabels)
            # print(allSuvs.shape)
            # print(allCts.shape)
            # print(allHighs.shape)
            # print(allLabels.shape)
            # print('--------')
            x= np.stack([allSuvs,allCts,allHighs],axis=3)



            y=allLabels[:,:,:,np.newaxis]
            # print(np.unique(y))
            # print('x')
            # print(x.shape)
            # print('y')
            # print(y.shape)
            yield x,y
            allSuvs=[]
            allCts=[]
            allHighs = []
            allLabels = []



# def dice(y_true, y_pre, smooth=1e-7):
#     # y_pre=K.clip(y_pre,0,1)
#     return (K.sum(2. * (y_true * y_pre)) + smooth) / (K.sum(y_true) + K.sum(y_pre) + smooth)
def dice(y_true, y_pre, smooth=1e-7):
    # y_pre=K.clip(y_pre,0,1)
    return (K.sum(2. * (y_true * K.round(y_pre))) + smooth) / (K.sum(y_true) + K.sum(K.round(y_pre)) + smooth)


def tversky(y_true, y_pred, smooth=1e-7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 1e-5*99999
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)


def focal_loss(gamma=2., alpha=.0089):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


if __name__ == '__main__':
    from keras_contrib.applications.densenet import DenseNetFCN

    # model = DenseNetFCN(input_shape=(256, 256, 3), nb_dense_block=3, nb_layers_per_block=3, dropout_rate=0.8,
    #                     reduction=0.5, initial_kernel_size=(3, 3))

    model = DenseNetFCN(input_shape=(256, 256, 3), )
    # model = Tiramisu(n_classes=1, input_shape=(256, 256, 3), )
    model.compile('adam', focal_tversky, metrics=['acc', dice, recall, precision])
    # model.compile('adam', binary_crossentropy, metrics=['acc', dice, recall, precision])

    cpr = os.path.join('experiment', 'checkPoint')
    if not os.path.exists(cpr):
        os.makedirs(cpr)
    mcp = ModelCheckpoint(
        os.path.join(cpr,
                     r'l_shallow_{epoch:03d}-{val_loss:.6f}-{val_dice:.6f}-{val_recall:.6f}-{val_precision:.6f}.hdf5'),
        'val_loss')
    logP = os.path.join(config.expRoot, 'log')
    if not os.path.exists(logP):
        os.makedirs(logP)
    logger = CSVLogger(os.path.join(logP, 'log .txt'))
    lrReduce = ReduceLROnPlateau(factor=config.lrReduceRate, patience=config.lrReducePatience, verbose=1)
    estp = EarlyStopping(patience=config.estpPatient, verbose=1, min_delta=config.estpDelta, )
    model.fit_generator(conGene('train'), steps_per_epoch=9787 / 10 * 8, epochs=config.epochs,
                        callbacks=[logger, mcp, lrReduce, estp, ], validation_data=datagene('test'),
                        validation_steps=9787 / 10 * 2)

    visualLoss(os.path.join(logP, 'log.txt'))
