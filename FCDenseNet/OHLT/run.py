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
 @Author = 'steven'   @DateTime = '2019/2/28 11:04'
'''
from keras import models
from keras.callbacks import LearningRateScheduler,  ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import RMSprop
import math

from Config import Config
from natsort import natsorted
from glob import glob
import numpy as np
import os
import keras.backend as K
from skimage import io as skio

from FCDenseNet.OHLT.fcDensenetModel import Tiramisu

config = Config()


def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.00001
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

lrate = LearningRateScheduler(step_decay)

def datagene(mode='train'):
    if mode == 'train':
        dataList = natsorted(glob(os.path.join(config.dataRootp , '*')))[:40]
    else:
        dataList = natsorted(glob(os.path.join(config.dataRootp , '*')))[40:45]

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
            cts=(cts-cts.min())/(cts.max()-cts.min())
            highAreas=np.log(highAreas, where=(highAreas != 0)) / np.log(100)
            suvs = np.pad(suvs, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)
            cts = np.pad(cts, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)
            labels = np.pad(labels, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)
            highAreas = np.pad(highAreas, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)


            for i in range(0, suvs.shape[0]):
                suv = suvs[i, :, :]
                highArea = highAreas[i, :, :]
                ct = cts[i, :, :]
                x.append(np.stack([suv, ct, highArea], axis=-1))
                y.append(labels[i, :, :])
            x = np.array(x)
            y = np.array(y)[:, :, :, np.newaxis]
            # y=to_categorical(np.array(y),num_classes=2)
            sliceNum = 1
            for i in range(0, suvs.shape[0], sliceNum):
                yield x[i:i + sliceNum, :, :, :], y[i:i + sliceNum, :, :]


def dice(y_true, y_pre, smooth=1e-7):
    # y_pre=K.clip(y_pre,0,1)
    return (K.sum(2. * (y_true *  K.round(y_pre))) + smooth) / (K.sum(y_true) + K.sum( K.round(y_pre)) + smooth)

def tversky(y_true, y_pred,smooth=1e-7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)
def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)


if __name__ == '__main__':
    config.dataRootp = r'E:\pyWorkspace\CAE\res\cp250'
    config.epochs = 2000
    config.lrReduceRate = 0.1
    config.lrReducePatience = 20
    config.estpPatient = 30
    config.estpDelta = 5e-5
    config.expRoot = 'experiment'

    # with open('tiramisu_fc_dense103_model.json') as model_file:
    #     tiramisu = models.model_from_json(model_file.read())

    tiramisu=Tiramisu(1,img_dim=(250,250,3))
    optimizer = RMSprop(lr=0.001, decay=0.0000001)

    tiramisu.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    cpr = os.path.join('experiment', 'checkPoint')
    if not os.path.exists(cpr):
        os.makedirs(cpr)
    mcp = ModelCheckpoint(
        os.path.join(cpr,
                     r'tlms158_{epoch:03d}-{val_loss:.6f}-{val_dice:.6f}-{val_recall:.6f}-{val_precision:.6f}.hdf5'),
        'val_loss')
    logP = os.path.join(config.expRoot, 'log')
    if not os.path.exists(logP):
        os.makedirs(logP)
    logger = CSVLogger(os.path.join(logP, 'log.txt'))
    lrReduce = ReduceLROnPlateau(factor=config.lrReduceRate, patience=config.lrReducePatience, verbose=1)
    estp = EarlyStopping(patience=config.estpPatient, verbose=1, min_delta=config.estpDelta, )
    tiramisu.fit_generator(datagene('train'), steps_per_epoch=558 * 40, epochs=config.epochs,
                        callbacks=[logger, mcp, lrReduce, estp, ], validation_data=datagene('test'),
                        validation_steps=558 * 5)