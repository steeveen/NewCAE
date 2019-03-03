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
 @Author = 'steven'   @DateTime = '2019/3/1 10:43'
'''
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
 @Author = 'steven'   @DateTime = '2019/2/28 22:16'
'''

import os
from natsort import natsorted
from glob import glob
import numpy as np
from skimage import io as skio
from Config import Config
from kerasTools import visualLoss, recall, precision
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from keras.losses import binary_crossentropy, categorical_crossentropy
import keras.backend as K

config = Config()
config.dataRootp = r'E:\pyWorkspace\CAE\res\cp250'
config.epochs = 2000
config.lrReduceRate = 0.1
config.lrReducePatience = 20
config.estpPatient = 30
config.estpDelta = 5e-5
config.expRoot = 'experimentFirstVGG'


def datagene(mode='train'):
    if mode == 'train':
        dataList = natsorted(glob(os.path.join(config.dataRootp, '*')))[:70]
    else:
        dataList = natsorted(glob(os.path.join(config.dataRootp, '*')))[70:80]

    while True:
        for _patientRoot in dataList:
            x = []
            y = []
            suvs = np.stack([skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'suv', '*')))])
            # cts = np.stack([skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'ct', '*')))])
            # highAreas = np.stack(
            #     [skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'highAreaInfo', '*')))])
            labels = np.stack(
                [skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'labelClear', '*')))])
            # cts = (cts - cts.min()) / (cts.max() - cts.min())
            # highAreas = np.log(highAreas, where=(highAreas != 0)) / np.log(100)
            suvs = np.pad(suvs, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)
            suvs = np.clip(suvs, 0, 500)
            suvs = np.log(suvs, where=(suvs != 0)) / np.log(10)
            # cts = np.pad(cts, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)
            labels = np.pad(labels, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)
            # highAreas = np.pad(highAreas, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)

            for i in range(0, suvs.shape[0]):
                suv = suvs[i, :, :]
                # highArea = highAreas[i, :, :]
                # ct = cts[i, :, :]
                # x.append(np.stack([suv, ct, highArea], axis=-1))
                # x.append(np.stack([suv,highArea], axis=-1))
                x.append(np.stack([suv, suv, suv], axis=-1))
                y.append(np.any(labels[i, :, :] >= 0.5))
            x = np.array(x)
            x=(x-x.min())/(x.max()-x.min())
            y = np.array(y)
            # y=to_categorical(np.array(y),num_classes=2)
            sliceNum = 3
            # yield x,y
            print(x.shape)
            print(y.shape)
            for i in range(0, suvs.shape[0], sliceNum):
                yield x[i:i + sliceNum, :, :, :], y[i:i + sliceNum]


def buildModel():
    from keras_contrib.applications import ResNet101, ResNet152
    from keras.applications import VGG19
    from keras.layers import Dense, Flatten, BatchNormalization
    from keras.models import Model
    vgg = VGG19(input_shape=(256, 256, 3), classes=1, include_top=False,
                weights=r'C:\Users\lenovo\Desktop\vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    vgg.trainable = False
    o = Flatten()(vgg.output)
    o = Dense(2048, activation='relu')(o)
    o = BatchNormalization()(o)
    o = Dense(2048, activation='relu')(o)
    o = BatchNormalization()(o)
    o = Dense(1024, activation='relu')(o)
    o = BatchNormalization()(o)
    o = Dense(512, activation='relu')(o)
    o = BatchNormalization()(o)
    o = Dense(1, activation='sigmoid')(o)
    model = Model(vgg.inputs, o)
    return model


def train():
    model = buildModel()
    model.compile('adam', binary_crossentropy, metrics=['acc', recall, precision])

    cpr = os.path.join(config.expRoot, 'checkPoint')
    if not os.path.exists(cpr):
        os.makedirs(cpr)
    mcp = ModelCheckpoint(os.path.join(cpr, r'{epoch:03d}-{val_loss:.6f}-{val_recall:.6f}-{val_precision:.6f}.hdf5'),
                          'val_loss', period=1)
    logP = os.path.join(config.expRoot, 'log')
    if not os.path.exists(logP):
        os.makedirs(logP)
    logger = CSVLogger(os.path.join(logP, 'log .txt'))
    lrReduce = ReduceLROnPlateau(factor=config.lrReduceRate, patience=config.lrReducePatience, verbose=1)
    estp = EarlyStopping(patience=config.estpPatient, verbose=1, min_delta=config.estpDelta, )
    model.fit_generator(datagene('train'), class_weight={0: 1, 1: 3}, steps_per_epoch=70 * 558 / 3,
                        epochs=config.epochs,
                        callbacks=[logger, mcp, lrReduce, estp, ], validation_data=datagene('test'),
                        validation_steps=10 * 558 / 3)

    visualLoss(os.path.join(logP, 'log.txt'))


if __name__ == '__main__':
    # from keras_contrib.applications.densenet import DenseNetFCN
    train()
