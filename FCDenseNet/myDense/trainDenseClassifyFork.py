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
 @Author = 'steven'   @DateTime = '2019/3/7 13:45'
'''
import keras.backend as K
from keras.optimizers import Adam
import os

from keras.utils import plot_model
from natsort import natsorted
from glob import glob
from skimage.transform import resize
import numpy as np
from Config import Config
from kerasTools import visualLoss, recall, precision
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from FCDenseNet.myDense.dense import dense3DSemi
from keras.losses import categorical_crossentropy, binary_crossentropy, mse
import pickle as pkl

config = Config()
config.dataRootp = r'E:\pyWorkspace\NewCAE\data\res\highSuvBlock'
config.epochs = 2000
config.lrReduceRate = 0.1
config.lrReducePatience = 20
config.estpPatient = 30
config.estpDelta = 5e-5
config.expRoot = 'classExperiment'


def openPkl(filePath):
    with open(filePath, 'rb') as f:
        data = pkl.load(f)
        dy = data['y']
        dx = data['x']
        if dx.shape != (32, 32, 32, 3):
            dx = np.stack([resize(data['x'][:, :, :, 0], (32, 32, 32), preserve_range=True),
                           resize(data['x'][:, :, :, 1], (32, 32, 32), preserve_range=True),
                           resize(data['x'][:, :, :, 2], (32, 32, 32), preserve_range=True)], axis=-1)
        return dx, dy


def semiDatagene(mode='train', batchSize=5):
    if mode == 'train':
        x1Paths = natsorted(glob(os.path.join(r'E:\pyWorkspace\CAE\res\highSuvArgued', '*')))
        np.random.shuffle(x1Paths)
        x2Paths = natsorted(glob(os.path.join(r'E:\pyWorkspace\CAE\res\highSuvBlock\test', '*')))
        np.random.shuffle(x2Paths)
        x2Paths = x2Paths[:len(x1Paths)]
    else:
        x1Paths = natsorted(glob(os.path.join(r'E:\pyWorkspace\CAE\res\highSuvBlock\test', '*')))
        x2Paths = natsorted(glob(os.path.join(r'E:\pyWorkspace\CAE\res\highSuvBlock\test', '*')))
    index = 0
    x1 = []
    x2 = []
    y = []
    iterTimes = 0

    while True:
        _p1 = x1Paths[index]
        _p2 = x2Paths[index]
        dx1, dy1 = openPkl(_p1)
        dx2, _ = openPkl(_p2)
        x1.append(dx1)
        x2.append(dx2)
        y.append((np.sum(dy1) > 2).astype(np.int))

        if len(x1) == batchSize:
            yield [np.array(x1), np.array(x2)], [np.array(y), np.array(x2)]
            iterTimes += 1
            x1 = []
            x2 = []
            y = []
        if index + 1 >= len(x1Paths):
            print(mode + ' itertimes:' + str(iterTimes))
        index = (index + 1) % len(x1Paths)


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(y_true * K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = (true_positives + K.epsilon() * (1e-3)) / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = (true_positives + K.epsilon() * (1e-3)) / (predicted_positives + K.epsilon())
    return precision


def mmse(y_true, y_pred):
    pass


if __name__ == '__main__':
    from keras.layers import Dense, MaxPool3D, Conv3D, Flatten
    from keras.models import Model

    batchSize = 2
    model = dense3DSemi(input_shape=(32, 32, 32, 3), dropout_rate=0.4, nb_dense_block=5, nb_layers_per_block=9,
                        growth_rate=16, semi_growth_rate=1, semi_layers_per_block=2,
                        classes=512, activation='elu', initDis='RandomNormal')
    model.compile(Adam(lr=1e-3), {'dense_5': binary_crossentropy, 'conv3d_2': mse},
                  loss_weights={'dense_5': 0.99, 'conv3d_2': 0.01}, metrics={'dense_5': [recall, precision]})

    plot_model(model, 'semiArcFork.png', show_shapes=True, rankdir='TB')
    model.summary()
    cpr = os.path.join(config.expRoot, 'checkPoint')
    if not os.path.exists(cpr):
        os.makedirs(cpr)
    mcp = ModelCheckpoint(
        os.path.join(cpr,
                     r'AugClassifyFork_{epoch:03d}-{val_loss:.6f}-{val_dense_5_loss:.6f}-{val_conv3d_2_loss:.6f}-{dense_5_recall:.6f}-{dense_5_precision:.6f}.hdf5'),
        'val_loss', period=2)
    logP = os.path.join(config.expRoot, 'log')
    if not os.path.exists(logP):
        os.makedirs(logP)
    logger = CSVLogger(os.path.join(logP, 'log .txt'))
    lrReduce = ReduceLROnPlateau(factor=config.lrReduceRate, patience=config.lrReducePatience, verbose=1)
    estp = EarlyStopping(patience=config.estpPatient, verbose=1, min_delta=config.estpDelta, )
    model.fit_generator(semiDatagene(mode='train', batchSize=batchSize), steps_per_epoch=int(np.ceil(3020 / batchSize)),
                        epochs=config.epochs,
                        callbacks=[logger, mcp, lrReduce, estp, ],
                        validation_data=semiDatagene(mode='test', batchSize=batchSize),
                        validation_steps=int(np.ceil(4809 / batchSize)))

    visualLoss(os.path.join(logP, 'log.txt'))
