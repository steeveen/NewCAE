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
 @Author = 'steven'   @DateTime = '2019/3/23 16:39'
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
from FCDenseNet.myDense.dense import dense2DSemi
from keras.layers import Dropout
from keras.losses import categorical_crossentropy, binary_crossentropy, mse
import pickle as pkl
from keras.optimizers import RMSprop

config = Config()
config.epochs = 2000
config.batchSize = 24
config.lrReduceRate = 0.1
config.lrReducePatience = 20
config.estpPatient = 30
config.estpDelta = 5e-5
config.expRoot = 'classExperiment2D3'
# config.testNum=1971
# config.trainNum=17696
config.testNum = 1068
config.trainNum = 17696


def openPkl(filePath):
    # print(filePath)
    try:

        with open(filePath, 'rb') as f:
            data = pkl.load(f)
            dy = (np.sum(data['gt']) > 4).astype(np.int8)
            dx = data['suv'].transpose((1, 2, 0))
            dxPET=data['ct'].transpose((1, 2, 0))
            return np.concatenate([dx,dxPET],axis=-1), dy

    except pkl.UnpicklingError:

        print('Unpick error:' + filePath)
    except:
        print('donnot know exception.' + filePath)


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


def tversky(y_true, y_pred, smooth=1e-7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.8
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    # gamma = 2
    # gamma = 2
    return K.pow((1 - pt_1), 1 / gamma)


def semiDatagene(mode='train', batchSize=5):
    if mode == 'train':
        x1Paths = natsorted(glob(os.path.join(r'E:\pyWorkspace\CAE\res\BigTomurData\train', '*')))
        np.random.seed(0)
        np.random.shuffle(x1Paths)
        x2Paths = natsorted(glob(os.path.join(r'E:\pyWorkspace\CAE\res\BigTomurData\train', '*')))
        np.random.seed(0)
        np.random.shuffle(x2Paths)
    else:
        x1Paths = natsorted(glob(os.path.join(r'E:\pyWorkspace\CAE\res\BigTomurData\test', '*')))
        np.random.seed(0)
        np.random.shuffle(x1Paths)
        x2Paths = natsorted(glob(os.path.join(r'E:\pyWorkspace\CAE\res\BigTomurData\test', '*')))
        np.random.seed(0)
        np.random.shuffle(x2Paths)
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
        y.append(dy1)

        if len(x1) == batchSize:
            yield [(np.array(x1)-np.min(x1))/(np.max(x1)-np.min(x)), (np.array(x2)-np.mean(x2))/np.std(x2)], [np.array(y), (np.array(x2)-np.mean(x2))/np.std(x2)]
            iterTimes += 1
            x1 = []
            x2 = []
            y = []
        if index + 1 >= len(x1Paths):
            print(mode + ' itertimes:' + str(iterTimes))
        index = (index + 1) % len(x1Paths)


def bse(y_t, y_p):
    return binary_crossentropy(y_t, y_p) + mse(y_t, y_p)


if __name__ == '__main__':

    model = dense2DSemi(input_shape=(32, 32, 6), dropout_rate=0.5, nb_dense_block=5, nb_layers_per_block=7,
                        growth_rate=64, semi_growth_rate=8, semi_layers_per_block=5, transition_pooling='max',
                        classes=512, activation='elu', initDis='he_uniform')
    # model.compile(Adam(lr=1e-3), {'dense_5': binary_crossentropy, 'conv2d_2': mse},
    #               loss_weights={'dense_5': 0.99, 'conv2d_2': 0.01}, metrics={'dense_5': [recall, precision]})
    model.compile(RMSprop(lr=1e-3, decay=0.995, ), {'dense_5': bse, 'conv2d_2': mse},
                  loss_weights={'dense_5': 0.99, 'conv2d_2': 0.01}, metrics={'dense_5': [recall, precision]})

    plot_model(model, 'semiArcFork2D.png', show_shapes=True, rankdir='TB')
    model.summary()
    cpr = os.path.join(config.expRoot, 'checkPoint')

    if not os.path.exists(cpr):
        os.makedirs(cpr)
    mcp = ModelCheckpoint(
        os.path.join(cpr,
                     r'AugClassifyFork2D_{epoch:03d}-{val_loss:.6f}-{val_dense_5_loss:.6f}-{val_conv2d_2_loss:.6f}-{dense_5_recall:.6f}-{dense_5_precision:.6f}.hdf5'),
        'val_loss', period=2)
    logP = os.path.join(config.expRoot, 'log')
    if not os.path.exists(logP):
        os.makedirs(logP)
    logger = CSVLogger(os.path.join(logP, 'log.csv'))
    lrReduce = ReduceLROnPlateau(factor=config.lrReduceRate, patience=config.lrReducePatience, verbose=1)
    estp = EarlyStopping(patience=config.estpPatient, verbose=1, min_delta=config.estpDelta, )
    model.fit_generator(semiDatagene(mode='train', batchSize=config.batchSize),
                        steps_per_epoch=int(np.ceil(config.trainNum / config.batchSize)),
                        epochs=config.epochs, class_weight={0: 1, 1: 8},
                        callbacks=[logger, mcp, lrReduce, estp, ],
                        validation_data=semiDatagene(mode='test', batchSize=config.batchSize),
                        validation_steps=int(np.ceil(config.testNum / config.batchSize)))

    # model.fit_generator(semiDatagene(mode='train', batchSize=batchSize), steps_per_epoch=10,
    #                     epochs=config.epochs,
    #                     callbacks=[logger, mcp, lrReduce, estp, ],
    #                     validation_data=semiDatagene(mode='test', batchSize=batchSize),
    #                     validation_steps=10)

    visualLoss(os.path.join(logP, 'log.csv'))
