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
from keras_contrib.applications.densenet import DenseNetImageNet161
import os
from natsort import natsorted
from glob import glob
from skimage.transform import resize
import numpy as np
from skimage import io as skio
from Config import Config
from kerasTools import visualLoss, recall, precision
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
import keras.backend as K
from FCDenseNet.myDense.dense import dense3DClassify
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.utils import plot_model
import pickle as pkl

config = Config()
config.dataRootp = r'E:\pyWorkspace\NewCAE\data\res\highSuvBlock'
config.epochs = 2000
config.lrReduceRate = 0.1
config.lrReducePatience = 20
config.estpPatient = 30
config.estpDelta = 5e-5
config.expRoot = 'classExperiment'


def tversky(y_true, y_pred, smooth=1e-7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.4
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    # gamma = 0.75
    # gamma = 2
    gamma = 2
    return K.pow((1 - pt_1), 1 / gamma)


def localPrecision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    y_true = y_true[:, 1]
    y_pred = y_pred[:, 1]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = (true_positives + K.epsilon()) / (predicted_positives + K.epsilon())
    return precision


def localRecall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    y_true = y_true[:, 1]
    y_pred = y_pred[:, 1]
    true_positives = K.sum(K.round(y_true * K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = (true_positives + K.epsilon()) / (possible_positives + K.epsilon())
    return recall


def dice(y_true, y_pre, smooth=1e-7):
    return (K.sum(2. * (y_true * y_pre)) + smooth) / (K.sum(y_true) + K.sum(y_pre) + smooth)

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

def datagene(mode='train'):
    if mode == 'train':
        allPaths = natsorted(glob(os.path.join(r'E:\pyWorkspace\CAE\res\highSuvArgued', '*')))
        np.random.shuffle(allPaths)
    else:
        allPaths = natsorted(glob(os.path.join(r'E:\pyWorkspace\CAE\res\highSuvBlock\test', '*')))
    sliceNum = 5
    index = 0
    x = []
    y = []
    iterTimes = 0
    while True:
        _p = allPaths[index]
        dx,dy=openPkl(_p)

        x.append(dx)
        y.append((np.sum(dy) > 2).astype(np.int))

        if len(x) == sliceNum:
            yield np.array(x), np.array(y)
            iterTimes += 1
            x = []
            y = []
        if index + 1 >= len(allPaths):
            print(mode + ' itertimes:' + str(iterTimes))
        index = (index + 1) % len(allPaths)




# def datagene(mode='train'):
#     import pickle as pkl
#     if mode == 'train':
#         allPaths = natsorted(glob(os.path.join(r'E:\pyWorkspace\CAE\res\highSuvArgued', '*')))
#         np.random.shuffle(allPaths)
#     else:
#         allPaths = natsorted(glob(os.path.join(r'E:\pyWorkspace\CAE\res\highSuvBlock\test', '*')))
#     sliceNum = 5
#     index = 0
#     x = []
#     y = []
#     iterTimes = 0
#     while True:
#         _p = allPaths[index]
#         with open(_p, 'rb') as f:
#             data = pkl.load(f)
#             dy = (np.sum(data['y']) > 2).astype(np.int)
#             dx = data['x']
#             if dx.shape != (32, 32, 32, 3):
#                 dx = np.stack([resize(data['x'][:, :, :, 0], (32, 32, 32), preserve_range=True),
#                                resize(data['x'][:, :, :, 1], (32, 32, 32), preserve_range=True),
#                                resize(data['x'][:, :, :, 2], (32, 32, 32), preserve_range=True)], axis=-1)
#                 # dx = np.stack([
#                 #                resize(data['x'][:, :, :, 1], (32, 32, 32), preserve_range=True)
#                 #               ], axis=-1)
#
#             x.append(dx)
#             y.append(dy)
#             # if dy==0:
#             #     y.append([1,0])
#             # else:
#             #     y.append([0,1])
#
#         if len(x) == sliceNum:
#             yield np.array(x), np.array(y)
#             iterTimes += 1
#             x = []
#             y = []
#         if index + 1 >= len(allPaths):
#             print(mode + ' itertimes:' + str(iterTimes))
#         index = (index + 1) % len(allPaths)


def focal_loss(classes_num, gamma=2., alpha=.25, e=0.1):
    # classes_num contains sample number of each classes
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''
        import tensorflow as tf
        from tensorflow.python.ops import array_ops
        from keras import backend as K

        # 1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(tf.greater(target_tensor, zeros), target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))

        # 2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [total_num / ff for ff in classes_num]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ff / sum_ for ff in classes_w_t1]  # scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

        # 3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_sum(balanced_fl)

        # 4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1 - e) * balanced_fl + e * K.categorical_crossentropy(
            K.ones_like(prediction_tensor) / nb_classes, prediction_tensor)

        return fianal_loss

    return focal_loss_fixed


if __name__ == '__main__':
    from keras.layers import Dense, MaxPool3D, Conv3D, Flatten
    from keras.models import Model

    top = dense3DClassify(input_shape=(32, 32, 32, 3), dropout_rate=0.4, nb_dense_block=5, nb_layers_per_block=9,
                          growth_rate=16, weight_decay=1e-2,
                          classes=512, activation='elu', initDis='glorot_normal')
    x = MaxPool3D((2, 2, 2), strides=[2, 2, 2], padding='valid')(top.output)
    x = Conv3D(640, (1, 1, 1), activation='elu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(128, activation='elu')(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(top.inputs, out)
    model.compile(Adam(lr=1e-3), binary_crossentropy, metrics=['acc', recall, precision])
    model.summary()
    plot_model(model, 'classfy.png', show_shapes=True)
    cpr = os.path.join(config.expRoot, 'checkPoint')
    if not os.path.exists(cpr):
        os.makedirs(cpr)
    mcp = ModelCheckpoint(
        os.path.join(cpr,
                     r'AugClassify_{epoch:03d}-{val_loss:.6f}-{val_recall:.6f}-{val_precision:.6f}.hdf5'),
        'val_loss', period=2)
    logP = os.path.join(config.expRoot, 'log')
    if not os.path.exists(logP):
        os.makedirs(logP)
    logger = CSVLogger(os.path.join(logP, 'log .txt'))
    lrReduce = ReduceLROnPlateau(factor=config.lrReduceRate, patience=config.lrReducePatience, verbose=1)
    estp = EarlyStopping(patience=config.estpPatient, verbose=1, min_delta=config.estpDelta, )
    model.fit_generator(datagene(mode='train'), steps_per_epoch=3020 // 5, epochs=config.epochs,
                        callbacks=[logger, mcp, lrReduce, estp, ], validation_data=datagene(mode='test'),
                        validation_steps=4809 // 5)
    visualLoss(os.path.join(logP, 'log.txt'))
