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
 @Author = 'steven'   @DateTime = '2019/3/7 17:39'
'''
from keras.models import load_model
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

config = Config()
# config.dataRootp = r'E:\pyWorkspace\CAE\res\highSuvBlock\test'
config.dataRootp = r'E:\pyWorkspace\CAE\res\highSuvArgued'
config.y_true = []


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
    gamma = 4
    return K.pow((1 - pt_1), gamma)


def dice(y_true, y_pre, smooth=1e-7):
    return (K.sum(2. * (y_true * y_pre)) + smooth) / (K.sum(y_true) + K.sum(y_pre) + smooth)


def datagene():
    import pickle as pkl
    allPaths = natsorted(glob(os.path.join(config.dataRootp,'*')))
    sliceNum = 5
    index = 0
    x = []
    y = []
    iterTimes=0
    while True:
        _p = allPaths[index]
        with open(_p, 'rb') as f:
            data = pkl.load(f)
            dy = (np.sum(data['y']) > 2).astype(np.float)
            dx = data['x']
            if dx.shape != (32, 32, 32):
                dx = np.stack([resize(data['x'][:, :, :, 0], (32, 32, 32), preserve_range=True),
                               resize(data['x'][:, :, :, 1], (32, 32, 32), preserve_range=True),
                               resize(data['x'][:, :, :, 2], (32, 32, 32), preserve_range=True)], axis=-1)
                # dx = np.stack([
                #                resize(data['x'][:, :, :, 1], (32, 32, 32), preserve_range=True)
                #               ], axis=-1)

            x.append(dx)
            y.append(dy)

        if len(x) == sliceNum:
            yield np.array(x)
            config.y_true+=y
            iterTimes+=1
            x = []
            y = []
        if index+1>=len(allPaths):
            print(' itertimes:'+str(iterTimes))

        index = (index + 1) % len(allPaths)


def getY():
    import pickle as pkl
    allPaths = natsorted(glob(os.path.join(config.dataRootp,'*')))
    for _p in allPaths:
        with open(_p, 'rb') as f:
            data = pkl.load(f)
            dy = (np.sum(data['y']) > 2).astype(np.float)
            config.y_true.append(dy)

if __name__ == '__main__':
    import pickle as pkl
    getY()
    m = load_model(
        r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\classExperiment\checkPoint\AugClassify_002-0.192912-0.962539-1.000000.hdf5',
        custom_objects={'recall': recall, 'precision': precision})
    # result = m.predict_generator(datagene(mode='test'),2233 )
    result = m.predict_generator(datagene(), 3020//5)
    with open(r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\classExperiment\AugTrainResult.pkl', 'wb') as f:
        pkl.dump({'y_p': result, 'y_t': config.y_true}, f)
