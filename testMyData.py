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
 @Author = 'steven'   @DateTime = '2019/2/25 20:13'
'''
from natsort import natsorted
from Config import Config
from glob import glob
import os
from skimage import io as skio
import numpy as np
import pickle as pkl
import keras.backend as K
from kerasTools import recall,precision
config = Config()
config.dataRootp = r'E:\pyWorkspace\CAE\res\cp250'
# config.dataBatch = 10
config.epochs = 2000
config.lrReduceRate = 0.1

config.lrReducePatience = 20
config.estpPatient = 30
config.estpDelta = 5e-5
config.expRoot = 'experiment'

rootp = r'E:\pyWorkspace\CAE\res\cp250\75'
from keras.models import load_model
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

def dice(y_true, y_pre, smooth=0.0000001):
    return (K.sum(2 * (y_true * y_pre)) + smooth) / (K.sum(y_true) + K.sum(y_pre) + smooth)


def testDatagene():
    dataList = natsorted(glob(os.path.join(config.dataRootp, '*')))
    _patientRoot = dataList[0]
    while True:
        suvs = np.stack([skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'suv', '*')))])
        cts = np.stack([skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'ct', '*')))])
        highAreas = np.stack(
            [skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'highAreaInfo', '*')))])

        cts = (cts - cts.min()) / (cts.max() - cts.min())
        highAreas = np.log(highAreas, where=(highAreas != 0)) / np.log(100)

        suvs = np.pad(suvs, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)
        cts = np.pad(cts, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)
        highAreas = np.pad(highAreas, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)

        x = []
        for i in range(0, suvs.shape[0]):
            suv = suvs[i, :, :]
            highArea = highAreas[i, :, :]
            ct = cts[i, :, :]
            x.append(np.stack([suv, ct, highArea], axis=-1))
        x = np.array(x)

        sliceNum = 1
        for i in range(0, suvs.shape[0], sliceNum):
            yield x[i:i + sliceNum, :, :, :]

def evaDatagene():
    _patientRoot = natsorted(glob(os.path.join(config.dataRootp, '*')))[4]
    # while True:
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

    x = []
    y = []
    for i in range(0, suvs.shape[0]):
        suv = suvs[i, :, :]
        highArea = highAreas[i, :, :]
        ct = cts[i, :, :]
        x.append(np.stack([suv, ct, highArea], axis=-1))
        y.append(labels[i, :, :])
    x = np.array(x)
    y = np.array(y)[:, :, :, np.newaxis]
    print(y.mean())
    sliceNum = 1
    for i in range(0, suvs.shape[0], sliceNum):
        yield x[i:i + sliceNum, :, :, :],y[i:i + sliceNum, :, :]

def loadData(_patientRoot ):
    suvs = np.stack([skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'suv', '*')))])
    cts = np.stack([skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'ct', '*')))])
    highAreas = np.stack(
        [skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'highAreaInfo', '*')))])

    cts = (cts - cts.min()) / (cts.max() - cts.min())
    highAreas = np.log(highAreas, where=(highAreas != 0)) / np.log(100)

    suvs = np.pad(suvs, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)
    cts = np.pad(cts, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)
    highAreas = np.pad(highAreas, [[0, 0], [3, 3], [3, 3]], mode='constant', constant_values=0)

    x = []
    for i in range(0, suvs.shape[0]):
        suv = suvs[i, :, :]
        highArea = highAreas[i, :, :]
        ct = cts[i, :, :]
        x.append(np.stack([suv, ct, highArea], axis=-1))
    x = np.array(x)
    return x[197:198,:,:,:]

if __name__ == '__main__':

    op = r'E:\pyWorkspace\NewCAE\FCDenseNet\experiment\0'
    os.mkdir(op) if not os.path.exists(op) else None
    model = load_model(r'E:\pyWorkspace\NewCAE\FCDenseNet\experiment\checkPoint\l_002-0.998921-0.000649-1.000000-0.000325.hdf5',
                       custom_objects={'dice': dice,'tversky_loss':tversky_loss,'recall':recall,'precision':precision})
    result = model.predict_generator(testDatagene(), steps=420, )
    # result=model.predict(loadData(r'E:\pyWorkspace\CAE\res\cp250\0'))
    # result = model.evaluate_generator(evaDatagene(),steps=512 )
    # print(result)
    with open(op + '.pkl', 'wb') as f:
        pkl.dump(result, f)

    print(np.shape(result))
    print(np.sum(result))
    with open(op + '.pkl', 'rb') as f:
        result = pkl.load(f)
        for i in range(result.shape[0]):
            skio.imsave(os.path.join(op, str(i) + '.bmp'), np.round(result[i, :, :, 0] * 255))

