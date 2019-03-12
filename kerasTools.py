# -*- coding: utf-8 -*-
'''
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
 @Belong = 'StevenTools'  @MadeBy = 'PyCharm'
 @Author = 'lenovo'   @DateTime = '2018/7/4 14:36'
'''
import keras.backend as K


def visModel(model, to_file):
    '''
    对模型进行可视化
    :param model:model对象
    :param to_file: 可视化图片存放的位置
    :return: 将model可视化后的图像放到to_file位置
    '''
    from keras.utils import plot_model
    plot_model(model, to_file=to_file, show_shapes=True)


def log4GraphCb(logp):
    '''
    为了之后生成训练与测试损失图而进行的对每次epoch的损失记录
    :param logp: 损失记录保存的位置 是txt文件
    :return:
    '''
    from keras.callbacks import CSVLogger
    return CSVLogger(logp, append=True)


def visualLoss(logp, targets=None):
    from matplotlib import pyplot as plt
    import os
    import pandas as pd
    '''可视化loss
    :param logp:log文件地址
    :return:
    '''

    if os.path.exists(logp):
        tableXEpoch = pd.read_csv(logp).set_index('epoch')
        targetVailds = tableXEpoch.keys()[:len(tableXEpoch.keys()) // 2] if targets == None else [i for i in targets if
                                                                                                  i in tableXEpoch.keys()]
        for _y in targetVailds:
            plt.plot(tableXEpoch[_y].values, label='training ' + _y)
            plt.plot(tableXEpoch['val_' + _y].values, label='validation ' + _y)
            plt.ylabel(_y)
            plt.xlabel('Epoch')
            plt.xlim((0, len(tableXEpoch.index)))
            plt.legend(loc='best')
            plt.savefig(logp[:-4] + '_' + _y + '.jpg')
            plt.close()


import numpy as np


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(y_true *K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall =  (true_positives+ K.epsilon()*(1e-3)) / (possible_positives + K.epsilon())
    return recall
def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = (true_positives+ K.epsilon()*(1e-3)) / (predicted_positives + K.epsilon())
    return precision


tpr = recall
sensitivity = tpr


def fpr(y_true, y_pred):
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    true_neg = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return fp / true_neg


def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2, 3))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2, 3)) + beta * K.sum(p1 * g0, (0, 1, 2, 3))

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl - T


def tverskyMean(y_true, y_pred):
    alpha = 0.5
    beta = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2, 3))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2, 3)) + beta * K.sum(p1 * g0, (0, 1, 2, 3))

    T = K.sum(num / den)
    result = T / K.cast(K.shape(y_true)[-1], 'float32')  # when summing over classes, T has dynamic range [0 Ncl]
    return result



def createIndpWeightTversky(t0=0.002, t1=0.749, t2=0.250):
    def independentTverskyLoss(y_true, y_pred, ):
        alpha = 0.5
        beta = 0.5

        ones = K.ones(K.shape(y_true))
        p0 = y_pred  # proba that voxels are class i
        p1 = ones - y_pred  # proba that voxels are not class
        g0 = y_true
        g1 = ones - y_true

        def classTver(classIndex):
            num = K.sum((p0 * g0)[:, :, :, classIndex])
            den = num + alpha * K.sum((p0 * g1)[:, :, :, classIndex]) + beta * K.sum((p1 * g0)[:, :, :, classIndex])
            return K.sum(num / den)

        T = classTver(0) * t0 + classTver(1) * t1 + classTver(2) * t2
        return 1 - T

    return independentTverskyLoss


def tverskyMeanWLoss(y_true, y_pred):
    return 1 - tverskyMeanW(y_true, y_pred)


def tverskyMeanW(y_true, y_pred, t0=0.99992, t1=0.00002, t2=0.00006):
    alpha = 0.5
    beta = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class
    g0 = y_true
    g1 = ones - y_true

    num = K.sum((p0 * g0)[:, :, :, 0] * t0 + (p0 * g0)[:, :, :, 1] * t1 + (p0 * g0)[:, :, :, 2] * t2)
    den = num + alpha * K.sum(
        (p0 * g1)[:, :, :, 0] * t0 + (p0 * g1)[:, :, :, 1] * t1 + (p0 * g1)[:, :, :, 2] * t2) + beta * K.sum(
        (p1 * g0)[:, :, :, 0] * t0 + (p1 * g0)[:, :, :, 1] * t1 + (p1 * g0)[:, :, :, 2] * t2)
    return K.sum(num / den)


def show2DDice(classIndex, smooth=1e-5):
    def classDice(y_true, y_pred):
        intersection = K.sum(K.round((y_true * y_pred)[:, :, :, classIndex]))
        den = K.sum(K.round(y_true[:, :, :, classIndex])) + K.sum(K.round(y_pred[:, :, :, classIndex]))
        return (intersection * 2 + smooth) / (den + smooth)

    return classDice
def weightedLoss(originalLossFunc, weightsList):

    def lossFunc(true, pred):

        axis = -1 #if channels last


        #argmax returns the index of the element with the greatest value
        #done in the class axis, it returns the class index
        classSelectors = K.argmax(true, axis=axis)

        #considering weights are ordered by class, for each class
        #true(1) if the class index is equal to the weight index
        classSelectors = [K.equal(np.int64(i), classSelectors) for i in range(len(weightsList))]

        #casting boolean to float for calculations
        #each tensor in the list contains 1 where ground true class is equal to its index
        #if you sum all these, you will get a tensor full of ones.
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

        #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel,w in zip(classSelectors, weightsList)]

        #sums all the selections
        #result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]


        #make sure your originalLossFunc only collapses the class axis
        #you need the other axes intact to multiply the weights tensor
        loss = originalLossFunc(true,pred)
        loss = loss * weightMultiplier

        return loss
    return lossFunc


def show3DDice(classIndex, smooth=1e-5):
    def classDice(y_true, y_pred):
        intersection = K.sum((y_true * y_pred)[:, :, :, :, classIndex])
        den = K.sum(y_true[:, :, :, :, classIndex]) + K.sum(y_pred[:, :, :, :, classIndex])
        return (intersection * 2 + smooth) / (den + smooth)

    return classDice


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred, smooth=1.):
    return 1. - dice_coef(y_true, y_pred, smooth)





if __name__ == '__main__':
    visualLoss(r'E:\pyWorkspace\LymDetection\log\6.txt')
