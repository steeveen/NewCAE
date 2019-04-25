# -*- coding: utf-8 -*-
'''计算去除掉头和膀胱的假阳性后的总dice
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
 @Author = 'steven'   @DateTime = '2019/4/22 14:34'
'''
preRoot = r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\experimentCleanSlice3StackFocalLossSegFork\result_tts4-1(5)_1bfl030mse_bfl3,01_nlpb5_ndb5_red05_icf16_gr16_lr1e-4_dp05_Allr4'
maskRoot = r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\experimentCleanSlice3StackFocalLossSegFork\result_tts4-1(5)_1bfl030mse_bfl3,01_nlpb5_ndb5_red05_icf16_gr16_lr1e-4_dp05_Allr3'
gtRoot = r'E:\pyWorkspace\CAE\res\cp250'
lst = [68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79]
thresholdDefault = 0.5
op = 'calculateTheDice_' + str(thresholdDefault) + '.csv'

import os
import pickle as pkl
import numpy as np
import csv
from skimage import io as skio
from natsort import natsorted
from glob import glob
from matplotlib import pyplot as plt


def wrtieData(titleNames, data, op='calculateTheDice_' + str(thresholdDefault) + '.csv'):
    with open(op, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(titleNames)
        for i in data:
            w.writerow(i)


def calculateMetrics(threshold):
    metrics = []
    for i in lst:
        print(i)
        preP = os.path.join(preRoot, str(i) + '.pkl')
        with open(preP, 'rb') as f:
            pre = pkl.load(f)
        gts = np.stack([(skio.imread(_) / 255).astype(np.uint8) for _ in
                        natsorted(glob(os.path.join(gtRoot, str(i), 'labelClear', '*')))])[1:-1, :, :]

        maskP = os.path.join(maskRoot, 'mask_' + str(i) + '.pkl')
        binPre = pre > threshold
        # 如果有头和膀胱的掩码，就先做掩码处理一下
        if os.path.exists(maskP):
            with open(maskP, 'rb') as f:
                mask = pkl.load(f)[1:-1, :, :]
            print('binPre shape:' + str(binPre.shape))
            print('mask shape:' + str(mask.shape))
            binPre = binPre * (1 - mask)

        _dice = 2 * np.sum(binPre * gts) / (np.sum(binPre) + np.sum(gts))
        _iou = np.sum(binPre * gts) / np.sum(np.logical_or(binPre, gts))
        tp = np.sum(binPre * gts)
        _recall = tp / np.sum(gts)
        _precision = tp / (np.sum(binPre)+1e-7)
        metrics.append([str(i), _dice, _iou, _recall, _precision])

    meanDice = np.mean([i[1] for i in metrics])
    meanIou = np.mean([i[2] for i in metrics])
    meanRecall = np.mean([i[3] for i in metrics])
    meanPrecision = np.mean([i[4] for i in metrics])

    metrics.append(['mean', meanDice, meanIou, meanRecall, meanPrecision])
    return metrics


def drawPic():
    x = np.linspace(0.3, 0.9, 13)
    y = [[_]+ calculateMetrics(_)[-1][1:] for _ in x]
    wrtieData(['threshold', 'dice', 'iou', 'recall', 'precision'], y,'iou_thr.csv')
    yDice = [_[1] for _ in y]
    yIou = [_[2] for _ in y]
    yRecall = [_[3] for _ in y]
    yPrecision = [_[4] for _ in y]
    plt.plot(x, yDice)
    plt.plot(x, yIou)
    plt.plot(x, yRecall)
    plt.plot(x, yPrecision)
    plt.show()


if __name__ == '__main__':
    # metrics = calculateMetrics(thresholdDefault)
    # wrtieData(['id', 'dice', 'recall', 'precision'], metrics)
    drawPic()
