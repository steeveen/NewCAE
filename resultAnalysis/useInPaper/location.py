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
 @Author = 'steven'   @DateTime = '2019/5/5 11:32'
'''
import numpy as np
from skimage import io as skio
from skimage.morphology import label as skl
from skimage.measure import regionprops as skr
from Config import Config
from natsort import natsorted
from glob import glob
import pickle as pkl
import os

config = Config()
config.labelRoot = r'E:\pyWorkspace\CAE\res\cp250'
config.resultRoot = r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\experimentCleanSlice3StackFocalLossSegFork\result_tts4-1(5)_1bfl030mse_bfl3,01_nlpb5_ndb5_red05_icf16_gr16_lr1e-4_dp05_Allr3 - 副本'
config.trainId = [68, 69, 70]
config.iouT = 0.5
config.csvLogger = r'locLog.csv'


def tp(gtLabel, preLabel):
    tp = 0
    for i in range(1,np.max(gtLabel) + 1):
        for j in range(1,np.max(preLabel) + 1):
            iou = np.sum(np.logical_and((gtLabel == i), (preLabel == j))) / np.sum(
                np.logical_or((gtLabel == i), (preLabel == j)))
            if iou > config.iouT:
                tp += 1
                break
    return tp


csvLogger = ['patientId,tp,gtN,preN,recall,precision,']
for i in config.trainId:
    print('---------------------------')
    gt = np.stack(
        [skio.imread(_) for _ in natsorted(glob(os.path.join(config.labelRoot, str(i), 'labelClear', '*')))[1:-1]]) > 0
    pre = np.stack(
        [skio.imread(_) for _ in natsorted(glob(os.path.join(config.resultRoot, str(i), '[1-9]*preMd.bmp')))]) > 0
    gtLabel = skl(gt, connectivity=1)
    preLabel = skl(pre, connectivity=1)
    tpValue = tp(gtLabel, preLabel)
    print('tp:%d' % tpValue)
    print('gt:%d' % np.max(gtLabel))
    print('pre:%d' % np.max(preLabel))
    recall = tpValue / np.max(gtLabel)
    precision = tpValue / np.max(preLabel)
    print(r'patient %d  tp:%d  gt:%d  pre:%d  recall: %06f  precision: %06f ' % (i,tpValue,np.max(gt),np.max(pre), recall, precision))
    csvLogger.append(r'%d,%d,%d,%d,%f,%f,' % (i,tpValue,np.max(gtLabel),np.max(preLabel), recall, precision))
with open(config.csvLogger, 'w') as f:
    for line in csvLogger:
        f.write(line + '\n')
