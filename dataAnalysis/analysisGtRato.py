# -*- coding: utf-8 -*-
'''分析trainDenseSegFork中，切片序列中的正例像素比例
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
 @Author = 'steven'   @DateTime = '2019/4/16 10:24'
'''
import os
from natsort import natsorted
from glob import glob
import numpy as np
from skimage import io as skio
from Config import Config
from Tools.kerasTools import visualLoss, recall, precision
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
import keras.backend as K
from keras.optimizers import Adam, RMSprop
from skimage.transform import resize
from Tools.losses import binary_focal_loss, focal_tversky, dice
from keras.utils import plot_model
from keras_contrib.applications.densenet import DenseNetFCN
from FCDenseNet.myDense.dense import DenseNetFCNSemi
from keras.losses import binary_crossentropy, mse

config = Config()
config.dataRootp = r'E:\pyWorkspace\CAE\res\cleanSliceMore'
config.op = r'analysisGtRato.csv'

config.suvps = []
config.ctps = []
config.gtps = []

config.ids = [1, 4, 7, 11, 15, 17, 20, 23, 24, 25, 26, 31,
              39, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43,
              44, 45, 46, 47, 48, 49, 50, 51, 52, 53,  55,
              56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
              68, 69, 70, 71, 72, 74, 76, 77, 78, 79, 73, 75, ]


def analysis():
    allLst = natsorted(glob(os.path.join(config.dataRootp, '*')))

    r = []
    for _p in allLst:
        if int(os.path.basename(_p)) in config.ids:
            r.append(_p)
    allLst = r
    print(allLst)
    with open(config.op, 'w') as f:
        f.write('id,oneGtPixel,threeGtPixel,oneTotalPixel,threeTotalPixel,oneGtRato,threeGtRato,\n')
        oneAllTotalP = 0
        threeAllTotalP = 0
        oneAllGtP = 0
        threeAllGtP = 0
        for _p in allLst:
            i = 0
            oneTotalP = 0
            threeTotalP = 0
            oneGtP = 0
            threeGtP = 0
            while True:
                imgps = natsorted(glob(os.path.join(_p, 'labelClear', str(i) + '_*')))
                if len(imgps) == 0:
                    break
                for j in range(1, len(imgps) - 1):
                    s1 = (skio.imread(imgps[j - 1]) > 0).astype(np.int)
                    s2 = (skio.imread(imgps[j]) > 0).astype(np.int)
                    s3 = (skio.imread(imgps[j + 1]) > 0).astype(np.int)

                    oneTotalP += s2.size
                    threeTotalP += s2.size * 3
                    oneGtP += np.sum(s2)
                    threeGtP += (np.sum(s1) + np.sum(s2) + np.sum(s3))
                i += 1
            f.write(os.path.basename(_p) + ',' + str(oneGtP) + ',' + str(threeGtP) + ',' + str(oneTotalP) + ',' + str(
                threeTotalP) + ','+str(oneGtP / oneTotalP) + ',' + str(threeGtP / threeTotalP) + ',\n')
            oneAllGtP += oneGtP
            threeAllGtP += threeGtP
            oneAllTotalP += oneTotalP
            threeAllTotalP += threeTotalP
        f.write('mean,' + str(oneAllGtP/len(allLst)) + ',' + str(threeAllGtP/len(allLst)) + ',' + str(oneAllTotalP/len(allLst)) + ',' + str(
            threeAllTotalP/len(allLst)) + ',' + str(oneAllGtP / oneAllTotalP) + ',' + str(threeAllGtP / threeAllTotalP) + ',\n')


if __name__ == '__main__':
    analysis()
