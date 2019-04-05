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
 @Author = 'steven'   @DateTime = '2019/2/23 20:42'
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
import tensorflow as tf
from keras.optimizers import Adam
from skimage.transform import resize
from Tools.losses import binary_focal_loss, focal_tversky, dice

config = Config()
config.dataRootp = r'E:\pyWorkspace\CAE\res\cleanSliceMore'
config.epochs = 2000
config.batchSize = 5
config.lrReduceRate = 0.1
config.lrReducePatience = 20
config.estpPatient = 30
config.estpDelta = 5e-5
config.lr = 1e-4
config.thr = 1
config.expRoot = 'experimentCleanSlice3StackFocolLossTest'
config.tts = 0.8
# config.trainIds = [4, 7, 15, 17, 20, 23, 24, 26, 31, 32,
#                    33, 34, 35, 36, 37, 38, 40, 42, 44, 45,
#                    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
#                    57, 58, 60, 61, 62, 63, 65, 66, 68, 69,
#                    70, 71, 72, 74, 76, 77, 78, 79, ]
# config.testIds = [1, 11, 25, 39, 41, 43, 56, 59, 64, 67, 73, 75]

# config.trainIds = [4, 7, 15, 17, 20, 23, 24, 26, 31, 32,
#                    33, 34, 35, 36, 37, 38, 40, 42, 44, 45,
#                    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
#                    57, 58, 60, 61, 62, 63, 65, 66, 68, 69,
#                    70, 71, 72, 74, 76, 77, 78, 79, 73, 75
#                    ]
# config.testIds = [1, 11, 25, 39, 41, 43, 56, 59, 64, 67,
#                   ]

config.trainIds = [20, 23, 24, 25, 26, 31, 39, 32, 33, 34,
                   35, 36, 37, 38, 40, 41, 42, 43, 44, 45,
                   46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                   56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                   66, 67, 68, 69, 70, 71, 72, 74, 76, 77,
                   78, 79, 73, 75,
                   ]
config.testIds = [1, 4, 7, 11, 15, 17, ]

trainSuvps = []
trainCtps = []
trainGtps = []
testSuvps = []
testCtps = []
testGtps = []


def genePath():
    def gene(root, modi, lst, mode):
        allLst = natsorted(glob(os.path.join(root, '*')))
        # np.random.seed(0)
        # np.random.shuffle(allLst)

        if mode == 'train':
            # allLst = allLst[:int(len(allLst) * config.tts)]
            r = []
            for _p in allLst:
                if int(os.path.basename(_p)) in config.trainIds:
                    r.append(_p)
            allLst = r
        else:
            # allLst = allLst[int(len(allLst) * config.tts):]
            r = []
            for _p in allLst:
                if int(os.path.basename(_p)) in config.testIds:
                    r.append(_p)
            allLst = r
        print('*' * 8 + mode + '*' * 8)
        print(allLst)
        for _p in allLst:
            i = 0
            while True:
                imgps = natsorted(glob(os.path.join(_p, modi, str(i) + '_*')))
                if len(imgps) == 0:
                    break
                for j in range(1, len(imgps) - 1):
                    lst.append(imgps[j - 1:j + 2])
                i += 1

    gene(config.dataRootp, 'suv', trainSuvps, 'train')
    gene(config.dataRootp, 'ct', trainCtps, 'train')
    gene(config.dataRootp, 'labelClear', trainGtps, 'train')
    gene(config.dataRootp, 'suv', testSuvps, 'test')
    gene(config.dataRootp, 'ct', testCtps, 'test')
    gene(config.dataRootp, 'labelClear', testGtps, 'test')


def dataGene(batchSize, mode='train'):
    if mode == 'train':
        suvps = trainSuvps
        ctps = trainCtps
        gtps = trainGtps
    else:
        suvps = testSuvps
        ctps = testCtps
        gtps = testGtps
    index = 0
    suvs = []
    cts = []
    gts = []
    gtSum = 0
    while True:
        # suv = np.stack([skio.imread(_) for _ in suvps[index]], axis=-1)
        # ct = np.stack([skio.imread(_) for _ in ctps[index]], axis=-1)
        # gt = (skio.imread(gtps[index][1]) / 255)[:,:,np.newaxis]

        suv = np.stack([resize(skio.imread(_), (128, 128), preserve_range=True) for _ in suvps[index]], axis=-1)
        ct = np.stack([resize(skio.imread(_), (128, 128), preserve_range=True) for _ in ctps[index]], axis=-1)
        gt = (resize(skio.imread(gtps[index][1]) / 255, (128, 128), preserve_range=True) > 0).astype(np.int)[:, :,
             np.newaxis]
        if index == 0:
            print(mode + ' gt rato:' + str(gtSum / (len(suvps) * 128 * 128)))
        gtSum += np.sum(gt)

        suvs.append(suv)
        cts.append(ct)
        gts.append(gt)
        if len(gts) == batchSize:
            suvs = np.array(suvs)
            suvs = suvs * (suvs > config.thr)
            suvs = np.clip(suvs, 0, 10)
            suvs = np.log(suvs, where=(suvs > 0)) / np.log(10)
            # suvs = np.log(suvs, where=(suvs > 0)) / np.log(100)
            # suvs = (suvs - suvs.min()) / (suvs.max() - suvs.min())
            cts = np.array(cts)
            cts = (cts + 250) / 500
            # cts = (cts - cts.min()) / (cts.max() - cts.min())
            gts = np.array(gts)

            x = np.concatenate([suvs, cts], axis=-1)
            # x=suvs
            yield x, gts
            suvs = []
            cts = []
            gts = []
        index = (index + 1) % len(suvps)


if __name__ == '__main__':
    from keras.utils import plot_model
    from keras_contrib.applications.densenet import DenseNetFCN

    genePath()
    print(len(trainGtps))
    print(len(trainCtps))
    print(len(trainSuvps))
    print(len(testGtps))
    print(len(testCtps))
    print(len(testSuvps))

    model = DenseNetFCN(input_shape=(128, 128, 6), nb_layers_per_block=5, dropout_rate=0.5, nb_dense_block=4,
                        reduction=0.5,
                        initial_kernel_size=(3, 3),
                        init_conv_filters=16, growth_rate=16, classes=1, activation='sigmoid')
    model.compile(Adam(config.lr), binary_focal_loss(gamma=2, alpha=0.1), metrics=['acc', dice, recall, precision])
    # model.compile('adam', focal_tversky, metrics=['acc', dice, recall, precision])
    plot_model(model, 'tiramisu128x3.png', show_shapes=True)
    model.summary()
    cpr = os.path.join(config.expRoot, 'checkPoint')

    if not os.path.exists(cpr):
        os.makedirs(cpr)
    # mcp = ModelCheckpoint(
    #     os.path.join(cpr,
    #                  r'{epoch:03d}-{val_loss:.6f}-{val_dice:.6f}-{val_recall:.6f}-{val_precision:.6f}.hdf5'),
    #     'val_loss')
    logP = os.path.join(config.expRoot, 'log')
    if not os.path.exists(logP):
        os.makedirs(logP)
    logger = CSVLogger(os.path.join(logP, 'log.csv'))
    lrReduce = ReduceLROnPlateau(factor=config.lrReduceRate, patience=config.lrReducePatience, verbose=1)
    estp = EarlyStopping(patience=config.estpPatient, verbose=1, min_delta=config.estpDelta, )

    # model.fit_generator(dataGene(config.batchSize, 'train'), steps_per_epoch=np.ceil(len(trainGtps) / config.batchSize),
    #                     epochs=config.epochs,
    #                     callbacks=[logger, mcp, lrReduce, estp, ], validation_data=dataGene(config.batchSize, 'test'),
    #                     validation_steps=np.ceil(len(testGtps) / config.batchSize))

    model.fit_generator(dataGene(config.batchSize, 'train'), steps_per_epoch=np.ceil(len(testGtps) / config.batchSize),
                        epochs=config.epochs,
                        callbacks=[logger, lrReduce, estp, ], validation_data=dataGene(config.batchSize, 'test'),
                        validation_steps=np.ceil(len(trainGtps) / config.batchSize))

    visualLoss(os.path.join(logP, 'log.csv'))
