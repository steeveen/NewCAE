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
 @Author = 'steven'   @DateTime = '2019/4/11 8:52'
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
config.epochs = 2000
config.batchSize = 5
config.lrReduceRate = 0.1
config.lrReducePatience = 20
config.estpPatient = 45
config.estpDelta = 5e-5
# config.lr = 1e-4
config.lr = 1e-4

config.thr = 1
config.expRoot = 'experimentCleanSlice3StackFocalLossSegFork'
config.tts = 0.8

config.trainSuvps = []
config.trainCtps = []
config.trainGtps = []
config.testSuvps = []
config.testCtps = []
config.testGtps = []

config.trainIds = [1, 4, 7, 11, 15, 17, 20, 23, 24, 25, 26, 31,
                   39, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43,
                   44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55,
                   56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
                   ]
config.testIds = [68, 69, 70, 71, 72, 74, 76, 77, 78, 79, 73, 75, ]


def genePath():
    def gene(root, modi, lst, mode):
        allLst = natsorted(glob(os.path.join(root, '*')))

        if mode == 'train':
            r = []
            for _p in allLst:
                if int(os.path.basename(_p)) in config.trainIds:
                    r.append(_p)
            allLst = r
        else:
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

    gene(config.dataRootp, 'suv', config.trainSuvps, 'train')
    gene(config.dataRootp, 'ct', config.trainCtps, 'train')
    gene(config.dataRootp, 'labelClear', config.trainGtps, 'train')
    gene(config.dataRootp, 'suv', config.testSuvps, 'test')
    gene(config.dataRootp, 'ct', config.testCtps, 'test')
    gene(config.dataRootp, 'labelClear', config.testGtps, 'test')


def dataGene(batchSize, mode='train'):
    if mode == 'train':
        suvps1 = config.trainSuvps
        ctps1 = config.trainCtps
        gtps1 = config.trainGtps
        suvps2 = config.testSuvps
        ctps2 = config.testCtps
    else:
        suvps1 = config.testSuvps
        ctps1 = config.testCtps
        gtps1 = config.testGtps
        suvps2 = config.testSuvps
        ctps2 = config.testCtps
    index1 = 0
    index2 = 0

    suvStackLst1 = []
    ctStackLst1 = []
    gtStackLst1 = []
    suvStackLst2 = []
    ctStackLst2 = []

    gtSum = 0
    while True:
        suvStack1 = np.stack([resize(skio.imread(_), (128, 128), preserve_range=True) for _ in suvps1[index1]], axis=-1)
        ctStack1 = np.stack([resize(skio.imread(_), (128, 128), preserve_range=True) for _ in ctps1[index1]], axis=-1)
        gtStack1 = (resize(skio.imread(gtps1[index1][1]) / 255, (128, 128), preserve_range=True) > 0).astype(np.int)[:,
                   :,
                   np.newaxis]

        suvStack2 = np.stack([resize(skio.imread(_), (128, 128), preserve_range=True) for _ in suvps2[index2]], axis=-1)
        ctStack2 = np.stack([resize(skio.imread(_), (128, 128), preserve_range=True) for _ in ctps2[index2]], axis=-1)
        if index1 == 0:
            print(mode + ' gt rato:' + str(gtSum / (len(suvps1) * 128 * 128)))
        gtSum += np.sum(gtStack1)

        suvStackLst1.append(suvStack1)
        ctStackLst1.append(ctStack1)
        gtStackLst1.append(gtStack1)

        suvStackLst2.append(suvStack2)
        ctStackLst2.append(ctStack2)
        if len(gtStackLst1) == batchSize:
            suvStackLst1 = np.array(suvStackLst1)
            suvStackLst1 = suvStackLst1 * (suvStackLst1 > config.thr)
            suvStackLst1 = np.clip(suvStackLst1, 0, 10)
            suvStackLst1 = np.log(suvStackLst1, where=(suvStackLst1 > 0)) / np.log(10)
            ctStackLst1 = np.array(ctStackLst1)
            ctStackLst1 = (ctStackLst1 + 250) / 500
            gtStackLst1 = np.array(gtStackLst1)

            suvStackLst2 = np.array(suvStackLst2)
            suvStackLst2 = suvStackLst2 * (suvStackLst2 > config.thr)
            suvStackLst2 = np.clip(suvStackLst2, 0, 10)
            suvStackLst2 = np.log(suvStackLst2, where=(suvStackLst2 > 0)) / np.log(10)
            ctStackLst2 = np.array(ctStackLst2)
            ctStackLst2 = (ctStackLst2 + 250) / 500

            x1 = np.concatenate([suvStackLst1, ctStackLst1], axis=-1)
            x2 = np.concatenate([suvStackLst2, ctStackLst2], axis=-1)
            yield [x1, x2], [gtStackLst1, x2]
            suvStackLst1 = []
            ctStackLst1 = []
            gtStackLst1 = []
            suvStackLst2 = []
            ctStackLst2 = []
        if index1 + 1 > len(suvps1):
            print(mode + ' complete')
        index1 = (index1 + 1) % len(suvps1)
        index2 = (index2 + 1) % len(suvps2)


if __name__ == '__main__':

    genePath()
    print(len(config.trainGtps))
    print(len(config.trainCtps))
    print(len(config.trainSuvps))
    print(len(config.testGtps))
    print(len(config.testCtps))
    print(len(config.testSuvps))

    model = DenseNetFCNSemi(input_shape=(128, 128, 6), nb_layers_per_block=7,
                            dropout_rate=0.5,
                            nb_dense_block=4,
                            reduction=0.5,
                            initial_kernel_size=(3, 3),
                            init_conv_filters=8, growth_rate=8, classes=1, activation='sigmoid')
    model.summary()
    plot_model(model, 'denseNetFCNSemi2D.png', show_shapes=True)
    # model.compile(Adam(config.lr), loss={'output_1': binary_focal_loss(gamma=3, alpha=0.011), 'output_2': mse},
    #               metrics={'output_1': ['acc', dice, recall, precision]},loss_weights={'output_1':1,'output_2':0.02})
    model.compile(Adam(config.lr), loss={'output_1': binary_focal_loss(gamma=4, alpha=0.1), 'output_2': mse},
                  metrics={'output_1': ['acc', dice, recall, precision]},
                  loss_weights={'output_1': 1, 'output_2': 0.3})

    cpr = os.path.join(config.expRoot, 'checkPoint')

    if not os.path.exists(cpr):
        os.makedirs(cpr)
    logP = os.path.join(config.expRoot, 'log')
    if not os.path.exists(logP):
        os.makedirs(logP)
    logger = CSVLogger(os.path.join(logP, 'log.csv'))
    lrReduce = ReduceLROnPlateau(factor=config.lrReduceRate, patience=config.lrReducePatience, verbose=1)
    estp = EarlyStopping(patience=config.estpPatient, verbose=1, min_delta=config.estpDelta, )

    mcp = ModelCheckpoint(
        os.path.join(cpr,
                     r'{epoch:03d}-{val_loss:.6f}-{val_output_1_dice:.6f}-{val_output_1_precision:.6f}-{val_output_1_recall:.6f}.hdf5'),
        'val_loss')

    model.fit_generator(dataGene(config.batchSize, 'train'),
                        steps_per_epoch=np.ceil(len(config.trainGtps) / config.batchSize),
                        epochs=config.epochs,
                        callbacks=[logger, lrReduce, estp, mcp], validation_data=dataGene(config.batchSize, 'test'),
                        validation_steps=np.ceil(len(config.testGtps) / config.batchSize))

    visualLoss(os.path.join(logP, 'log.csv'))
