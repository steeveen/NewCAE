# -*- coding: utf-8 -*-
'''用于可视化网络分割结果 数据与标签
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
 @Author = 'steven'   @DateTime = '2019/4/15 9:49'
'''
import os

from keras.engine.saving import load_model
from natsort import natsorted
from glob import glob
import numpy as np
from skimage import io as skio
from Config import Config
from skimage.transform import resize
import pickle as pkl
from matplotlib import cm

config = Config()
# config.dataRootp = r'E:\pyWorkspace\CAE\res\cleanSliceMore'
config.dataRootp = r'E:\pyWorkspace\CAE\res\cp250'
config.modelRootp = r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\experimentCleanSlice3StackFocalLossSegFork\checkPoint_tts4-1(5)_1bfl030mse_bfl3,01_nlpb5_ndb5_red05_icf16_gr16_lr1e-4_dp05\035-0.071842-0.615032-0.848771-0.546990.hdf5'
config.op = r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\experimentCleanSlice3StackFocalLossSegFork\result_tts4-1(5)_1bfl030mse_bfl3,01_nlpb5_ndb5_red05_icf16_gr16_lr1e-4_dp05_Allr4'
config.epochs = 2000
config.batchSize = 1
config.lrReduceRate = 0.1
config.lrReducePatience = 20
config.estpPatient = 40
config.estpDelta = 5e-5
# config.lr = 1e-4
config.lr = 1e-4
config.preThr = 0.5
config.thr = 1

config.suvps = []
config.ctps = []
config.gtps = []

config.ids = [68, 69, 70, 71, 72]
config.ids = [73, 74, 75, 76, 77, 78, 79, ]
config.ids = [79]
# test 2019 4 22 calculate dice
config.ids = [68, 69, 70, 72, 73, 74,]
config.ids = [75, 76, 77, 78, 79]


# config.ids = [75]


def geneWholePath():
    def gene(root, modi):
        lst = []
        allLst = natsorted(glob(os.path.join(root, '*')))

        r = []
        for _p in allLst:
            if int(os.path.basename(_p) if os.path.isdir(_p) else -1) in config.id:
                r.append(_p)
        allLst = r
        print(allLst)
        for _p in allLst:
            imgps = natsorted(glob(os.path.join(_p, modi, '*')))
            for j in range(1, len(imgps) - 1):
                lst.append(imgps[j - 1:j + 2])
        return lst

    config.suvps = gene(config.dataRootp, 'suv')
    config.ctps = gene(config.dataRootp, 'ct')
    config.gtps = gene(config.dataRootp, 'labelClear')


def genePath():
    def gene(root, modi, lst):
        allLst = natsorted(glob(os.path.join(root, '*')))

        r = []
        for _p in allLst:
            if int(os.path.basename(_p)) in config.id:
                r.append(_p)
        allLst = r
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

    gene(config.dataRootp, 'suv', config.suvps)
    gene(config.dataRootp, 'ct', config.ctps)
    gene(config.dataRootp, 'labelClear', config.gtps)

    print(config.suvps)
    print(config.ctps)
    print(config.gtps)


def dataGene(batchSize):
    suvps1 = config.suvps
    ctps1 = config.ctps

    index1 = 0

    suvStackLst1 = []
    ctStackLst1 = []

    while True:
        suvStack1 = np.stack([resize(skio.imread(_), (128, 128), preserve_range=True) for _ in suvps1[index1]], axis=-1)
        ctStack1 = np.stack([resize(skio.imread(_), (128, 128), preserve_range=True) for _ in ctps1[index1]], axis=-1)

        suvStackLst1.append(suvStack1)
        ctStackLst1.append(ctStack1)

        if len(ctStackLst1) == batchSize:
            suvStackLst1 = np.array(suvStackLst1)
            suvStackLst1 = suvStackLst1 * (suvStackLst1 > config.thr)
            suvStackLst1 = np.clip(suvStackLst1, 0, 10)
            suvStackLst1 = np.log(suvStackLst1, where=(suvStackLst1 > 0)) / np.log(10)
            ctStackLst1 = np.array(ctStackLst1)
            ctStackLst1 = (ctStackLst1 + 250) / 500

            x1 = np.concatenate([suvStackLst1, ctStackLst1], axis=-1)
            yield [x1, x1]
            suvStackLst1 = []
            ctStackLst1 = []
        # if index1 + 1 > len(suvps1):
        #     print('epoch data complete')
        # index1 = (index1 + 1) % len(suvps1)

        if index1 + 1 == len(suvps1):
            print('epoch data complete')
            yield [x1, x1]
            suvStackLst1 = []
            ctStackLst1 = []
            index1 = 0
        index1 = index1 + 1


def main():
    from FCDenseNet.myDense.trainDenseSegFork import binary_focal_loss, dice, recall, precision, mse
    # genePath()
    geneWholePath()

    model = load_model(config.modelRootp,
                       custom_objects={'focal_loss': binary_focal_loss(gamma=3, alpha=0.1), 'dice': dice,
                                       'recall': recall, 'precision': precision, 'mse': mse})
    #     #     # model = Model()
    oriResult, recontrib = model.predict_generator(dataGene(config.batchSize),
                                                   np.ceil(len(config.gtps) / config.batchSize),
                                                   verbose=1)
    oriResult=oriResult.astype(np.float16)
    recontrib=recontrib.astype(np.float16)
    oriResult = resize(oriResult[:, :, :, 0], (oriResult.shape[0], 250, 250), preserve_range=True)

    print('result number:' + str(oriResult.shape[0]))
    print('ct number:' + str(len(config.ctps)))

    recontribSUV = resize(recontrib[:, :, :, 1], (recontrib.shape[0], 250, 250), preserve_range=True)
    recontribSUV = ((recontribSUV - recontribSUV.min()) / (recontribSUV.max() - recontribSUV.min()) * 255).astype(
        np.uint8)
    recontribCT = resize(recontrib[:, :, :, 4], (recontrib.shape[0], 250, 250), preserve_range=True)
    recontribCT = ((recontribCT - recontribCT.min()) / (recontribCT.max() - recontribCT.min()) * 255).astype(
        np.uint8)

    cts = np.stack([((skio.imread(i[1]) + 250) / 500 * 255).astype(np.uint8) for i in config.ctps], axis=0)
    suvs = np.stack([(np.clip(skio.imread(i[1]), 0, 10) / 10 * 255).astype(np.uint8) for i in config.suvps], axis=0)
    gts = np.stack([(skio.imread(i[1])).astype(np.uint8) for i in config.gtps], axis=0)

    print(oriResult.dtype)
    print(recontrib.dtype)
    recontrib=None
    print(recontribSUV.dtype)
    print(recontribCT.dtype)
    print(cts.dtype)
    print(suvs.dtype)
    print(gts.dtype)

    for i, ctStackp in enumerate(config.ctps):
        pl = ctStackp[1].split('\\')
        patientOp = os.path.join(config.op, pl[-3])
        if not os.path.exists(patientOp):
            os.mkdir(patientOp)
            with open(patientOp + '.pkl', 'wb') as f:
                pkl.dump(oriResult, f)

        skio.imsave(os.path.join(patientOp, pl[-1].replace('.tif', '_pre.bmp')),
                    (oriResult[i, :, :] * (oriResult[i, :, :] > config.preThr) * 255).astype(np.uint8))
        skio.imsave(os.path.join(patientOp, pl[-1].replace('.tif', '_preHeat.bmp')),
                    cm.jet(oriResult[i, :, :]))
        skio.imsave(os.path.join(patientOp, pl[-1].replace('.tif', '_recSUV.jpg')),
                    recontribSUV[i, :, :])
        skio.imsave(os.path.join(patientOp, pl[-1].replace('.tif', '_recCt.jpg')),
                    recontribCT[i, :, :])
        skio.imsave(os.path.join(patientOp, pl[-1].replace('.tif', '_ct.tif')), cts[i, :, :])
        skio.imsave(os.path.join(patientOp, pl[-1].replace('.tif', '_suv.tif')), suvs[i, :, :])
        skio.imsave(os.path.join(patientOp, pl[-1].replace('.tif', '_gt.bmp')), gts[i, :, :])

    for i in range(cts.shape[1]):
        patientOp = os.path.join(config.op, config.ctps[0][1].split('\\')[-3])

        skio.imsave(os.path.join(patientOp, 'col_' + str(i) + '_pre.bmp'),
                    (oriResult[:, i, :] * (oriResult[:, i, :] > config.preThr) * 255).astype(np.uint8))
        skio.imsave(os.path.join(patientOp, 'col_' + str(i) + '_preHeat.bmp'), cm.jet(oriResult[:, i, :]))
        skio.imsave(os.path.join(patientOp, 'col_' + str(i) + '_recSUV.jpg'), recontribSUV[:, i, :])
        skio.imsave(os.path.join(patientOp, 'col_' + str(i) + '_recCt.jpg'), recontribCT[:, i, :])

        skio.imsave(os.path.join(patientOp, 'col_' + str(i) + '_ct.tif'), cts[:, i, :])
        skio.imsave(os.path.join(patientOp, 'col_' + str(i) + '_suv.tif'), suvs[:, i, :])
        skio.imsave(os.path.join(patientOp, 'col_' + str(i) + '_gt.bmp'), gts[:, i, :])


if __name__ == '__main__':
    os.mkdir(config.op) if not os.path.exists(config.op) else None
    for _ in config.ids:
        config.id = [_]
        main()
    # convertSliceToCol()
