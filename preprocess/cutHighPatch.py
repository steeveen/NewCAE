# -*- coding: utf-8 -*-
'''对患者先进行3D上的小连通区去除，再进行2.5D的切块
每个人的高代谢highLabels由thres定义(猛哥写的，放在txt中的肝脏血池suv平均值、标准差、中位数
    forbinden是不能用的患者的id号，
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
 @Author = 'steven'   @DateTime = '2019/3/18 16:12'
'''
import os
import numpy as np
from natsort import natsorted
from glob import glob
from skimage.morphology import label
from skimage.measure import regionprops
from skimage import io as skio
import pickle as pkl
from skimage.transform import resize
from List import forbinden,thres



def cutBlockCenter(cts, suvs, labels, bbox, sizeMode):
    nBBox = [0] * 6
    allShape = cts.shape
    sizeMode = min(sizeMode, 64)

    nBBox[0]=bbox[0]
    nBBox[3]=bbox[3]
    for i in range(1,3):

        nBBox[i] = bbox[i] - (sizeMode - (bbox[i + 3] - bbox[i])) // 2
        nBBox[i + 3] = nBBox[i] + sizeMode
    ct = np.pad(cts[max(nBBox[0], 0):min(nBBox[3], allShape[0]),
                max(nBBox[1], 0):min(nBBox[4], allShape[1]),
                max(nBBox[2], 0):min(nBBox[5], allShape[2])],
                ((max(0, -nBBox[0]), max(0, nBBox[3] - allShape[0])),
                 (max(0, -nBBox[1]), max(0, nBBox[4] - allShape[1])),
                 (max(0, -nBBox[1]), max(0, nBBox[5] - allShape[2]))), mode='reflect')
    suv = np.pad(suvs[max(nBBox[0], 0):min(nBBox[3], allShape[0]),
                 max(nBBox[1], 0):min(nBBox[4], allShape[1]),
                 max(nBBox[2], 0):min(nBBox[5], allShape[2])],
                 ((max(0, -nBBox[0]), max(0, nBBox[3] - allShape[0])),
                  (max(0, -nBBox[1]), max(0, nBBox[4] - allShape[1])),
                  (max(0, -nBBox[1]), max(0, nBBox[5] - allShape[2]))),
                 mode='reflect')
    # volume = np.pad(volumes[max(nBBox[0], 0):min(nBBox[3], allShape[0]),
    #                 max(nBBox[1], 0):min(nBBox[4], allShape[1]),
    #                 max(nBBox[2], 0):min(nBBox[5], allShape[2])],
    #                 ((max(0, -nBBox[0]), max(0, nBBox[3] - allShape[0])),
    #                  (max(0, -nBBox[1]), max(0, nBBox[4] - allShape[1])),
    #                  (max(0, -nBBox[1]), max(0, nBBox[5] - allShape[2]))),
    #                 mode='reflect')
    label = np.pad(labels[max(nBBox[0], 0):min(nBBox[3], allShape[0]),
                   max(nBBox[1], 0):min(nBBox[4], allShape[1]),
                   max(nBBox[2], 0):min(nBBox[5], allShape[2])],
                   ((max(0, -nBBox[0]), max(0, nBBox[3] - allShape[0])),
                    (max(0, -nBBox[1]), max(0, nBBox[4] - allShape[1])),
                    (max(0, -nBBox[1]), max(0, nBBox[5] - allShape[2]))),
                   mode='reflect')

    return ct, suv, label


def main(ip, op):
    os.makedirs(op, exist_ok=True)

    bigPatchRoot = os.path.join(op, 'bigPatch')
    os.makedirs(bigPatchRoot, exist_ok=True)
    smallPatchRoot = os.path.join(op, 'smallPatch')
    os.makedirs(smallPatchRoot, exist_ok=True)

    for i, _p in enumerate(natsorted(glob(os.path.join(ip, '*')))):
        if os.path.basename(_p) in forbinden:
            continue
        cts = np.stack([skio.imread(_) for _ in natsorted(glob(os.path.join(_p, 'ct', '*')))])
        suvs = np.stack([skio.imread(_) for _ in natsorted(glob(os.path.join(_p, 'suv', '*')))])
        gts = np.stack([skio.imread(_) for _ in natsorted(glob(os.path.join(_p, 'labelClear', '*')))])

        # 获得要生成的切片
        highLabels = suvs > (thres[i][2] - thres[i][1])
        for _r in regionprops(label(highLabels, connectivity=1)):
            if _r.area <= 4:
                print(_p + ' too small ' + str(_r.area))
                highLabels[highLabels == _r.label] = 0
        for si in range(1, highLabels.shape[0] - 1):
            sliceLabels = highLabels[si - 1:si + 2, :, :]
            for _sr in regionprops(label(sliceLabels, connectivity=1)):
                ll = max(_sr.bbox[4] - _sr.bbox[1], _sr.bbox[5] - _sr.bbox[2])
                if ll < 3:
                    print(_p + ' too small ' + str(_r.area))
                    continue
                if ll < 6:
                    ct, suv, gt = cutBlockCenter(cts, suvs, gts,
                                                 (si - 1, _sr.bbox[1], _sr.bbox[2], si + 2, _sr.bbox[4], _sr.bbox[5]),
                                                 sizeMode=8)
                    patch = {'ct': ct, 'suv': suv, 'gt': gt}
                    with open(os.path.join(smallPatchRoot, 'Len' + str(ll) + 'Siz8' + 'Loc' + str(
                            (si - 1, _sr.bbox[1], _sr.bbox[2], si + 2, _sr.bbox[4],
                             _sr.bbox[5])) + 'Pth' + os.path.basename(_p) + 'Gt' + str(int(np.sum(gt) > 4)) + '.pkl'),
                              'wb') as f:
                        pkl.dump(patch, f)
                elif ll < 28:
                    ct, suv, gt = cutBlockCenter(cts, suvs, gts,
                                                 (si - 1, _sr.bbox[1], _sr.bbox[2], si + 2, _sr.bbox[4], _sr.bbox[5]),
                                                 sizeMode=ll + 4)
                    patch = {'ct': ct, 'suv': suv, 'gt': gt}
                    with open(os.path.join(smallPatchRoot, 'Len' + str(ll) + 'Siz' + str(ll + 4) + 'Loc' + str(
                            (si - 1, _sr.bbox[1], _sr.bbox[2], si + 2, _sr.bbox[4],
                             _sr.bbox[5])) + 'Pth' + os.path.basename(_p) + 'Gt' + str(int(np.sum(gt) > 4)) + '.pkl'),
                              'wb') as f:
                        pkl.dump(patch, f)

                else:
                    patch = {'ct': resize(
                        cts[si - 1:si + 2, _sr.bbox[1]:_sr.bbox[4], _sr.bbox[2]:_sr.bbox[5]], (3, 32, 32),
                        preserve_range=True),
                        'suv': resize(
                            suvs[si - 1:si + 2, _sr.bbox[1]:_sr.bbox[4], _sr.bbox[2]:_sr.bbox[5]], (3, 32, 32),
                            preserve_range=True),
                        'gt': (resize(
                            gts[si - 1:si + 2, _sr.bbox[1]:_sr.bbox[4], _sr.bbox[2]:_sr.bbox[5]], (3, 32, 32),
                            preserve_range=True) > 0.5).astype(np.int)}
                    with open(os.path.join(bigPatchRoot, 'L' + str(ll) + 'S' + str(
                            (si - 1, _sr.bbox[1], _sr.bbox[2], si + 2, _sr.bbox[4],
                             _sr.bbox[5])) + 'P' + os.path.basename(_p) + 'G' + str(
                        int(np.sum(patch['gt']) > 4)) + '.pkl'),
                              'wb') as f:
                        pkl.dump(patch, f)


if __name__ == '__main__':
    ip = r'E:\pyWorkspace\CAE\res\cp250'
    op = r'E:\pyWorkspace\CAE\res\highSuvPatch'
    main(ip, op)
