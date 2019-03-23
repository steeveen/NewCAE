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

thres = [[1.0, 0, 1],
         [1.0, 0, 1],
         [1.68675, 0.270291, 1.685],
         [1.78862, 0.20133, 1.765],
         [2.02084, 0.419678, 1.905],
         [2.70377, 0.308884, 2.705],
         [1.86265, 0.225558, 1.865],
         [1.82487, 0.318216, 1.785],
         [2.30139, 0.19442, 2.285],
         [2.02857, 0.377839, 2.095],
         [1.81117, 0.198819, 1.795],
         [1.99146, 0.178486, 1.985],
         [1.25853, 0.144098, 1.235],
         [2.39119, 0.311516, 2.395],
         [1.7631, 0.0969169, 1.755],
         [2.46485, 0.581147, 2.535],
         [1.4100, 0.41, 1.4000],
         [1.6100, 0.41, 1.6],
         [2.37, 0.4, 2.3001],
         [1.85, 0.40, 1.8],
         [1.21, 0.3, 1.20],
         [2.09735, 0.344403, 2.055],
         [2.00127, 0.189155, 1.995],
         [2.27763, 0.456008, 2.235],
         [1.70926, 0.219484, 1.695],
         [1.9981, 0.23742, 1.975],
         [1.0, 0, 1],
         [1.99561, 0.27911, 1.975],
         [2.27263, 0.222743, 2.275],
         [2.22544, 0.310421, 2.195],
         [2.001, 0.2, 2.001],
         [2.32624, 0.283438, 2.305],
         [2.38536, 0.34569, 2.375],
         [2.66131, 0.348425, 2.645],
         [2.15941, 0.298108, 2.145],
         [1.0, 0, 1],
         [2.54272, 0.382998, 2.555],
         [2.70791, 0.352986, 2.665],
         [2.40256, 0.252971, 2.385],
         [2.09735, 0.344403, 2.055],
         [1.12, 0.2, 1.12],
         [1.75, 0.3, 1.75],
         [1.56853, 0.26133, 1.545],
         [2.38536, 0.34569, 2.375],
         [2.06371, 0.403317, 2.055],
         [1.700, 0.3, 1.75],
         [2.46485, 0.581147, 2.535],
         [1.0, 0, 1],
         [2.02857, 0.377839, 2.095],
         [2.0, 0.3, 2.001],
         [2.0, 0.2, 2.0],
         [1.0, 0, 1],
         [1.83376, 0.327339, 1.795],
         [2.70377, 0.308884, 2.705],
         [2.24014, 0.243368, 2.225],
         [2.0001, 0.2, 2.01],
         [1.7645, 0.211, 1.7443],
         [2.78805, 0.262183, 2.775],
         [2.72396, 0.288798, 2.695],
         [2.17603, 0.581752, 2.125],
         [2.70441, 0.383908, 2.695],
         [1.0, 0, 1],
         [3.31165, 2.20634, 2.475],
         [1.75, 0.3, 1.77],
         [2.23887, 0.342719, 2.235],
         [2.37406, 0.274724, 2.355],
         [2.37406, 0.274724, 2.355],
         [2.27653, 0.246097, 2.275],
         [3.10535, 0.370482, 3.105],
         [2.30156, 0.196611, 2.285],
         [2.23633, 0.231912, 2.215],
         [1.0, 0, 1],
         [2.0078, 0.187011, 2.005],
         [1.005, 0.211, 1.006],
         [1.94519, 0.333655, 1.915],
         [1.73983, 0.185099, 1.745],
         [1.92518, 0.31164, 1.915],
         [2.20173, 0.320697, 2.205],
         [1.68257, 0.187831, 1.675],
         [1.82114, 0.154759, 1.805]]
forbinden = ['0', '2', '3', '5', '6',
             '8', '9', '10', '12', '13',
             '14', '16', '18', '19', '21',
             '22', '27', '28', '29', '30',
             '32', '33', '34', '38', '40',
             '41', '42', '43', '44', '45',
             '48', '49', '54', '58', '59',
             '60', '65', '68', '73', '78']


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
