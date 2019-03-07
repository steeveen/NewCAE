# -*- coding: utf-8 -*-
'''以阈值为界限，将高代谢的立方块剪切出来，立方块的大小为3个尺度
：（5,8}:剪切为8x8x8立方块，缺失立方块用0补齐
：（8,16];剪切为16x16x16立方块，缺失立方块用0补齐
:(16,+)：剪切为64x64x64 立方块，缺失立方块用0 补齐，超出64x64x64范围的缩小到64x64x64

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
 @Author = 'steven'   @DateTime = '2019/3/4 15:32'
'''
from natsort import natsorted
from glob import glob
import os
import numpy as np
from skimage.morphology import label
from skimage.measure import regionprops
from skimage import io as skio
import pickle as pkl
from skimage.transform import resize

ip = r'E:\pyWorkspace\CAE\res\cp250'
op = r'E:\pyWorkspace\NewCAE\data\res\highSuvBlock'


def cutBlock(cts, suvs, volumes, labels, bbox, size):
    nBBox = [0] * 6
    allShape = cts.shape

    for i in range(3):
        if bbox[i + 3] - size > 0:
            if bbox[i] + size < allShape[i]:
                nBBox[i] = (bbox[i] + bbox[i + 3] - size) // 2
                nBBox[i + 3] = nBBox[i] + size
            else:
                nBBox[i + 3] = bbox[i + 3]
                nBBox[i] = bbox[i + 3] - size
        else:
            nBBox[i] = bbox[i]
            nBBox[i + 3] = bbox[i] + size
    return np.stack([cts[nBBox[0]:nBBox[3], nBBox[1]:nBBox[4], nBBox[2]:nBBox[5]],
                     suvs[nBBox[0]:nBBox[3], nBBox[1]:nBBox[4], nBBox[2]:nBBox[5]],
                     volumes[nBBox[0]:nBBox[3], nBBox[1]:nBBox[4], nBBox[2]:nBBox[5]]],axis=-1), labels[nBBox[0]:nBBox[3],
                                                                                   nBBox[1]:nBBox[4], nBBox[2]:nBBox[5]]


def cutBigBlock(cts, suvs, volumes, labels, maxLength, bbox, size):
    nBBox = [0] * 6
    allShape = cts.shape
    oriSize = size
    if maxLength <= 30:  # 如果小于等于32，先按照32切，之后放大回去
        size = 32
    if maxLength > 62:
        size = maxLength + 2

    for i in range(3):
        if bbox[i + 3] - size > 0:
            if bbox[i] + size < allShape[i]:
                nBBox[i] = (bbox[i] + bbox[i + 3] - size) // 2
                nBBox[i + 3] = nBBox[i] + size
            else:
                nBBox[i + 3] = bbox[i + 3]
                nBBox[i] = bbox[i + 3] - size
        else:
            nBBox[i] = bbox[i]
            nBBox[i + 3] = bbox[i] + size

    y = labels[nBBox[0]:nBBox[3], nBBox[1]:nBBox[4], nBBox[2]:nBBox[5]]
    ct = cts[nBBox[0]:nBBox[3], nBBox[1]:nBBox[4], nBBox[2]:nBBox[5]]
    suv = suvs[nBBox[0]:nBBox[3], nBBox[1]:nBBox[4], nBBox[2]:nBBox[5]]
    volume = volumes[nBBox[0]:nBBox[3], nBBox[1]:nBBox[4], nBBox[2]:nBBox[5]]
    if maxLength <= 32 or maxLength > 62:
        ct = resize(ct, (oriSize, oriSize, oriSize), preserve_range=True)
        suv = resize(suv, (oriSize, oriSize, oriSize), preserve_range=True)
        volume = resize(volume, (oriSize, oriSize, oriSize), preserve_range=True)
        y = (resize(y, (oriSize, oriSize, oriSize), preserve_range=True) > 0.5).astype(np.int8)
    x = np.stack([ct, suv, volume],axis=-1)
    return x, y


def main(mode='train'):
    patients = natsorted(glob(os.path.join(ip, '*')))
    if mode == 'train':
        patients = patients[:60]
    else:
        patients = patients[60:]
    index = 0
    for _p in patients:
        print(_p)
        suvs = np.clip(np.stack([skio.imread(_) for _ in natsorted(glob(os.path.join(_p, 'suv', '*')))]), 0, 90)
        # suv在得到高代谢区域后才能归一化
        cts = (np.stack([skio.imread(_) for _ in natsorted(glob(os.path.join(_p, 'ct', '*')))]) + 250) / 500
        labels = np.stack(
            [(skio.imread(_) > 0.5).astype(np.int8) for _ in natsorted(glob(os.path.join(_p, 'labelCLear', '*')))])
        volumes = np.stack([skio.imread(_) for _ in natsorted(glob(os.path.join(_p, 'highAreaInfo', '*')))])
        volumes = np.log(volumes + 1) / np.log(100)
        volumes = volumes / volumes.max()

        highs = suvs > 1.0
        suvs = suvs / 90
        highLabel = label(highs, connectivity=2)
        regions = regionprops(highLabel)
        for region in regions:
            maxLength = np.max([region.bbox[i + 3] - region.bbox[i] for i in range(3)])
            if maxLength < 5:
                # 小于5的肿瘤不要
                continue
            elif maxLength <= 8 - 2:  # 肿瘤大小最长为5或6的
                pass
                block, blockLabel = cutBlock(cts, suvs, volumes, labels, region.bbox, 8, )
                with open(os.path.join(r'E:\pyWorkspace\NewCAE\data\res\highSuvBlock', mode,
                                       str(index) + '_p' + os.path.basename(_p) + '_s' + str(maxLength) + '.pkl'),
                          'wb') as f:
                    pkl.dump({'x': block, 'y': blockLabel}, f)
            elif maxLength <= 16 - 2:
                block, blockLabel = cutBlock(cts, suvs, volumes, labels, region.bbox, 16, )
                with open(os.path.join(r'E:\pyWorkspace\NewCAE\data\res\highSuvBlock', mode,
                                       str(index) + '_p' + os.path.basename(_p) + '_m' + str(maxLength) + '.pkl'),
                          'wb') as f:
                    pkl.dump({'x': block, 'y': blockLabel}, f)
            else:
                pass
                block, blockLabel = cutBigBlock(cts, suvs, volumes, labels, maxLength, region.bbox, 64, )

                with open(os.path.join(r'E:\pyWorkspace\NewCAE\data\res\highSuvBlock', mode,
                                       str(index) + '_p' + os.path.basename(_p) + '_b' + str(maxLength) + '.pkl'),
                          'wb') as f:
                    pkl.dump({'x': block, 'y': blockLabel}, f)

            index += 1


if __name__ == '__main__':
    mode = 'train'
    main(mode)
    mode = 'test'
    main(mode)
