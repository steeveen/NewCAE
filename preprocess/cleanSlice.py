# -*- coding: utf-8 -*-
'''屏蔽一些不好的病例，将SUV>肝血池阈值 作为掩码，与原数据标签相乘，去掉小区域，只保留切片，切片最好保留起始index
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
 @Author = 'steven'   @DateTime = '2019/3/29 16:03'
'''
from natsort import natsorted
from glob import glob
import os
from skimage import morphology as sm
from skimage import measure as smea
from skimage import io as skio
import numpy as np
import shutil

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
# forbinden = ['0', '2', '3', '5', '6',
#              '8', '9', '10', '12', '13',
#              '14', '16', '18', '19', '21',
#              '22', '27', '28', '29', '30',
#              '32', '33', '34', '38', '40',
#              '41', '42', '43', '44', '45',
#              '48', '49', '54', '58', '59',
#              '60', '65', '68', '73', '78']
forbinden = ['0', '2', '3', '5', '6',
             '8', '9', '10', '12', '13',
             '14', '16', '18', '19', '21',
             '22', '27', '28', '29', '30',]
ip = r'E:\pyWorkspace\CAE\res\cp250'
op = r'E:\pyWorkspace\CAE\res\cleanSliceMore'

os.makedirs(op, exist_ok=True)

patientps = natsorted(glob(os.path.join(ip, '*')))
for pi, _p in enumerate(patientps):
    if os.path.basename(_p) in forbinden:
        continue
    else:
        paOp = os.path.join(op, os.path.basename(_p))
        if not os.path.exists(paOp): os.mkdir(paOp)
        labels = np.stack([skio.imread(_i) / 255 for _i in natsorted(glob(os.path.join(_p, 'labelClear', '*')))])
        suvs = np.stack([skio.imread(_i) for _i in natsorted(glob(os.path.join(_p, 'suv', '*')))])
        thresP = thres[pi][2]
        highSuv = suvs > thresP
        masked = highSuv * labels
        maskLabel = sm.label(masked, connectivity=1, )
        maskLabelRegions = smea.regionprops(maskLabel)

        for region in maskLabelRegions:
            llength = max([region.bbox[i + 3] - region.bbox[i] for i in range(3)])

            if llength <= 4:
                masked[maskLabel == region.label] = 0
        # 用两个for循环是因为，当大的连通区与小的连通区在同一切片序列中时，小连通区的标签没来得及去掉，就被大联通区写到了大联通区中的label中了
        sliceIndex = set()
        maskLabel2=sm.label(masked, connectivity=1)
        for ri, region in enumerate(smea.regionprops(maskLabel2)):
            suvIn = os.path.join(_p, 'suv')
            suvOut = os.path.join(paOp, 'suv')
            if not os.path.exists(suvOut): os.mkdir(suvOut)
            ctIn = os.path.join(_p, 'ct')
            ctOut = os.path.join(paOp, 'ct')
            if not os.path.exists(ctOut): os.mkdir(ctOut)
            labelIn = os.path.join(_p, 'labelClear')
            labelOut = os.path.join(paOp, 'labelClear')
            if not os.path.exists(labelOut): os.mkdir(labelOut)
            sliceMaxIndex = min(region.bbox[3] + 1, maskLabel2.shape[0])
            sliceMinIndex = max(region.bbox[0] - 1, 0)
            sliceIndex = sliceIndex | set(range(sliceMinIndex, sliceMaxIndex))
        lainxuIndex = -1
        continues = -1
        sliceIndex = list(sliceIndex)
        sliceIndex.sort()
        for i in sliceIndex:
            if continues != i:
                lainxuIndex += 1
                continues = i
            shutil.copy(os.path.join(suvIn, str(i) + '.tif'),
                        os.path.join(suvOut, str(lainxuIndex) + '_' + str(i) + '.tif'))
            shutil.copy(os.path.join(ctIn, str(i) + '.tif'),
                        os.path.join(ctOut, str(lainxuIndex) + '_' + str(i) + '.tif'))
            skio.imsave(os.path.join(labelOut, str(lainxuIndex) + '_' + str(i) + '.bmp'),
                        (masked[i, :, :] * 255).astype(np.uint8))
            continues += 1
