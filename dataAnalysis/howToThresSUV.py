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
 @Author = 'steven'   @DateTime = '2019/3/17 20:17'
'''
from natsort import natsorted
from glob import glob
import os
from skimage import io as skio
import numpy as np

thres = [[1.0, 0, 1],
         [1.0, 0, 1],
         [1.68675, 0.270291, 1.685],
         [

             1.78862, 0.20133, 1.765], [

             2.02084, 0.419678, 1.905], [

             2.70377, 0.308884, 2.705], [

             1.86265, 0.225558, 1.865], [

             1.82487, 0.318216, 1.785], [

             2.30139, 0.19442, 2.285],
         [

             2.02857, 0.377839, 2.095],
         [

             1.81117, 0.198819, 1.795],
         [

             1.99146, 0.178486, 1.985],
         [

             1.25853, 0.144098, 1.235],
         [

             2.39119, 0.311516, 2.395],
         [

             1.7631, 0.0969169, 1.755],
         [

             2.46485, 0.581147, 2.535],
         [

             1.4100, 0.41, 1.4000],
         [

             1.6100, 0.41, 1.6],
         [

             2.37, 0.4, 2.3001],
         [

             1.85, 0.40, 1.8],
         [

             1.21, 0.3, 1.20],
         [

             2.09735, 0.344403, 2.055],
         [

             2.00127, 0.189155, 1.995],
         [

             2.27763, 0.456008, 2.235],
         [

             1.70926, 0.219484, 1.695],
         [

             1.9981, 0.23742, 1.975],
         [

             1.0, 0, 1],
         [
             1.99561, 0.27911, 1.975],
         [

             2.27263, 0.222743, 2.275],
         [

             2.22544, 0.310421, 2.195],
         [

             2.001, 0.2, 2.001],
         [

             2.32624, 0.283438, 2.305],
         [

             2.38536, 0.34569, 2.375],
         [

             2.66131, 0.348425, 2.645],
         [

             2.15941, 0.298108, 2.145],
         [

             1.0, 0, 1],
         [
             2.54272, 0.382998, 2.555],
         [

             2.70791, 0.352986, 2.665
         ],
         [
             2.40256, 0.252971, 2.385
         ],
         [
             2.09735, 0.344403, 2.055
         ],
         [
             1.12, 0.2, 1.12],
         [

             1.75, 0.3, 1.75],
         [

             1.56853, 0.26133, 1.545],
         [

             2.38536, 0.34569, 2.375],
         [

             2.06371, 0.403317, 2.055],
         [

             1.700, 0.3, 1.75],
         [

             2.46485, 0.581147, 2.535],
         [

             1.0, 0, 1],
         [
             2.02857, 0.377839, 2.095
         ],
         [
             2.0, 0.3, 2.001
         ],
         [
             2.0, 0.2, 2.0
         ],
         [
             1.0, 0, 1],
         [
             1.83376, 0.327339, 1.795],
         [

             2.70377, 0.308884, 2.705],
         [

             2.24014, 0.243368, 2.225],
         [

             2.0001, 0.2, 2.01],
         [

             1.7645, 0.211, 1.7443],
         [

             2.78805, 0.262183, 2.775],
         [

             2.72396, 0.288798, 2.695],
         [

             2.17603, 0.581752, 2.125],
         [

             2.70441, 0.383908, 2.695],
         [

             1.0, 0, 1],
         [
             3.31165, 2.20634, 2.475],
         [

             1.75, 0.3, 1.77
         ],
         [
             2.23887, 0.342719, 2.235],
         [

             2.37406, 0.274724, 2.355
         ],
         [
             2.37406, 0.274724, 2.355
         ],
         [
             2.27653, 0.246097, 2.275
         ],
         [
             3.10535, 0.370482, 3.105
         ],
         [
             2.30156, 0.196611, 2.285
         ],
         [
             2.23633, 0.231912, 2.215
         ],
         [
             1.0, 0, 1],
         [
             2.0078, 0.187011, 2.005
         ],
         [
             1.005, 0.211, 1.006
         ],
         [
             1.94519, 0.333655, 1.915
         ],
         [
             1.73983, 0.185099, 1.745
         ],
         [
             1.92518, 0.31164, 1.915
         ],
         [
             2.20173, 0.320697, 2.205
         ],
         [
             1.68257, 0.187831, 1.675
         ],
         [
             1.82114, 0.154759, 1.805]]


def show(suvM, gtM, m):
    cross = suvM * gtM
    miss = np.sum((cross.astype(np.int8) - gtM.astype(np.int8)) < 0)
    print(
        _p + ' ' + m + ' gtNum:' + str(np.sum(gtM)) + '\tmiss:%d/%d %.4f' % (miss, np.sum(gtM), miss / np.sum(gtM)) +
        '\t keep:%d/%d %.4f' % (np.sum(cross), np.sum(gtM), np.sum(cross) / np.sum(gtM)) +
        ' \trubbish:%d/%d %.4f' % (
            np.sum(suvM.astype(np.int8) - cross), np.sum(gtM), np.sum(suvM.astype(np.int8) - cross) / np.sum(gtM)))
    return np.sum(gtM), miss, np.sum(cross), np.sum(suvM.astype(np.int8) - cross)


if __name__ == '__main__':
    ip = r'E:\pyWorkspace\CAE\res\cp250'
    from skimage import morphology as sm

    totalGtm = 0
    totalmiss = 0
    totalKeep = 0
    totalRubbish = 0

    totalTempGtm = 0
    totalTempmiss = 0
    totalTempKeep = 0
    totalTempRubbish = 0

    for i, _p in enumerate(natsorted(glob(os.path.join(ip, '*')))):
        if os.path.basename(_p) in ['0', '2', '3', '5', '6',
                                    '8', '9', '10', '12', '13',
                                    '14', '16', '18', '19', '21',
                                    '22', '27', '28', '29', '30',
                                    '32', '33', '34', '38', '40',
                                    '41', '42', '43', '44', '45',
                                    '48', '49', '54', '58', '59',
                                    '60', '65', '68', '73', '78']:
            continue
        print('---------------'+_p+'-----------------')
        suvs = np.stack([skio.imread(_) for _ in natsorted(glob(os.path.join(_p, 'suv', '*')))])

        gts = np.stack([skio.imread(_) for _ in natsorted(glob(os.path.join(_p, 'labelClear', '*')))]) > 0

        thr=thres[i][0]-thres[i][1]
        print('suvThre:' + str(thr))

        gtm, miss, keep, rubbish = show(suvs > thr, gts, 'media-0.3')
        totalGtm += gtm
        totalmiss += miss
        totalKeep += keep
        totalRubbish += rubbish

        # gts = sm.erosion(gts, sm.ball(1))
        thr = thres[i][2] - thres[i][1]
        print('suvThre:' + str(thr))
        gtm, miss, keep, rubbish = show(suvs > thr, gts, 'media+e2')
        totalTempGtm += gtm
        totalTempmiss += miss
        totalTempKeep += keep
        totalTempRubbish += rubbish
    print(totalmiss / totalGtm)
    print(totalKeep / totalGtm)
    print(totalRubbish / totalGtm)
    print('--------------------')
    print(totalTempmiss / totalTempGtm)
    print(totalTempKeep / totalTempGtm)
    print(totalTempRubbish / totalTempGtm)