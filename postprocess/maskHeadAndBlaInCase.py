# -*- coding: utf-8 -*-
'''根据手动标注的头和膀胱进行分割结果中的头和膀胱假阳性去除
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
 @Author = 'steven'   @DateTime = '2019/4/23 17:20'
'''
import os
from natsort import natsorted
from glob import glob
from skimage import io as skio
import pickle as pkl
from matplotlib import cm

threshold = 0.5
root = r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\experimentCleanSlice3StackFocalLossSegFork\result_tts4-1(5)_1bfl030mse_bfl3,01_nlpb5_ndb5_red05_icf16_gr16_lr1e-4_dp05_Allr3 - 副本'
maskps = natsorted(glob(os.path.join(root, 'mask_*.pkl')))
for _mp in maskps:
    patientId = os.path.basename(_mp).split('.')[0].split('_')[1]
    patientOp = os.path.join(root, patientId)
    with open(_mp, 'rb') as f:
        mask = pkl.load(f)[1:-1, :, :]
        print(mask.max())
    with open(os.path.join(root, str(patientId) + '.pkl'), 'rb') as f:
        pre = pkl.load(f)
    maskedPre = pre * (1 - mask)
    with open(_mp.replace('mask_','masked_'),'wb') as f:
        pkl.dump(maskedPre,f)

    for i in range(maskedPre.shape[0]):
        skio.imsave(os.path.join(patientOp, str(i + 1) + '_preMdHeat.bmp'), cm.jet(maskedPre[i, :, :]))
        skio.imsave(os.path.join(patientOp, str(i + 1) + '_preMd.bmp'), (maskedPre[i, :, :] > threshold) * 255)
    for i in range(maskedPre.shape[1]):
        skio.imsave(os.path.join(patientOp, 'col_' + str(i) + '_preMdHeat.bmp'), cm.jet(maskedPre[:, i, :]))
        skio.imsave(os.path.join(patientOp, 'col_' + str(i) + '_preMd.bmp'), (maskedPre[:, i, :] > threshold) * 255)
