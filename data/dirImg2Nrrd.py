# -*- coding: utf-8 -*-
''' 将图像文件夹转为nii，用于使用itksnap打标签（头部和膀胱），并将标签的nii转为pickle
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
 @Author = 'steven'   @DateTime = '2019/4/21 18:45'
'''
from glob import glob
from natsort import natsorted
import pickle as pkl
import os
import nrrd
import numpy as np
import nrrd as nd
import nibabel as nib
from skimage import io as skio


def convertImg2Nii(ip, op):
    data = np.stack([skio.imread(_) for _ in natsorted(glob(os.path.join(ip, '*')))], axis=0)
    print(data.shape)
    nib.save(nib.Nifti1Image(data, affine=np.eye(4), ), op + '.nii')


def convertNii2Pkl(ip, op):
    maskNii = nib.load(ip)
    maskN = np.array(maskNii.dataobj)
    maskN = (maskN > 0).astype(np.uint8)
    with open(op, 'wb') as f:
        pkl.dump(maskN, f)


def convertImg2NiiAll():
    name = [68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79]
    for i in name:
        ip = os.path.join(r'E:\pyWorkspace\CAE\res\cp250', str(i), 'suv')
        print(ip)
        op = os.path.join(
            r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\experimentCleanSlice3StackFocalLossSegFork\result_tts4-1(5)_1bfl030mse_bfl3,01_nlpb5_ndb5_red05_icf16_gr16_lr1e-4_dp05_Allr3',
            str(i) + '_suv')
        convertImg2Nii(ip, op)


def convertNii2PklAll():
    name = [68, 69, 70, 72, 73, 74, 75, 76, 77, 79]
    for i in name:
        ip = os.path.join(
            r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\experimentCleanSlice3StackFocalLossSegFork\result_tts4-1(5)_1bfl030mse_bfl3,01_nlpb5_ndb5_red05_icf16_gr16_lr1e-4_dp05_Allr3',
            'mask_' + str(i) + '.nii')
        op = os.path.join(
            r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\experimentCleanSlice3StackFocalLossSegFork\result_tts4-1(5)_1bfl030mse_bfl3,01_nlpb5_ndb5_red05_icf16_gr16_lr1e-4_dp05_Allr3',
            'mask_' + str(i) + '.pkl')
        print(ip)
        convertNii2Pkl(ip, op)


if __name__ == '__main__':
    # convertImg2NiiAll
    convertNii2PklAll()
    # ip = r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\experimentCleanSlice3StackFocalLossSegFork\result_tts4-1(5)_1bfl030mse_bfl3,01_nlpb5_ndb5_red05_icf16_gr16_lr1e-4_dp05_Allr3\68Mask.nii'
    # op = r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\experimentCleanSlice3StackFocalLossSegFork\result_tts4-1(5)_1bfl030mse_bfl3,01_nlpb5_ndb5_red05_icf16_gr16_lr1e-4_dp05_Allr3\68_mask.pkl'
    # Nii2pickle(ip,op)
