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
 @Author = 'steven'   @DateTime = '2019/5/13 14:46'
'''
import  numpy as np
from natsort import natsorted
from glob import glob
import os
import pickle as pkl
from skimage import io as skio

def convertGtimg2Pkl(ip,op):
    gt= np.stack([(skio.imread(_) / 255).astype(np.uint8) for _ in
                        natsorted(glob(os.path.join(ip, '*')))])[1:-1, :, :]
    # 去掉第一张和最后一张
    with open(op,'wb') as f:
        pkl.dump(gt,f)

if __name__ == '__main__':
    gtRoot= r'E:\pyWorkspace\CAE\res\cp250'
    opRoot=r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\experimentCleanSlice3StackFocalLossSegFork\result_tts4-1(5)_1bfl030mse_bfl3,01_nlpb5_ndb5_red05_icf16_gr16_lr1e-4_dp05_Allr3 - 副本'
    lst = [68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79]
    for i in lst:
        ip=os.path.join(gtRoot,str(i),'labelClear')
        op=os.path.join(opRoot,str(i)+'_gt.pkl')
        convertGtimg2Pkl(ip,op)