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
 @Author = 'steven'   @DateTime = '2019/5/14 16:14'
'''
from natsort import natsorted
from glob import  glob
import os
import  pickle as pkl
import numpy as np
from skimage import io as skio
from skimage .morphology import label
def ef():
    pass
def main():

    con=1
    gtRoot=r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\experimentCleanSlice3StackFocalLossSegFork\reference'
    preRoot=r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\experimentCleanSlice3StackFocalLossSegFork\result_tts4-1(5)_1bfl030mse_bfl3,01_nlpb5_ndb5_red05_icf16_gr16_lr1e-4_dp05_Allr3 - 副本'
    dataRoot=r'E:\pyWorkspace\CAE\res\cp250'
    lst=[68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79]
    # gtps=natsorted(glob(os.path.join(gtRoot,'*_gt.pkl')))
    # preps=natsorted(glob(os.path.join(preRoot,'[0-9]*.pkl')))
    for pid in lst:
        with open( os.path.join(gtRoot,str(pid)+'_gt.pkl') ,'rb') as f:
            gt=pkl.load(f)
        with open(os.path.join(preRoot,str(pid)+'.pkl'),'rb') as f:
            pre=pkl.load(f)
        suv=np.stack([skio.imread(_) for _ in natsorted(glob(os.path.join(dataRoot,str(pid),'suv*')))[1:-1]])
        ct=np.stack([skio.imread(_) for _ in natsorted(glob(os.path.join(dataRoot,str(pid),'ct*')))[1:-1]])
        preLabel=label(pre,connectivity=con)
        for i in range(1, np.max(preLabel)+1):
            prei=(preLabel==i).astype(np.uint8)
            feature=ef(prei,ct,suv)
            if np.sum(prei*gt)/np.sum(prei)>0.3:
                clazz=1
            else:
                clazz=0

if __name__ == '__main__':
    main()



