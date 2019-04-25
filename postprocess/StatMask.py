# -*- coding: utf-8 -*-
'''统计头和膀胱的手动标签，对所有病例进行取并集。
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
 @Author = 'steven'   @DateTime = '2019/4/23 16:25'
'''
from natsort import natsorted
from glob import glob
import os
import pickle as pkl
from skimage.transform import resize
import numpy as np
from skimage import io as skio

maskRoot = r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\experimentCleanSlice3StackFocalLossSegFork\result_tts4-1(5)_1bfl030mse_bfl3,01_nlpb5_ndb5_red05_icf16_gr16_lr1e-4_dp05_Allr3'
lst = natsorted(glob(os.path.join(maskRoot, 'mask*.pkl')))
final=np.zeros((558,250,250))
for _ in lst:
    with open(_, 'rb') as f:
        data = pkl.load(f)
        final=np.logical_or(resize(data, (558, 250, 250), preserve_range=True)>0,final)
final=final.astype(np.uint8)
final*=255
with open(r'HeaBlaMask.pkl','wb') as f:
    pkl.dump(final,f)
os.mkdir(r'HeaBlaMask') if not os.path.exists(r'HeaBlaMask') else None
for i in range(final.shape[1]):
    skio.imsave(os.path.join(r'HeaBlaMask',str(i)+'.bmp'),final[:,i,:])