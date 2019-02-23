# -*- coding: utf-8 -*-
'''分析使用阈值过滤后，病人的连通区情况
数据来源：E:\pyWorkspace\CAE\res\cp250\0
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
 @Belong = 'Fuck'  @MadeBy = 'PyCharm'
 @Author = 'steven'   @DateTime = '2019/2/23 14:56'
'''
import os
from natsort import natsorted
from glob import glob
from skimage import io as skio
import numpy     as np
from skimage.morphology import label

from skimage.measure import regionprops
from skimage.color import label2rgb
suvp=r'E:\pyWorkspace\CAE\res\cp250\0\suv'
imgps=natsorted(glob(os.path.join(suvp,'*.tif')))
suvs=np.stack([skio.imread(_p)  for _p in imgps])
op=r'E:\pyWorkspace\CAE\try0\tryThres2.5'
thresholdV=2.5
ts=label2rgb(label(suvs>thresholdV))
from skimage import color
for i in range(ts.shape[0]):
    skio.imsave(os.path.join(op,str(i)+'.bmp'),np.round(ts[i,:,:,:]*255).astype(np.uint8))

