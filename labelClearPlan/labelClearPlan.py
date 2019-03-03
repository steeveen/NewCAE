# -*- coding: utf-8 -*-
'''标签清除计划
咨询过猛哥， 标签中，体积小于5x5x5的连通区，很大可能上是标签标注时候的谬误。
所以此代码对这些小的区域进行消除
数据：E:\pyWorkspace\CAE\res\cp250
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
 @Author = 'steven'   @DateTime = '2019/2/23 16:08'
'''
from skimage import  io as skio
from natsort import natsorted
import os
from glob import glob
import numpy as np
from skimage.measure import label
from skimage.measure import regionprops
p=r'E:\pyWorkspace\CAE\res\cp250'
patientps=natsorted(glob(os.path.join(p,'*','label')))
for _p in patientps:
    imgps=natsorted(glob(os.path.join(_p,'*')))
    gts=np.stack(skio.imread(_imgp)/255 for _imgp in imgps)
    gtLabel=label(gts,connectivity=2)
    gtRegions=regionprops(gtLabel)
    for _region in gtRegions:
        l=np.max([_region.bbox[i+3]-_region.bbox[i] for i in range(3)])
        if l<5:
            gtLabel[gtLabel==_region.label]=0
    maskdLabel=(gtLabel>0).astype(np.uint8)*255
    patientOp=_p+'Clear'
    os.mkdir(patientOp) if not os.path.exists(patientOp) else None
    for i in range(maskdLabel.shape[0]):
        skio.imsave(os.path.join(patientOp,str(i)+'.bmp'),maskdLabel[i,:,:])

