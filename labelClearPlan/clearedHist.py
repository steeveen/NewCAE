# -*- coding: utf-8 -*-
'''清除小的label后，看看大小分布的直方图
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
 @Author = 'steven'   @DateTime = '2019/2/23 18:48'
'''
from skimage import  io as skio
from natsort import natsorted
import os
from glob import glob
import numpy as np
from skimage.measure import label
from skimage.measure import regionprops
from matplotlib import pyplot as plt

p=r'E:\pyWorkspace\CAE\res\cp250'
patientps=natsorted(glob(os.path.join(p,'*','labelClear')))
allregionArea=[]
for _p in patientps:

    imgps=natsorted(glob(os.path.join(_p,'*')))
    gts=np.stack(skio.imread(_imgp)/255 for _imgp in imgps)
    gtLabel=label(gts)
    gtRegions=regionprops(gtLabel)
    for _region in gtRegions:
        print(_region.area)
        allregionArea.append(_region.area)
    print('---------------------------------')
print('+++++++++++++++++++++++++++++')
print(allregionArea.__len__())
print(np.mean(allregionArea))
print(max(allregionArea))
print(min(allregionArea))
plt.hist(allregionArea,100)
plt.show()