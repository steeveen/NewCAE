# -*- coding: utf-8 -*-
'''生成每个切片的高代谢面积掩模
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
 @Author = 'steven'   @DateTime = '2019/2/24 13:26'
'''
import os
from glob import glob
from natsort import natsorted
from skimage import io as skio
import numpy as np
from skimage.measure import regionprops
from skimage.morphology import label
rootP=r'E:\pyWorkspace\CAE\res\cp250'
patientps=natsorted(glob(os.path.join(rootP,'*')))
for _patientp in patientps:
    suvs=np.stack([skio.imread(_imgp) for _imgp in natsorted(glob( os.path.join(_patientp,'suv','*')))])
    high=(suvs>2.5).astype(np.uint8)
    highLabel=label(high)

    result=np.copy( highLabel).astype(np.uint64)
    regions=regionprops(highLabel)
    for _region in regions:
        result[result==_region.label]=_region.area

    op=os.path.join(_patientp,'highAreaInfo')
    os.mkdir(op) if not os.path.exists(op) else None
    for i in range(result.shape[0]):
        skio.imsave(os.path.join(op,str(i)+'.png'),result[i,:,:])