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
 @Author = 'steven'   @DateTime = '2019/2/24 14:01'
'''
import os
from natsort import natsorted
from glob import glob
patientps=natsorted(glob(os.path.join(r'E:\pyWorkspace\CAE\res\cp250','*')))
for _patientp in patientps:
    wrongImg=natsorted(glob(os.path.join(_patientp,'highAreaInfo','*.bmp')))
    for _p in wrongImg:
        os.remove(_p)
    wrongImg = natsorted(glob(os.path.join(_patientp, 'highAreaInfo', '*.png')))
    for _p in wrongImg:
        os.remove(_p)