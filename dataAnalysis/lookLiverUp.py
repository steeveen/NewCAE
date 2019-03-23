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
 @Author = 'steven'   @DateTime = '2019/3/17 21:11'
'''
from natsort import natsorted
from glob import glob
import os
from skimage import io as skio
import numpy as np
for _p in natsorted(glob(os.path.join(r'G:\stage_one','*'))):
    if os.path.exists(os.path.join(_p,'liver_uptake.txt')):
        with open(os.path.join(_p,'liver_uptake.txt'),'r') as f:
            line=f.readlines()[0]
            print(line)
    else :
        print(' 1.0 ,0,1'  )