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
 @Author = 'steven'   @DateTime = '2019/4/3 16:38'
'''
from FCDenseNet.tiramisu.runFCDenseNet import genePath
from natsort import natsorted
from  glob import glob
import os


trainIds = [4, 7, 15, 17, 20, 23, 24, 26, 31, 32,
            33, 34, 35, 36, 37, 38, 40, 42, 44, 45,
            ]
testIds = [46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
           57, 58, 60, 61, 62, 63, 65, 66, 68, 69,
           1, 11, 25, 39, 41, 43, 56, 59, 64, 67,
           70, 71, 72, 74, 76, 77, 78, 79, 73, 75]
ipRoot = r'E:\pyWorkspace\CAE\res\cleanSliceMore'


