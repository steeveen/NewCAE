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
 @Author = 'steven'   @DateTime = '2019/2/23 20:42'
'''
from FCDenseNet.tiramisu_net import Tiramisu
import os
from natsort import natsorted
from glob import glob
import  numpy as np
from keras.callbacks import  CSVLogger
from keras.losses import  categorical_crossentropy
from skimage import io as skio
def datagene(mode='train'):
    if mode=='train':
        dataList=natsorted(glob(os.path.join(r'E:\pyWorkspace\CAE\res\cp250','*')))[0:70]
    else:
        dataList=natsorted(glob(os.path.join(r'E:\pyWorkspace\CAE\res\cp250','*')))[70:80]
    while True:

    suv= np.stack([skio.imread() for  _p in dataList])


model=Tiramisu(n_classes=2,)
model.compile('adam',)
