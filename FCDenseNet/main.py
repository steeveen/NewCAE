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
from skimage.morphology import label
from skimage.measure import regionprops
dataRootp=r'D:\testData'
def datagene(mode='train'):
    thickness=64
    if mode=='train':
        dataList=natsorted(glob(os.path.join(r'dataRootp','*')))[:2]
    else:
        dataList=natsorted(glob(os.path.join(r'dataRootp','*')))[2:]
    while True:
        for _patientRoot in dataList:
            suvs=np.stack([skio.imread(_p) for  _p in natsorted(glob(os.path.join(_patientRoot,'suv')))])
            cts = np.stack([skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'ct')))])
            labels = np.stack([skio.imread(_p) for _p in natsorted(glob(os.path.join(_patientRoot, 'labelClear')))])/255
            suvs=np.pad(suvs,[0,3,3],mode='constant',constant_values=0)
            cts=np.pad(cts,[0,3,3],mode='constant',constant_values=0)
            labels=np.pad(labels,[0,3,3],mode='constant',constant_values=0)
            highAreaLabel=label((suvs>2.5).astype(np.uint8))
            highRegions=regionprops(highAreaLabel)
            for _hRegion in highRegions:
                highAreaLabel[highAreaLabel==_hRegion.label]=_hRegion.area
            x=[]
            y=[]
            for i in range(0,suvs.shape[0],thickness):
                if i +thickness<suvs.shape[0]:
                    suv=suvs[i:i+thickness,:,:]
                    _highAreaLabel=highAreaLabel[i:i+thickness,:,:]
                    ct=cts[i:i+thickness,:,:]
                    x.append(np.stack(suv,ct,_highAreaLabel))
                    y.append(labels[i:i+thickness,:,:])
                else:

                    suv = suvs[suvs.shape[0]-thickness:suvs.shape[0], :, :]
                    _highAreaLabel = highAreaLabel[suvs.shape[0]-thickness:suvs.shape[0], :, :]
                    ct = cts[suvs.shape[0]-thickness:suvs.shape[0], :, :]
                    x.append(np.stack(suv, ct, _highAreaLabel))
                    y.append(labels[i:i + 64, :, :])
            yield x,y


model=Tiramisu(n_classes=2,)
model.compile('adam',categorical_crossentropy,metrics=['acc'])
model.fit_generator(datagene('train'),steps_per_epoch=2,epochs=2,validation_data=datagene('test'),validation_steps=2)
