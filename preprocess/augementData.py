# -*- coding: utf-8 -*-
'''增强数据，对cutHighBlock生成的数据，正例进行旋转扩充， 反例随机消减，计划先只做train的
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
 @Author = 'steven'   @DateTime = '2019/3/10 14:47'
'''
from natsort import natsorted
from glob import glob
import os
import numpy as np
import pickle as pkl
import shutil


def dataArgement(ipRoot,opRoot ,times=10, pklBatchSize=1024):
    from skimage.transform import rotate

    if not os.path.exists(opRoot):
        os.mkdir(opRoot)
    lst = natsorted(glob(os.path.join(ipRoot, '*')))

    fileIndex=0
    negFiles=[]

    for _lst in lst:
        print(_lst)
        with open(_lst, 'rb') as f:
            data = pkl.load(f)
            ct = data['x'][:, :, :, 0]
            suv = data['x'][:, :, :, 0]
            volume = data['x'][:, :, :, 0]
            gt = data['y']
            needed = np.sum(gt) > 0
            if needed:
                for ai in range(0, 360, 360 // times):
                    newCt = rotate(ct, ai, preserve_range=True, order=3)
                    newSuv = rotate(suv, ai, preserve_range=True, order=3)
                    newVolume = rotate(volume, ai, preserve_range=True, order=3)
                    newGt = (rotate(gt, ai, preserve_range=True, order=3) > 0.5).astype(np.int8)
                    with open(os.path.join(opRoot,'aug='+str(fileIndex)+'_'+os.path.basename(_lst)),'wb') as f:
                        pkl.dump({'x':np.stack([newCt,newSuv,newVolume],axis=-1),'y':newGt},f)
                        fileIndex+=1
            else:
                negFiles.append(_lst)
    #根据正例数量，随机保留反例数量
    np.random.shuffle(negFiles)
    negFiles=negFiles[:fileIndex]
    print(fileIndex)
    for _p in negFiles:
        shutil.copy(_p,os.path.join(opRoot,'aug='+str(fileIndex)+'_'+os.path.basename(_p)))
if __name__ == '__main__':
    dataArgement(r'E:\pyWorkspace\CAE\res\highSuvBlock\train',r'E:\pyWorkspace\CAE\res\highSuvArgued')
