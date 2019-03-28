# -*- coding: utf-8 -*-
'''在E:\pyWorkspace\CAE\res\highSuvPatch\bigPatch中，有1285块正例，9387块负例
    按比例从其中抽取1/9作为验证集，8/9作为训练集
    所有pickle都是3x32x32的
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
 @Author = 'steven'   @DateTime = '2019/3/23 16:55'
'''
import os
from natsort import natsorted
from glob import glob
import pickle as pkl
import numpy as np
import shutil


def select(ip, op, ttS):
    '''
    将使用到的数据从ip中挑选出来并分成train和test，按照tts比率
    :param ip:
    :param op:
    :return:
    '''

    os.mkdir(op) if not os.path.exists(op) else None

    def splitAndSave(srcRoot, opRoot, ttS, pattern):
        trainOp = os.path.join(opRoot, 'train')
        testOp = os.path.join(opRoot, 'test')
        os.mkdir(trainOp) if not os.path.exists(trainOp) else None
        os.mkdir(testOp) if not os.path.exists(testOp) else None

        lst = natsorted(glob(os.path.join(srcRoot, pattern)))
        np.random.seed(0)
        np.random.shuffle(lst)
        splitIndex = int(ttS * len(lst))
        trainLst = lst[:splitIndex]
        testLst = lst[splitIndex:]
        for _p in trainLst:
            shutil.copy(_p, os.path.join(trainOp, os.path.basename(_p)))
        for _p in testLst:
            shutil.copy(_p, os.path.join(testOp, os.path.basename(_p)))

    splitAndSave(ip, op, ttS, '*G1.pkl')
    splitAndSave(ip, op, ttS, '*G0.pkl')


def arguData(ipRoot, opRoot, times):
    from skimage.transform import rotate

    if not os.path.exists(opRoot):
        os.mkdir(opRoot)
    lst = natsorted(glob(os.path.join(ipRoot,'*G1.pkl')))

    for _lst in lst:
        print(_lst)
        with open(_lst, 'rb') as f:
            data = pkl.load(f)
            ct = data['ct']
            suv = data['suv']
            gt = data['gt']
            for ai in range(0, 360, 360 // times):
                newCt = rotate(ct.transpose((1, 2, 0)), ai, preserve_range=True, order=3, ).transpose((2, 0, 1))
                newSuv = rotate(suv.transpose((1, 2, 0)), ai, preserve_range=True, order=3).transpose((2, 0, 1))
                newGt = (rotate(gt.transpose((1, 2, 0)), ai, preserve_range=True, order=3) > 0.5).astype(
                    np.int8).transpose((2, 0, 1))
                with open(os.path.join(opRoot, 'r' + str(ai) + '_' + os.path.basename(_lst)), 'wb') as f:
                    pkl.dump({'ct': newCt, 'suv': newSuv, 'gt': newGt}, f)
        os.remove(_lst)

def main():
    ip = r'E:\pyWorkspace\CAE\res\highSuvPatch\bigPatch'
    op = r'E:\pyWorkspace\CAE\res\BigTomurData'
    select(ip, op, 0.9)
    arguData(os.path.join(op,'train'), os.path.join(op,'train'), times=8)
    # arguData(os.path.join(op, 'test'), os.path.join(op, 'test'), times=8)


if __name__ == '__main__':
    main()
