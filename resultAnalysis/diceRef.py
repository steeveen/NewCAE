# -*- coding: utf-8 -*-
'''计算分割结果的DICEref、DICEglo、VOLUME、SENSITIVITY
    指标参考：3D lymphoma segmentation in PET/CT images based on fully connected CRFs
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
 @Author = 'steven'   @DateTime = '2019/5/13 14:56'
'''
from skimage.morphology import label
from skimage.measure import regionprops
import  numpy as np
import pickle as pkl
from natsort import natsorted
from glob import glob
import csv
import os

def calSen1(gt,pre,c):
    '''
    第一种算法，算个数，所有的金标准连通区中，能与pre的交集大于它自身的50%时，作为tp
    :param gt:
    :param pre:
    :param c:
    :return:
    '''
    sen = 0
    gtLabel = label(gt, connectivity=c)
    for i in range(1, np.max(gtLabel) + 1):
        gi = (gtLabel == i).astype(np.uint8)
        if np.sum(gi * pre) / np.sum(gi) > iouThr:
            sen += 1
    return sen / np.max(gtLabel)
def calSen2(gt,pre,c):
    '''
    第二种算法，算体积，所有的金标准连通区中，能与pre的交集大于它自身的50%时，面积作为tp
    :param gt:
    :param pre:
    :param c:
    :return:
    '''
    sen = 0
    gtLabel = label(gt, connectivity=c)
    for i in range(1, np.max(gtLabel) + 1):
        gi = (gtLabel == i).astype(np.uint8)
        if np.sum(gi * pre) / np.sum(gi) > iouThr:
            sen += np.sum(gi * pre)
    return sen / np.sum(gt)
def calSen3(gt,pre,c):
    '''
    第三种算法，不管连通区与预测的交集面积多大，所有交集区域面积都作为tp
    :param gt:
    :param pre:
    :param c:
    :return:
    '''
    return np.sum(gt*pre)/np.sum(gt)



preThr=0.5
iouThr=0.5
connect=3
calSen=calSen2
# root=r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\experimentCleanSlice3StackFocalLossSegFork\result_tts4-1(5)_1bfl030mse_bfl3,01_nlpb5_ndb5_red05_icf16_gr16_lr1e-4_dp05_Allr3 - 副本'
preRoot=r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\experimentCleanSlice3StackFocalLossSegFork\result_tts4-1(5)_1bfl030mse_bfl3,01_nlpb5_ndb5_red05_icf16_gr16_lr1e-4_dp05_Allr3 - 副本'
gtRoot=r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\experimentCleanSlice3StackFocalLossSegFork\reference'
contents=['DI-G,DI-R,Volume,Sen,\n']
lst = [68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79]


if __name__ == '__main__':
    digg=[]
    dirr=[]
    vv=[]
    senn=[]
    for id in lst:
        gtp=os.path.join(gtRoot,str(id)+'_gt.pkl')
        prep=os.path.join(preRoot,'masked_'+str(id)+'.pkl')
        if not os.path.exists(prep):
            prep=os.path.join(preRoot,str(id)+'.pkl')
        with open(gtp,'rb') as f:
            gt=pkl.load(f)
        with open(prep  ,'rb') as f:
            pre=pkl.load(f)
            pre=(pre>=preThr).astype(np.uint8)
        DIG=2*np.sum(gt*pre)/(np.sum(gt)+np.sum(pre))
        digg.append(DIG)

        preLabel=label(pre,connectivity=connect)
        b=pre.copy()
        s=np.zeros(pre.shape)
        for i in range(1,np.max(preLabel)+1):
            bi=(preLabel==i).astype(np.uint8)
            if np.sum( bi*gt)==0:
                b[bi==1]=0
                s[bi==1]=1
        DIR=2*np.sum(gt*b)/(np.sum(gt)+np.sum(b))
        dirr.append(DIR)
        v=np.sum(s)/(np.sum(pre))
        vv.append(v)

        # sen=0
        # gtLabel=label(gt,connectivity=connect)
        # for i in range(1,np.max(gtLabel)+1):
        #     gi=(gtLabel==i).astype(np.uint8)
        #     if np.sum(gi*pre)/np.sum(gi)>iouThr:
        #         sen+=1
        # sen=sen/np.max(gtLabel)
        sen=calSen(gt,pre,connect)
        senn.append(sen)
        print('id:%d, dig:%.4f, dir:%.4f, v:%.4f, sen:%.4f'%(id,DIG,DIR,v,sen))
        contents.append(str(DIG)+','+str(DIR)+','+str(v)+','+str(sen)+',\n')
    contents.append(str(np.mean(digg))+','+str(np.mean(dirr))+','+str(np.mean(vv))+','+str(np.mean(senn))+',\n')
    print('mean, dig:%.4f, dir:%.4f, v:%.4f, sen:%.4f' % (np.mean(digg), np.mean(dirr), np.mean(vv), np.mean(senn)))
    with open(r'diceAnalysis_preThr=%f_iouThr=%f_connect=%d_calSen=%s.csv'%(preThr,iouThr,connect,calSen.__name__),'w') as f:
        for line in contents:
            f.write(line)