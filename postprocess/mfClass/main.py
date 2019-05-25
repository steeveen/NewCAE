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
 @Author = 'steven'   @DateTime = '2019/5/14 17:27'
'''
import pickle as pkl
from sklearn.model_selection import train_test_split
dataP=r''
def loadData():
    with open(dataP,'rb') as f:
        data=pkl.load(f)
        x=data['x']
        y=data['y']
    return x,y
def main():
    x,y=loadData()
    trainX,testX,trainY,testY=train_test_split(x,y,0.7,stratify=y)
if __name__ == '__main__':
    main()
