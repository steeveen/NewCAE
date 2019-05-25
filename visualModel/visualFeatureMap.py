from keras import backend as K
import os
import numpy as np
from keras.models import load_model
from FCDenseNet.myDense.trainDenseSegFork import binary_focal_loss, dice, recall, precision, mse
import pickle as pkl
from skimage import io as skio
from matplotlib import  pyplot as plt
import matplotlib as mpl


def deprocessImg(x):
    x -= x.mean()
    x /= (x.std() + (1e-5))
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x*=255
    x=np.clip(x,0,255).astype(np.uint8)
    return x


def visualModel(model, layerName,  inputShape, iterTimes=40,filterIndex=None, opRoot='visual', iterStep=1,useMean=False):
    '''

    :param model: 模型
    :param layerName: 可视化哪一层，层名（在summary的打印结果中能找到）
    :param inputShape: 模型输入的形状（比如（256,256），三通道是（256,256,3））
    :param filterIndex: 默认就好
    :param iterTimes: 迭代的次数，理论上来说，次数越大越好，默认40
    :param iterStep: 类似于学习率，默认1 ，用默认1就好
    :param opRoot: 输出图像的文件夹路径，默认为当前目录的visual文件夹下
    :return:
    '''
    os.mkdir(opRoot) if not os.path.exists(opRoot) else None
    layout_op = model.get_layer(layerName).output


    if filterIndex==None:
        filterIndex=range(layout_op.shape[-1])
    if isinstance(filterIndex,int):

        filterIndex=[filterIndex]
    print('filterIndex is '+str(filterIndex))

    norm = mpl.colors.Normalize(vmin=0, vmax=255)

    for _filterIndex in filterIndex:
        loss = K.mean(layout_op[:, :, :, _filterIndex])

        grads = K.gradients(loss, model.get_input_at(0))[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        iterate = K.function([model.get_input_at(0)], [loss, grads])

        inputImg = np.random.random((1,)+inputShape)
        print(inputImg.shape)
        for i in range(iterTimes):
            loss_value, grads_value = iterate([inputImg])
            inputImg += grads_value * iterStep
        img =deprocessImg (inputImg[0])

        picklePath=os.path.join(opRoot,'pkl')
        os.mkdir(picklePath) if not os.path.exists(picklePath) else None
        with open(os.path.join(picklePath , layerName+'_its%d_%d.pkl' % (iterTimes,_filterIndex)), 'wb') as f:
            pkl.dump(img, f)


        if useMean:
            skio.imsave(os.path.join( opRoot ,layerName+'_its%d_f%d_Mean.png' % (iterTimes,_filterIndex)),  norm(np.mean(img,axis=-1)))
        else:
            for i in range(img.shape[-1]):
                skio.imsave(os.path.join( opRoot ,layerName+'_its%d_f%d_c%d.png' % (iterTimes,_filterIndex,i)), norm(img[:,:,i]))
    return img


if __name__ == '__main__':
    modelp = r'E:\pyWorkspace\NewCAE\FCDenseNet\myDense\experimentCleanSlice3StackFocalLossSegFork\checkPoint_tts4-1(5)_1bfl030mse_bfl3,01_nlpb5_ndb5_red05_icf16_gr16_lr1e-4_dp05\035-0.071842-0.615032-0.848771-0.546990.hdf5'
    model = load_model(modelp,
                       custom_objects={'focal_loss': binary_focal_loss(gamma=3, alpha=0.1), 'dice': dice,
                                       'recall': recall, 'precision': precision, 'mse': mse})
    tar=model.get_layer('model_1')
    tar.summary()

    featureMap = visualModel(tar, 'dense_0_1_conv2D', ( 128, 128, 6),iterTimes=500 ,useMean=True,opRoot='visualMeanBmp')
