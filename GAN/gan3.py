# -*- coding: utf-8 -*-
'''gan的修改版，借鉴了wgan-gp的思想
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
 @Author = 'steven'   @DateTime = '2019/3/16 13:02'
'''

import pickle as pkl

from keras.optimizers import RMSprop
from skimage import io as skio
import keras
from keras.layers import Input, Dense, BatchNormalization, Activation, Flatten, Conv2D, UpSampling2D, LeakyReLU, \
    Dropout, Reshape
from keras.layers.merge import _Merge
import numpy as np
from natsort import natsorted
from glob import glob
import os
from skimage.transform import resize
import time
import keras.backend as K
from keras.utils import plot_model
from functools import partial
from keras.models import Model


class RandomWeightAverage(_Merge):
    'provides  a  random weighted average between real and generated image samples'

    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 32, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def acc(y_true, y_pred):
    import keras.backend as K
    return K.mean(K.equal(y_true > 0, y_pred > 0), axis=-1)


class MyGan():
    def __init__(self, realRoot, fakeRoot, chRoot, geneImgPath):
        self.trainRato = 0.9
        self.realRoot = realRoot
        self.fakeRoot = fakeRoot
        self.half_batch_size = 4
        self.epoch = 200
        self.saveporid = 2
        self.trainDTimes = 5
        self.valBatchSize = 2
        self.gModelBeginEpochThres = 0
        self.clip_value = 0.5
        self.bnm = 0.8

        os.makedirs(chRoot, exist_ok=True)
        os.makedirs(geneImgPath, exist_ok=True)

        self.geneImgPath = geneImgPath
        self.chechPointRoot = os.path.join(chRoot, '%d_dLoss=%.6f_dAcc=%.6f_gLoss=%.6f_gAcc=%.6f')
        self.logPath = os.path.join(chRoot, 'log3.csv')
        if not os.path.exists(self.logPath):
            with open(self.logPath, 'w') as f:
                f.write('epoch,dLoss,dAcc,gLoss,gAcc\n')
        else:
            with open(self.logPath, 'a') as f:
                f.write('epoch,dLoss,dAcc,gLoss,gAcc\n')
        self.ImgOp = r'E:\pyWorkspace\NewCAE\GAN\geneImg3'
        os.makedirs(self.ImgOp, exist_ok=True)

        self.realPaths = natsorted(glob(os.path.join(realRoot, '*')))
        np.random.seed(1)
        np.random.shuffle(self.realPaths)

        self.fakePaths = natsorted(glob(os.path.join(fakeRoot, '*')))
        np.random.seed(1)
        np.random.shuffle(self.fakePaths)
        self.fakeTrainPaths = self.fakePaths[:int(self.trainRato * len(self.fakePaths))]
        self.fakeValPaths = self.fakePaths[int(self.trainRato * len(self.fakePaths)):]

        print('build discriminator')
        self.dModel = self.getDiscriminator()

        print('build generator')
        self.gModel = self.getGenerator()

        realImg = Input(shape=(32, 32, 1))
        fakeInput = Input(shape=(8, 8, 1))

        fakeImg = self.gModel(fakeInput)

        fake = self.dModel(fakeImg)
        valid = self.dModel(realImg)

        interpolated_img = RandomWeightAverage()([realImg, fakeImg])
        validity_interpolated = self.dModel(interpolated_img)

        partial_gp_loss = partial(self.gradient_penalty_loss, averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'

        optimizer = RMSprop(lr=0.00005)

        self.dModelModel = Model(inputs=[realImg, fakeInput],
                                 outputs=[valid, fake, validity_interpolated],
                                 name='dModelModel')
        self.dModelModel.compile(loss=[self.wLoss, self.wLoss, partial_gp_loss], optimizer=optimizer,
                                 loss_weights=[1, 1, 10], metrics=['acc'])

        self.dModel.trainable = False
        self.gModel.trainable = True

        z_gen = Input((8, 8, 1))
        img = self.gModel(z_gen)
        valid = self.dModel(img)
        self.gModelModel = Model(z_gen, valid, name='gModelModel')
        self.gModelModel.compile(loss=self.wLoss, optimizer=optimizer,metrics=['acc'])

        plot_model(self.gModelModel, 'gModelModel3.png', show_shapes=True)
        plot_model(self.dModelModel, 'dModelModel3.png', show_shapes=True)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        '''
        梯度惩罚
        :param y_true:
        :param y_pred:
        :param averaged_samples:
        :return:
        '''
        gradients = K.gradients(y_pred, averaged_samples)
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.get_shape())))
        gradients_l2_norm = K.sqrt(gradients_sqr_sum)
        gradients_penalty = K.square(1 - gradients_l2_norm)
        return K.mean(gradients_penalty)

    def getGenerator(self):
        '''
        生成器（generator）
        首先，创建一个“生成器（generator）”模型，它将一个矢量（从潜在空间 - 在训练期间随机采样）转换为候选图像。
        GAN通常出现的许多问题之一是generator卡在生成的图像上，看起来像噪声。一种可能的解决方案是在鉴别器（discriminator）
        和生成器（generator）上使用dropout。
        '''
        # generator_input = keras.Input(shape=(None, None,1))
        generator_input = keras.Input(shape=(8, 8, 1))

        x = UpSampling2D((2, 2), interpolation='bilinear')(generator_input)
        x = Conv2D(128, 5, padding='same')(x)
        x = BatchNormalization(momentum=self.bnm)(x)
        x = LeakyReLU()(x)
        # x = Conv2D(128, 5, padding='same')(x)
        # x = BatchNormalization(momentum=self.bnm)(x)
        # x = LeakyReLU()(x)
        # x = Conv2D(256, 5, padding='same')(x)
        # x = BatchNormalization(momentum=self.bnm)(x)
        # x = LeakyReLU()(x)

        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Conv2D(256, 5, padding='same')(x)
        x = BatchNormalization(momentum=self.bnm)(x)
        x = LeakyReLU()(x)
        # x = Conv2D(256, 5, padding='same')(x)
        # x = BatchNormalization(momentum=self.bnm)(x)
        # x = LeakyReLU()(x)
        # x = Conv2D(256, 5, padding='same')(x)
        # x = BatchNormalization(momentum=self.bnm)(x)
        # x = LeakyReLU()(x)

        # 添加更多的卷积层
        # x = Conv2D(256, 5, padding='same')(x)
        # x = BatchNormalization(momentum=self.bnm)(x)
        # x = LeakyReLU()(x)
        x = Conv2D(128, 5, padding='same')(x)
        x = BatchNormalization(momentum=self.bnm)(x)
        x = LeakyReLU()(x)
        # x = Conv2D(512, 5, padding='same')(x)
        # x = BatchNormalization(momentum=self.bnm)(x)
        # x = LeakyReLU()(x)

        # 生成一个 32x32 1-channel 的feature map
        x = Conv2D(1, 7, activation='tanh', padding='same')(x)
        generator = keras.models.Model(generator_input, x, name='gModel')
        generator.summary()
        plot_model(generator, 'gModel3.png', show_shapes=True)
        return generator

    def wLoss(self, y_t, y_p):
        return K.mean(y_t * y_p)

    def getDiscriminator(self):
        '''
        discriminator(鉴别器)
        创建鉴别器模型，它将候选图像（真实的或合成的）作为输入，并将其分为两类：“生成的图像”或“来自训练集的真实图像”。
        '''
        # discriminator_input = Input(shape=(None, None,1))
        discriminator_input = keras.Input(shape=(32, 32, 1))
        leakAlpha = 0.7
        ac = 'tanh'
        # ac = LeakyReLU()
        x = Conv2D(128, 3, padding='same')(discriminator_input)
        x = BatchNormalization(momentum=self.bnm)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(128, 5, padding='same')(x)
        x = BatchNormalization(momentum=self.bnm)(x)
        x = LeakyReLU(alpha=0.2)(x)

        # x = Conv2D(256, 4, strides=2)(x)  # (32-4)//2+1=15
        # x = BatchNormalization(momentum=self.bnm)(x)
        # x = LeakyReLU(alpha=0.2)(x)
        #
        x = Conv2D(256, 5, padding='same')(x)
        x = BatchNormalization(momentum=self.bnm)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(256, 4, strides=2)(x)  # (15-4)//2+1=6
        x = BatchNormalization(momentum=self.bnm)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(256, 5, padding='same')(x)
        x = BatchNormalization(momentum=self.bnm)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(128, 7, strides=1, padding='same', activation='tanh')(x)
        x = BatchNormalization(momentum=self.bnm)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Dropout(0.8)(x)
        x = Flatten()(x)
        x = Dense(1)(x)

        discriminator = keras.models.Model(discriminator_input, x, name='dModel')
        plot_model(discriminator, 'dModel3.png', show_shapes=True)
        discriminator.summary()
        # 为了训练稳定，在优化器中使用学习率衰减和梯度限幅（按值）。
        # discriminator_optimizer = keras.optimizers.RMSprop(lr=8e-4, clipvalue=1.0, decay=1e-8)
        # discriminator.compile(optimizer=discriminator_optimizer, loss=self.wLoss, metrics=[acc])

        return discriminator

    def trainGan(self):

        self.save_imgs(self.gModel, self.fakeValPaths[:10], -1)
        for e in range(self.epoch):
            print('Epoch:' + str(e) + '/' + str(self.epoch))

            for _ in range(0, len(self.realPaths), self.half_batch_size):
                beginTime = time.time()
                trainedRate = _ / len(self.realPaths)
                '''
                   train  D
                '''
                realx = []
                endIndex = min([_ + self.half_batch_size, len(self.realPaths), len(self.fakeTrainPaths)])

                for _p in self.realPaths[_:endIndex]:
                    with open(_p, 'rb') as f:
                        realx.append(pkl.load(f)['suv'][1, :, :, np.newaxis] / 45 - 1)
                fakex = []
                for _p in self.fakeTrainPaths[_:endIndex]:
                    with open(_p, 'rb') as f:
                        x = pkl.load(f)['suv'][1, :, :]
                        if x.shape != (8, 8):
                            x = resize(x, (8, 8), order=1, preserve_range=True, )
                        fakex.append(x[:, :, np.newaxis] / 45 - 1)
                for xxx in range(self.trainDTimes):  # 判别器多训练几次
                    # dBatchLoss, dBatchAcc = self.dModelModel.train_on_batch(x=[np.array(realx), np.array(fakex)],
                    #                                                         y={'dModel': -np.ones((len(realx), 1)),
                    #                                                            'dModel': np.ones((len(realx), 1)),
                    #                                                            'dModel': np.zeros((len(realx), 1)),
                    #                                                            })
                    totalLoss, dBatchRealLoss, dBatchFakeLoss, interLoss, dBatchRealAcc, dBatchFakeAcc, interAcc = self.dModelModel.train_on_batch(
                        x=[np.array(realx), np.array(fakex)],
                        y=[-np.ones((len(realx))),
                           np.ones((len(realx))),
                           np.zeros((len(realx)))
                           ])
                    self.dModel.trainable = False
                    self.gModel.trainable = True
                    ganBatchLoss, ganBatchAcc = self.gModelModel.train_on_batch(np.array(fakex),
                                                                                [-np.ones((len(realx), 1))])
                    batchTime = time.time() - beginTime
                    processBar = '%04d/%04d' % (e, self.epoch) + '  [' + '=' * int(20 * trainedRate) + ' ' * int(
                        20 * (1 - trainedRate)) + '] ' + '%.2f' % trainedRate + ' leftTime:' + str(batchTime * len(
                        self.realPaths) // self.half_batch_size * (1 - trainedRate))
                    print(
                        processBar + " [dILoss=%.6f,dRLoss=%.6f，dFLoss=%.6f,dRAcc=%.6f，dFAcc=%.6f] [gloss: %.6f,gAcc: %.6f]" % (
                            interLoss, dBatchRealLoss, dBatchFakeLoss, dBatchRealAcc, dBatchFakeAcc,
                            ganBatchLoss, ganBatchAcc))

                    # dEpochRealLoss, dEpochRealAcc = self.dModel.evaluate_generator(
                    #     self.dRealGenerator(self.realPaths, batchSize=self.valBatchSize),
                    #     int(np.ceil(len(self.realPaths) / self.valBatchSize)))
                    # dEpochFakeLoss, dEpochFakeAcc = self.gan.evaluate_generator(
                    #     self.fakeGenerator(self.fakeTrainPaths, batchSize=self.valBatchSize),
                    #     int(np.ceil(len(self.fakeTrainPaths) / self.valBatchSize)))
                    # dEpochFakeValLoss, dEpochFakeValAcc = self.gan.evaluate_generator(
                    #     self.fakeGenerator(self.fakeValPaths, batchSize=self.valBatchSize),
                    #     int(np.ceil(len(self.fakeValPaths) / self.valBatchSize)))
                    # print(
                    #     '%d' % e + ' epoch end:' + 'dRealLoss=%.6f  dRealAcc=%.6f  dFakeLoss=%.6f  dFakeAcc=%.6f  valLoss=%.6f  valAcc=%.6f' % (
                    #         dEpochRealLoss, dEpochRealAcc, dEpochFakeLoss, dEpochFakeAcc, dEpochFakeValLoss,
                    #         dEpochFakeValAcc))
                    with open(self.logPath, 'a') as f:
                        f.write('%d,%.6f,%.6f,%.6f,%.6f\n' % (e, interLoss, interAcc, ganBatchLoss, ganBatchAcc,))
                    if e % self.saveporid == 0:
                        # 在验证集上输出效果
                        # 判别器在训练集上的效果

                        self.dModel.save(
                            self.chechPointRoot % (e, interLoss, interAcc, ganBatchLoss, ganBatchAcc) + '_d')
                    self.gModel.save(self.chechPointRoot % (e, interLoss, interAcc, ganBatchLoss, ganBatchAcc) + '_g')

                    self.save_imgs(self.gModel, self.fakeValPaths[:10], e)

    from keras.applications.resnet50 import ResNet50
    def fakeGenerator(self, paths, batchSize):
        for _ in range(0, len(paths), batchSize):
            xs = []
            ys = []
            for _p in paths[_:_ + batchSize]:
                with open(_p, 'rb') as f:
                    x = pkl.load(f)['suv'][1, :, :]
                    if x.shape != (8, 8):
                        x = resize(x, (8, 8), order=1, preserve_range=True, )
                    xs.append(x[:, :, np.newaxis] / 45 - 1)
                    ys.append(0)
            yield np.array(xs), np.array(ys)

    def dRealGenerator(self, paths, batchSize):

        for _ in range(0, len(paths), batchSize):
            xs = []
            ys = []
            for _p in paths[_:_ + batchSize]:
                with open(_p, 'rb') as f:
                    xs.append(pkl.load(f)['suv'][1, :, :, np.newaxis] / 45 - 1)
                    ys.append(1)  # 真实数据用1表示,并在标签中加入一些随机
            yield np.array(xs), np.array(ys)

    def save_imgs(self, generator, imgPaths, e):
        xs = []
        if e == -1:
            for i, _p in enumerate(imgPaths):
                with open(_p, 'rb') as f:
                    x = pkl.load(f)['suv'][1, :, :]
                    skio.imsave(os.path.join(self.geneImgPath, 'e=%d_i=%d_p=%s.tif' % (e, i, os.path.basename(_p))), x)
        else:
            for i, _p in enumerate(imgPaths):
                with open(_p, 'rb') as f:
                    x = pkl.load(f)['suv'][1, :, :]
                    if x.shape != (8, 8):
                        x = resize(x, (8, 8), order=1, preserve_range=True, )
                    xs.append(x[:, :, np.newaxis] / 45 - 1)
            res = (generator.predict(np.array(xs)) + 1) * 45
            for i in range(res.shape[0]):
                skio.imsave(os.path.join(self.geneImgPath, 'e=%d_i=%d.tif' % (e, i)), res[i, :, :, 0])


if __name__ == '__main__':
    realRoot = r'E:\pyWorkspace\CAE\res\highSuvPatch\bigPatch'
    fakeRoot = r'E:\pyWorkspace\CAE\res\highSuvPatch\smallPatch'
    gan = MyGan(realRoot, fakeRoot, chRoot=r'E:\pyWorkspace\NewCAE\GAN\chp3',
                geneImgPath=r'E:\pyWorkspace\NewCAE\GAN\geneImg3')
    gan.trainGan()
    # from keras.applications import vgg16
    #
    # m = vgg16.VGG16()
