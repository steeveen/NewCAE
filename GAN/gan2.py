# -*- coding: utf-8 -*-
'''gan的修改版，借鉴了wgan的思想
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

import os
from keras.preprocessing import image
import pickle as pkl
from skimage import io as skio
import keras
from keras.layers import Input, Dense, Flatten, Conv2D, UpSampling2D, LeakyReLU, Dropout, Reshape
import numpy as np
from natsort import natsorted
from glob import glob
import os
from skimage.transform import resize
import time
import keras.backend as K
from keras.utils import plot_model
from keras.metrics import binary_accuracy


def acc(y_true, y_pred):
    import keras.backend as K
    return K.mean(K.equal(y_true > 0, y_pred > 0), axis=-1)


class MyGan():
    def __init__(self, realRoot, fakeRoot, chRoot, geneImgPath):
        self.trainRato = 0.9
        self.realRoot = realRoot
        self.fakeRoot = fakeRoot
        self.half_batch_size = 128
        self.epoch = 200
        self.saveporid = 2
        self.trainDTimes = 5
        self.valBatchSize = 2
        self.gModelBeginEpochThres = 0
        self.clip_value = 0.5

        os.makedirs(chRoot, exist_ok=True)
        os.makedirs(geneImgPath, exist_ok=True)

        self.geneImgPath = geneImgPath
        self.chechPointRoot = os.path.join(chRoot, '%d_gLoss=%.6f_gAcc=%.6f')
        self.logPath = os.path.join(chRoot, 'log2.csv')
        if not os.path.exists(self.logPath):
            with open(self.logPath, 'w') as f:
                f.write('epoch,dRealLoss,dRealAcc,dFakeLoss,dFakeAcc,valLoss,valAcc\n')
        else:
            with open(self.logPath, 'a') as f:
                f.write('epoch,dRealLoss,dRealAcc,dFakeLoss,dFakeAcc,valLoss,valAcc\n')
        self.ImgOp = r'E:\pyWorkspace\NewCAE\GAN\geneImg'
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
        z = Input(shape=(8, 8, 1))
        geneImg = self.gModel(z)
        valid = self.dModel(geneImg)
        self.gan = keras.models.Model(z, valid)
        optimizers = keras.optimizers.RMSprop(lr=8e-4, clipvalue=1.0, decay=1e-8)
        self.gan.compile(loss=self.wLoss, optimizer=optimizers, metrics=[acc])

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
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)

        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)

        # 添加更多的卷积层
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv2D(512, 5, padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv2D(512, 5, padding='same')(x)
        x = LeakyReLU()(x)

        # 生成一个 32x32 1-channel 的feature map
        x = Conv2D(1, 7, activation='tanh', padding='same')(x)
        generator = keras.models.Model(generator_input, x)
        generator.summary()
        plot_model(generator, 'gModel.png', show_shapes=True)
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
        # x = Conv2D(128, 3, padding='same')(discriminator_input)
        # x = LeakyReLU(leakAlpha)(x)
        # x = Conv2D(128, 4, strides=2)(x)  # (32-4)//2+1=15
        # x = LeakyReLU(leakAlpha)(x)
        # x = Conv2D(128, 4, strides=2)(x)  # (15-4)//2+1=6
        # x = LeakyReLU(leakAlpha)(x)
        # x = Conv2D(128, 4, strides=2)(x)  # (6-4)//2+1=2
        # x = LeakyReLU(leakAlpha)(x)
        # x = Conv2D(128, 2, strides=2)(x)
        # # x = LeakyReLU()(x)
        # x = Conv2D(128, 7, strides=1, padding='same')(x)
        # # x = LeakyReLU()(x)
        # x = Dropout(0.8)(x)
        ac = 'tanh'
        # ac = LeakyReLU()
        x = Conv2D(128, 3, padding='same', activation=ac)(discriminator_input)
        x = Conv2D(256, 5, padding='same', activation=ac)(x)
        x = Conv2D(256, 4, strides=2, activation=ac)(x)  # (32-4)//2+1=15
        x = Conv2D(256, 5, padding='same', activation=ac)(x)
        x = Conv2D(256, 4, strides=2, activation=ac)(x)  # (15-4)//2+1=6
        x = Conv2D(256, 5, padding='same', activation=ac)(x)
        x = Conv2D(256, 4, strides=2, activation=ac)(x)  # (6-4)//2+1=2
        x = Conv2D(256, 5, padding='same', activation=ac)(x)
        x = Conv2D(512, 2, strides=2, activation=ac)(x)
        x = Conv2D(512, 5, padding='same', activation=ac)(x)
        x = Conv2D(256, 7, strides=1, padding='same', activation='tanh')(x)
        x = Dropout(0.8)(x)

        # 分类层
        x = Reshape((-1,))(x)
        x = Dense(1)(x)

        discriminator = keras.models.Model(discriminator_input, x)
        plot_model(discriminator,'dModel.png',show_shapes=True)
        discriminator.summary()
        # 为了训练稳定，在优化器中使用学习率衰减和梯度限幅（按值）。
        discriminator_optimizer = keras.optimizers.RMSprop(lr=8e-4, clipvalue=1.0, decay=1e-8)
        discriminator.compile(optimizer=discriminator_optimizer, loss=self.wLoss, metrics=[acc])

        return discriminator

    # def getGan(self, gModel, dModel):
    #     '''
    #     The adversarial network:对抗网络
    #     最后，设置GAN，它链接生成器（generator）和鉴别器（discrimitor）。 这是一种模型，经过训练，
    #     将使生成器（generator）朝着提高其愚弄鉴别器（discrimitor）能力的方向移动。 该模型将潜在的空间点转换为分类决策，
    #     “假的”或“真实的”，并且意味着使用始终是“这些是真实图像”的标签来训练。 所以训练`gan`将以一种方式更新
    #     “发生器”的权重，使得“鉴别器”在查看假图像时更可能预测“真实”。 非常重要的是，将鉴别器设置为在训练
    #     期间被冻结（不可训练）：训练“gan”时其权重不会更新。 如果在此过程中可以更新鉴别器权重，那么将训练鉴别
    #     器始终预测“真实”。
    #     '''
    #     # 将鉴别器（discrimitor）权重设置为不可训练（仅适用于`gan`模型）
    #     dModel.trainable = False
    #
    #     # gan_input = keras.Input(shape=(None, None,1))
    #     gan_input = keras.Input(shape=(None, None))
    #     gan_output = dModel(gModel(gan_input))
    #     gan = keras.models.Model(gan_input, gan_output)
    #
    #     gan_optimizer = keras.optimizers.RMSprop(lr=4e-4, clipvalue=1.0, decay=1e-8)
    #     gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy',)
    #     return gan

    def trainGan(self):
        '''
          开始训练了。
          每个epoch：
           *在潜在空间中绘制随机点（随机噪声）。
           *使用此随机噪声生成带有“generator”的图像。
           *将生成的图像与实际图像混合。
           *使用这些混合图像训练“鉴别器”，使用相应的目标，“真实”（对于真实图像）或“假”（对于生成的图像）。
           *在潜在空间中绘制新的随机点。
           *使用这些随机向量训练“gan”，目标都是“这些是真实的图像”。 这将更新发生器的权重（仅因为鉴别器在“gan”内被冻结）
           以使它们朝向获得鉴别器以预测所生成图像的“这些是真实图像”，即这训练发生器欺骗鉴别器。
        '''

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

                for _p in self.realPaths[_:_ + self.half_batch_size]:
                    with open(_p, 'rb') as f:
                        realx.append(pkl.load(f)['suv'][1, :, :, np.newaxis] / 45 - 1)
                fakex = []
                for _p in self.fakeTrainPaths[_:_ + self.half_batch_size]:
                    with open(_p, 'rb') as f:
                        x = pkl.load(f)['suv'][1, :, :]
                        if x.shape != (8, 8):
                            x = resize(x, (8, 8), order=1, preserve_range=True, )
                        fakex.append(x[:, :, np.newaxis] / 45 - 1)
                self.dModel.trainable=True
                self.gModel.trainable=False
                for xxx in range(self.trainDTimes):  # 判别器多训练几次
                    dBatchRealLoss, dBatchRealAcc = self.dModel.train_on_batch(np.array(realx),
                                                                               -np.ones((len(realx), 1)))
                    geneX = self.gModel.predict(np.array(fakex))
                    dBatchFakeLoss, dBatchFakeAcc = self.dModel.train_on_batch(geneX,
                                                                               np.ones((len(fakex), 1)))

                    for l in self.dModel.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                        l.set_weights(weights)

                    batchTime = time.time() - beginTime
                    processBar = '%04d/%04d' % (e, self.epoch) + '  [' + '=' * int(20 * trainedRate) + ' ' * int(
                        20 * (1 - trainedRate)) + '] ' + '%.2f' % trainedRate + ' leftTime:' + str(batchTime * len(
                        self.realPaths) // self.half_batch_size * (1 - trainedRate))
                    print(
                        processBar + " [dRealLoss=%.6f,dRealAcc=%.6f][dFakeLoss=%.6f,dFakeAcc=%.6f] " % (
                            dBatchRealLoss, dBatchRealAcc, dBatchFakeLoss, dBatchFakeAcc,
                        ))
                self.dModel.trainable = False
                self.gModel.trainable = True
                ganBatchLoss, ganBatchAcc = self.gan.train_on_batch(np.array(fakex), -np.ones((np.shape(fakex)[0], 1)))
                batchTime = time.time() - beginTime
                processBar = '%04d/%04d' % (e, self.epoch) + '  [' + '=' * int(20 * trainedRate) + ' ' * int(
                    20 * (1 - trainedRate)) + '] ' + '%.2f' % trainedRate + ' leftTime:' + str(batchTime * len(
                    self.realPaths) // self.half_batch_size * (1 - trainedRate))
                print(
                    processBar + " [dRealLoss=%.6f,dRealAcc=%.6f][dFakeLoss=%.6f,dFakeAcc=%.6f] [gloss: %.6f,gAcc: %.6f]" % (
                        dBatchRealLoss, dBatchRealAcc, dBatchFakeLoss, dBatchFakeAcc,
                        ganBatchLoss, ganBatchAcc))

            dEpochRealLoss, dEpochRealAcc = self.dModel.evaluate_generator(
                self.dRealGenerator(self.realPaths, batchSize=self.valBatchSize),
                int(np.ceil(len(self.realPaths) / self.valBatchSize)))
            dEpochFakeLoss, dEpochFakeAcc = self.gan.evaluate_generator(
                self.fakeGenerator(self.fakeTrainPaths, batchSize=self.valBatchSize),
                int(np.ceil(len(self.fakeTrainPaths) / self.valBatchSize)))
            dEpochFakeValLoss, dEpochFakeValAcc = self.gan.evaluate_generator(
                self.fakeGenerator(self.fakeValPaths, batchSize=self.valBatchSize),
                int(np.ceil(len(self.fakeValPaths) / self.valBatchSize)))
            print(
                '%d' % e + ' epoch end:' + 'dRealLoss=%.6f  dRealAcc=%.6f  dFakeLoss=%.6f  dFakeAcc=%.6f  valLoss=%.6f  valAcc=%.6f' % (
                    dEpochRealLoss, dEpochRealAcc, dEpochFakeLoss, dEpochFakeAcc, dEpochFakeValLoss,
                    dEpochFakeValAcc))
            with open(self.logPath, 'a') as f:
                f.write('%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n' % (e,
                                                                dEpochRealLoss, dEpochRealAcc, dEpochFakeLoss,
                                                                dEpochFakeAcc, dEpochFakeValLoss,
                                                                dEpochFakeValAcc))
            if e % self.saveporid == 0:
                # 在验证集上输出效果
                # 判别器在训练集上的效果

                self.dModel.save(self.chechPointRoot % (e, dEpochFakeValLoss, dEpochFakeValAcc) + '_d')
                self.gModel.save(self.chechPointRoot % (e, dEpochFakeValLoss, dEpochFakeValAcc) + '_g')

                self.save_imgs(self.gModel, self.fakeValPaths[:10], e, )

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
    gan = MyGan(realRoot, fakeRoot, chRoot=r'E:\pyWorkspace\NewCAE\GAN\chp2',
                geneImgPath=r'E:\pyWorkspace\NewCAE\GAN\geneImg2')
    gan.trainGan()
    from keras.applications import vgg16
    m=vgg16.VGG16()