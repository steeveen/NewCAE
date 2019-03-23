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
 @Author = 'steven'   @DateTime = '2019/3/19 10:04'
'''
from GAN.gan import getGan, getGenerator, getDiscriminator, trainGan


def main():
    g = getGenerator()
    d = getDiscriminator()
    smallRoot = r'E:\pyWorkspace\CAE\res\highSuvPatch\smallPatch'
    bigRoot = r'E:\pyWorkspace\CAE\res\highSuvPatch\bigPatch'
    trainGan(g,d,realRoot=bigRoot,feakRoot=smallRoot )


if __name__ == '__main__':
    main()
