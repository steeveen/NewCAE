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
 @Author = 'steven'   @DateTime = '2019/2/25 17:06'
'''

def make(n,a,used=0,tag=0,forehead=''):
    if used==n and tag==0:
        a.append(forehead)
    else:
        if used<n:
            make(n,a,used+1,tag+1,forehead+'(')
        if tag>0:
            make(n,a,used,tag-1,forehead+')')
def run(n):
    a=[]
    make(n,a)
    return a
if __name__ == '__main__':
    r=run(3)
    print(r)

