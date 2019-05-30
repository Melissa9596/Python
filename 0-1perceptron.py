# 0-1perceptron.py
#单层感知器的作用是进行分类，局限性：无法很好的解决异或问题

import numpy as np
import matplotlib.pyplot as plt

#输入数据
X = np.array([[1,3,3],
            [1,4,3],
            [1,1,1]])
#标签
Y = np.array([1,1,-1])

#权值初始化,1行3列，取值范围为-1到1
W = (np.random.random(3)-0.5)*2  #中间的random为随机数，其区间[0,1)
print(W)

#学习率设置
lr = 0.11

#计算迭代次数
n = 0

#神经网络输出
O = 0

def update():
    global X, Y, W,lr, n
    n += 1
    O = np.sign( np.dot(X, W.T) )
    W_C = (lr*(Y-O.T).dot(X))/int(X.shape[0])
    W = W+W_C

for _ in range (100):
    update()
    print(W)
    print(n)
    O = np.sign(np.dot(X, W.T))
    if(O == Y.T).all():
        print('Finished')
        print('epoch:',n)
        break

#正样本
x1 = [3,4]
y1 = [3,3]

#负样本
x2 = [1]
y2 = [1]

#计算分界线的斜率和截距
k = -W[1] / W[2]
d = -W[0] / W[2]
print('k = ',k)
print('d = ',d)

xdata = np.linspace(0,5)
plt.figure()
plt.plot(xdata, xdata*k+d,'r')
plt.plot(x1,y1,'bo')
plt.plot(x2,y2,'yo')
plt.show()
