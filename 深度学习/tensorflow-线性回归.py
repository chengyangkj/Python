#coding:utf-8
##线性回归
#各种算法定损的方法
#算法     定损方法      优化
#线性回归   均方误差     梯度下降
#逻辑回归   对数似然损失  梯度下降 二分类
#神经网络   交叉熵损失    反向传播算法（梯度下降）
import tensorflow as tf

#线性回归的模型：
#y = w*x + b
#训练其实就是传入x 的值 和 y对应的值，通过训练w和b的值，使通过线性函数计算的y_值趋近y
#定义x特征值  占位符类型 x的值在运行的过程中进行动态传入
x = tf.placeholder(tf.float32)
#定义w 变量类型 一维
W = tf.Variable(tf.zeros([1]))
#定义b
b = tf.Variable(tf.zeros([1]))
#定义每次训练传入的目标值
y_ = tf.placeholder(tf.float32)

#线性模型
y = W * x +b

#计算每次训练的丢失值 使用均方误差进行定损
lost = tf.reduce_mean(tf.square(y_ - y))

#优化函数 优化函数代表我们要通过什么方式去优化我们需要学习的值，即优化更改我们要学习的w和b让学习结果和线性结果趋于相同 这里我们采用梯度下降的函数
#后面传入的值是学习效率。一般是一个小于1的数。越小收敛越慢，但并不是越大收敛越快，取值太大甚至可能不收敛了。
optimizer = tf.train.GradientDescentOptimizer(0.0000001)
#这里填写我们每次训练的目的 即我们是通过优化函数的对象optimizer进行减小lost的值
train_step = optimizer.minimize(lost)
#打开一个session
with tf.Session() as sess:
    #在session下初始化 所有的变量
    init = tf.global_variables_initializer()
    sess.run(init)
    #训练的次数
    steps = 1000
    for i in range(steps):
        #模拟传入训练数据
        #每次训练传入的数据x的列表
        xs = [i]
        #y的列表
        ys = [3 * i]
        #构造传参列表
        feed = {x: xs, y_: ys}
        #运行 传入训练目的 和变量列表
        sess.run(train_step, feed_dict=feed)
        if i % 100 == 0:
            print("经过 %d 训练:" % i)
            print("W: %f" % sess.run(W))
            print("b: %f" % sess.run(b))
            print("lost: %f" % sess.run(lost, feed_dict=feed))