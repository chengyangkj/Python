#coding:utf-8
import tensorflow as tf
#创建常数值张量
zer = tf.zeros([2, 3])
tensor5 = tf.constant(0.5, shape = [2, 3])
#创建随机值张量

with tf.Session() as sees:
    print(sees.run(tensor5))
    print(sees.run(zer))
