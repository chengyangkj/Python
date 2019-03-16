#coding:utf-8
import tensorflow as tf

plt = tf.placeholder(tf.float32,[None,3])
#打印tensor的形状
print(plt.shape)
#静态改变形状
plt.set_shape([2,3])
print(plt.shape)
#plt.set_shape([3,2]) 会报错，因为静态形状改变只适用在形状不固定的张量

#动态形状改变
plt_reshape = tf.reshape(plt, [3,2])
print(plt_reshape.shape)