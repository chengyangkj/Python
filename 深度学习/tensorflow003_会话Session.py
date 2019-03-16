#coding:utf-8
#会话（session）作用：
#1.运行图结构 2.分配资源计算 3.掌握资源

import tensorflow as tf
#实现一个加法运算
a = tf.constant(3.0)
b = tf.constant(4.0)
sum1 = tf.add(a, b)
#run op
with tf.Session() as sess:
    #run一个op
    print(sess.run(sum1))
    #run多个op（使用列表）：
    print(sess.run([a,b,sum1]))
#有重载机制，默认给运算符重载为op类型
var1 = 2
sum2 = a + var1
with tf.Session() as sess:
    print(sess.run(sum2))
#placeholder是一个占位符
#数据类型为float32，大小为两行三列
plt = tf.placeholder(tf.float32, [2, 3])
#使用feed_dit 传入占位的字典字典数据类型数据
#这里由于传入的参数只有一个为占位符类型的数据plt，所以在feed_dit 里面只填一个键值对类型的数据，键为plt，值为行三列的列表
with tf.Session() as sess:
    print(sess.run(plt, feed_dict={plt: [[1, 2, 3, ], [4, 5, 6,]]}))