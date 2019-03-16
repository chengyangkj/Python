#coding:utf-8
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   #设置报错的等级，避免爆红
#创建一张图包括一组op和tensor，上下文环境
#op:只要使用了tensorflow定义的的api都是op
#tensor就指代的是数据

#实现一个加法运算
a = tf.constant(3.0)
b = tf.constant(4.0)
sum1 = tf.add(a, b)

#程序默认会给图进行注册,即给图分配地址
# 获取默认图的调用
graph = tf.get_default_graph()
#打印
print(graph)
#打印op的图的调用
print (a.graph)
print (sum1.graph)
#  #创建一个会话用于运行图：
with tf.Session() as sess:
    #打印sess的图的调用
    print(sess.graph)
    #运行图的结果
    print(sess.run(sum1))
#创建一张新图,并获取新图的对象
g = tf.Graph()
#打印新图的地址
print (g)
#使用上下文环境，将g作为默认的图：
with g.as_default():
    c = tf.constant(11.0)
    d = tf.constant(12.0)
    sum2 = tf.add(c, d)
    print(c.graph)
#离开上下文环境，打印tf默认图的地址
print(tf.get_default_graph())
#打开新图的会话：
with tf.Session(graph=g) as sess:
    print(sess.run(sum2))

