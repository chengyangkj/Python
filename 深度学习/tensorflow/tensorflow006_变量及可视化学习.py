import tensorflow as tf

a = tf.constant(3.0)
b = tf.constant(4.0)
sum1 = tf.add(a, b)

#将tensor放入变量中
sum_vis = tf.Variable(sum1)
#初始化所有变量
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  print(sess.run(init_op))
  sess.run(sum1)

#将程序的图写入事件文件
summary_writer = tf.summary.FileWriter('./tmp/tensorflow/test1', graph=sum1.graph)
#在控制台通过tensorboard --logdir="./tmp/tensorflow/test1"  启动