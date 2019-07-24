import paddle.fluid as fluid
import paddle
import paddle.dataset.mnist as mnist
import numpy as np
from PIL import Image
import shutil
import os
#定义卷积神经网络

def convolutional_neural_network(input):
    #第一个卷积层 卷积核的个数为32 卷积核大小为3 卷积步长为1 传入28 28 1 这次卷积后大小为 (28-3)/1 +1 = 26
    conv1 = fluid.layers.conv2d(
        input=input,
        num_filters=32,
        filter_size=3,
        stride=1,
    )
    #第一个池化层 进行降特征 大小2*2 步长1 最大池化
    pool1 = fluid.layers.pool2d(
        input= conv1,
        pool_size=2,
        pool_stride=1,
        pool_type="max"

    )
    #第二个卷积层
    conv2 = fluid.layers.conv2d(
        input=pool1,
        num_filters=64,
        filter_size=3,
        stride=1
    )
    #第二个池化层 卷积核3*3 一共64个卷积核
    pool2 = fluid.layers.pool2d(
        input=conv2,
        pool_size=2,
        pool_stride=1,
        pool_type='max',
    )
    #以softmax为激活函数的全连接层 大小为label大小
    fc  = fluid.layers.fc(input=pool2,size=10,act='softmax')
    return fc

image = fluid.layers.data(name='image',shape=[1,28,28],dtype='float32')
label = fluid.layers.data(name='label',shape=[1],dtype='int64')

#调取卷积神经网络
model = convolutional_neural_network(image)

#获取预测程序
infer_program = fluid.default_main_program().clone(for_test=True)

#获取损失函数和准确率函数 交叉商损失
cost = fluid.layers.cross_entropy(input=model,label=label)
#计算平均损失
avg_cost = fluid.layers.mean(cost)
#获取准确率
acc = fluid.layers.accuracy(input=model,label=label)
# 获取测试程序
test_program = fluid.default_main_program().clone(for_test=True)

#定义优化的方法 Adam 优化
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
ops = optimizer.minimize(avg_cost)


#获取Mnist数据 存入管道进行批处理
train_reader = paddle.batch(mnist.train(),batch_size=128)
test_reader = paddle.batch(mnist.test(),batch_size=128)

#定义一个使用CPU的执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)

#进行参数初始化
exe.run(fluid.default_startup_program())

#s使用feeder读取数据，并且定义数据的维度
feeder = fluid.DataFeeder(place=place,feed_list=[image,label])

#开始训练和测试

for pass_id in range(50):
    #取出管道中的数据
    for batch_id,data in enumerate(train_reader()):
        train_cost,train_acc = exe.run(
            program=fluid.default_main_program(),
            feed=feeder.feed(data),
            fetch_list=[avg_cost,acc]
        )
        # 每100个batch打印一次信息
        if batch_id % 10 == 0:
            print('Pass:%d, Batch:%d, 损失:%0.5f, 准确率:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))


    # 进行测试
    test_accs = []
    test_costs = []
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_cost, acc])
        test_accs.append(test_acc[0])
        test_costs.append(test_cost[0])
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, 损失:%0.5f,准确率:%0.5f' % (pass_id, test_cost, test_acc))
    # 保存预测模型
    save_path = 'models/infer_model/'
    # 删除旧的模型文件
    shutil.rmtree(save_path, ignore_errors=True)
    # 创建保持模型文件目录
    os.makedirs(save_path)
    # 保存预测模型
    fluid.io.save_inference_model(save_path, feeded_var_names=[image.name], target_vars=[model], executor=exe)

# 对图片进行预处理
def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = im / 255.0 * 2.0 - 1.0
    return im


# 加载数据并开始预测
img = load_image('image/test.png')
results = exe.run(program=infer_program,
                  feed={'image': img},
                  fetch_list=[model])
# 获取概率最大的标签
lab = np.argsort(results)[0][0][-1]
print('test.png infer result: %d' % lab)

