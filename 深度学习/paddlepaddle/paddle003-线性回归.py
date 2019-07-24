import paddle.fluid as fluid
import paddle
import numpy as np

#定义一个简单的线性网络
#data 和 create_tensor类似 都是一个张量
#定义x
x = fluid.layers.data(name='x',shape=[1],dtype='float32')
#定义全连接层网络
hidden = fluid.layers.fc(input=x,size=100,act='relu')
#定义输出层网络
net =fluid.layers.fc(input=hidden,size=1,act=None)

#获得预测程序
infer_program = fluid.default_main_program().clone(for_test=True)

#定义目标值
y = fluid.layers.data(name='y',shape=[1],dtype='float32')
#定义损失函数
#计算单个误差 均方误差
cost = fluid.layers.square_error_cost(input=net,label=y)
#计算平均误差
avg_cost = fluid.layers.mean(cost)

#复制一个主程序 用于预测
test_program = fluid.default_main_program().clone(for_test=True)

#定义优化方法
#梯度下降法 设置学习率为0.01
optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01)

#根据本次误差作为凭据进行梯度下降
ops = optimizer.minimize(avg_cost)

#创建一个cup解析器
place  = fluid.CPUPlace()
exe = fluid.Executor(place)

#进行参数初始化
exe.run(fluid.default_startup_program())
#定义训练和预测数据
x_data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]).astype('float32')
y_data = np.array([[3.0], [5.0], [7.0], [9.0], [11.0]]).astype('float32')
test_data = np.array([[30.0]]).astype('float32')

#训练100次
for pass_id in range(1000):
    train_cost = exe.run(program=fluid.default_main_program(),
                         feed={'x':x_data,'y':y_data},
                         fetch_list=[avg_cost]
                         )
    print('经过：'+str(pass_id)+"训练,误差为："+str(train_cost[0]))


#预测
result = exe.run(
    program=infer_program,
    feed={'x':test_data},
    fetch_list=[net]
)
print("当x为30，预测值为："+str(result))