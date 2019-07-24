import paddle.fluid as fluid
import numpy as np
#1,计算常量相加
#定义两个常量张量

x1 = fluid.layers.fill_constant(shape=[2,2],value=1,dtype='int64')

x2 = fluid.layers.fill_constant(shape=[2,2],value=1,dtype='int64')

#讲两个张量相加
y1 = fluid.layers.sum(x=[x1,x2])

#创建一个执行器 类似tensorflow的session
#可以指定计算使用CPU或GPU 当使用CPUPlace()使用CPU 当使用CUDAPlace使用GPU

palce = fluid.CPUPlace()
#创建解析器 解析器是用来进行运算的
exe = fluid.executor.Executor(palce)
#解析器进行参数初始化
exe.run(fluid.default_startup_program())

#进行运算 fetch_list 是指定解析器在进行run之后要输出的值
result = exe.run(program=fluid.default_main_program(),
                 fetch_list=[y1])
print(result)


#二，计算变量相加
#上面的为常量 这里来定义变量
#定义两个张量
a = fluid.layers.create_tensor(dtype='int64',name="a")
b = fluid.layers.create_tensor(dtype='int64',name='b')

#讲两张量求和
y = fluid.layers.sum(x=[a,b])

#创建cpu执行器
Place = fluid.CPUPlace()
exe = fluid.executor.Executor(palce)

#初始化参数
exe.run(fluid.default_startup_program())

#定义要进行计算的两变量
a1 = np.array([3,2]).astype('int64')
b1 = np.array([1,1]).astype('int64')

#feed 填入变量的参数

out_a,out_b,result =exe.run(program=fluid.default_main_program(),
                            feed={'a':a1,'b':b1},
                            fetch_list=[a,b,y],
                            )
print(out_a,out_b,result)