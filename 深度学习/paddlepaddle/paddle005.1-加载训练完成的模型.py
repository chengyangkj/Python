import paddle.fluid as fluid
import os
from PIL import  Image
import  numpy as np






# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

#加载保存的模型
# 加载之前训练过的参数模型
save_path = 'models/infer_model/'
if os.path.exists(save_path):
    print('使用参数模型作为预训练模型')
    # 从模型中获取预测程序、输入数据名称列表、分类器
    [infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)


# 对图片进行预处理
def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = im / 255.0 * 2.0 - 1.0
    return im


# 加载数据并开始预测
img = load_image('image/3.png')
results = exe.run(program=infer_program,
                  feed={'image': img},
                  fetch_list=target_var)

lab = np.argsort(results)[0][0][-1]
print('test.png infer result: %d' % lab)
