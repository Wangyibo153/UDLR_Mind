import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import VGG16
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import neuron.layers as layers
import time
import os


config = tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=False)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
"""
"""
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
"""


class TopModel(object):
    def __init__(self):
        super(TopModel, self).__init__()
        # 初始化参数
        self.batch_size = 8
        self.vol_shape = (256, 256)
        self.channels = 1
        self.samples_num = 24
        self.moving_num_per_sample = 100
        self.src_path = '../datause/train_data/'
        self.weight_path = '../myweight/top_model/'
        self.filterList = [16, 32, 64]
        self.loss = ['mse', 'mse']
        self.loss_weights = [1, 1]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.epochs = 10
        self.steps_per_epoch = 20
        self.test_src_path = '../datause/test_data/'
        self.test_samples_num = 12
        self.test_moving_num_per_people = 31

    def create_model(self, filterList):
        # 输入层
        Input_T1 = Input(shape=[*self.vol_shape, self.channels],
                         name='input_T1')
        Input_T2 = Input(shape=[*self.vol_shape, self.channels],
                         name='input_T2')

        # T1 3层 Conv Block
        x1 = Conv2D(filters=filterList[0],
                    kernel_size=3,
                    padding='same',
                    kernel_initializer='he_normal',
                    strides=1,
                    name='top_conv1')(Input_T1)
        x1 = LeakyReLU(0.2,
                       name='top_relu1')(x1)

        x1 = Conv2D(filters=filterList[1],
                    kernel_size=3,
                    padding='same',
                    kernel_initializer='he_normal',
                    strides=1,
                    name='top_conv2')(x1)
        x1 = LeakyReLU(0.2,
                       name='top_relu2')(x1)

        x1 = Conv2D(filters=filterList[2],
                    kernel_size=3,
                    padding='same',
                    kernel_initializer='he_normal',
                    strides=1,
                    name='top_conv3')(x1)
        x1 = LeakyReLU(0.2,
                       name='top_relu3')(x1)

        Output_T1 = Conv2D(filters=1,
                           kernel_size=3,
                           padding='same',
                           kernel_initializer='he_normal',
                           strides=1,
                           name='T1_struc')(x1)

        # T2 3层 Conv Block
        x2 = Conv2D(filters=filterList[0],
                    kernel_size=3,
                    padding='same',
                    kernel_initializer='he_normal',
                    strides=1)(Input_T2)
        x2 = LeakyReLU(0.2)(x2)

        x2 = Conv2D(filters=filterList[1],
                    kernel_size=3,
                    padding='same',
                    kernel_initializer='he_normal',
                    strides=1)(x2)
        x2 = LeakyReLU(0.2)(x2)

        x2 = Conv2D(filters=filterList[2],
                    kernel_size=3,
                    padding='same',
                    kernel_initializer='he_normal',
                    strides=1)(x2)
        x2 = LeakyReLU(0.2)(x2)

        Output_T2 = Conv2D(filters=1,
                           kernel_size=3,
                           padding='same',
                           kernel_initializer='he_normal',
                           strides=1,
                           name='T2_struc')(x2)

        return Model(inputs=[Input_T1, Input_T2], outputs=[Output_T1, Output_T2])

    def data_generator(self):
        while True:
            for i in range(self.batch_size):
                # 随机指数产生
                sample_index = np.random.randint(0, self.samples_num)
                moving_index = np.random.randint(0, self.moving_num_per_sample)

                # 路径产生
                T1_fixed_path = self.src_path + 'sample_' + str(sample_index) + '/T1_fixed.png'
                T1_fixed_struc_path = self.src_path + 'sample_' + str(sample_index) + '/T1_fixed_struc.png'
                T2_fixed_path = self.src_path + 'sample_' + str(sample_index) + '/T2_fixed.png'
                T2_fixed_struc_path = self.src_path + 'sample_' + str(sample_index) + '/T2_fixed_struc.png'
                T1_moving_path = self.src_path + 'sample_' + str(sample_index) + '/T1_moving_' + str(moving_index) + '.png'
                T1_moving_struc_path = self.src_path + 'sample_' + str(sample_index) + '/T1_moving_struc_' + str(moving_index) + '.png'
                T2_moving_path = self.src_path + 'sample_' + str(sample_index) + '/T2_moving_' + str(moving_index) + '.png'
                T2_moving_struc_path = self.src_path + 'sample_' + str(sample_index) + '/T2_moving_struc_' + str(moving_index) + '.png'

                # 读取图片
                T1_fixed = np.array(Image.open(T1_fixed_path)) / 255
                T1_fixed_struc = np.array(Image.open(T1_fixed_struc_path)) / 255
                T2_fixed = np.array(Image.open(T2_fixed_path)) / 255
                T2_fixed_struc = np.array(Image.open(T2_fixed_struc_path)) / 255
                T1_moving = np.array(Image.open(T1_moving_path)) / 255
                T1_moving_struc = np.array(Image.open(T1_moving_struc_path)) / 255
                T2_moving = np.array(Image.open(T2_moving_path)) / 255
                T2_moving_struc = np.array(Image.open(T2_moving_struc_path)) / 255

                # 变形
                T1_fixed = np.reshape(T1_fixed, (1, self.vol_shape[0], self.vol_shape[1], 1))
                T1_fixed_struc = np.reshape(T1_fixed_struc, (1, self.vol_shape[0], self.vol_shape[1], 1))
                T2_fixed = np.reshape(T2_fixed, (1, self.vol_shape[0], self.vol_shape[1], 1))
                T2_fixed_struc = np.reshape(T2_fixed_struc, (1, self.vol_shape[0], self.vol_shape[1], 1))
                T1_moving = np.reshape(T1_moving, (1, self.vol_shape[0], self.vol_shape[1], 1))
                T1_moving_struc = np.reshape(T1_moving_struc, (1, self.vol_shape[0], self.vol_shape[1], 1))
                T2_moving = np.reshape(T2_moving, (1, self.vol_shape[0], self.vol_shape[1], 1))
                T2_moving_struc = np.reshape(T2_moving_struc, (1, self.vol_shape[0], self.vol_shape[1], 1))

                if i == 0:
                    T1_inputs = T1_fixed
                    T1_outputs = T1_fixed_struc
                    T2_inputs = T2_fixed
                    T2_outputs = T2_fixed_struc
                else:
                    T1_inputs = np.concatenate([T1_inputs, T1_moving], axis=0)
                    T1_outputs = np.concatenate([T1_outputs, T1_moving_struc], axis=0)
                    T2_inputs = np.concatenate([T2_inputs, T2_moving], axis=0)
                    T2_outputs = np.concatenate([T2_outputs, T2_moving_struc], axis=0)

            inputs = [T1_inputs, T2_inputs]
            outputs = [T1_outputs, T2_outputs]

            yield inputs, outputs

    def create_mind_model(self):
        Input_T1 = Input(shape=[*self.vol_shape, self.channels])
        Input_T2 = Input(shape=[*self.vol_shape, self.channels])

        # 实例化Mind
        Mind1 = layers.Mind(sigma=0.8, c1=1.2, c2=1.2, name='T1_struc')
        Mind2 = layers.Mind(sigma=0.8, c1=1.2, c2=1.2, name='T2_struc')

        Output_T1 = Mind1(Input_T1)
        Output_T2 = Mind2(Input_T2)

        return Model(inputs=[Input_T1, Input_T2], outputs=[Output_T1, Output_T2])

    def run(self):
        # 实例化模型
        model = self.create_model(self.filterList)
        # 打印模型构成
        model.summary()
        # 导入已经训练的权重
        model.load_weights(self.weight_path + 'test.h5')
        # 编译模型
        model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)
        # 开始训练
        model.fit_generator(self.data_generator(), steps_per_epoch=self.steps_per_epoch, epochs=self.epochs, verbose=2)
        # 保存权重
        model.save_weights(self.weight_path + 'test.h5')

    def predict(self):
        T1_path = '../datause/test_data/sample_0/T1_fixed.png'
        T1 = np.array(Image.open(T1_path)) / 255
        T1 = np.reshape(T1, (1, self.vol_shape[0], self.vol_shape[1], 1))
        T2_path = '../datause/test_data/sample_0/moving_0.png'
        T2 = np.array(Image.open(T2_path)) / 255
        T2 = np.reshape(T2, (1, self.vol_shape[0], self.vol_shape[1], 1))
        inputs = [T1, T2]
        model = self.create_model(self.filterList)
        model.load_weights(self.weight_path + 'test.h5')
        predicts = model.predict(inputs)
        images = [inputs[0][0, ..., 0], inputs[1][0, ..., 0], predicts[0][0, ..., 0], predicts[1][0, ..., 0]]
        names = ['Input:T1', 'Input:T2', 'Output:T1', 'Output:T2']
        self.plot(images, names)

    @staticmethod
    def plot(images, names):
        for i in range(1, 5):
            plt.subplot(1, 4, i)
            plt.imshow(images[i - 1], cmap='gray')
            plt.title(names[i - 1])
            plt.xticks([])
            plt.yticks([])
            plt.colorbar(fraction=0.05, pad=0.05)
        plt.show()

    def compute_mind_time(self):
        # 实例化模型
        model = self.create_mind_model()
        total_time = 0
        # 手敲数据生成器
        for i in range(self.test_samples_num):
            for j in range(self.test_moving_num_per_people):
                T1_moving_path = self.test_src_path + 'sample_' + str(i) + '/' + 'T1_moving_' + str(j) + '.png'
                T1_moving = np.array(Image.open(T1_moving_path)) / 255
                T1_moving = np.reshape(T1_moving, (1, self.vol_shape[0], self.vol_shape[1], 1))
                T2_moving_path = self.test_src_path + 'sample_' + str(i) + '/' + 'T2_moving_' + str(j) + '.png'
                T2_moving = np.array(Image.open(T2_moving_path)) / 255
                T2_moving = np.reshape(T2_moving, (1, self.vol_shape[0], self.vol_shape[1], 1))
                inputs = [T1_moving, T2_moving]
                startTime = time.time()
                temp = model.predict(inputs)
                endTime = time.time()
                duringTime = endTime - startTime

                if (i + j) != 0:
                    total_time += duringTime

        finalTime = total_time / (self.test_moving_num_per_people * self.test_samples_num - 1)
        finalStr = 'Mind的时间是：%s ms' % (finalTime * 1000)
        print(finalStr)

    def compute_conv_time(self):
        # 实例化模型
        model = self.create_model(self.filterList)
        model.load_weights(self.weight_path + 'test.h5')
        total_time = 0
        # 手敲数据生成器
        for i in range(self.test_samples_num):
            for j in range(self.test_moving_num_per_people):
                T1_moving_path = self.test_src_path + 'sample_' + str(i) + '/' + 'T1_moving_' + str(j) + '.png'
                T1_moving = np.array(Image.open(T1_moving_path)) / 255
                T1_moving = np.reshape(T1_moving, (1, self.vol_shape[0], self.vol_shape[1], 1))
                T2_moving_path = self.test_src_path + 'sample_' + str(i) + '/' + 'T2_moving_' + str(j) + '.png'
                T2_moving = np.array(Image.open(T2_moving_path)) / 255
                T2_moving = np.reshape(T2_moving, (1, self.vol_shape[0], self.vol_shape[1], 1))
                inputs = [T1_moving, T2_moving]
                startTime = time.time()
                temp = model.predict(inputs)
                endTime = time.time()
                duringTime = endTime - startTime

                if (i + j) != 0:
                    total_time += duringTime

        finalTime = total_time / (self.test_moving_num_per_people * self.test_samples_num - 1)
        finalStr = 'Conv的时间是：%s ms' % (finalTime * 1000)
        print(finalStr)


if __name__ == '__main__':
    demo = TopModel()
    demo.run()
