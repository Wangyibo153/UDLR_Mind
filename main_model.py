import tensorflow as tf
from src import losses
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, UpSampling2D, concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import VGG16
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from neuron.layers import SpatialTransformer
from deform_conv.layers import ConvOffset2D
import time

config = tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=False)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


class MainModel(object):
    def __init__(self):
        super(MainModel, self).__init__()
        # 初始化参数
        self.batch_size = 8
        self.vol_shape = (256, 256)
        self.channels = 1
        self.samples_num = 24
        self.moving_num_per_sample = 100
        self.src_path = '../datause/train_data/'
        self.weight_path = '../myweight/main_model/'
        self.loss = ['mse', losses.Grad('l2').loss]
        self.loss_weights = [1, 0.08]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
        self.epochs = 100
        self.steps_per_epoch = 20
        self.test_src_path = '../datause/test_data/'
        self.test_samples_num = 12
        self.test_moving_num_per_people = 31
        self.enc_nf = [16, 32, 32, 32, 32]
        self.dec_nf = [32, 32, 32, 32, 16]
        self.enc_nf1 = [24, 48, 48, 48, 48]
        self.dec_nf1 = [48, 48, 48, 48, 24]

    def create_morph(self, enc_nf, dec_nf):
        # 输入层
        moving_images = Input(shape=[*self.vol_shape, self.channels],
                              name='moving_images')
        fixed_images = Input(shape=[*self.vol_shape, self.channels],
                             name='fixed_images')
        # 链接
        x_in = concatenate([moving_images, fixed_images])
        # 卷积实现下采样
        x_enc = [x_in]

        for i in range(len(enc_nf)):
            if i == 0:
                x_enc.append(self.conv_block(x_enc[-1], enc_nf[i]))
            else:
                x_enc.append(self.conv_block(x_enc[-1], enc_nf[i], 2))

        # 上采样
        x = UpSampling2D()(x_enc[-1])
        x = concatenate([x, x_enc[-2]])
        x = self.conv_block(x, dec_nf[0])

        x = UpSampling2D()(x)
        x = concatenate([x, x_enc[-3]])
        x = self.conv_block(x, dec_nf[1])

        x = UpSampling2D()(x)
        x = concatenate([x, x_enc[-4]])
        x = self.conv_block(x, dec_nf[2])

        x = self.conv_block(x, dec_nf[3])

        x = UpSampling2D()(x)
        x = concatenate([x, x_enc[-5]])
        x = self.conv_block(x, dec_nf[4])

        # 特征形成形变场
        Fai = Conv2D(filters=2,
                     kernel_size=3,
                     padding='same',
                     kernel_initializer='he_normal',
                     strides=1,
                     name='Fai')(x)

        # 恢复
        moved_images = SpatialTransformer(interp_method='linear',
                                          indexing='ij',
                                          name='moved_images')([moving_images, Fai])
        return Model([moving_images, fixed_images],
                     [moved_images, Fai])

    def morph_main_body(self, enc_nf, dec_nf):
        # 输入层
        moving_images = Input(shape=[*self.vol_shape, self.channels],
                              name='moving_images')
        fixed_images = Input(shape=[*self.vol_shape, self.channels],
                             name='fixed_images')
        # 链接
        x_in = concatenate([moving_images, fixed_images])
        # 卷积实现下采样
        x_enc = [x_in]

        for i in range(len(enc_nf)):
            if i == 0:
                x_enc.append(self.conv_block(x_enc[-1], enc_nf[i]))
            else:
                x_enc.append(self.conv_block(x_enc[-1], enc_nf[i], 2))

        # 上采样
        x = UpSampling2D()(x_enc[-1])
        x = concatenate([x, x_enc[-2]])
        x = self.conv_block(x, dec_nf[0])

        x = UpSampling2D()(x)
        x = concatenate([x, x_enc[-3]])
        x = self.conv_block(x, dec_nf[1])

        x = UpSampling2D()(x)
        x = concatenate([x, x_enc[-4]])
        x = self.conv_block(x, dec_nf[2])

        x = self.conv_block(x, dec_nf[3])

        x = UpSampling2D()(x)
        x = concatenate([x, x_enc[-5]])
        x = self.conv_block(x, dec_nf[4])

        # 特征形成形变场
        Fai = Conv2D(filters=2,
                     kernel_size=3,
                     padding='same',
                     kernel_initializer='he_normal',
                     strides=1,
                     name='Fai')(x)

        return Model([moving_images, fixed_images],
                     [Fai])

    def off_morph_main_body(self, enc_nf, dec_nf):
        moving_images = Input(shape=[*self.vol_shape, self.channels],
                              name='moving_images')
        fixed_images = Input(shape=[*self.vol_shape, self.channels],
                             name='fixed_images')
        x_in = concatenate([moving_images, fixed_images])
        # 卷积实现下采样
        x_enc = [x_in]

        for i in range(len(enc_nf)):
            if i == 0:
                x_enc.append(self.off_conv_block(x_enc[-1], enc_nf[i]))
            else:
                x_enc.append(self.conv_block(x_enc[-1], enc_nf[i], 2))

        # 上采样
        x = UpSampling2D()(x_enc[-1])
        x = concatenate([x, x_enc[-2]])
        x = self.conv_block(x, dec_nf[0])

        x = UpSampling2D()(x)
        x = concatenate([x, x_enc[-3]])
        x = self.conv_block(x, dec_nf[1])

        x = UpSampling2D()(x)
        x = concatenate([x, x_enc[-4]])
        x = self.conv_block(x, dec_nf[2])

        x = self.conv_block(x, dec_nf[3])

        x = UpSampling2D()(x)
        x = concatenate([x, x_enc[-5]])
        x = self.off_conv_block(x, dec_nf[4])

        # 特征形成形变场
        Fai = Conv2D(filters=2,
                     kernel_size=3,
                     padding='same',
                     kernel_initializer='he_normal',
                     strides=1,
                     name='Fai')(x)

        return Model([moving_images, fixed_images],
                     [Fai])

    def build_off_morph(self, enc_nf, dec_nf):
        # 实例化morph main body
        morph = self.off_morph_main_body(enc_nf, dec_nf)
        # 指定输入
        moving_images, fixed_images = morph.input
        # 实例化空间变化 并且指定输出
        moved_images = SpatialTransformer(interp_method='linear',
                                          indexing='ij',
                                          name='moved_images')([moving_images, morph.output])
        # 返回模型
        return Model([moving_images, fixed_images],
                     [moved_images, morph.output])

    def build_morph(self, enc_nf, dec_nf):
        # 实例化morph main body
        morph = self.morph_main_body(enc_nf, dec_nf)
        # 指定输入
        moving_images, fixed_images = morph.input
        # 实例化空间变化 并且指定输出
        moved_images = SpatialTransformer(interp_method='linear',
                                          indexing='ij',
                                          name='moved_images')([moving_images, morph.output])
        # 返回模型
        return Model([moving_images, fixed_images],
                     [moved_images, morph.output])

    @staticmethod
    def conv_block(x_input, filters, strides=1):
        x = Conv2D(filters=filters,
                   kernel_size=3,
                   padding='same',
                   kernel_initializer='he_normal',
                   strides=strides)(x_input)
        x_output = LeakyReLU(0.2)(x)
        return x_output

    @staticmethod
    def off_conv_block(x_input, filters, strides=1):
        x = Conv2D(filters=filters,
                   kernel_size=3,
                   padding='same',
                   kernel_initializer='he_normal',
                   strides=strides)(x_input)
        x = ConvOffset2D(filters=filters)(x)
        x_output = LeakyReLU(0.2)(x)
        return x_output

    def data_generator(self):
        """这个生成器是T1 moving往T2 fixed配准 注意形变场!!!!"""
        zeros_tensors = np.zeros((self.batch_size, *self.vol_shape, 2))
        while True:
            for i in range(self.batch_size):
                sample_index = np.random.randint(0, self.samples_num)
                moving_index = np.random.randint(0, self.moving_num_per_sample)

                T1_fixed_struc_path = self.src_path + 'sample_' + str(sample_index) + '/T1_fixed_struc.png'
                T1_moving_struc_path = self.src_path + 'sample_' + str(sample_index) + '/' + 'T1_moving_struc_' + str(moving_index) + '.png'
                T2_fixed_struc_path = self.src_path + 'sample_' + str(sample_index) + '/T2_fixed_struc.png'

                T1_fixed_struc = np.array(Image.open(T1_fixed_struc_path)) / 255
                T1_moving_struc = np.array(Image.open(T1_moving_struc_path)) / 255
                T2_fixed_struc = np.array(Image.open(T2_fixed_struc_path)) / 255

                T1_fixed_struc = np.reshape(T1_fixed_struc, (1, self.vol_shape[0], self.vol_shape[1], 1))
                T1_moving_struc = np.reshape(T1_moving_struc, (1, self.vol_shape[0], self.vol_shape[1], 1))
                T2_fixed_struc = np.reshape(T2_fixed_struc, (1, self.vol_shape[0], self.vol_shape[1], 1))

                if i == 0:
                    T1_fixed_strucs = T1_fixed_struc
                    T1_moving_strucs = T1_moving_struc
                    T2_fixed_strucs = T2_fixed_struc
                else:
                    T1_fixed_strucs = np.concatenate([T1_fixed_strucs, T1_fixed_struc], axis=0)
                    T1_moving_strucs = np.concatenate([T1_moving_strucs, T1_moving_struc], axis=0)
                    T2_fixed_strucs = np.concatenate([T2_fixed_strucs, T2_fixed_struc], axis=0)

            inputs = [T1_moving_strucs, T2_fixed_strucs]
            outputs = [T1_fixed_strucs, zeros_tensors]

            yield inputs, outputs

    def run(self):
        # 实例化模型
        model = self.create_morph(self.enc_nf, self.dec_nf)
        # 打印信息
        model.summary()
        # 打开权重
        model.load_weights(self.weight_path + 'test.h5')
        # 编译模型
        model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)
        # 学习率衰减
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='moved_images_loss',
                                                         patience=5,
                                                         verbose=1,
                                                         factor=0.5,
                                                         min_lr=1e-6)
        # 开始训练
        model.fit_generator(self.data_generator(),
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=self.epochs,
                            verbose=2,
                            callbacks=[reduce_lr])
        # 保存权重
        model.save_weights(self.weight_path + 'test.h5')

    def train_morph(self):
        # 实例化模型
        model = self.build_morph(self.enc_nf1, self.dec_nf1)
        # 打印信息
        model.summary()
        # 打开权重
        model.load_weights(self.weight_path + 'test1.h5')
        # 编译模型
        model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)
        # 学习率衰减
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='moved_images_loss',
                                                         patience=5,
                                                         verbose=1,
                                                         factor=0.5,
                                                         min_lr=1e-6)
        # 开始训练
        model.fit_generator(self.data_generator(),
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=self.epochs,
                            verbose=2,
                            callbacks=[reduce_lr])
        # 保存权重
        model.save_weights(self.weight_path + 'test1.h5')

    def run(self):
        # 实例化模型
        model = self.create_morph(self.enc_nf, self.dec_nf)
        # 打印信息
        model.summary()
        # 打开权重
        model.load_weights(self.weight_path + 'test.h5')
        # 编译模型
        model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)
        # 学习率衰减
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='moved_images_loss',
                                                         patience=5,
                                                         verbose=1,
                                                         factor=0.5,
                                                         min_lr=1e-6)
        # 开始训练
        model.fit_generator(self.data_generator(),
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=self.epochs,
                            verbose=2,
                            callbacks=[reduce_lr])
        # 保存权重
        model.save_weights(self.weight_path + 'test.h5')

    def train_off_morph(self):
        # 实例化模型
        model = self.build_off_morph(self.enc_nf, self.dec_nf)
        # 打印信息
        model.summary()
        # 打开权重
        # model.load_weights(self.weight_path + 'test.h5')
        # 编译模型
        model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)
        # 学习率衰减
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='moved_images_loss',
                                                         patience=5,
                                                         verbose=1,
                                                         factor=0.5,
                                                         min_lr=1e-6)
        # 开始训练
        model.fit_generator(self.data_generator(),
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=self.epochs,
                            verbose=2,
                            callbacks=[reduce_lr])
        # 保存权重
        model.save_weights(self.weight_path + 'offtest.h5')

    @staticmethod
    def plot(images, names):
        num = len(images)
        for i in range(1, num + 1):
            plt.subplot(1, num, i)
            plt.imshow(images[i - 1], cmap='gray')
            plt.title(names[i - 1])
            plt.xticks([])
            plt.yticks([])
            plt.colorbar(fraction=0.05, pad=0.05)
        plt.show()

    def predict(self):
        model = self.build_morph(self.enc_nf1, self.dec_nf1)
        model.load_weights(self.weight_path + 'test1.h5')
        T1_path = '../datause/train_data/sample_0/T1_moving_struc_0.png'
        T1 = np.array(Image.open(T1_path)) / 255
        T1 = np.reshape(T1, (1, self.vol_shape[0], self.vol_shape[1], 1))
        T2_path = '../datause/train_data/sample_0/T2_fixed_struc.png'
        T2 = np.array(Image.open(T2_path)) / 255
        T2 = np.reshape(T2, (1, self.vol_shape[0], self.vol_shape[1], 1))
        inputs = [T1, T2]
        predicts = model.predict(inputs)
        images = [inputs[0][0, ..., 0], inputs[1][0, ..., 0], predicts[0][0, ..., 0], predicts[1][0, ..., 0], predicts[1][0, ..., 0]]
        names = ['Input:T1 moving', 'Input:T2 fixed', 'Output:T1 fixed', 'Output:Tensor x', 'Output:Tensor y']
        self.plot(images, names)


if __name__ == '__main__':
    demo = MainModel()
    demo.predict()
