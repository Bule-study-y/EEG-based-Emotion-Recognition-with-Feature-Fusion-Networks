import h5py
import os
import paddle
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Layer, Pool2D, Linear, Sequential, to_variable, BatchNorm
from paddle.fluid.layers import reshape, concat
import time


def read_h5(model):
    feature_list = ['_PSD_1S.h5', '_HJORTH_1S.h5', '_SEN_1S.h5', '_DE_1S.h5', '_label_V.h5']

    def reader():
        for name in range(1, 33):
            if name < 10:
                for idex, feature in enumerate(feature_list):
                    if idex == 0:
                        name_path = 's0' + str(name) + feature
                        f = h5py.File(name_path, 'r')
                        PSD = np.array(f['data'])
                        f.close()
                    elif idex == 1:
                        name_path = 's0' + str(name) + feature
                        f = h5py.File(name_path, 'r')
                        HJORTH = np.array(f['data'])
                        f.close()
                    elif idex == 2:
                        name_path = 's0' + str(name) + feature
                        f = h5py.File(name_path, 'r')
                        SEN = np.array(f['data'])
                        f.close()
                    elif idex == 3:
                        name_path = 's0' + str(name) + feature
                        f = h5py.File(name_path, 'r')
                        DE = np.array(f['data'])
                        f.close()
                    elif idex == 4:
                        name_path = 's0' + str(name) + feature
                        f = h5py.File(name_path, 'r')
                        label = np.array(f['data'])
                        f.close()
                # print(PSD.shape,HJORTH.shape,SEN.shape,DE.shape)
                if model == 'train':
                    for n in range(int(PSD.shape[0] * 0.9)):
                        yield PSD[n], HJORTH[n], SEN[n], DE[n], label[n]

                if model == 'val':
                    for n in range(int(PSD.shape[0] * 0.9), PSD.shape[0]):
                        yield PSD[n], HJORTH[n], SEN[n], DE[n], label[n]

            if name >= 10:
                for idex, feature in enumerate(feature_list):
                    if idex == 0:
                        name_path = 's' + str(name) + feature
                        f = h5py.File(name_path, 'r')
                        PSD = np.array(f['data'])
                        f.close()
                    elif idex == 1:
                        name_path = 's' + str(name) + feature
                        f = h5py.File(name_path, 'r')
                        HJORTH = np.array(f['data'])
                        f.close()
                    elif idex == 2:
                        name_path = 's' + str(name) + feature
                        f = h5py.File(name_path, 'r')
                        SEN = np.array(f['data'])
                        f.close()
                    elif idex == 3:
                        name_path = 's' + str(name) + feature
                        f = h5py.File(name_path, 'r')
                        DE = np.array(f['data'])
                        f.close()
                    elif idex == 4:
                        name_path = 's' + str(name) + feature
                        f = h5py.File(name_path, 'r')
                        label = np.array(f['data'])
                        f.close()
                if model == 'train':
                    for n in range(int(PSD.shape[0] * 0.9)):
                        yield PSD[n], HJORTH[n], SEN[n], DE[n], label[n]

                if model == 'val':
                    for n in range(int(PSD.shape[0] * 0.9), PSD.shape[0]):
                        yield PSD[n], HJORTH[n], SEN[n], DE[n], label[n]

    return reader



class AlexNet(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_classes=1):
        super(AlexNet, self).__init__(name_scope)

        # AlexNet与LeNet一样也会同时使用卷积和池化层提取图像特征
        # 与LeNet不同的是激活函数换成了‘relu’
        w_param_attrs = fluid.ParamAttr(
            learning_rate=0.1,
            regularizer=fluid.regularizer.L1Decay(regularization_coeff=0.5),
            trainable=True)
        self.conv1 = Conv2D(num_channels=5, num_filters=1, filter_size=[1, 3], stride=[1, 3], param_attr=w_param_attrs,
                            act='relu')
        self.norm1 = BatchNorm(num_channels=1)
        self.conv2 = Conv2D(num_channels=5, num_filters=1, filter_size=[1, 2], param_attr=w_param_attrs, act='relu')
        self.norm2 = BatchNorm(num_channels=1)
        self.conv3 = Conv2D(num_channels=1, num_filters=1, filter_size=[32, 1], param_attr=w_param_attrs, act='relu')
        self.norm3 = BatchNorm(num_channels=1)
        self.conv4 = Conv2D(num_channels=1, num_filters=1, filter_size=[16, 1], stride=[16, 1],
                            param_attr=w_param_attrs, act='relu')
        self.norm4 = BatchNorm(num_channels=1)

        self.yconv1 = Conv2D(num_channels=32, num_filters=1, filter_size=5, param_attr=w_param_attrs, act='relu')
        self.ynorm1 = BatchNorm(num_channels=1)
        self.yconv2 = Conv2D(num_channels=16, num_filters=8, filter_size=3, param_attr=w_param_attrs, act='relu')
        self.ynorm2 = BatchNorm(num_channels=8)
        self.yconv3 = Conv2D(num_channels=8, num_filters=1, filter_size=1, param_attr=w_param_attrs, act='relu')
        self.ynorm3 = BatchNorm(num_channels=1)

        self.mconv1 = Conv2D(num_channels=5, num_filters=1, filter_size=[1, 3], param_attr=w_param_attrs, act='relu')
        self.mnorm1 = BatchNorm(num_channels=1)
        self.mconv2 = Conv2D(num_channels=5, num_filters=1, filter_size=[1, 2], param_attr=w_param_attrs, act='relu')
        self.mnorm2 = BatchNorm(num_channels=1)
        self.mconv3 = Conv2D(num_channels=1, num_filters=1, filter_size=[32, 1], param_attr=w_param_attrs, act='relu')
        self.mnorm3 = BatchNorm(num_channels=1)
        self.mconv4 = Conv2D(num_channels=1, num_filters=1, filter_size=[16, 1], stride=[16, 1],
                             param_attr=w_param_attrs, act='relu')
        self.mnorm4 = BatchNorm(num_channels=1)

        self.nconv1 = Conv2D(num_channels=5, num_filters=1, filter_size=[1, 3], param_attr=w_param_attrs, act='relu')
        self.nnorm1 = BatchNorm(num_channels=1)
        self.nconv2 = Conv2D(num_channels=5, num_filters=1, filter_size=[1, 2], param_attr=w_param_attrs, act='relu')
        self.nnorm2 = BatchNorm(num_channels=1)
        self.nconv3 = Conv2D(num_channels=1, num_filters=1, filter_size=[32, 1], param_attr=w_param_attrs, act='relu')
        self.nnorm3 = BatchNorm(num_channels=1)
        self.nconv4 = Conv2D(num_channels=1, num_filters=1, filter_size=[16, 1], stride=[16, 1],
                             param_attr=w_param_attrs, act='relu')
        self.nnorm4 = BatchNorm(num_channels=1)

        self.fc1 = Linear(input_dim=8363, output_dim=1080, param_attr=w_param_attrs, act='relu')
        self.drop_ratio1 = 0.5
        self.fc2 = Linear(input_dim=4096, output_dim=4096, param_attr=w_param_attrs, act='relu')
        self.drop_ratio2 = 0.5
        self.fc3 = Linear(input_dim=4096, output_dim=2048, param_attr=w_param_attrs, act='relu')
        self.drop_ratio3 = 0.5
        self.fc4 = Linear(input_dim=2048, output_dim=1080, param_attr=w_param_attrs, act='relu')
        self.drop_ratio4 = 0.5
        self.fc5 = Linear(input_dim=1080, output_dim=num_classes)
        self.fc6 = Linear(input_dim=6944, output_dim=num_classes)

    def forward(self, x, y, m, n):
        x1 = self.conv1(x)
        x1 = self.norm1(x1)
        x2 = self.conv2(x)
        x2 = self.norm2(x2)
        # x3 = self.conv3(x)
        x3 = self.conv3(fluid.layers.concat([x1, x2], axis=-1))
        x3 = self.norm3(x3)
        # x4 = self.conv3(x)
        x4 = self.conv4(fluid.layers.concat([x1, x2], axis=-1))
        x4 = self.norm4(x4)
        x = fluid.layers.concat([x3, x4], axis=2)
        x = fluid.layers.reshape(x, [x.shape[0], -1])

        y = self.yconv1(y)
        y = self.ynorm1(y)
        # y = self.yconv2(y)
        # y = self.ynorm2(y)
        # y = self.yconv3(y)
        # y = self.ynorm3(y)
        y = fluid.layers.reshape(y, [y.shape[0], -1])
        out = self.fc6(y)

        m1 = self.mconv1(m)
        m1 = self.mnorm1(m1)
        m2 = self.mconv2(m)
        m2 = self.mnorm2(m2)
        m3 = self.mconv3(fluid.layers.concat([m1, m2], axis=-1))
        m3 = self.mnorm3(m3)
        m4 = self.mconv4(fluid.layers.concat([m1, m2], axis=-1))
        m4 = self.mnorm4(m4)
        m = fluid.layers.concat([m3, m4], axis=2)
        m = fluid.layers.reshape(m, [m.shape[0], -1])

        n1 = self.nconv1(n)
        n1 = self.mnorm1(n1)
        n2 = self.nconv2(n)
        n2 = self.mnorm1(n2)
        n3 = self.nconv3(fluid.layers.concat([n1, n2], axis=-1))
        n3 = self.mnorm1(n3)
        n4 = self.nconv4(fluid.layers.concat([n1, n2], axis=-1))
        n4 = self.mnorm1(n4)
        n = fluid.layers.concat([n3, n4], axis=2)
        n = fluid.layers.reshape(n, [n.shape[0], -1])

        x = fluid.layers.concat([x, y, m, n], axis=1)
        x = self.fc1(x)
        # 在全连接之后使用dropout抑制过拟合
        x = fluid.layers.dropout(x, self.drop_ratio1)
        # x = self.fc2(x)
        # # 在全连接之后使用dropout抑制过拟合
        # x = fluid.layers.dropout(x, self.drop_ratio2)
        # x = self.fc3(x)
        # # 在全连接之后使用dropout抑制过拟合
        # x = fluid.layers.dropout(x, self.drop_ratio3)
        # x = self.fc4(x)
        # # 在全连接之后使用dropout抑制过拟合
        # x = fluid.layers.dropout(x, self.drop_ratio4)
        x = self.fc5(x)

        return x, out


    def train(model):
        print('start training ... ')
        model.train()
        epoch_num = 20
        opt = fluid.optimizer.SGDOptimizer(learning_rate=0.1,parameter_list=model.parameters())
        # 使用Paddle自带的数据读取器

        # train_loader = paddle.batch(paddle.reader.shuffle(read_h5(model='train'),buf_size=32*8), batch_size=60)
        # valid_loader = paddle.batch(paddle.reader.shuffle(read_h5(model='val'),buf_size=16), batch_size=8)
        train_loader = paddle.batch(read_h5(model='train'), batch_size=28)
        valid_loader = paddle.batch(read_h5(model='val'), batch_size=8)
        for epoch in range(epoch_num):
            start = time.time()
            for batch_id, data in enumerate(train_loader()):
                # 调整输入数据形状和类型
                y_data = np.array([item[0] for item in data], dtype='float32').reshape(-1, 32,128,60 )
                x_data = np.array([item[1] for item in data], dtype='float32').reshape(-1, 5,32,180)
                m_data = np.array([item[2] for item in data], dtype='float32').reshape(-1, 5,32,60)
                n_data = np.array([item[3] for item in data], dtype='float32').reshape(-1, 5,32,60)
                z_data = np.array([item[4] for item in data], dtype='int64').reshape(-1, 1)
                # 将numpy.ndarray转化成Tensor
                x = fluid.dygraph.to_variable(x_data)
                y = fluid.dygraph.to_variable(y_data)
                m = fluid.dygraph.to_variable(m_data)
                n = fluid.dygraph.to_variable(n_data)
                label = fluid.dygraph.to_variable(z_data)
                # 计算模型输出
                logits,out = model(x,y,m,n)
                # 计算损失函数
                # loss = fluid.layers.softmax_with_cross_entropy(logits, label)+0.3*fluid.layers.softmax_with_cross_entropy(out, label)
                loss = fluid.layers.softmax_with_cross_entropy(logits, label)
                avg_loss = fluid.layers.mean(loss)
                if batch_id % 500 == 0:
                    print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
                avg_loss.backward()
                opt.minimize(avg_loss)
                model.clear_gradients()

            model.eval()
            accuracies = []
            losses = []
            for batch_id, data in enumerate(valid_loader()):
                # 调整输入数据形状和类型
                y_data = np.array([item[0] for item in data], dtype='float32').reshape(-1, 32,128,60 )
                x_data = np.array([item[1] for item in data], dtype='float32').reshape(-1, 5,32,180)
                m_data = np.array([item[2] for item in data], dtype='float32').reshape(-1, 5,32,60)
                n_data = np.array([item[3] for item in data], dtype='float32').reshape(-1, 5,32,60)
                z_data = np.array([item[4] for item in data], dtype='int64').reshape(-1, 1)
                # 将numpy.ndarray转化成Tensor
                x = fluid.dygraph.to_variable(x_data)
                y = fluid.dygraph.to_variable(y_data)
                m = fluid.dygraph.to_variable(m_data)
                n = fluid.dygraph.to_variable(n_data)
                label = fluid.dygraph.to_variable(z_data)
                # 计算模型输出
                logits,out = model(x,y,m,n)
                pred = fluid.layers.softmax(logits)
                # 计算损失函数
                loss = fluid.layers.softmax_with_cross_entropy(logits, label)
                acc = fluid.layers.accuracy(pred, label)
                accuracies.append(acc.numpy())
                losses.append(loss.numpy())
            print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
            endtime = time.time()
            print('use time',endtime-start)
            model.train()

        # 保存模型参数
        # fluid.save_dygraph(model.state_dict(), 'AlexNet_psd')


if __name__ == '__main__':
    # 创建模型
    with fluid.dygraph.guard():
        model = AlexNet("AlexNet", num_classes=2)
        #启动训练过程
        train(model)

