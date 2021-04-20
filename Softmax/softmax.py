# _*_ coding: utf-8 _*_
# @Time    :   2021/04/14 20:30:14
# @FileName:   softmax.py
# @Author  :   handy
# @Software:   VSCode
import tensorflow as tf
import matplotlib.pyplot as plt

# 针对fashion_mnist数据集，进行多分类任务(10)
# 训练图片部分是(60000, 28, 28) 60000张28X28的图片

(train_img, train_label), (test_img, test_label) = tf.keras.datasets.fashion_mnist.load_data()
# print(train_img.shape)
# 每个图片部分是0-255的数构成像素点,故需要变成0-1
train_img = train_img / 255
test_img = test_img / 255

model = tf.keras.Sequential()
# Dense层只支持一维数据,此处28x28需打平成一维28*28
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
# 可以多层间加dropout，参数为断开神经元比例
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='sigmoid'))

# 将标签one-hot编码
train_label = tf.keras.utils.to_categorical(train_label)
test_label = tf.keras.utils.to_categorical(test_label)

# model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
# 指定学习速率

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
            loss='categorical_crossentropy', 
            metrics=['acc'])

#  验证机，在训练过程中，同时打出验证集的loss等
# 此时history.history中有valid_loss、valid_acc等数据
history = model.fit(train_img, train_label, epochs=10, validation_data=(test_img, test_label))

# model.evaluate(test_img, test_label)

plt.plot(history.epoch, history.history.get('acc'), label='train_acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.show()