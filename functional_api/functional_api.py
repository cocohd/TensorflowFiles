# _*_ coding: utf-8 _*_
# @Time    :   2021/04/15 19:57:34
# @FileName:   functional_api.py
# @Author  :   handy
# @Software:   VSCode
import tensorflow as tf
import matplotlib.pyplot as plt

# 函数式API调用
# 可写多输入、多输出模型。如多个输入加个连接层……
(train_img, train_label), (test_img, test_label ) = tf.keras.datasets.fashion_mnist.load_data()

train_img = train_img / 255
test_img = test_img /255

# 此处train_img.shape 是(60000, 28, 28),是指60000张28*28大小的图片
# 28*28才是维度,60000是batch_size
input = tf.keras.Input((28, 28))
# 下列类都实现了__call__方法
x = tf.keras.layers.Flatten()(input)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
output = tf.keras.layers.Dense(10, activation='sigmoid')(x)

model = tf.keras.Model(inputs=input, outputs=output)

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
model.fit(train_img, train_label, epochs=20, validation_data=(test_img, test_label))
