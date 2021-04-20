# _*_ coding: utf-8 _*_
# @Time    :   2021/04/14 19:19:48
# @FileName:   logical_regression.py
# @Author  :   handy
# @Software:   VSCode
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
# 信用卡数据，最后一列值为1/-1，需要改为1/0

# 该数据没有表头，即首行就是数据
data = pd.read_csv('../data/credit-a.csv', header=None)
X, Y = data.iloc[:, :-1], data.iloc[:, -1].replace(-1, 0)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(4, input_shape=(15,), activation='relu'))
# 第二层input_shape可以不用加，自动
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X, Y, epochs=100)

# plt.title('Accuracy')
# plt.xlabel('epoches')
# plt.ylabel('Acc')

plt.figure(figsize=(1, 2))
ax_1 = plt.subplot(1, 2, 1)
ax_2 = plt.subplot(1, 2, 2)
ax_1.plot(history.epoch, history.history.get('acc'))
ax_2.plot(history.epoch, history.history.get('loss'))
# ax_1.setlabel('ACCURACY')
# history.history里面存储字典,包含每次的loss、acc等
# plt.plot(history.epoch, history.history.get('acc'))
# plt.plot(history.epoch, history.history.get('loss'))
plt.show()