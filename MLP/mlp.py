# _*_ coding: utf-8 _*_
# @Time    :   2021/04/14 17:02:35
# @FileName:   mlp.py
# @Author  :   handy
# @Software:   VSCode

import tensorflow as tf
import pandas as pd

data = pd.read_csv('../data/Advertising.csv')
X, Y = data.iloc[:, 1:-1], data.iloc[:, -1]

model = tf.keras.Sequential(
    [
        # 输入大小即为输入X的维度(即列的大小)
        tf.keras.layers.Dense(10, input_shape=(3,), activation='relu',),
        tf.keras.layers.Dense(1)
    ]
)

model.summary()
model.compile(optimizer='Adam', loss='mse')
model.fit(X, Y, epochs=500)

pre_data = data.iloc[:10, 1:-1]
print(model.predict(pre_data))