# _*_ coding: utf-8 _*_
# @Time    :   2021/04/14 17:02:41
# @FileName:   linear_regression.py
# @Author  :   handy
# @Software:   VSCode
import tensorflow as tf
import pandas as pd

# f(x) = a * x + b

x = [1, 2, 3]
y = [6, 15, 23]

linear_model = tf.keras.Sequential()
linear_model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
# 查看模型信息(参数)
linear_model.summary()
# 编译模型
linear_model.compile(optimizer='Adam', loss='mse')
# 训练模型
linear_model.fit(x, y, epochs=20)
# 使用模型预测
print(linear_model.predict(pd.Series([5])))

