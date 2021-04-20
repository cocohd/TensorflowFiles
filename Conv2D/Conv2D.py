import tensorflow as tf
import numpy as np

# 卷积
(train_img, train_label), (test_img, test_label) = tf.keras.datasets.fashion_mnist.load_data()
# 卷积需要四维
train_img = tf.expand_dims(train_img, -1)
test_img = tf.expand_dims(test_img, -1)
print(train_img)

model = tf.keras.Sequential()
# Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1)
# filters代表卷积核个数，指卷积后的'厚度'(个数)，padding默认valid，即不填充(维度会变小)，填充='same'的话大小不变
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
# 默认为2x2
model.add(tf.keras.layers.MaxPool2D())
print(model.output_shape)
# >>> (None, 13, 13, 32)
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# 全局平均池化，将此处(None, 11, 11, 64)的变为(None, 64)  整张'图'池化
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
            loss='sparse_categorical_crossentropy', 
            metrics=['acc'])

model.fit(train_img, train_label, epochs=30, validation_data=(test_img, test_label))