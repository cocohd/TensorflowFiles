# tf.dataset.from_tensor_slices(),接受一个列表、字典或numpy做参数，
# 返回的是里面每个元素的张量大小，如[1, 2, 3] 返回Tensor shape()

import tensorflow as tf

(train_img, train_label), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

train_imgs = tf.data.Dataset.from_tensor_slices(train_img)
train_labels = tf.data.Dataset.from_tensor_slices(train_label)
# print(train_imgs)
# >>> <TensorSliceDataset shapes: (28, 28), types: tf.uint8>

# 将训练数据和标签整合在一起，然后便于打乱顺序
train_img_label = tf.data.Dataset.zip((train_imgs, train_labels))
# print(train_img_label)

# 直接from_tensor_slices也可
# train_labels = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))


# 打乱顺序，重复打乱(不然每次都一样的顺序)，batch_size=64即每次输出64个样本
train_img_label = train_img_label.shuffle(10000).repeat().batch(64)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
            loss='sparse_categorical_crossentropy', 
            metrics=['acc'])
# 样本数量 // batch_size    60000//64
step = train_img.shape[0] // 64
model.fit(train_img_label, epochs=5, steps_per_epoch=step)
# 同样可以添加validation_data
# model.fit(train_img_label, epochs=5, steps_per_epoch=step, validation_data=xxx, validation_steps=xx // 64)