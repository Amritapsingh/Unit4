# Amrit Singh
# TCSS 435

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(6, (5, 5), activation='sigmoid', input_shape=(28, 28, 1), padding="valid"),
    tf.keras.layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Conv2D(16, (5, 5), activation='sigmoid', padding="valid"),
    tf.keras.layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

print(probability_model(x_test[:5]))