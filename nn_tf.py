import tensorflow as tf
from tensorflow.keras import layers, models


(train_images, train_activityels), (test_images, test_activityels) = tf.keras.datasets.fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_activityels, epochs=10, batch_size=32)

test_loss, test_acc = model.evaluate(test_images, test_activityels)

print(f'Teast Accuracy: {test_acc}')


