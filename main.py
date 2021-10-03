import fiftyone.zoo as foz
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation
from keras.preprocessing.image import load_img
# load and display an image with Matplotlib
from matplotlib import pyplot

if __name__ == '__main__':
    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        split="validation",
        label_types=["detections", "segmentations"],
        classes=["Cat", "Dog"],
        max_samples=500,
    )
    dataset.persistent = True
    resized_imgs = []
    for sample in dataset:
        img = load_img(sample.filepath).resize((64, 64))
        img_array = np.array(img)
        resized_imgs.append(img_array)

    resized_imgs_array = np.array(resized_imgs)
    training_images = resized_imgs_array[0:400]
    test_images = resized_imgs_array[400:500]

    # Set input shape
    sample_shape = training_images[0].shape
    img_width, img_height = sample_shape[0], sample_shape[1]
    input_shape = (img_width, img_height, 3)

    # Нормализируем лист массивов без цикла - каждый элемент делим
    training_images = training_images / 255.0
    test_images = test_images / 255.0

    # Reshape data
    training_images = training_images.reshape(len(training_images), input_shape[0], input_shape[1], input_shape[2])
    test_images = test_images.reshape(len(test_images), input_shape[0], input_shape[1], input_shape[2])

    # Создаем архитектуру модели
    model = tf.keras.models.Sequential([
        Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        # Dropout(0.5, noise_shape=None, seed=None),
        BatchNormalization(),
        Flatten(),
        Dense(128),
        Activation(activation=tf.nn.relu),
        Dense(10),
        Activation(activation=tf.nn.softmax)
    ])

    # Задаем модели функции потерь и алгоритм для перестройки весов
    # (back propagation algoritm, в нашем случае - градиентный спуск)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    tf.keras.utils.plot_model(model, to_file="my_model.png", show_shapes=True)
    model.summary()

    # Тренируем модель
    # model.fit(training_images, training_labels, epochs=5)

    # Проверяем на тестовой выборке
    # model.evaluate(test_images, test_labels)

    # classifications = model.predict(test_images)

    # print(classifications[0])
    # print(test_labels[0])
