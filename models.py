import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten


def simple_cnn(input_shape, num_layers=12):
    # build network architecture using convolutional layers
    model = keras.models.Sequential()

    # 1st conv layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
                            kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'))

    # 2nd conv layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'))

    # 3rd conv layer
    model.add(layers.Conv2D(32, (2, 2), activation='relu',
                            kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding='same'))

    # flatten output and feed into dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    layers.Dropout(0.3)

    # softmax output layer
    model.add(layers.Dense(num_layers, activation='softmax'))
    return model


def cnn_tutorial(input_shape, X_train, num_layers=12):
    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(X_train)
    model = keras.models.Sequential([
        layers.Input(shape=input_shape),
        # preprocessing.Resizing(32, 32),
        # norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_layers),
    ])
    return model


def VGG16(input_shape, num_layers=12):
    model = keras.models.Sequential()
    # first part
    model.add(Conv2D(input_shape=(input_shape), filters=64,
                     kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # 2nd part
    model.add(Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # 3rd part
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # 4th part
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # 5th part
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # final
    model.add(Flatten())
    model.add(Dense(units=1024, activation="relu"))
    model.add(Dense(units=1024, activation="relu"))
    model.add(layers.Dense(num_layers, activation='softmax'))
    return model


def VGG16_twist(input_shape, num_layers=12):
    inputs = keras.Input(shape=(input_shape))
    # first part
    x = Conv2D(filters=64,
               kernel_size=(3, 3), padding="same", activation="relu")(inputs)
    x = Conv2D(filters=64,
               kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    # 2nd part
    x = Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu")(x)
    x = Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu")(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    # 3rd part
    x = Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu")(x)
    x = Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu")(x)
    x = Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu")(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    # 4th part
    x = Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu")(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    # 5th part
    x = Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu")(x)
    x = Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu")(x)
    x = Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu")(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    # Additional global avg + pooling layer
    max_x = layers.GlobalMaxPooling2D()(x)
    avg_x = layers.GlobalAveragePooling2D()(x)
    x = layers.Concatenate(axis=1)([max_x, avg_x])
    # final
    x = Flatten()(x)
    x = Dense(units=1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = Dense(units=1024, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(num_layers, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=out, name="vgg1")
    return model


def VGG11(input_shape, num_layers=12):
    model = keras.models.Sequential()
    # first part
    model.add(Conv2D(input_shape=(), filters=64,
                     kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # 2nd part
    model.add(Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # 3rd part
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # 4th part
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # 5th part
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # final
    model.add(Flatten())
    model.add(Dense(units=1024, activation="relu"))
    model.add(Dense(units=1024, activation="relu"))
    model.add(layers.Dense(num_layers, activation='softmax'))
    return model


if __name__ == "__main__":
    m = VGG16_twist((128, 128, 1))
    print(m.summary())
