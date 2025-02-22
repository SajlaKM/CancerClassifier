import tensorflow as tf
from tensorflow.keras import Model, layers, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# Define ResUNet Model
def resunet(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)

    def residual_block(x, filters):
        res = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        res = layers.BatchNormalization()(res)
        res = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(res)
        res = layers.BatchNormalization()(res)
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(x)
        shortcut = layers.BatchNormalization()(shortcut)
        out = layers.Add()([res, shortcut])
        return layers.ReLU()(out)

    # Encoder
    conv1 = residual_block(inputs, 64)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    conv2 = residual_block(pool1, 128)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    conv3 = residual_block(pool2, 256)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    conv4 = residual_block(pool3, 512)

    # Decoder
    up1 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(conv4)
    up1 = layers.Concatenate()([up1, conv3])
    conv5 = residual_block(up1, 256)
    up2 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv5)
    up2 = layers.Concatenate()([up2, conv2])
    conv6 = residual_block(up2, 128)
    up3 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv6)
    up3 = layers.Concatenate()([up3, conv1])
    conv7 = residual_block(up3, 64)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv7)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])
    return model
seg_model = resunet()
seg_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Print model summary
seg_model.summary()
