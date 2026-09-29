import tensorflow as tf


def build_3d_cnn(input_shape, num_classes):
    return tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv3D(8, 3, activation="relu"),
        tf.keras.layers.MaxPooling3D(2),
        tf.keras.layers.Conv3D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling3D(2),
        tf.keras.layers.Conv3D(64, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling3D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
