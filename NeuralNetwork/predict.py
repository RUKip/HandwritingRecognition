from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
# import makepretraindataset as md
# import os
import cv2
from pathlib import Path

tf.logging.set_verbosity(tf.logging.ERROR)
# tf.logging.set_verbosity(tf.logging.INFO)


class MyNetwork:
    def __init__(self):
        self.path = Path.cwd() / "NeuralNetwork"

        self.my_checkpointing_config = tf.estimator.RunConfig(
            save_checkpoints_secs=10 * 60,  # Save checkpoints every 10 minutes.
            keep_checkpoint_max=1,  # Retain the 10 most recent checkpoints.
        )

        self.mnist_classifier = tf.estimator.Estimator(
            model_fn=self.cnn_model_fn,
            model_dir=self.path.as_posix(),
            config=self.my_checkpointing_config)

        # Set up logging for predictions
        self.tensors_to_log = {"probabilities": "softmax_tensor"}

        self.NumberToLabel = {
            5: 'Alef',
            24: 'Ayin',
            9: 'Bet',
            25: 'Dalet',
            20: 'Gimel',
            14: 'He',
            21: 'Het',
            12: 'Kaf',
            13: 'Kaf-final',
            22: 'Lamed',
            4: 'Mem',
            19: 'Mem-medial',
            17: 'Nun-final',
            6: 'Nun-medial',
            10: 'Pe',
            0: 'Pe-final',
            8: 'Qof',
            1: 'Resh',
            23: 'Samekh',
            18: 'Shin',
            16: 'Taw',
            15: 'Tet',
            2: 'Tsadi-final',
            26: 'Tsadi-medial',
            11: 'Waw',
            7: 'Yod',
            3: 'Zayin'
        }

    def predict_image(self, image):
        # image = cv2.imread('letter.jpg', 0)
        image = cv2.resize(image, (227, 227))
        image = np.float32(image)
        image = np.array(image)
        image = image.reshape((-1, 227, 227, 1))
        im_pred = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(image)},
            y=None,
            batch_size=1,
            num_epochs=1,
            shuffle=False)

        # print(image.shape)

        pred = list(self.mnist_classifier.predict(input_fn=im_pred))
        # print(pred)
        pred_class = [p["classes"] for p in pred]
        predicted_classes = [p["probabilities"] for p in pred]
        # pr = pred_class[0]
        # print(
        #     "New Samples, Class Predictions:    {}\n"
        #     .format(predicted_classes))
        # print(pred_class[0])
        return predicted_classes, self.NumberToLabel[pred_class[0]]

    def cnn_model_fn(self, features, labels, mode):
        """Model function for CNN."""
        # Input Layer
        input_layer = tf.reshape(features["x"], [-1, 227, 227, 1])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=96,
            kernel_size=[11, 11],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=256,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=384,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[3, 3], strides=2)

        conv4 = tf.layers.conv2d(
            inputs=conv3,
            filters=256,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[3, 3], strides=2)
        # Dense Layer
        pool4_flat = tf.reshape(pool4, [-1, 27 * 27 * 256])
        dense = tf.layers.dense(inputs=pool4_flat, units=4096, activation=tf.nn.relu)
        dense2 = tf.layers.dense(inputs=dense, units=4096, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=27)
        # print(logits.shape)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # "classes": tf.nn.softmax(logits),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            #print("!!!!!!!!!!!!!!pred: ", predictions["probabilities"])
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
