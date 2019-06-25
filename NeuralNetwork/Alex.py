from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
#import makepretraindataset as md
import os

tf.logging.set_verbosity(tf.logging.INFO)

NPY_STORAGE = "gdrive/My Drive/HWR/numpy_aug/"


def cnn_model_fn(features, labels, mode):
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
  #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!',pool4.shape)
  # Dense Layer
  pool4_flat = tf.reshape(pool4, [-1, 27*27*256])
  dense = tf.layers.dense(inputs=pool4_flat, units=4096, activation=tf.nn.relu)
  dense2 = tf.layers.dense(inputs=dense, units=4096, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  #print(dropout.shape)




  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=27)
  #print(logits.shape)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

  if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=cross_entropy,
            eval_metric_ops=eval_metric_ops,
            evaluation_hooks=None)





# Load training and eval data
#train_data, train_labels = md.make_predataset2()
train_data = np.load(NPY_STORAGE + "trainData.npy")
train_labels = np.load(NPY_STORAGE + "trainLabels.npy")

print('train lables shape',train_labels.shape)

validationData = np.load(NPY_STORAGE + "testData.npy")
validationLabels = np.load(NPY_STORAGE + "testLabels.npy")



train_data = train_data/np.float32(255)
train_labels = train_labels.astype(np.int32)  # not required

eval_data = np.load(NPY_STORAGE + "evalData.npy")
eval_labels = np.load(NPY_STORAGE + "evalLabels.npy")


eval_data = eval_data/np.float32(255)
eval_labels = eval_labels.astype(np.int32)  # not required

#eval_data = eval_data/np.float32(255)
#eval_labels = eval_labels.astype(np.int32)  # not required

my_checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 10*60,  # Save checkpoints every 20 minutes.
    keep_checkpoint_max = 1,       # Retain the 10 most recent checkpoints.
)


# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir="gdrive/My Drive/HWR",
    config=my_checkpointing_config)


# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}



# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=32,
    num_epochs=10,
    shuffle=True)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    batch_size = 32,
    num_epochs=1,
    shuffle=False)

for i in range(10):
      print(i)
      mnist_classifier.train(input_fn=train_input_fn, steps = 100)
      print('done training')
      metrics = mnist_classifier.evaluate(input_fn=eval_input_fn)
      print(metrics)
