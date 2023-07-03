# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.training import coordinator


def feature_numeric(key):
  return tf.feature_column.numeric_column(key=key, default_value=0)

def feature_indicator(key, vocabulary):
  return tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list(
      key=key, vocabulary_list=vocabulary ))


labels = ['Label1','Label2','Label3']

model = tf.estimator.DNNClassifier(
  feature_columns=[
    feature_numeric("number"),
    # feature_indicator("indicator", ["A","B","C"]),
  ],
  hidden_units=[64, 16, 8],
  model_dir='./models',
  n_classes=len(labels),
  label_vocabulary=labels)

def train(inputs, training):
  model.train(
    input_fn = numpy_io.numpy_input_fn(x=inputs, y=training, batch_size=2, shuffle=False, num_epochs=1),
    steps=1)

inputs = {
  "number": np.array([1,2,3,4,5]),
  # "indicator": np.array([
  #     ["A"],
  #     ["B"],
  #     ["C"],
  #     ["A", "A"],
  #     ["A", "B", "C"],
  #   ]),
}

training = np.array(['Label1','Label2','Label3','Label2','Label1'])

train(inputs, training)

 