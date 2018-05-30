#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

batch_size = 100;
train_steps = 1000;

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

def create_model(train_x,train_y,test_x,test_y, words, num_songs):

    # Fetch the data
    # Feature columns describe how to use the input.
    my_feature_columns = []
    my_feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(key="Words", vocabulary_list= words))
    my_feature_columns.append(tf.feature_column.categorical_column_with_identity(key="Songs",num_buckets= num_songs))
    my_feature_columns.append(tf.feature_column.categorical_column_with_identity(key="Labels",num_buckets= num_songs))


    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[32, 32,32,32],
        # The model must choose between 3 classes.
        n_classes=num_songs)

    # Train the Model.
    classifier.train(
        input_fn=lambda:train_input_fn(train_x, train_y,
                                                 batch_size),
        steps=train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(test_x, test_y,
                                                batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    #expected = ['Setosa', 'Versicolor', 'Virginica']
    #predict_x = {
    #   'SepalLength': [5.1, 5.9, 6.9],
    #    'SepalWidth': [3.3, 3.0, 3.1],
    #    'PetalLength': [1.7, 4.2, 5.4],
    #    'PetalWidth': [0.5, 1.5, 2.1],
    #}

    ###predictions = classifier.predict(
    #  input_fn=lambda:eval_input_fn(predict_x,
    #                                            labels=None,
    ###                                            batch_size=args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    #for pred_dict, expec in zip(predictions, expected):
    #    class_id = pred_dict['class_ids'][0]
    #    probability = pred_dict['probabilities'][class_id]

    #   print(template.format(iris_data.SPECIES[class_id],
    #                          100 * probability, expec))

