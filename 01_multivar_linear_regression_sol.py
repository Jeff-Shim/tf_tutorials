"""
Simple linear regression example in TensorFlow
This program tries to predict the final test score
of general psychology based on previous exam scores
"""

import numpy as np
import tensorflow as tf
import csv

DATA_FILE = 'data/test_scores.csv'

# Step 1: read data
with open(DATA_FILE, 'r') as f:
    data = []
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        if i == 0:
            continue
        data.append(row)
    n_samples = len(data)
    data = np.asarray(data, dtype='float32')

# Step 2: create placeholders
X = tf.placeholder(tf.float32, shape=[None, 3], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')

# Step 3: create weight and bias
W = tf.Variable(tf.random_normal([3, 1]), name='weights')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Step 4: build model to predict Y
Y_predicted = tf.matmul(X, W) + b

# Step 5: use the square error as the loss function
loss = tf.reduce_mean(tf.square(Y - Y_predicted, name='loss'))

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=2e-7)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())

    # Step 8: train the model
    for i in range(100):  # train the model 100 times
        total_loss = 0
        for x1, x2, x3, y in data:
            # Session runs train_op and fetch values of loss
            _, l = sess.run([train, loss], feed_dict={X: [[x1, x2, x3]], Y: [[y]]})
            total_loss += l
        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    # Step 9: output the values of w and b
    W_value, b_value = sess.run([W, b])

    # predict
    for i in [2, 13, 18]:
        y_h = sess.run(Y_predicted, feed_dict={X: np.expand_dims(data[i][:-1], 0)})
        print('X1: {}, X2: {}, X3: {}'.format(data[i][0], data[i][1], data[i][2]))
        print('Y_predicted: {}'.format(np.squeeze(y_h)))
        print('Y: {}'.format(data[i][-1]))
