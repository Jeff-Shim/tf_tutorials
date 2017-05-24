'''
Build your own deep network using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Code references:
https://github.com/shouvikmani/Tensorflow-Deep-Learning-Tutorial/blob/master/tutorial.ipynb
https://github.com/aymericdamien/TensorFlow-Examples/

The source code modified modified by S.W. Oh.
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# import Dense (fully-connected) layer and Convolution layer
from util.layer import Dense, Conv2D, BatchNorm

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 5
batch_size = 10
display_step = 1

###### Build graph ######################################################
# Place holders
x = tf.placeholder(tf.float32, [None,28,28,1]) # mnist data image of shape [28,28,1]
y = tf.placeholder(tf.float32, [None,10]) # 0-9 digits recognition => 10 classes
is_train = tf.placeholder(tf.bool, shape=[]) # Train flag


######################################################################

# your code here !!

# Layer Usages:
#     h = Conv2D(h, [3,3,1,8], [1,1,1,1], 'SAME', 'conv1')
#     h = BatchNorm(h, is_train, decay=0.9, name='bn1')
#     h = tf.nn.relu(h)
#     h = tf.nn.max_pool(h, [1,2,2,1], [1,2,2,1], 'SAME')
#     h = Dense(h, [8,10], 'fc1')

#######################################################################




pred = tf.nn.softmax(logit) # Softmax

# Directly compute loss from logit (to ensure stability and avoid overflow)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=y))

# Define optimizer and train_op
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#########################################################################



###### Start Training ###################################################
# Open a Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = np.reshape(batch_xs, [batch_size,28,28,1])
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, cost], feed_dict={x: batch_xs, y: batch_ys, is_train: True})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: np.reshape(mnist.test.images, [-1,28,28,1]), y: mnist.test.labels, is_train: False}))