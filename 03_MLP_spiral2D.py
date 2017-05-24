'''
A MLP algorithm example using TensorFlow library.
This example is using generate random distribution
(http://cs231n.github.io/neural-networks-case-study/)

Code references:
https://github.com/shouvikmani/Tensorflow-Deep-Learning-Tutorial/blob/master/tutorial.ipynb
https://github.com/aymericdamien/TensorFlow-Examples/
http://cs231n.github.io/neural-networks-case-study/

The source code modified modified by S.W. Oh.
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# import Dense (fully-connected) layer
from util.layer import Dense

###### Generate 2D spiral random data and Plot ###################################
N = 200 # number of points per class
D = 2 # dimensionality
K = 4 # number of classes
X_train = np.zeros((N*K,D)) # data matrix (each row = single example)
y_train = np.zeros((N*K,K)) # class labels
yc = np.zeros(N*K, dtype='uint8')
for j in range(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4.8,(j+1)*4.8,N) + np.random.randn(N)*0.2 # theta
    X_train[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y_train[ix,j] = 1
    yc[ix] = j

# lets visualize the data:
plt.scatter(X_train[:, 0], X_train[:, 1], c=yc, s=40, cmap=plt.cm.Spectral)
plt.show()

# Random shuffle
perm = np.random.permutation(len(y_train))
X_train = X_train[perm,:]
y_train = y_train[perm,:]
yc = yc[perm]

# Parameters
learning_rate = 0.01
training_epochs = 500
batch_size = 10
display_step = 1

###### Build graph ######################################################

# Place holders
x = tf.placeholder(tf.float32, [None, 2]) # 2 dimensional input
y = tf.placeholder(tf.float32, [None, 4]) # 4 classes 

# Construct MLP with two hidden layer
h = Dense(x, [2,64], 'ih')
h = tf.nn.relu(h)
h = Dense(h, [64,64], 'hh')
h = tf.nn.relu(h)
logit = Dense(h, [64,4], 'hl')

pred = tf.nn.softmax(logit) # Softmax

# Directly compute loss from logit (to ensure stability and avoid overflow)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))

# Define optimizer and train_op
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


###### Start Training ###################################################
# Open a Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(y_train)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = X_train[i:i+batch_size,:]
            batch_ys = y_train[i:i+batch_size,:]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, cost], feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Visualize Dicision boundary
    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = sess.run(pred, feed_dict={x: np.c_[xx.ravel(), yy.ravel()]})

    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
   
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=yc, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()
 