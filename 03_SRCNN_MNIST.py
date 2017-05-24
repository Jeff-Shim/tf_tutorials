'''
A SRCNN example using TensorFlow library.
Recover High-resololution image given low-resolution image. 

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

# import Convolution layer
from util.layer import Conv2D

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 10
display_step = 1

###### Build graph ######################################################
# Now setup simple linear model 

# Place holders
H = tf.placeholder(tf.float32, [None,28,28,1]) # Input image
L = tf.image.resize_bicubic(H, [7,7]) # Downsample input inside graph
L = tf.image.resize_nearest_neighbor(L, [28,28]) # and upsample back 

# Construct CNN 
h = Conv2D(L, [3,3,1,8], [1,1,1,1], 'SAME', 'conv1') # out_shape: [Batch,28,28,8]
h = tf.nn.relu(h)  

h = Conv2D(h, [3,3,8,8], [1,1,1,1], 'SAME', 'conv2') # out_shape: [Batch,28,28,8]
h = tf.nn.relu(h) 
  
pred = Conv2D(h, [3,3,8,1], [1,1,1,1], 'SAME', 'conv3') # out_shape: [Batch,28,28,1]

# L2 (Euclidean) distance as cost function
cost = tf.reduce_mean((pred - H)**2)

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
            batch_H, _ = mnist.train.next_batch(batch_size)
            batch_H = np.reshape(batch_H, [batch_size,28,28,1])

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, cost], feed_dict={H: batch_H})
            # Compute average loss
            avg_cost += c / total_batch

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

##### Visualizing the results ######################################################
    # select some of test images
    num_image=20
    image_shape=[28,28]
    test_H = np.reshape(mnist.test.images, [-1,28,28,1])[:num_image] 
    test_L, test_pred = sess.run([L, pred], feed_dict={H:test_H})
    # Plot
    canvas = np.zeros((3*image_shape[0], 20*image_shape[1]))
    for i in range(num_image):
        canvas[0:image_shape[0], image_shape[1]*i:image_shape[1]*i+image_shape[1]] = test_H[i,:,:,0]
        canvas[image_shape[0]:2*image_shape[0], image_shape[1]*i:image_shape[1]*i+image_shape[1]] = test_L[i,:,:,0]   
        canvas[2*image_shape[0]:3*image_shape[0], image_shape[1]*i:image_shape[1]*i+image_shape[1]] = test_pred[i,:,:,0]    

    plt.figure(0)
    plt.imshow(canvas, interpolation='nearest', vmin=0, vmax=1, cmap='gray')
    plt.show()

