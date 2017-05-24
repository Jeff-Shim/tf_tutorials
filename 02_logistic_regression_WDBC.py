'''
A logistic regression learning algorithm example using TensorFlow library.
This example uses WDBC dataset (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

Code refercences:
https://github.com/shouvikmani/Tensorflow-Deep-Learning-Tutorial/blob/master/tutorial.ipynb
https://github.com/aymericdamien/TensorFlow-Examples/

The source code modified m by S.W. Oh.
'''

import tensorflow as tf
import numpy as np

###### Load WDBC data and preprocess. ################################### 
import csv
X = np.empty((569, 30))
y = np.empty((569, 1))
with open('data/cancer_data.csv', 'r') as csvfile:
	reader = csv.reader(csvfile)
	idx = 0
	for row in reader:
		if row[0] != 'id':
			X[idx] = np.asarray(row[2:], dtype=np.float32)           # Features
			y[idx] = np.asarray((row[1] == 'M'), dtype=np.float32)   # Labels: 0 -> benign (양성), 1 -> malignant (악성)
			idx += 1

# Random shuffle and split to train/test set
perm = np.random.permutation(len(y))
trainIndices = perm[:500]
testIndices = perm[500:]
X_train = X[trainIndices,:]
y_train = y[trainIndices]
X_test = X[testIndices,:]
y_test = y[testIndices]

# Normalize features (zero mean, unit variance)
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

#########################################################################

# Parameters
learning_rate = 0.01
training_epochs = 100
batch_size = 10
display_step = 1

###### Build graph ######################################################
# Now setup simple linear model 

# Place holders
x = tf.placeholder(tf.float32, [None, 30]) # Inputs: a batch of features (30 dims)
y = tf.placeholder(tf.float32, [None, 1]) # Labels: a batch of labels (0 -> benign (양성), 1 -> malignant (악성))

# Set model weights 
W = tf.Variable(tf.zeros([30, 1]))
b = tf.Variable(tf.zeros([1]))

# Construct model (y = W*X + b)
logit = tf.matmul(x, W) + b
pred = tf.nn.sigmoid(logit)

# Directly compute loss from logit (to ensure stability and avoid overflow)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=y))

# Define optimizer and train_op
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


###### Start Training ###################################################
# Open a Session
with tf.Session() as sess:
	# Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(X_train.shape[0]/batch_size)
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

    # Test model
    correct_prediction = tf.equal(tf.round(pred), y)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: X_test, y: y_test}))

