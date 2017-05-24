"""
Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from 
the number of fire in the city of Chicago
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv

DATA_FILE = 'data/fire_theft.csv'

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

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
# YOUR CODE HERE

# Step 3: create weight and bias, initialized to 0
# YOUR CODE HERE

# Step 4: build model to predict Y
# YOUR CODE HERE

# Step 5: use the square error as the loss function
# YOUR CODE HERE

# Step 6: using gradient descent with learning rate of 0.001 to minimize loss
# YOUR CODE HERE

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    # YOUR CODE HERE

    # Step 8: train the model
    for i in range(100):  # train the model 100 times
        total_loss = 0
        for x, y in data:
            # Session runs train_op and fetch values of loss
            # YOUR CODE HERE
            total_loss += l
        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    # Step 9: output the values of w and b
    w_value, b_value = sess.run([w, b])

# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w_value + b_value, 'r', label='Predicted data')
plt.legend()
plt.show()
