import csv
import tensorflow as tf
import numpy as np
from src.autoencoder import autoencoder

# Read in CSV file
brca = []
count = 0

# brca is 192 long
with open("brca_toronto_collab_mutect_123_030617.csv") as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        row.pop()
        if count != 0:
            floatRow = [float(i) for i in row]
            brca.append(floatRow)
        count += 1
csvFile.close()

# Get the length of brca
lenBRCA = len(brca[0])
halfBRCA = lenBRCA / 2

# Initialize weight and adjustment vectors
x11 = tf.placeholder(tf.float32, [None,lenBRCA])
b11 = tf.Variable(tf.zeros([halfBRCA]))
b12 = tf.Variable(tf.zeros([lenBRCA]))
W11 = tf.get_variable('W11', shape=[lenBRCA, halfBRCA], initializer = tf.contrib.layers.xavier_initializer())
W12 = tf.get_variable('W12', shape=[halfBRCA, lenBRCA], initializer = tf.contrib.layers.xavier_initializer())
y11 = tf.nn.relu(tf.matmul(x11, W11) + b11)
y12 = tf.nn.relu(tf.matmul(y11, W12) + b12)

# Print array dimensions
print("The shape of x11 is: ", x11.get_shape())
print("The shape of b11 is: ", b11.get_shape())
print("The shape of b12 is: ", b12.get_shape())
print("The shape of W11 is: ", W11.get_shape())
print("The shape of W12 is: ", W12.get_shape())

# Calculate cross entropy and define training step
square_difference1 = tf.reduce_sum(tf.square(x11 - y12))
train_step1 = tf.train.AdamOptimizer().minimize(square_difference1)

# Start tensorflow session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train autoencoder
for i in range(10000):
    for j in range(len(brca)):
        # Convert brca array into numpy array for tensorflow
        inputArray = np.array(brca[i], dtype=float).reshape(1, lenBRCA)
        # print("\nInput Array\n", inputArray)
        sess.run(train_step1, feed_dict={x11: inputArray})
        # print("\nW11\n", sess.run(W11))
        # print("\nb11\n", sess.run(b11))
        # print("\nW12\n", sess.run(W12))
        # print("\nb12\n", sess.run(b12))

# Calculate difference between input and ouput
accuracy = tf.reduce_sum(tf.square(x11 - y12))

# Print difference
inputArray = np.array(brca[0], dtype=float).reshape(1, lenBRCA)
print(sess.run(accuracy, feed_dict={x11: inputArray}))
