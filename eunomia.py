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

# Convert brca array into numpy array for tensorflow
inputArray = np.array(brca[0], dtype=float)

# Initialize weight and adjustment vectors
x = tf.placeholder(tf.float32, [None, lenBRCA])
b = tf.Variable(tf.zeros([lenBRCA]))
W = tf.get_variable('W', shape=[lenBRCA, lenBRCA], initializer = tf.contrib.layers.xavier_initializer())
y = tf.matmul(x, W) + b

# Print array dimensions
print(a.get_shape())
print(b.get_shape())
print(W.get_shape())

# Calculate cross entropy and define training step
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = x, logits = y))
train_step = tf.train.AdamOptimizer(0.5).minimize(cross_entropy)

# Start tensorflow session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
print(sess.run(train_step, feed_dict={x: inputArray}))
