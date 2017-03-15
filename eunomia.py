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

# Initialize weight and adjustment vectors
x11 = tf.placeholder(tf.float32, [None,lenBRCA])
b11 = tf.Variable(tf.zeros([lenBRCA]))
b12 = tf.Variable(tf.zeros([lenBRCA]))
W11 = tf.get_variable('W11', shape=[lenBRCA, lenBRCA], initializer = tf.contrib.layers.xavier_initializer())
W12 = tf.get_variable('W12', shape=[lenBRCA, lenBRCA], initializer = tf.contrib.layers.xavier_initializer())
y11 = tf.nn.relu(tf.matmul(x11, W11) + b11)
y12 = tf.nn.relu(tf.matmul(y11, W12) + b12)

# Print array dimensions
print("The shape of x11 is: ", x11.get_shape())
print("The shape of b11 is: ", b11.get_shape())
print("The shape of b12 is: ", b12.get_shape())
print("The shape of W11 is: ", W11.get_shape())
print("The shape of W12 is: ", W12.get_shape())

# Calculate cross entropy and define training step
cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = x11, logits = y12))
train_step1 = tf.train.AdamOptimizer(0.1).minimize(cross_entropy1)

# Start tensorflow session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(100):
    for j in range(len(brca)):
        # Convert brca array into numpy array for tensorflow
        inputArray = np.array(brca[j], dtype=float).reshape(1, lenBRCA)
        print("\nInput Array\n", inputArray)
        sess.run(train_step1, feed_dict={x11: inputArray})
        print("\nb1\n", sess.run(b11))
        print("\nb1\n", sess.run(b12))
        print("\nW1\n", sess.run(W11))
        print("\nW1\n", sess.run(W12))

