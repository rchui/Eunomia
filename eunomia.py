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
x1 = tf.placeholder(tf.float32, [None,lenBRCA])
b1 = tf.Variable(tf.zeros([lenBRCA]))
W1 = tf.get_variable('W', shape=[lenBRCA, lenBRCA], initializer = tf.contrib.layers.xavier_initializer())
y1 = tf.matmul(x1, W1) + b1

# Print array dimensions
print("The shape of x1 is: ", x1.get_shape())
print("The shape of b1 is: ", b1.get_shape())
print("The shape of W1 is: ", W1.get_shape())

# Calculate cross entropy and define training step
cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = x1, logits = y1))
train_step1 = tf.train.AdamOptimizer(0.5).minimize(cross_entropy1)

# Start tensorflow session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(len(brca)):
    # Convert brca array into numpy array for tensorflow
    inputArray = np.array(brca[i], dtype=float).reshape(1, lenBRCA)
    # print("\nInput Array\n", inputArray)
    sess.run(train_step1, feed_dict={x1: inputArray})
    # print("\nb1\n", sess.run(b1))
    # print("\nW1\n", sess.run(W1))

correct_prediction = tf.equal(tf.argmax(x1, 1), tf.argmax(y1, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
inputArray = np.array(brca[0], dtype=float).reshape(1, lenBRCA)
print(sess.run(accuracy, feed_dict={x1: inputArray}))
