import csv
import tensorflow as tf
import numpy as np
from src.autoencoder import autoencoder

# Regularization factor
beta = 0.01

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

z11 = tf.matmul(x11, W11) + b11
y11 = tf.nn.relu(z11)
z12 = tf.matmul(y11, W12) + b12
y12 = tf.nn.relu(z12)

# Print array dimensions
print("The shape of x11 is: ", x11.get_shape())
print("The shape of b11 is: ", b11.get_shape())
print("The shape of b12 is: ", b12.get_shape())
print("The shape of W11 is: ", W11.get_shape())
print("The shape of W12 is: ", W12.get_shape())

W21 = tf.get_variable('W21', shape=[halfBRCA, 8], initializer = tf.contrib.layers.xavier_initializer())
W22 = tf.get_variable('W22', shape=[8, halfBRCA], initializer = tf.contrib.layers.xavier_initializer())
b21 = tf.Variable(tf.zeros([8]))
b22 = tf.Variable(tf.zeros([halfBRCA]))

z21 = tf.matmul(y11, W21) + b21
y21 = tf.nn.relu(z21)
z22 = tf.matmul(y21, W22) + b22
y22 = tf.nn.relu(z22)

# Print array dimensions
print("The shape of b21 is: ", b21.get_shape())
print("The shape of b22 is: ", b22.get_shape())
print("The shape of W21 is: ", W21.get_shape())
print("The shape of W22 is: ", W22.get_shape())

# Calculate square difference
square_difference1 = tf.reduce_sum(tf.square(x11 - y12))
square_difference2 = tf.reduce_sum(tf.square(y11 - y22))

# Regularization
reg1 = tf.nn.l2_loss(W11) + tf.nn.l2_loss(W12)
loss1 = square_difference1 + beta * reg1

reg2 = tf.nn.l2_loss(W21) + tf.nn.l2_loss(W22)
loss2 = square_difference2 + beta * reg2

# Optimization
train_step1 = tf.train.AdamOptimizer().minimize(loss1)
train_step2 = tf.train.AdamOptimizer().minimize(loss2)

# Start tensorflow session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train autoencoder
for i in range(100):
    for j in range(len(brca)):
        # Convert brca array into numpy array for tensorflow
        inputArray = np.array(brca[j], dtype=float).reshape(1, lenBRCA)
        # print("\nInput Array\n", inputArray)
        sess.run(train_step1, feed_dict={x11: inputArray})
# print("\nW11\n", sess.run(W11))
# print("\nb11\n", sess.run(b11))
# print("\nW12\n", sess.run(W12))
# print("\nb12\n", sess.run(b12))

for i in range(100):
    for j in range(len(brca)):
        inputArray = np.array(brca[j], dtype=float).reshape(1, lenBRCA)
        print("\nInput Array\n", inputArray)
        sess.run(train_step2, feed_dict={x11: inputArray})
print("\nW11\n", sess.run(W21))
print("\nb11\n", sess.run(b21))
print("\nW12\n", sess.run(W22))
print("\nb12\n", sess.run(b22))

# Calculate difference between input and ouput
accuracy1 = tf.reduce_sum(tf.square(x11 - y12))
accuracy2 = tf.reduce_sum(tf.square(y11 - y22))

# Print difference
inputArray = np.array(brca[0], dtype=float).reshape(1, lenBRCA)
print("Accuracy for layer 1: ", sess.run(accuracy1, feed_dict={x11: inputArray}))
print("Accuracy for layer 2: ", sess.run(accuracy2, feed_dict={x11: inputArray}))
