import csv
import random
import sys
import tensorflow as tf
import numpy as np
from src.autoencoder import autoencoder

# Read in CSV file
brca = []
labelInput = []
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

for i in range(5):
    labelInput.append([1.0, 0.0])
for i in range(12):
    labelInput.append([0.0, 1.0])
for i in range(11):
    labelInput.append([1.0, 0.0])
for i in range(95):
    labelInput.append([0.0, 1.0])

# Get the length of brca
lenBRCA = len(brca[0])
halfBRCA = lenBRCA / 2

# Initialize weight and adjustment layer 1 vectors
x11 = tf.placeholder(tf.float32, [None,lenBRCA])
b11 = tf.Variable(tf.zeros([halfBRCA]))
b12 = tf.Variable(tf.zeros([lenBRCA]))
W11 = tf.get_variable('W11', shape=[lenBRCA, halfBRCA], initializer = tf.contrib.layers.xavier_initializer(), regularizer = tf.contrib.layers.l2_regularizer(0.01))
W12 = tf.get_variable('W12', shape=[halfBRCA, lenBRCA], initializer = tf.contrib.layers.xavier_initializer(), regularizer = tf.contrib.layers.l2_regularizer(0.01))

z11 = tf.matmul(x11, W11) + b11
y11 = tf.nn.relu(z11)
z12 = tf.matmul(y11, W12) + b12

# Print array dimensions
print("\nThe shape of x11 is: ", x11.get_shape())
print("The shape of b11 is: ", b11.get_shape())
print("The shape of b12 is: ", b12.get_shape())
print("The shape of W11 is: ", W11.get_shape())
print("The shape of W12 is: ", W12.get_shape())
print("The shape of y11 is: ", y11.get_shape())

# Initialize weight and adjustment layer 2 vector
W21 = tf.get_variable('W21', shape=[halfBRCA, 16], initializer = tf.contrib.layers.xavier_initializer(), regularizer = tf.contrib.layers.l2_regularizer(0.01))
W22 = tf.get_variable('W22', shape=[16, halfBRCA], initializer = tf.contrib.layers.xavier_initializer(), regularizer = tf.contrib.layers.l2_regularizer(0.01))
b21 = tf.Variable(tf.zeros([16]))
b22 = tf.Variable(tf.zeros([halfBRCA]))

z21 = tf.matmul(y11, W21) + b21
y21 = tf.nn.relu(z21)
z22 = tf.matmul(y21, W22) + b22

# Print array dimensions
print("\nThe shape of b21 is: ", b21.get_shape())
print("The shape of b22 is: ", b22.get_shape())
print("The shape of W21 is: ", W21.get_shape())
print("The shape of W22 is: ", W22.get_shape())
print("The shape of y21 is: ", y21.get_shape())

#Output Layer
Wo = tf.get_variable('Wo', shape=[16, 2], initializer = tf.contrib.layers.xavier_initializer(), regularizer = tf.contrib.layers.l2_regularizer(0.01))
bo = tf.Variable(tf.zeros([2]))
zo = tf.matmul(y21, Wo) + bo
yo = tf.nn.softmax(zo)
labelTensor = tf.placeholder(tf.float32, [None, 2]);

print("\nThe shape of bo is: ", bo.get_shape())
print("The shape of Wo is: ", Wo.get_shape())
print("The shape of yo is: ", yo.get_shape())

# Calculate square difference
square_difference1 = tf.reduce_sum(tf.square(x11 - z12))
# square_difference1 = tf.reduce_sum(tf.square(x11 - y12))
square_difference2 = tf.reduce_sum(tf.square(y11 - z22))
# square_difference2 = tf.reduce_sum(tf.square(y11 - y22))

# Output layer loss
loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labelTensor, logits=zo))

# Optimization
train_step1 = tf.train.AdamOptimizer().minimize(square_difference1)
train_step2 = tf.train.AdamOptimizer().minimize(square_difference2)
train_step3 = tf.train.AdamOptimizer().minimize(loss3)

# Start tensorflow session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train autoencoder layer 1
for j in range(1):
    for i in range(len(brca)):
        # Convert brca array into numpy array for tensorflow
        inputArray = np.array(brca[i], dtype=float).reshape(1, lenBRCA)
        # print("\nInput Array\n", inputArray)
sess.run(train_step1, feed_dict={x11: inputArray})

print("\nW11\n", sess.run(W11))
print("\nb11\n", sess.run(b11))
# print("\nW12\n", sess.run(W12))
# print("\nb12\n", sess.run(b12))

# Train autoencoder layer 2
for j in range(1):
    for i in range(len(brca)):
        inputArray = np.array(brca[i], dtype=float).reshape(1, lenBRCA)
        # print("\nInput Array\n", inputArray)
        sess.run(train_step2, feed_dict={x11: inputArray})

print("\nW21\n", sess.run(W21))
print("\nb21\n", sess.run(b21))
# print("\nW12\n", sess.run(W22))
# print("\nb12\n", sess.run(b22))

# Train autoencoder output layer
for j in range(1000):
    for i in range(len(brca)):
        inputArray = np.array(brca[i], dtype = float).reshape(1, lenBRCA)
        # print(labelInput[i])
        labelArray = np.array(labelInput[i], dtype = float).reshape(1, 2)
        sess.run(train_step3, feed_dict={x11: inputArray, labelTensor: labelArray})

print("\nWo\n", sess.run(Wo))
print("\nbo\n", sess.run(bo))

# Print output of each layer
for i in range(len(brca)):
    inputArray = np.array(brca[i], dtype = float).reshape(1, lenBRCA)
    # print("\n")
    # print("y11 tensor, sample ", i, ": \n", sess.run(y11, feed_dict={x11: inputArray}))
    # print("y21 tensor, sample ", i, ": \n", sess.run(y21, feed_dict={x11: inputArray}))
    # print("yo tensor, sample ", i, ": \n", sess.run(yo, feed_dict={x11: inputArray}))

outputList = []
for i in range(len(brca)):
    inputArray = np.array(brca[i], dtype = float).reshape(1, lenBRCA)
    outputList.append(sess.run(yo, feed_dict={x11: inputArray}))

num1 = 0
num2 = 0
iterations = 0
for i in outputList:
    if i[0][0] > i[0][1]:
        num1 += 1
    else:
        num2 += 1
    iterations += 1

print("Number of 1: ", num1)
print("Number of 2: ", num2)

# Calculate difference between input and ouput
# accuracy1 = tf.reduce_sum(tf.square(x11 - y12))
accuracy1 = tf.reduce_sum(tf.square(x11 - z12))
# accuracy2 = tf.reduce_sum(tf.square(y11 - y22))
accuracy2 = tf.reduce_sum(tf.square(y11 - z22))

# Print difference
inputArray = np.array(brca[0], dtype=float).reshape(1, lenBRCA)
print("Squared difference for layer 1: ", sess.run(accuracy1, feed_dict={x11: inputArray}))
print("Squared difference for layer 2: ", sess.run(accuracy2, feed_dict={x11: inputArray}))

