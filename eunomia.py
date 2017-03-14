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
inputArray = np.array(brca[0], dtype=tf.float32)

# Initialize weight and adjustment vectors
a = tf.placeholder(tf.float32, [None, lenBRCA])
b = tf.Variable(tf.zeros([lenBRCA]))
W = tf.get_variable('W', shape=[lenBRCA, lenBRCA], initializer = tf.contrib.layers.xavier_initializer())

# Print array dimensions
print(a.get_shape())
print(b.get_shape())
print(W.get_shape())



# Start tensorflow session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
autoencoder.printTensor(b)
autoencoder.printTensor(W)
