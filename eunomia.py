import csv
import tensorflow as tf
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

# Initialize weight and adjustment vectors
b = tf.Variable(tf.zeros([192]))
W = tf.get_variable('W', shape=[192, 192], initializer = tf.contrib.layers.xavier_initializer())

# Start tensorflow session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
autoencoder.printTensor(W)
