import sys
import tensorflow as tf
import numpy as np

# Class to represent the autoencoder input layer.
class InputLayer:
    def __init__(self, outSize):
        """
        @params:
            outSize - size of the input layer
        """
        self.outSize = outSize
        self.inputLayer = tf.placeholder(tf.float32, [None, outSize])

    def printLayerShape(self):
        """ Prints the dimensions of the layer's tensors. """
        print("\nThe shape of input is: ", self.inputLayer.get_shape())

# Class to represent the autoencoder hidden layers.
class HiddenLayer:
    def __init__(self, outSize, layerInput, shape = None, initializer = None, regularizer = None):
        """
        @params:
            outSize - size of the hidden layer
            layerInput - tensor being fed to the hidden layer
            shape - shape of the layers (default None)
            initalizer - weight initialization function (default None)
            regularizer - l2 loss regularizer (default None)
        """
        self.layerInput = layerInput
        self.outSize = outSize
        num_rows, num_cols = layerInput.get_shape().as_list()
        self.inSize = num_cols

        self.w1 = tf.get_variable('w1', shape = [self.inSize, self.outSize], 
                                  initializer = tf.contrib.layers.xavier_initializer())
        self.b1 = tf.Variable(tf.zeros(self.outSize))
        self.w2 = tf.get_variable('w2', shape = [self.outSize, self.inSize], 
                                  initializer = tf.contrib.layers.xavier_initializer())
        self.b2 = tf.Variable(tf.zeros(self.inSize))
        
        self.z1 = tf.matmul(self.layerInput, self.w1) + self.b1
        self.y1 = tf.nn.sigmoid(self.z1)
        self.z2 = tf.matmul(self.y1, self.w2) + self.b2
        self.y2 = tf.nn.sigmoid(self.z2)

    def buildTrainer(self, alpha, beta, rho):
        """ Trains the hidden layer. 
        @params:
            alpha -- scaling factor for sparsity function
            beta -- scaling factor for l2 loss function
            rho -- sparsity parameter
        """
        self.squareDifference = tf.reduce_sum(tf.square(self.layerInput - self.y2))
        self.l2 = beta * tf.nn.l2_loss(self.w1) + beta * tf.nn.l2_loss(self.w2)

        self.rho = rho
        self.rhoHat = (tf.reduce_sum(self.y1, 0) / self.outSize) + self.rho
        self.sparsity = alpha * tf.reduce_sum(self.rho * tf.log(self.rho / self.rhoHat) +
                                (1 - self.rho) * tf.log((1 - self.rho) / (1 - self.rhoHat)))

        self.loss = self.squareDifference + self.l2 + self.sparsity
        self.trainStep = tf.train.AdamOptimizer().minimize(self.loss)

    def printLayerShape(self):
        """ Prints the dimensions of the layer's tensors. """
        print("\nThe shape of the x is: ", self.layerInput.get_shape())
        print("The shape of w1 is: ", self.w1.get_shape())
        print("The shape of b1 is: ", self.b1.get_shape())
        print("The shape of w2 is: ", self.w2.get_shape())
        print("The shape of b2 is: ", self.b2.get_shape())
        print("The shape of z1 is: ", self.z1.get_shape())
        print("The shape of y1 is: ", self.y1.get_shape())
        print("The shape of z2 is: ", self.z2.get_shape())
        print("The shape of y2 is: ", self.y2.get_shape())

# Class to represent the autoencoder output layer.
class OutputLayer:
    def __init__(self, outSize, layerInput, shape = None, initalizer = None, regularizer = None):
        """
        @params:
            outSize -- size of the output layer
            layerInput -- tensor being fed to the output layer
            shape -- dimensions of the output layer (default None)
            initializer -- weight initalization function (default None)
            regularizer -- l2 loss regularizer (default None)
        """
        self.outSize = outSize
        self.layerInput = layerInput
        num_rows, num_cols = layerInput.get_shape().as_list()
        self.inSize = num_cols

        self.wo = tf.get_variable('wo', shape = [self.inSize, self.outSize],
                              initializer = tf.contrib.layers.xavier_initializer())
        self.bo = tf.Variable(tf.zeros(self.outSize))
        self.zo = tf.matmul(layerInput, self.wo) + self.bo
        self.yo = tf.nn.sigmoid(self.zo)

    def buildTrainer(self):
        """ Trains the hidden layer. """
        self.labelTensor = tf.placeholder(tf.float32, [None, 2])
        self.softmax = tf.nn.softmax_cross_entropy_with_logits(labels = self.labelTensor, 
                                                               logits = self.zo)
        self.loss = tf.reduce_mean(self.softmax)
        self.trainStep = tf.train.AdamOptimizer().minimize(self.loss)

    def printLayerShape(self):
        """ Prints the dimensions of the layer's tensors. """
        print("\nThe shape of x is: ", self.layerInput.get_shape())
        print("The shape of wo is: ", self.wo.get_shape())
        print("The shape of bo is: ", self.bo.get_shape())
        print("The shape of zo is: ", self.zo.get_shape())
        print("The shape of yo is: ", self.yo.get_shape())
