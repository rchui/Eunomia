import tensorflow as tf
import numpy as np

class Autoencoder:
    # Starts a session
    def startSession():
        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)
        return sess

    # Converts a list into a tensor
    def listToTensor(inputList):
        return tf.convert_to_tensor(inputList, dtype=tf.float32)

    # Prints a tensor
    def printTensor(inputTensor):
        inputTensor = tf.Print(inputTensor, [inputTensor], message = "Printing Tensor: ")
        scratch = tf.add(inputTensor, inputTensor).eval()

    
