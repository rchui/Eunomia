import tensorflow as tf
import numpy as np

class autoencoder:
    def startSession():
        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)
        return sess

    def listToTensor(inputList):
        return tf.convert_to_tensor(inputList, dtype=tf.float32)

    def printTensor(inputTensor):
        inputTensor = tf.Print(inputTensor, [inputTensor], message = "Printing Tensor: ")
        scratch = tf.add(inputTensor, inputTensor).eval()

    
