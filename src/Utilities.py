import csv
import sys
import tensorflow as tf
import numpy as np

class Utilities:
    def readData():
        """ Reads in data passed by the user from a CSV file. """
        count = 0
        fileName = sys.argv[1]
        inputArray = []
        with open(fileName) as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                row.pop()
                if count != 0:
                    floatRow = [float(i) for i in row]
                    inputArray.append(floatRow)
                count += 1
        csvFile.close()
        return inputArray

    def numpyReshape(array, batchSize):
        """ Reshapes a given array to match tensor dimensions.
        @params:
            array -- the array to be reshaped and turned into a numpy array
        """
        return np.array(array, dtype = float).reshape(batchSize, len(array))

    def startSession():
        """ Starts an interactive session to run the tensorflow graph. """
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        return sess

    def progress(count, total, status=''):
        """ Displays a progress bar that updates with the program's progress.
        @params:
            count -- distance traveled so far
            total -- total distance to travel
            status -- updates on the progress of the program
        """
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        
        if count == total: 
            sys.stdout.write('[%s] %s%s ...%s%s\r' % (bar, percents, '%', status, '\n'))
            sys.stdout.flush()
        else:
            sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
            sys.stdout.flush()
