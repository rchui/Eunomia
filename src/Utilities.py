import csv
import sys
import tensorflow as tf
import numpy as np

class Utilities:
    def readData():
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

    def numpyReshape(array):
        return np.array(array, dtype = float).reshape(1, len(array))

    def startSession():
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        return sess

    def progress(count, total, status=''):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        
        if count == total: 
            sys.stdout.write('[%s] %s%s ...%s%s' % (bar, percents, '%', status, '\n'))
            sys.stdout.flush()
        else:
            sys.stdout.write('[%s] %s%s ...%s%s' % (bar, percents, '%', status, '\n'))
            sys.stdout.flush()
