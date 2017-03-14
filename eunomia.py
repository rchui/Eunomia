import csv
from src.autoencoder import autoencoder

# Read in CSV file
brca = []
count = 0

# brca is  long
with open("brca_toronto_collab_mutect_123_030617.csv") as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        row.pop()
        if count != 0:
            floatRow = [float(i) for i in row]
            brca.append(floatRow)
        count += 1
csvFile.close()

print("Size of BRCA is: " + str(len(brca[0])))

sess = autoencoder.startSession()
brcaTensor = autoencoder.listToTensor(brca[0])
autoencoder.printTensor(brcaTensor)
