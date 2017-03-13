import csv
from src.autoencoder import autoencoder

# Read in CSV file
csvFile = open('./brca_toronto_collab_mutect_123_030617.csv', 'rb')
reader = csv.reader(csvFile)

brca = []
for row in reader:
    brca.append(row)
    brca.pop()
csvFile.close()

x = autoencoder("Hello World")
x.printHolder()
