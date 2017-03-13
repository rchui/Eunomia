import csv
from src.autoencoder import autoencoder

# Read in CSV file
csvFile = open('brca_toronto_collab_mutect_123_030617.csv', 'rb')
reader = csv.reader(csvFile)
brca = []
count = 0

for row in reader:
    if count == 0:
        brca.append(row)
        brca[count].pop()
        count += 1
csvFile.close()


