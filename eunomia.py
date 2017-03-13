import csv

# Read in CSV file
csvFile = open('./brca_toronto_collab_mutect_123_030617.csv', 'rb')
reader = csv.reader(csvFile)

brca = []
for row in reader:
    brca.append(row)
csvFile.close()

