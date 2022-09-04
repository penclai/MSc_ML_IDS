import itertools
import csv

entries = []
entries2 = []
with open('./data/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv', 'r') as f:
    mycsv = csv.reader(f)
    i=0
    for row in itertools.islice(mycsv, 300002):
        if i == 0:
            entries.append(row)
            entries2.append(row)
            i = i+1
        elif i> 100001 and i <= 200001:
            entries.append(row)
            i = i+1
        elif i> 200001:
            entries2.append(row)
        else:
            i=i+1

with open('./data/trim/test1.csv', 'w', newline='') as f2:
    writer = csv.writer(f2)
    writer.writerows(entries)

with open('./data/trim/test2.csv', 'w', newline='') as f3:
    writer = csv.writer(f3)
    writer.writerows(entries2)


print(entries)