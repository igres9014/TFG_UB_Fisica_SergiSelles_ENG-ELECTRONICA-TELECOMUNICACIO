

import matplotlib.pyplot as plt

"""
with open('coordinatesHVD.txt', 'r') as f:
    for item in coord:  
        print(item[1])
        print(item[2])
        #plt.scatter(item[1], item[2])
        #plt.show()
"""

data = []
with open('coordinatesHVD0F2.txt','r') as f:
    for line in f:
        fl = float(line)
        data.append(fl)

data2 = []
data3 = []
color = []
#print(data[-3])
for i in range(0, len(data), 3):
    color.append(data[i]/data[-3])
    data2.append(data[i+1])
    data3.append(data[i+2])

#print(data2)
plt.scatter(data2, data3, s= 100, c=color)
plt.xlim([-5, 5])
plt.show()
