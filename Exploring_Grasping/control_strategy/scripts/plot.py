#!/usr/bin/python3
import csv
import matplotlib.pyplot as plt
force_x = []
force_y = []
x = []

filename = '/home/robot/explore_pushing_and_grasping/src/Exploring_Grasping/control_strategy/logs/2022-02-15--16:04:16/wrench/data.csv'
with open(filename) as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        force_x.append(row[0])
        force_y.append(row[1])
        x.append(i)
        i+=1
plt.plot(x,force_x)
plt.show()

