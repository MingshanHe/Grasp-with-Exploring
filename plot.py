import matplotlib.pyplot as plt

filename = r'/home/robot/explore_grasp_ws/logs/2022-03-12@15-36-58/data/force-sensor-data/foce_data.csv'
force,torque,x = [],[],[]
force_x = []
force_y = []
force_z = []
i = 0
with open(filename, 'r') as f:#1
    lines = f.readlines()#2
    for line in lines:#3
        force.append([line[0], line[1]])#5
        force_x.append(line[0])
        force_y.append(line[1])
        # force_z.append(value[2])
        torque.append([line[3], line[4], line[5]])
        x.append(i)
        i+=1

plt.plot(x, force_x)
plt.show()