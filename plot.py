import matplotlib.pyplot as plt

filename = r'E:\Code\exploring-pushing-grasping\logs\2022-02-11@20-29-06\data\force-sensor-data\1.force_sensor_data.txt'
force,torque,x = [],[],[]
force_x = []
force_y = []
force_z = []
i = 0
with open(filename, 'r') as f:#1
    lines = f.readlines()#2
    for line in lines:#3
        value = [float(s) for s in line.split()]#4
        force.append([value[0], value[1]])#5
        force_x.append(value[0])
        force_y.append(value[1])
        # force_z.append(value[2])
        torque.append([value[3], value[4], value[5]])
        x.append(i)
        i+=1

plt.plot(x, force)
plt.show()