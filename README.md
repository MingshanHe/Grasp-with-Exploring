1. Robustness: move the original position of object
2. force and torque sensor: 
    force data(3)->grasp position(change) and grasp orientation(only z axis and provide original data)
    torque data(3)->grasp orientation(x y z axis and change)
3. situation:
    1) multipule objects
    2) unknown object

4. problem:
    1) force/torque sensor data error
    2) 

## Requirement
pip3 install math3d

'python3 main.py --use_sim --train_axis 'x' --iterations 100'