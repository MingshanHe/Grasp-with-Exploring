
import numpy as np
class NeuralNetwork():
    def __init__(self):
        self.force = np.zeros((1,2))
        self.w_x = np.zeros((2,1))
        self.w_y = np.zeros((2,1))
        self.b = np.zeros((2,1))

        self.w_x[0][0] = 0.0101
        self.w_x[1][0] = 0.001
        self.w_y[0][0] = 0.001
        self.w_y[1][0] = 0.0101

        self.b[0][0] = -0.0101
        self.b[1][0] = -0.0101

    def forward(self, force):
        self.force[0][0] = force[0]
        self.force[0][1] = force[1]

        pos_x = np.dot(self.force, self.w_x) + self.b[0]
        pos_y = np.dot(self.force, self.w_y) + self.b[1]

        return (pos_x, pos_y)
