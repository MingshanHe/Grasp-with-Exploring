
import numpy as np
class NeuralNetwork():
    def __init__(self, sizes):
        """
        sizes = [2,4,2]
        """

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        self.heatmap = np.zeros([500, 500])

    def upate_heatmap(self, workspace_limits, position, force):
        map_y = int((position[1] - workspace_limits[1][0]) * 500/np.fabs(workspace_limits[1][0]-workspace_limits[1][1]))
        map_x = int((position[0] - workspace_limits[0][0]) * 500/np.fabs(workspace_limits[0][0]-workspace_limits[0][1]))

        for i in range(100):
            for j in range(100):
                i_ = i-50
                j_ = j-50
                if ((force[1]*i_)+(force[0]*j_))>=0:
                    if np.sqrt(i_**2+j_**2) <= 10:
                        self.heatmap[map_x+i_][map_y+j_] = 255
                    elif np.sqrt(i_**2+j_**2) <= 20:
                        self.heatmap[map_x+i_][map_y+j_] = 200
                    elif np.sqrt(i_**2+j_**2) <= 30:
                        self.heatmap[map_x+i_][map_y+j_] = 100
                    elif np.sqrt(i_**2+j_**2) <= 40:
                        self.heatmap[map_x+i_][map_y+j_] = 50
                    elif np.sqrt(i_**2+j_**2) <= 50:
                        self.heatmap[map_x+i_][map_y+j_] = 25
        # for i in range(30):
        #     for j in range(30):
        #         self.heatmap[map_x+i][map_y+j] = 55

        # for i in range(20):
        #     for j in range(20):
        #         self.heatmap[map_x+i][map_y+j] = 155

        # for i in range(10):
        #     for j in range(10):
        #         self.heatmap[map_x+i][map_y+j] = 255
        return self.heatmap

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for input, output in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(input, output)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.biases     = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
            self.weights    = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]

    def update(self, input, output):
        """
        Update: using input and output data
        update the neural network
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        delta_nabla_b, delta_nabla_w = self.backprop(input, output)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # print("Network Biases: ")
        # print(self.biases)
        # print("Network Weights: ")
        # print(self.weights)
        self.biases     = [b-0.1*nb for b, nb in zip(self.biases, nabla_b)]
        self.weights    = [w-0.1*nw for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, input, output):
        """
        update the params in network
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = input
        activations = [input]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], output) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, np.asarray(activations[-2]).T)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def forward(self, data):
        try:
            for b, w in zip(self.biases, self.weights):
                data = self.sigmoid(np.dot(w, data)+b)
            return data
        except:
            print("Layer Error.")

    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives
        """
        return (output_activations - y)

    def evaluate(self, data):
        results = [(np.argmax(self.forward(input)), output) for (input, output) in data]
        return sum(int(nn_output==output) for (nn_output, output) in results)

    def sigmoid(self,val):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-val)) - 0.5

    def sigmoid_prime(self,val):
        """Derivative of the sigmoid function."""
        return self.sigmoid(val)*(1-self.sigmoid(val))