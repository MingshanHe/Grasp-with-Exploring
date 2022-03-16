
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

    def SGD(self, training_data, mini_batch_size, eta, test_data=None, epochs=1):
        self.forward(training_data)
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            # np.random.shuffle(training_data)
            # mini_batches = [
            #     training_data[k:k+mini_batch_size]
            #     for k in range(0, n, mini_batch_size)
            # ]
            for minibatch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1}/{2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

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