
from turtle import forward
import numpy as np

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import CrossEntropyLoss2d
from model import reinforcement_net
from scipy import ndimage

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

    def upate_heatmap(self, workspace_limits, position, force, current_angle):
        map_y = int((position[1] - workspace_limits[1][0]) * 500/np.fabs(workspace_limits[1][0]-workspace_limits[1][1]))
        map_x = int((position[0] - workspace_limits[0][0]) * 500/np.fabs(workspace_limits[0][0]-workspace_limits[0][1]))
        tmp_fx = force[0] * np.cos(-current_angle) - force[1] * np.sin(-current_angle)
        tmp_fy = force[0] * np.sin(-current_angle) + force[1] * np.cos(-current_angle)
        force[0] = tmp_fx
        force[1] = tmp_fy
        for i in range(100):
            for j in range(100):
                i_ = i-50
                j_ = j-50
                if ((force[1]*i_)+(force[0]*j_))>=0:
                    if np.sqrt(i_**2+j_**2) <= 10:
                        if self.heatmap[map_x+i_][map_y+j_] < 255:
                            self.heatmap[map_x+i_][map_y+j_] = 255
                    elif np.sqrt(i_**2+j_**2) <= 20:
                        if self.heatmap[map_x+i_][map_y+j_] < 200:
                            self.heatmap[map_x+i_][map_y+j_] = 200
                    elif np.sqrt(i_**2+j_**2) <= 30:
                        if self.heatmap[map_x+i_][map_y+j_] < 100:
                            self.heatmap[map_x+i_][map_y+j_] = 100
                    elif np.sqrt(i_**2+j_**2) <= 40:
                        if self.heatmap[map_x+i_][map_y+j_] < 50:
                            self.heatmap[map_x+i_][map_y+j_] = 50
                    elif np.sqrt(i_**2+j_**2) <= 50:
                        if self.heatmap[map_x+i_][map_y+j_] < 25:
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



class Trainer(object):
    def __init__(self, force_cpu):

        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print("CUDA detected. Running GPU")
            self.use_cuda = True
        elif force_cpu:
            print("CUDA detected. Running CPU")
            self.use_cuda = False
        else:
            print("CUDA is *NOT* detected. Runing CPU")
            self.use_cuda = False

        self.heatmap = np.zeros([500, 500])

        # Fully convolutional Q network for deep reinforcement learning
        self.model = reinforcement_net(self.use_cuda)
        self.push_rewareds = None
        self.future_reward_discount = None

        # Initialize Huber loss
        self.criterion = torch.nn.SmoothL1Loss(reduce=False)
        if self.use_cuda:
            self.criterion = self.criterion.cuda()

        # Convert model from CPU to GPU
        if self.use_cuda:
            self.model = self.model.cuda()

        # Set model to training mode
        self.model.train()

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        self.iteration = 0

        # Initialize lists to save execution info and RL variables
        self.label_value_log = []
        self.reward_value_log = []
        self.predict_value_log = []

    # Compute forward pass through model to compute affordances/Q
    def forward(self, heat_map, is_volatile=False, specific_rotation=-1):

        # Apply 2x scale to input heightmaps
        heatmap_2x = ndimage.zoom(heat_map, zoom=[2,2], order=0)

        # Add exra padding (to handle rotations insede network)
        diag_length = float(heatmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length/32)*32
        padding_width = int((diag_length - heat_map.shape[0])/2)
        heatmap_2x = np.pad(heatmap_2x, padding_width, 'constant', constant_values=0)

        #? Pre-process heatmap image
        heatmap_2x.shape = (heatmap_2x.shape[0], heatmap_2x.shape[1], 1)
        input_heatmap_image = np.concatenate((heatmap_2x, heatmap_2x, heatmap_2x), axis=2)

        # Construct minibatch of size 1 (b, c, h, w)
        input_heatmap_image.shape = (input_heatmap_image.shape[0], input_heatmap_image.shapep[1], input_heatmap_image.shape[2], 1)
        input_heatmap_data = torch.from_numpy(input_heatmap_image.astype(np.float32)).permute(3,2,0,1)

        # Pass input data through model
        output_prob, state_feat = self.model.forward(input_heatmap_data, is_volatile, specific_rotation)

        # Return Q values (and remove extra padding)
        for rotate_idx in range(len(output_prob)):
            if rotate_idx == 0:
                grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(heatmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(heatmap_2x.shape[0]/2 - padding_width/2)]
            else:
                grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(heatmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(heatmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

        return grasp_predictions, state_feat

    # Compute labels and backpropagete
    def backprop(self, heatmap, best_pix_ind, label_value):

        # Compute labels
        label = np.zeros((1,320,320))
        action_area = np.zeros((224, 224))
        action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
        tmp_label = np.zeros((224, 224))
        tmp_label[action_area > 0] = label_value
        label[0, 48:(320-48), 48:(320-48)] = tmp_label

        # Compute label mask
        label_weights = np.zeros(label.shape)
        tmp_label_weights = np.zeros((224,224))
        tmp_label_weights[action_area > 0] = 1
        label_weights[0, 48:(320-48), 48:(320-48)] = tmp_label_weights

        # Compute loss and backward pass
        self.optimizer.zero_grad()
        loss_value = 0

        # Do forward pass with specified rotation (to save gradients)
        grasp_predictions, state_feat = self.forward(heatmap, is_volatile=False, specific_rotation=best_pix_ind[0])

        if self.use_cuda:
            loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
        else:
            loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)
        loss = loss.sum()
        loss.backward()
        loss_value = loss.cpu().data.numpy()

        opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations/2) % self.model.num_rotations

        grasp_predictions, state_feat = self.forward(heatmap, is_volatile=False, specific_rotation=opposite_rotate_idx)

        if self.use_cuda:
            loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
        else:
            loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)

        loss = loss.sum()
        loss.backward()
        loss_value = loss.cpu().data.numpy()

        loss_value = loss_value/2

        print('Training loss: %f' % (loss_value))
        self.optimizer.step()

    def get_label_value(self, grasp_success, change_detected,):

        # Compute current reward
        current_reward = 0

        if grasp_success:
            current_reward = 1.0

        # Compute future reward
        if not change_detected and not grasp_success:
            future_reward = 0
        else:
            next_grasp_predictions, next_state_feat = self.forward(next_heatmap, is_volatile=True)
            future_reward =  np.max(next_grasp_predictions)

            # # Experiment: use Q differences
            # push_predictions_difference = next_push_predictions - prev_push_predictions
            # grasp_predictions_difference = next_grasp_predictions - prev_grasp_predictions
            # future_reward = max(np.max(push_predictions_difference), np.max(grasp_predictions_difference))

        print('Current reward: %f' % (current_reward))
        print('Future reward: %f' % (future_reward))

        expected_reward = current_reward + self.future_reward_discount * future_reward
        print('Expected reward: %f + %f x %f = %f' % (current_reward, self.future_reward_discount, future_reward, expected_reward))

        return expected_reward, current_reward


    def grasp_heuristic(self, heatmap):

        num_rotations = 16

        for rotate_idx in range(num_rotations):
            rotated_heightmap = ndimage.rotate(heatmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            valid_areas = np.zeros(rotated_heightmap.shape)
            valid_areas[np.logical_and(rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0,-25], order=0) > 0.02, rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0,25], order=0) > 0.02)] = 1
            # valid_areas = np.multiply(valid_areas, rotated_heightmap)
            blur_kernel = np.ones((25,25),np.float32)/9
            valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
            tmp_grasp_predictions = ndimage.rotate(valid_areas, -rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            tmp_grasp_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])

            if rotate_idx == 0:
                grasp_predictions = tmp_grasp_predictions
            else:
                grasp_predictions = np.concatenate((grasp_predictions, tmp_grasp_predictions), axis=0)

        best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
        return best_pix_ind

    def upate_heatmap(self, workspace_limits, position, force, current_angle):
        map_y = int((position[1] - workspace_limits[1][0]) * 500/np.fabs(workspace_limits[1][0]-workspace_limits[1][1]))
        map_x = int((position[0] - workspace_limits[0][0]) * 500/np.fabs(workspace_limits[0][0]-workspace_limits[0][1]))
        tmp_fx = force[0] * np.cos(-current_angle) - force[1] * np.sin(-current_angle)
        tmp_fy = force[0] * np.sin(-current_angle) + force[1] * np.cos(-current_angle)
        force[0] = tmp_fx
        force[1] = tmp_fy
        for i in range(100):
            for j in range(100):
                i_ = i-50
                j_ = j-50
                if ((force[1]*i_)+(force[0]*j_))>=0:
                    if np.sqrt(i_**2+j_**2) <= 10:
                        if self.heatmap[map_x+i_][map_y+j_] < 255:
                            self.heatmap[map_x+i_][map_y+j_] = 255
                    elif np.sqrt(i_**2+j_**2) <= 20:
                        if self.heatmap[map_x+i_][map_y+j_] < 200:
                            self.heatmap[map_x+i_][map_y+j_] = 200
                    elif np.sqrt(i_**2+j_**2) <= 30:
                        if self.heatmap[map_x+i_][map_y+j_] < 100:
                            self.heatmap[map_x+i_][map_y+j_] = 100
                    elif np.sqrt(i_**2+j_**2) <= 40:
                        if self.heatmap[map_x+i_][map_y+j_] < 50:
                            self.heatmap[map_x+i_][map_y+j_] = 50
                    elif np.sqrt(i_**2+j_**2) <= 50:
                        if self.heatmap[map_x+i_][map_y+j_] < 25:
                            self.heatmap[map_x+i_][map_y+j_] = 25

        return self.heatmap
