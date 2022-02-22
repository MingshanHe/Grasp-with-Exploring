import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import reinforcement_net
from scipy import ndimage
import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, force_cpu):
        # Check if CUDA can be used
        if torch.cuda.is_available()() and not force_cpu:
            print("CUDA detected. Running with GPU acceleration.")
            self.use_cuda = True
        elif force_cpu:
            print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            self.use_cuda = False
        else:
            print("CUDA is *Not* detected. Running with only CPU.")
            self.use_cuda = False
        # Fully convolutional classification network for supervised learning
        self.model = reinforcement_net(self.use_cuda)
        self.push_rewards = 1#TODO: Add
        self.future_reward_discount = 1#TODO: Add

        # Initialize Huber loss
        self.criterion = torch.nn.SmoothL1Loss(reduce=False) #Huber loss
        if self.use_cuda:
            self.criterion = self.criterion.cuda()
        # Load pre-trained model
        if load_snapshot:
            self.model.load_state_dict(torch.load(snapshot_file)) #TODO:ADD
        # Convert model from CPU to GPU
        if self.use_cuda:
            self.model = self.model.cuda()
        # Set model to training model
        self.model.train()
        # Initialize optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(()), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        self.iteration = 0
        # Initialize lists to save execution info and RL variables
        # TODO: ADD

    def forward(self, predict_heightmap, is_volatile=False, specific_rotation=-1):
        # Apply 2x scale to input heightmaps
        predict_heightmap_2x = ndimage.zoom(predict_heightmap, zoom=[2,2], order=0)
        # Add extra padding (to handle rotations inside network)
        diag_length = float(predict_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length/32) * 32
        padding_width = int((diag_length - predict_heightmap_2x.shape[0])/2)
        predict_heightmap_2x = np.pad(predict_heightmap_2x, padding_width,'constant',constant_values=0)
        # Pre-process predict image (normalize)
        predict_heightmap_2x.shape = (predict_heightmap_2x.shape[0], predict_heightmap_2x.shape[1], 1)
        input_predict_image = np.concatenate((predict_heightmap_2x, predict_heightmap_2x, predict_heightmap_2x), axis=2)
        #TODO: Normalize
        # Construct minibatch of size 1 (b,c,h,w)
        input_predict_image.shape = (input_predict_image.shape[0], input_predict_image.shape[1], 1)
        input_predict_data = torch.from_numpy(input_predict_image.astype(np.float32)).permute(3,2,0,1)

        # Pass input data through model
        output_prob, state_feat = self.model.forward(input_predict_data, is_volatile, specific_rotation)

        # RL: return Q values(and remove extra padding)
        for rotate_idx in range(len(output_prob)):
            if rotate_idx == 0:
                grasp_predictions = output_probp[rotate_idx][0].cpu().data.numpy()[:, 0, int(padding_width/2):int(predict_heightmap_2x.shape[0]/2 - padding_width/2), int(padding_width/2):int(predict_heightmap_2x.shape[0]/2 - padding_width/2)]
            else:
                grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:, 0, int(padding_width/2):int(predict_heightmap_2x.shape[0]/2 - padding_width/2), int(padding_width/2):int(predict_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
        return grasp_predictions, state_feat

    def get_label_value(self, grasp_sucess, change_detected, prev_grasp_predictions, next_predict_heightmap):
        # RL: compute current reward
        current_reward = 0
        if grasp_sucess:
            current_reward = 1.0
        # Compute future reward
        if not change_detected and not grasp_sucess:
            future_reward = 0
        else:
            next_grasp_predictions, next_state_feat = self.forward(next_predict_heightmap, is_volatile=True)
            future_reward = np.max(next_grasp_predictions)
        print('Current reward: %f' % (current_reward))
        print('Future reward: %f' % (future_reward))
        expected_reward = current_reward + self.future_reward_discount * future_reward
        print('Expected reward: %f + %f x %f = %f' % (current_reward, self.future_reward_discount, future_reward, expected_reward))
        return expected_reward, current_reward

    def backprop(self, predict_heightmap, best_pix_ind, label_value):
        # RL: Compute labels
        label = np.zeros((1, 320, 320))     #TODO: NEED FIX
        action_area = np.zeros((244, 244))  #TODO: NEED FIX
        action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
        # Compute label mask
        label_weights = np.zeros(label.shape)
        tmp_label_weights = np.zeros((224,224)) #TODO: NEED FIX
        tmp_label_weights[action_area > 0] = 1
        label_weights[0, 48:(320-48), 48:(320-48)] = tmp_label_weights
        #Compute loss and backward pass
        self.optimizer.zero_grad()
        loss_value = 0
        grasp_predictions, state_feat = self.forward(predict_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])
        if self.use_cuda:
            loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
        else:
            loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
        loss = loss.sum()
        loss.backward()
        loss_value = loss.cpu().data.numpy()
        print('Training loss: %f' % (loss_value))
        self.optimizer.step()

    # def get_prediction_vis(): TODO: CHECK

    def grasp_heuristic(self, predict_heightmap):

        num_rotations = 16

        for rotate_idx in range(num_rotations):
            rotated_heightmap = ndimage.rotate(predict_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            valid_areas = np.zeros(rotated_heightmap.shape)
            valid_areas[np.logical_and(rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0,-25], order=0) > 0.02, rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0,25], order=0)>0.02)] = 1
            blur_kernel = np.ones((25,25), np.float32)/9
            valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
            tmp_grasp_predictions = ndimage.rotate(valid_areas, -rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            tmp_grasp_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])

            if rotated_idx == 0:
                grasp_predictions = tmp_grasp_predictions
            else:
                grasp_predictions = np.concatenate((grasp_predictions, tmp_grasp_predictions), axis=0)
        best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
        return best_pix_ind