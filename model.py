from audioop import bias
from collections import OrderedDict
from turtle import forward
from cv2 import rotate
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import time

class reinforcement_net(nn.Module):
    def __init__(self, use_cuda):
        super(reinforcement_net, self).__init__()
        self.use_cuda = use_cuda

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.grasp_heat_trunk = torchvision.models.densenet.densenet121(pretrained=False)

        self.num_rotations = 16

        # Construct network branches for grasping
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(1024)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(1024, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-norm1', nn.BatchNorm2d(64)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []

    def forward(self, input_heatmap_data, is_volatile=False, specific_rotation=-1):

        if is_volatile:
            with torch.no_grad():
                self.output_prob = []
                self.interm_feat = []

                # Apply rotations to images
                for rotate_idx in range(self.num_rotations):
                    rotate_theta = np.radians(rotate_idx * (360/self.num_rotations))

                    # Compute sample frid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([
                        [np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                        [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2,3,1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()

                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_heatmap_data.size())
                    else:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_heatmap_data.size())

                    # Rotate images clockwise
                    if self.use_cuda:
                        rotate_heat = F.grid_sample(Variable(input_heatmap_data, volatile=True).cuda(), flow_grid_before, mode='nearest')
                    else:
                        rotate_heat = F.grid_sample(Variable(input_heatmap_data, volatile=True), flow_grid_before, mode='nearest')

                    # Compute intermediate features
                    interm_grasp_feat = self.grasp_heat_trunk.features(rotate_heat)
                    self.interm_feat.append(interm_grasp_feat)

                    # Compute sample grid for rotation AFTER branches
                    affine_mat_after = np.asarray([
                        [np.cos(rotate_theta), np.sin(rotate_theta), 0],
                        [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2,3,1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_grasp_feat.data.size())
                    else:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_grasp_feat.data.size())

                    # Forward pass through branches, undo rotation on output predictions, upsample resuls
                    self.output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after, mode='nearest'))])

            return self.output_prob, self.interm_feat

        else:
            self.output_prob = []
            self.interm_feat = []

            # Apply rotations to images
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

            # Compute sample frid for rotation BEFORE neural network
            affine_mat_before = np.asarray([
                [np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2,3,1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()

            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_heatmap_data.size())
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_heatmap_data.size())

            # Rotate images clockwise
            if self.use_cuda:
                rotate_heat = F.grid_sample(Variable(input_heatmap_data, volatile=True).cuda(), flow_grid_before, mode='nearest')
            else:
                rotate_heat = F.grid_sample(Variable(input_heatmap_data, volatile=True), flow_grid_before, mode='nearest')

            # Compute intermediate features
            interm_grasp_feat = self.grasp_heat_trunk.features(rotate_heat)
            self.interm_feat.append(interm_grasp_feat)

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([
                [np.cos(rotate_theta), np.sin(rotate_theta), 0],
                [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2,3,1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_grasp_feat.data.size())
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_grasp_feat.data.size())

            # Forward pass through branches, undo rotation on output predictions, upsample resuls
            self.output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after, mode='nearest'))])

            return self.output_prob, self.interm_feat