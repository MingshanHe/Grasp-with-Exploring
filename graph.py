from calendar import c
import numpy as np

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import CrossEntropyLoss2d
from model import reinforcement_net
from scipy import ndimage

class Node(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.connect = []

class RootNode(Node):

    def __init__(self, x, y, angle, fx, fy):
        Node.__init__(self, x, y)
        self.angle = angle
        self.fx = fx
        self.fy = fy
        self.connect = []

class SubNode(Node):

    def __init__(self, x, y):
        Node.__init__(self, x, y)
        self.connect = []

class Graph():

    def __init__(self):
        self.node = None
        self.init = False

    def addNode(self, position, angle, force):
        node = Node()
        node.x = position[0]
        node.y = position[1]
        node.angle = angle
        node.fx = force[0]
        node.fy = force[1]

        self.generateSubGraph(node)

    def generateSubGraph(self, node):
        subnode = SubNode()
        subnode.x = node.x + (-node.force[0])
        subnode.y = node.y + (-node.force[1])
        node.connect.append(subnode)
        subnode.connect.append(node)
