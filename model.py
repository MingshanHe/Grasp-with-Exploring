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
import pandas as pd

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        # self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.global_alpha = 0.1
        self.custom_beta  = 2.5
        self.q_table = {}

    def choose_action(self, map_pos, explore_complete ,resolutions):
        state = str(map_pos[0])+','+str(map_pos[1])
        self.check_state_exist(state)

        global_actions = [0, 0, 0, 0] # x+ | x- | y+ | y-
        custom_actions = [0, 0, 0, 0]
        actions = [0, 0, 0, 0]
        # action selection
        if (map_pos[0]>=0 and map_pos[0] < int(resolutions/3)) and(map_pos[1]>=0 and map_pos[1] < int(resolutions/3)):
            global_actions[0] = (1-explore_complete[3])+(0.5-explore_complete[6])+(0.5-explore_complete[7])
            global_actions[2] = (1-explore_complete[1])+(0.5-explore_complete[2])+(0.5-explore_complete[5])
            for i in range(4):
                custom_actions[i] = np.exp(-self.q_table[state][i])
                actions[i] = global_actions[i] * self.global_alpha + custom_actions[i] * self.custom_beta + np.random.normal() * self.global_alpha


            global_actions = [0, 0, 0, 0]

        elif (map_pos[0]>=0 and map_pos[0] < int(resolutions/3)) and(map_pos[1]>=int(resolutions/3) and map_pos[1] < int(resolutions*2/3)):
            global_actions[0] = (1-explore_complete[4])+(0.25-explore_complete[6])+(0.5-explore_complete[7])+(0.25-explore_complete[8])
            global_actions[2] = (1 - explore_complete[2])
            global_actions[3] = (1 - explore_complete[0])
            for i in range(4):
                custom_actions[i] = np.exp(-self.q_table[state][i])
                actions[i] = global_actions[i] * self.global_alpha + custom_actions[i] * self.custom_beta + np.random.normal() * self.global_alpha


            global_actions = [0, 0, 0, 0]

        elif (map_pos[0]>=0 and map_pos[0] < int(resolutions/3)) and(map_pos[1]>=int(resolutions*2/3) and map_pos[1] < int(resolutions)):
            global_actions[0] = (1-explore_complete[5])+(0.25-explore_complete[7])+(0.5-explore_complete[8])
            global_actions[3] = (0.5-explore_complete[0])+(1-explore_complete[1])+(0.25-explore_complete[3])
            for i in range(4):
                custom_actions[i] = np.exp(-self.q_table[state][i])
                actions[i] = global_actions[i] * self.global_alpha + custom_actions[i] * self.custom_beta + np.random.normal() * self.global_alpha


            global_actions = [0, 0, 0, 0]

        elif (map_pos[0]>=int(resolutions/3) and map_pos[0] < int(resolutions*2/3)) and(map_pos[1]>=0 and map_pos[1] < int(resolutions/3)):
            global_actions[0] = (1 - explore_complete[6])
            global_actions[1] = (1 - explore_complete[0])
            global_actions[2] = (0.25-explore_complete[2])+(1-explore_complete[4])+(0.5-explore_complete[5])+(0.25-explore_complete[8])
            for i in range(4):
                custom_actions[i] = np.exp(-self.q_table[state][i])
                actions[i] = global_actions[i] * self.global_alpha + custom_actions[i] * self.custom_beta + np.random.normal() * self.global_alpha


            global_actions = [0, 0, 0, 0]

        elif (map_pos[0]>=int(resolutions/3) and map_pos[0] < int(resolutions*2/3)) and(map_pos[1]>=int(resolutions/3) and map_pos[1] < int(resolutions*2/3)):
            global_actions[0] = (1 - explore_complete[7])
            global_actions[1] = (1 - explore_complete[1])
            global_actions[2] = (1 - explore_complete[5])
            global_actions[3] = (1 - explore_complete[3])
            for i in range(4):
                custom_actions[i] = np.exp(-self.q_table[state][i])
                actions[i] = global_actions[i] * self.global_alpha + custom_actions[i] * self.custom_beta + np.random.normal() * self.global_alpha


            global_actions = [0, 0, 0, 0]

        elif (map_pos[0]>=int(resolutions/3) and map_pos[0] < int(resolutions*2/3)) and(map_pos[1]>=int(resolutions*2/3) and map_pos[1] < int(resolutions)):
            global_actions[0] = (1 - explore_complete[8])
            global_actions[1] = (1 - explore_complete[2])
            global_actions[3] = (0.25-explore_complete[0])+(0.5-explore_complete[3])+(1-explore_complete[4])+(0.25-explore_complete[6])
            for i in range(4):
                custom_actions[i] = np.exp(-self.q_table[state][i])
                actions[i] = global_actions[i] * self.global_alpha + custom_actions[i] * self.custom_beta + np.random.normal() * self.global_alpha


            global_actions = [0, 0, 0, 0]

        elif (map_pos[0]>=int(resolutions*2/3) and map_pos[0] < int(resolutions)) and(map_pos[1]>=0 and map_pos[1] < int(resolutions/3)):
            global_actions[1] = (0.5-explore_complete[0])+(0.25-explore_complete[1])+(1-explore_complete[3])
            global_actions[2] = (0.25-explore_complete[5])+(1-explore_complete[7])+(0.5-explore_complete[8])
            for i in range(4):
                custom_actions[i] = np.exp(-self.q_table[state][i])
                actions[i] = global_actions[i] * self.global_alpha + custom_actions[i] * self.custom_beta + np.random.normal() * self.global_alpha


            global_actions = [0, 0, 0, 0]

        elif (map_pos[0]>=int(resolutions*2/3) and map_pos[0] < int(resolutions)) and(map_pos[1]>=int(resolutions/3) and map_pos[1] < int(resolutions*2/3)):
            global_actions[1] = (0.25-explore_complete[0])+(0.5-explore_complete[1])+(0.25-explore_complete[2])+(1-explore_complete[4])
            global_actions[2] = (1 - explore_complete[8])
            global_actions[3] = (1 - explore_complete[6])
            for i in range(4):
                custom_actions[i] = np.exp(-self.q_table[state][i])
                actions[i] = global_actions[i] * self.global_alpha + custom_actions[i] * self.custom_beta + np.random.normal() * self.global_alpha


            global_actions = [0, 0, 0, 0]

        elif (map_pos[0]>=int(resolutions*2/3) and map_pos[0] < int(resolutions)) and(map_pos[1]>=int(resolutions*2/3) and map_pos[1] < int(resolutions)):
            global_actions[1] = (0.25-explore_complete[1])+(0.5-explore_complete[2])+(1-explore_complete[5])
            global_actions[3] = (0.25-explore_complete[3])+(0.5-explore_complete[6])+(1-explore_complete[7])
            for i in range(4):
                # custom_actions[i] = np.random.uniform()
                custom_actions[i] = np.exp(-self.q_table[state][i])
                actions[i] = global_actions[i] * self.global_alpha + custom_actions[i] * self.custom_beta + np.random.normal() * self.global_alpha


            global_actions = [0, 0, 0, 0]
        action = actions.index(max(actions))

        # self.q_table[state][action] += 1
        print(state, "action: ", self.q_table[state])

        return action

    def learn(self, s, a, r):
        state = str(s[0])+','+str(s[1])
        self.check_state_exist(state)

        self.q_table[state][a] += r

        print(state, 'action: ',self.q_table[state])

        # if s_ != 'terminal':

        #     q_target = r + self.gamma * self.q_table.loc[s_, :].max()

        # else:

        #     q_target = r # next state is terminal

        # self.q_table.loc[s,a] += self.lr * (q_target - q_predict) # update

    def check_state_exist(self, state):

        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0, 0]