# Model and Algorithhms

#! 1: Dyn-Q Algorithm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq
from copy import copy, deepcopy
import os


class Environment:
    def __init__(self, WIDTH=10, HEIGHT=10):
        # maze width and height
        self.WORLD_WIDTH = WIDTH
        self.WORLD_HEIGHT = HEIGHT

        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]

        # start state and goal state
        self.START_STATE = [2, 0]
        self.GOAL_STATES = [[0, 9]]

        # all obstacles
        # self.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
        self.obstacles = None
        self.old_obstacles = None
        self.new_obstacles = None

        # time to change obstacles
        self.obstacle_switch_time = None

        # the size of q value
        self.q_size = (self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions))

        # max steps
        self.max_steps = float('inf')

        # track the resolution for this environment
        self.resolution = 1

    def step(self, state, action, ends):
        '''
        take the action in the satte
        return the new state and reward
        '''
        x, y = state
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)

        # if [x, y] in self.obstacles:
        #     x, y = state

        if [x, y] in ends:
            reward = 1.0
        else:
            reward = 0.0
        return [x, y], reward

class DynaParams:
    '''
    DynaParams: The Class for parameters of Dyna Algorithms
    '''
    def __init__(self):
        # discount
        self.gamma = 0.95

        # probability for exploration
        self.epsilon = 0.1

        # step size
        self.alpha = 0.1

        # weight for elapsed time
        self.time_weight = 0

        # n-step planning
        self.planning_steps = 5

        # average over several independent runs
        self.runs = 10

        # algorithm names
        self.methods = ['Dyna-Q', 'Dyna-Q+']

        # threshold for priority queue
        self.theta = 0

class TrivialModel:
    '''
    TrivialModel: Trivial model for planning in Dyna-Q
    '''
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, rand=np.random):
        self.model = dict()
        self.rand = rand

    def feed(self, state, action, next_state, reward):
        '''
        Feed the model with previous experience
        '''
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()
        self.model[tuple(state)][action] = [list(next_state), reward]

    def sample(self):
        '''
        Randomly sample from previous experience
        '''
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return list(state), action, list(next_state), reward

# choose an action based on epsilon-greedy algorithm
def choose_action(state, q_value, maze, dyna_params):
    if np.random.binomial(1, dyna_params.epsilon) == 1:
        return np.random.choice(maze.actions)
    else:
        values = q_value[state[0], state[1], :]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])

# play for an episode for Dyna-Q algorithm
# @q_value: state action pair values, will be updated
# @model: model instance for planning
# @environment: a instance containing all information about the environment
# @dyna_params: several params for the algorithm
def dyna_q(q_value, model, environment, dyna_params):
    state = environment.START_STATE
    ends = copy(environment.GOAL_STATES)
    steps = 0
    while ends:

        # track the steps
        steps += 1

        # get action
        action = choose_action(state, q_value, environment, dyna_params)

        # take action
        next_state, reward = environment.step(state, action, ends)

        # Q-Learning update
        q_value[state[0], state[1], action] += \
            dyna_params.alpha * (reward + dyna_params.gamma * np.max(q_value[next_state[0], next_state[1], :]) -
                                 q_value[state[0], state[1], action])

        # feed the model with experience
        model.feed(state, action, next_state, reward)

        # sample experience from the model
        for t in range(0, dyna_params.planning_steps):
            state_, action_, next_state_, reward_ = model.sample()
            q_value[state_[0], state_[1], action_] += \
                dyna_params.alpha * (reward_ + dyna_params.gamma * np.max(q_value[next_state_[0], next_state_[1], :]) -
                                     q_value[state_[0], state_[1], action_])

        state = next_state

        # check whether it has exceeded the step limit
        if steps > environment.max_steps:
            break

        if state in ends:
            ends.remove(state)

    return steps

def dyna_q_action(q_value, model, environment, dyna_params):
    state = environment.START_STATE
    ends = copy(environment.GOAL_STATES)
    steps = 0
    actions_record = []
    while ends:

        # track the steps
        steps += 1

        # get action
        action = choose_action(state, q_value, environment, dyna_params)

        # take action
        next_state, reward = environment.step(state, action, ends)

        # Q-Learning update
        q_value[state[0], state[1], action] += \
            dyna_params.alpha * (reward + dyna_params.gamma * np.max(q_value[next_state[0], next_state[1], :]) -
                                 q_value[state[0], state[1], action])

        # feed the model with experience
        model.feed(state, action, next_state, reward)

        # sample experience from the model
        for t in range(0, dyna_params.planning_steps):
            state_, action_, next_state_, reward_ = model.sample()
            q_value[state_[0], state_[1], action_] += \
                dyna_params.alpha * (reward_ + dyna_params.gamma * np.max(q_value[next_state_[0], next_state_[1], :]) -
                                     q_value[state_[0], state_[1], action_])

        state = next_state

        # check whether it has exceeded the step limit
        if steps > environment.max_steps:
            break

        if state in ends:
            ends.remove(state)
        actions_record.append(action)

    return actions_record

def Dyn_Q(Start, Goal, Maze_Width, Maze_Height):
    # set up an instance for DynaMaze
    dyna_environment = Environment(WIDTH=Maze_Width, HEIGHT=Maze_Height)
    dyna_params = DynaParams()

    # Config
    dyna_environment.START_STATE = Start
    dyna_environment.GOAL_STATES = Goal

    runs = 1
    episodes = 50
    planning_steps = [50]
    steps = np.zeros((len(planning_steps), episodes))

    for run in tqdm(range(runs)):
        for i, planning_step in enumerate(planning_steps):
            dyna_params.planning_steps = planning_step
            q_value = np.zeros(dyna_environment.q_size)

            # generate an instance of Dyna-Q model
            model = TrivialModel()
            for ep in range(episodes):
                print('run:', run, 'planning step:', planning_step, 'episode:', ep)
                steps[i, ep] += dyna_q(q_value, model, dyna_environment, dyna_params)

    # averaging over runs
    steps /= runs

    for i in range(len(planning_steps)):
        plt.plot(steps[i, :], label='%d planning steps' % (planning_steps[i]))
    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.legend()
    plt.savefig(os.getcwd()+'/algorithms/images/exam.png')
    plt.close()
    actions = dyna_q_action(q_value, model, dyna_environment, dyna_params)
    print(actions)
    return actions

















## -----------------------------------------------------------------------------------##
#! previous work

#! 1. Not Used
class QLearningTable:
    '''
    Q Learning Table Class
    To record the Q learning Process
    '''
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
        '''
        Base on the Current map position, explore complete percentage and resolutions
        to calculate the action(x+, x-, y+, y-).
        '''
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
        '''
        Based on the State, Action and Reward
        to Learn and Calculate
        '''
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