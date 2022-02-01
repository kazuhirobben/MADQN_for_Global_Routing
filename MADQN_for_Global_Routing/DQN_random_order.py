
import numpy as np
import copy
import random
from matplotlib import pyplot as plt
import time
from collections import deque
import torch
from torch import nn
from torch import optim


class Grid_map():
    def __init__(self, grid_list, start_point, goal_point, reward, penalty, overflow_penalty, capacity, sparce_reward, seed):
        self.original_grid_list = grid_list
        self.current_grid_list = copy.deepcopy(self.original_grid_list)
        self.best_grid_list = []
        self.best_episode = 0
        self.original_start_point = start_point
        self.original_goal_point = goal_point
        self.start_point = start_point
        self.goal_point = goal_point
        self.reward = reward
        self.penalty = penalty
        self.overflow_penalty = overflow_penalty
        self.sparce_reward = sparce_reward
        self.seed = seed
        self.capacity = (capacity + 1) * -1
        self.r_capacity = capacity
        self.state = start_point[0]

        grid_size = len(grid_list)
        self.base_self_channel = [[0] * grid_size for i in range(grid_size)]
        self.base_capacity_channel = [[capacity] * grid_size for i in range(grid_size)]
        self.block_cost = -len(self.start_point)
        for i in range(grid_size):
            if i == 0 or i == grid_size - 1:
                self.base_capacity_channel[i] = [self.block_cost] * grid_size
            else:
                self.base_capacity_channel[i][0] = self.block_cost
                self.base_capacity_channel[i][grid_size - 1] = self.block_cost
                for k in range(grid_size):
                    if self.original_grid_list[i][k] == -2:
                        self.base_capacity_channel[i][k] -= 1
        self.current_capacity_channel = copy.deepcopy(self.base_capacity_channel)

        self.path_count = 0
        self.episode_count = 0
        self.episode_reward_list = []
        self.train_reward_list = []
        self.episode_list = []
        self.movable_vec = [[1, 0], [-1, 0], [0, 1], [0, -1]]  # 動ける方向ベクトル(down, up, right, left)

        self.route = []
        self.route_combo = []
        self.best_route = []
        self.pre_reward = -10000000
        self.log = []

    def display(self, point=None):
        field_data = copy.deepcopy(self.current_grid_list)
        if not point is None:
            y, x = point
            field_data[y][x] = "@@"
        else:
            point = ""
        for line in field_data:
            print("\t" + "%3s " * len(line) % tuple(line))

    def reset_episode_count(self):
        self.episode_count = 0

    def reset_path_count(self):
        self.path_count = -1

    def reset_only_grid(self):
        self.current_grid_list = copy.deepcopy(self.original_grid_list)
        self.current_capacity_channel = copy.deepcopy(self.base_capacity_channel)

    def reset_grid(self):
        self.get_route_combo()
        self.path_count += 1
        if self.path_count > len(self.start_point) - 1:  # if all path gets connected
            self.episode_count += 1
            # self.display()
            self.path_count = 0
            self.logger()
            print('Episode' + str(self.episode_count))
            print('Reward' + str(sum(self.episode_reward_list)))
            self.train_reward_list.append(sum(self.episode_reward_list))
            self.episode_list.append(self.episode_count)
            self.get_best_route()
            self.reset_route_combo()
            self.episode_reward_list.clear()

            self.current_grid_list = copy.deepcopy(self.original_grid_list)
            self.current_capacity_channel = copy.deepcopy(self.base_capacity_channel)
            self.suffle_order()

        self.reset_route()
        self.state = self.start_point[self.path_count]

    def get_route(self):
        self.route.append(self.state)

    def reset_route(self):
        self.route = []

    def get_route_combo(self):

        if len(self.route) != 0:
            if self.route[-1] == self.goal_point[self.path_count]:
                self.route.append('Goal')
            else:
                self.route.append('Fail')
            self.route_combo.append(self.route)

    def reset_route_combo(self):
        self.route_combo = []

    def get_best_route(self):
        current_reward = sum(self.episode_reward_list)
        if current_reward > self.pre_reward:
            self.best_route = copy.deepcopy(self.route_combo)
            self.best_grid_list = copy.deepcopy(self.current_grid_list)
            self.pre_reward = current_reward
            self.best_episode = self.episode_count

    def step(self, action):
        self.update_grid_info()
        row_step, column_step = self.get_actions(action)
        row = self.state[0] + row_step
        column = self.state[1] + column_step
        next_state = [row, column]
        situation = self.get_val(next_state)  # situation[approvable_grid, overflow_grid, block, goal]

        is_terminal = False
        flag = 1
        if situation == "block":
            next_state = self.state
            reward = self.sparce_reward
            flag = 0
        elif situation == "goal":
            is_terminal = True
            reward = self.reward
        elif situation == "overflow_grid":
            reward = self.overflow_penalty
        else: # approvable_grid
            reward = self.sparce_reward

        self.episode_reward_list.append(reward)

        self.state = next_state
        if flag: self.get_route()

        return next_state, reward, is_terminal

    def update_grid_info(self):
        row = self.state[0]
        column = self.state[1]
        if self.state != self.start_point[self.path_count] \
                and self.state != self.goal_point[self.path_count] \
                and self.current_grid_list[row][column] != '#':
            self.current_grid_list[row][column] -= 1  # increment of cost
            self.current_capacity_channel[row][column] -= 1

    def suffle_order(self):
        random.seed()
        l = list(range(len(self.start_point)))
        random.shuffle(l)
        new_start_point = [0] * len(self.start_point)
        new_goal_point = [0] * len(self.goal_point)
        for i in range(len(self.start_point)):
            new_start_point[i] = self.start_point[l[i]]
            new_goal_point[i] = self.goal_point[l[i]]
        self.start_point = new_start_point
        self.goal_point = new_goal_point

    def original_order(self):
        self.start_point = self.original_start_point
        self.goal_point = self.original_goal_point

    def state_observe(self):
        point_ob = []
        d_y = self.state[0] - self.goal_point[self.path_count][0]
        d_x = self.state[1] - self.goal_point[self.path_count][1]
        for v in self.movable_vec:  # vには移動可能な方向ベクトル(up, down, right, left)
            row = self.state[0] + v[0]
            column = self.state[1] + v[1]
            point_ob.append(self.current_capacity_channel[row][column])
        return [self.state[0], self.state[1], d_y, d_x] + point_ob

    def get_actions(self, action):
        # (up, down, right, left)
        row = self.movable_vec[action][0]
        column = self.movable_vec[action][1]
        return row, column

    def sample(self):
        return np.random.randint(4)

    def get_val(self, state): # situation[approvable_grid, overflow_grid, block, goal]
        y, x = state
        if state == self.goal_point[self.path_count]:
            return "goal"
        elif self.current_grid_list[y][x] == "#":
            return "block"
        elif self.current_grid_list[y][x] <= self.capacity:
            return "overflow_grid"
        else:
            return "approvable_grid"

    def plot_reward(self):
        plt.plot(self.train_reward_list)
        # save_as_png
        #plt.savefig('Solution_Single_agent_mod4//reward_seed='+str(self.seed)+'.png')
        # plt.savefig('Solution_Single_agent_mod4//reward_seed='+str(self.seed)+'.png')

    def logger(self):
        goal_count = 0
        length = 0
        route_combo = copy.deepcopy(self.route_combo)
        for item in route_combo:
            check = item[-1]
            if check == 'Goal':
                goal_count += 1
            item.pop(-1)
            for i in range(len(self.start_point)):
                if item[-1] == self.goal_point[i]:
                    item.insert(0, self.start_point[i])
        if goal_count == len(self.start_point):
            for item in route_combo:
                length += (len(item) - 1)
        else:
            pass
        reward = round(sum(self.episode_reward_list), 2)
        self.log.append([self.episode_count,
                         reward,
                         length,
                         goal_count])

    def give_reward_list(self):
        train_reward = self.train_reward_list
        return train_reward

    def print_best_route(self):
        print(self.best_route)

    def give_best_route(self):
        best_route = self.best_route
        return best_route

    def print_best_reward(self):
        print(max(self.train_reward_list))

    def give_best_reward(self):
        best_reward = max(self.train_reward_list)
        return best_reward

    def print_best_grid(self, point=None):
        field_data = self.best_grid_list

        if not point is None:
            y, x = point
            field_data[y][x] = "@@"
        else:
            point = ""
        for line in field_data:
            print("\t" + "%3s " * len(line) % tuple(line))

    def give_best_grid(self):
        return self.best_grid_list

    def give_best_episode(self):
        return self.best_episode

    def give_log(self):
        return self.log

class Linear_Acti(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.Linear = nn.Linear(in_features, out_features, bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.Linear(x)
        x = self.relu(x)
        return x

class Only_Linear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.Linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        x = self.Linear(x)
        return x

class QNetwork(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.ln0 = Linear_Acti(in_features, 128)
        self.ln1 = Linear_Acti(128, 128)
        self.ln2 = Only_Linear(128, out_features)

    def forward(self, x):
        x = self.ln0(x)
        x = self.ln1(x)
        x = self.ln2(x)
        return x

class DQN_Solver():
    def __init__(self, state_size, action_size, batch_size, path_number, Gride):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.path_number = path_number
        self.Gride = Gride

        self.current_loss = None

        self.memory = []
        self.burn_in_memory_max = 10000
        self.memory_max = 50000
        self.gamma = 0.9
        self.epsilon = 0.05
        self.e_decay = 0.9999
        self.e_min = 0.01
        self.learning_rate = 0.0001
        self.obsev_size = self.state_size*2 + action_size
        self.qNetwork = QNetwork(self.obsev_size, self.action_size)  # (trainable=True)  # 学習するネットワーク
        self.tNetwork = QNetwork(self.obsev_size, self.action_size)  # (trainable=False)  # 学習しないネットワーク
        self.optimizer = optim.RMSprop(self.qNetwork.parameters(), lr=self.learning_rate)
        self.copy_weight()

    def copy_weight(self,):
        w0 = copy.deepcopy(self.qNetwork.state_dict()['ln0.Linear.weight'])
        b0 = copy.deepcopy(self.qNetwork.state_dict()['ln0.Linear.bias'])
        w1 = copy.deepcopy(self.qNetwork.state_dict()['ln1.Linear.weight'])
        b1 = copy.deepcopy(self.qNetwork.state_dict()['ln1.Linear.bias'])
        w2 = copy.deepcopy(self.qNetwork.state_dict()['ln2.Linear.weight'])
        b2 = copy.deepcopy(self.qNetwork.state_dict()['ln2.Linear.bias'])

        self.tNetwork.ln0.Linear.weight = nn.Parameter(w0)
        self.tNetwork.ln0.Linear.bias = nn.Parameter(b0)
        self.tNetwork.ln1.Linear.weight = nn.Parameter(w1)
        self.tNetwork.ln1.Linear.bias = nn.Parameter(b1)
        self.tNetwork.ln2.Linear.weight = nn.Parameter(w2)
        self.tNetwork.ln2.Linear.bias = nn.Parameter(b2)


    def train_model(self, x, y):
        t_x = torch.tensor(x, dtype=torch.float32)
        t_y = torch.tensor(y, dtype=torch.float32)

        loss_fn = nn.MSELoss()
        optimizer = self.optimizer
        optimizer.zero_grad()
        y_pred = self.qNetwork(t_x)
        loss = loss_fn(y_pred, t_y)
        loss.backward()
        optimizer.step()

        self.current_loss = loss.item()

    def memory_tank(self, transition):  # 現在の状態・行動・報酬・次の状態・次の状態から移動できる座標・ゴールしたか否か
        self.memory.append(transition)
        if len(self.memory) > self.memory_max:
            self.memory.pop(0)

    def burn_in_memory(self):
        print('Start burn in...')
        self.Gride.reset_grid()
        for i in range(self.burn_in_memory_max):
            observation = self.Gride.state_observe()
            action = self.Gride.sample()
            next_state, reward, is_terminal = self.Gride.step(action)
            observation_next = self.Gride.state_observe()
            self.memory_tank([observation, action, reward, observation_next, is_terminal])
            if is_terminal:
                self.Gride.reset_grid()

        print('Burn in finished.')

    def replay_sample_batch(self, batch_size):
        index = np.random.randint(len(self.memory), size=batch_size)
        batch = [self.memory[i] for i in index]
        return batch

    def epsilon_greedy_policy(self, q_value):
        rnd = np.random.rand()
        if self.epsilon >= rnd:
            action = np.random.randint(4)
        else:
            action = np.argmax(q_value)
        return action

    def train(self, episodes, times):
        self.Gride.reset_path_count()
        self.Gride.reset_episode_count()
        self.Gride.reset_only_grid()
        self.Gride.reset_route()
        self.Gride.reset_route_combo()
        for e in range(episodes*self.path_number):
            self.Gride.reset_grid()
            is_terminal = False

            if e % 50 == 0:
                self.copy_weight()

            iter = 0
            while not is_terminal:
                iter += 1
                observation = self.Gride.state_observe()
                x = torch.tensor(observation, dtype=torch.float32)
                q_values = self.qNetwork(x).to('cpu').detach().numpy().copy()
                action = self.epsilon_greedy_policy(q_values)
                nextstate, reward, is_terminal = self.Gride.step(action)
                observation_next = self.Gride.state_observe()
                self.memory_tank([observation, action, reward, observation_next, is_terminal])
                # ここまでone step完了

                # ここからNNの学習
                batch = self.replay_sample_batch(self.batch_size)
                batch_observation = np.squeeze(np.array([trans[0] for trans in batch]))
                batch_action = np.array([trans[1] for trans in batch])
                batch_reward = np.array([trans[2] for trans in batch])
                batch_observation_next = np.squeeze(np.array([trans[3] for trans in batch]))
                batch_is_terminal = np.array([trans[4] for trans in batch])
                batch_x = torch.tensor(batch_observation, dtype=torch.float32)
                q_batch = self.qNetwork(batch_x).to('cpu').detach().numpy().copy()
                batch_next_x = torch.tensor(batch_observation_next, dtype=torch.float32)
                q_batch_next = self.tNetwork(batch_next_x).to('cpu').detach().numpy().copy()
                y_batch = batch_reward + self.gamma * (1 - batch_is_terminal) * np.max(q_batch_next, axis=1)

                targetQ = q_batch.copy()
                targetQ[np.arange(self.batch_size), batch_action] = y_batch

                self.train_model(batch_observation, targetQ)

                if iter > times:
                    is_terminal = True

        # print solution
        self.Gride.plot_reward()
        self.Gride.print_best_route()
        self.Gride.print_best_reward()
        self.Gride.print_best_grid()

    def solution(self):
        train_reward = self.Gride.give_reward_list()
        best_route = self.Gride.give_best_route()
        best_reward = self.Gride.give_best_reward()
        best_grid = self.Gride.give_best_grid()
        best_episode = self.Gride.give_best_episode()
        log = self.Gride.give_log()
        return train_reward, best_route, best_reward, best_grid, best_episode, log


batch_size = 32
episodes = 10000
times = 50  # agent`s step number per pin_pair
score = 0


def DQN_random_order(grid_list, start_point, goal_point, capacity, seed, reward, overflow_penalty, sparce_reward):
    penalty = overflow_penalty
    number_of_path = len(start_point)
    # build environment
    env = Grid_map(grid_list, start_point, goal_point, reward, penalty, overflow_penalty, capacity, sparce_reward, seed)
    # make agent
    dql_solver = DQN_Solver(state_size=2, action_size=4, batch_size=batch_size, path_number=number_of_path, Gride=env)
    # burn in memory
    dql_solver.burn_in_memory()
    # train_agent
    dql_solver.train(episodes, times)
    # get solution
    train_reward, best_route, best_reward, best_grid, best_episode, log = dql_solver.solution()

    goal_count = 0
    for item in best_route:
        check = item[-1]
        if check == 'Goal':
            goal_count += 1
        item.pop(-1)
        for i in range(len(start_point)):
            if item[-1] == goal_point[i]:
                item.insert(0, start_point[i])

    return best_route, best_grid, goal_count, train_reward, best_episode, log

