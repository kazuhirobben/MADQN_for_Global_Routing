#  multiple sets of start and goal

import numpy as np
import random
import copy
import itertools
import matplotlib
import matplotlib.pyplot as plt
import statistics
from collections import deque
from Path_logger import path_logger
import time
import torch
from torch import nn
from torch import optim


class Grid_map():
    def __init__(self, grid_list, start_point, goal_point, reward, penalty, overflow_penalty,capacity, sparce_reward, seed):
        self.original_grid_list = grid_list
        self.current_grid_list = copy.deepcopy(self.original_grid_list)
        self.best_grid_list = []
        self.best_episode = 0
        self.start_point = start_point
        self.goal_point = goal_point
        self.price = reward
        self.penalty = penalty
        self.overflow_penalty = overflow_penalty
        self.sparce_reward = sparce_reward
        self.seed = seed
        self.capacity = (capacity + 1) * -1
        self.r_capacity = capacity
        self.path_num = len(self.start_point)

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
        self.movable_vec = [[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]]  # 動ける方向ベクトル(down, up, right, left, stay)

        self.state = [0] * self.path_num
        self.is_terminal = [False] * self.path_num

        self.route = []

        self.route_combo = []
        self.best_route = []
        self.pre_reward = -10000000
        self.log = []

    def display(self, point=None):
        field_data = copy.deepcopy(self.current_grid_list)
        for line in field_data:
            print("\t" + "%3s " * len(line) % tuple(line))

    def reset_episode_count(self):
        self.episode_count = 0

    def reset_path_count(self):
        self.path_count = -1

    def reset_only_grid(self):
        self.current_grid_list = copy.deepcopy(self.original_grid_list)

    def reset_grid(self, burnin):

        if burnin == False:
            # self.get_route_combo()
            self.episode_count += 1
            # self.display()
            print('Episode:' + str(self.episode_count) + '    ' + 'Reward:' + str(sum(self.episode_reward_list))
                  + '    ' + 'Path:' + str(self.is_terminal) + '/' + str(self.path_num))
            self.logger()
            self.train_reward_list.append(sum(self.episode_reward_list))
            self.episode_list.append(self.episode_count)
            self.get_best_route()
            # self.reset_route_combo()
            #self.episode_reward_list.clear()
        self.episode_reward_list.clear()
        self.current_grid_list = copy.deepcopy(self.original_grid_list)
        self.current_capacity_channel = copy.deepcopy(self.base_capacity_channel)
        self.reset_route()
        for i in range(self.path_num):
            self.state[i] = self.start_point[i]

    def get_route(self, agent):

        if len(self.route[agent]) > 1:

            if self.route[agent][-2] == self.state[agent]:  # if same grid as previous step
                return

            if len(self.route[agent]) == 2 and self.state[agent] == self.start_point[agent]:
                row, column = self.route[agent][-2]
                self.current_grid_list[row][column] += 1
                self.current_capacity_channel[row][column] += 1

                row, column = self.start_point[agent]
                self.current_grid_list[row][column] += 1
                self.current_capacity_channel[row][column] += 1
                self.route[agent] = [0]
                return

            if len(self.route[agent]) > 2:
                if self.route[agent][-3] == self.state[agent]:  # if returned to the same grid
                    row, column = self.route[agent][-2]
                    self.current_grid_list[row][column] += 1
                    self.current_capacity_channel[row][column] += 1

                    row, column = self.route[agent][-3]
                    self.current_grid_list[row][column] += 1
                    self.current_capacity_channel[row][column] += 1
                    self.route[agent].pop(-2)
                    return
        else:
            if self.state[agent] == self.start_point[agent]:
                return

        self.route[agent][-1] = self.state[agent]
        if self.state[agent] == self.goal_point[agent]:
            self.route[agent].append(1)
        else:
            self.route[agent].append(0)

    def reset_route(self):
        self.route = [0] * self.path_num
        for i in range(len(self.route)):
            self.route[i] = [0]

    def reset_route_combo(self):
        self.route_combo = []

    def get_best_route(self):
        current_reward = sum(self.episode_reward_list)
        if current_reward > self.pre_reward:
            self.best_route = copy.deepcopy(self.route)
            self.best_grid_list = copy.deepcopy(self.current_grid_list)
            self.pre_reward = current_reward
            self.best_episode = self.episode_count

    def step(self, action):
        next_state = [0]*self.path_num
        reward = [0]*self.path_num  # sparse team reward
        is_terminal = [False]*self.path_num

        for agent in range(self.path_num):  # if agent is already at goal
            bool = self.state[agent] == self.goal_point[agent]
            is_terminal[agent] = bool

        for agent in range(self.path_num):
            if is_terminal[agent] == True:  # if already at goal
                next_state[agent] = self.state[agent]

            elif action[agent] == 4:  # if "stay"
                reward[agent] += 0 # self.sparce_reward
                next_state[agent] = self.state[agent]

            else:
                row_step, column_step = self.get_actions(action[agent])
                row = self.state[agent][0] + row_step
                column = self.state[agent][1] + column_step
                next_state[agent] = [row, column]
                if next_state[agent] == self.goal_point[agent]:  # if a agent reaches to goal
                    reward[agent] += self.price
                    is_terminal[agent] = True
                elif self.current_grid_list[row][column] == "#":  # if bumped in to blockage
                    reward[agent] += self.sparce_reward
                    next_state[agent] = self.state[agent]
                elif self.current_grid_list[row][column] <= self.capacity:
                    reward[agent] += self.overflow_penalty
                    self.update_grid_info(next_state[agent], agent)
                else:
                    reward[agent] += self.sparce_reward
                    self.update_grid_info(next_state[agent], agent)

            self.state[agent] = next_state[agent]
            self.get_route(agent)

        sum_reward = np.sum(reward)
        False_count = is_terminal.count(False)

        #sum_reward += -self.sparce_reward  # sparse team reward
        #sum_reward += -self.sparce_reward*False_count  # parial team reward

        self.episode_reward_list.append(sum_reward)
        self.is_terminal = is_terminal.count(True)

        return next_state, reward, is_terminal

    def update_grid_info(self, next_state, agent):

        if len(self.route[agent])>2 and next_state == self.route[agent][-2]:
            row, column = self.state[agent]
            self.current_grid_list[row][column] += 1
            self.current_capacity_channel[row][column] += 1
        else:
            row, column = next_state
            self.current_grid_list[row][column] -= 1
            self.current_capacity_channel[row][column] -= 1

    def state_observe(self, agent):
        point_ob = []

        dys = self.start_point[agent][0] - self.state[agent][0]
        dxs = self.start_point[agent][1] - self.state[agent][1]

        dyg = self.goal_point[agent][0] - self.state[agent][0]
        dxg = self.goal_point[agent][1] - self.state[agent][1]

        current_capacity = copy.deepcopy(self.get_capacity_channel())

        one_neighbor_capacity_list = []
        two_neighbor_capacity_list = []
        vector = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        for v in vector:
            row = self.state[agent][0] + v[0]
            column = self.state[agent][1] + v[1]
            capacity_list = []
            one_neighbor_capacity_list.append(current_capacity[row][column])
            capacity_list.append(current_capacity[row][column])
            for w in vector:
                y = row + w[0]
                x = column + w[1]
                if (0 <= y and y <= (len(current_capacity)-1)) and (0 <= x and x <= (len(current_capacity)-1)):
                    capacity_list.append(current_capacity[y][x])
                else:
                    pass
            two_neighbor_capacity_list.append(statistics.mean(capacity_list))
        l = [dys, dxs, dyg, dxg] + one_neighbor_capacity_list + two_neighbor_capacity_list

        observe = l
        return observe

    def get_capacity_channel(self):
        return self.current_capacity_channel

    def get_actions(self, action):
        # (up, down, right, left, stay)
        row = self.movable_vec[action][0]
        column = self.movable_vec[action][1]
        return row, column

    def sample(self, agent):
        if self.state[agent] == self.goal_point[agent]:
            return 4
        else:
            return np.random.randint(5)

    def get_val(self, state, agent):
        y, x = state

        if state == self.goal_point[agent]:
            return round(self.price, 2)
        elif self.current_grid_list[y][x] == "#":
            return round(self.penalty, 2)
        elif self.current_grid_list[y][x] <= self.capacity:
            return round(self.overflow_penalty, 2)
        else:
            return round(self.current_grid_list[y][x] * self.sparce_reward, 2)

    def plot_reward(self):
        plt.plot(self.train_reward_list)
        # save_as_png
        # plt.savefig('Solution_MADQN_mod4//reward_seed='+str(self.seed)+'.png')

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

    def logger(self):
        goal_count = 0
        length = 0
        route_combo = copy.deepcopy(self.route)
        try:
            for item in route_combo:
                check = item[-1]
                if check == 1:
                    goal_count += 1
                item.pop(-1)
                for i in range(len(self.start_point)):
                    if item[-1] == self.goal_point[i]:
                        item.insert(0, self.start_point[i])
        except:
            pass
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

    def give_best_grid(self):
        best_grid = self.best_grid_list
        return best_grid

    def give_best_episode(self):
        return self.best_episode

    def give_log(self):
        return self.log

    def give_route(self):
        route =[]
        route = copy.deepcopy(self.route)
        for i in range(self.path_num):
            route[i].insert(0, self.start_point[i])
        return route

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

class MADQN_Solver():
    def __init__(self, state_size, action_size, batch_size, path_number, Gride):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.path_number = path_number
        self.Gride = Gride
        self.agent_number = 0

        self.current_loss = None

        self.memory = []
        self.burn_in_memory_max = 10000
        self.memory_max = 50000
        self.gamma = 0.9
        self.epsilon = 0.05
        self.e_decay = 0.9999
        self.e_min = 0.01
        self.learning_rate = 0.0001
        self.obsev_size = 12

        input_size = self.obsev_size
        self.qNetwork = QNetwork(in_features=input_size, out_features=self.action_size)  # (trainable=True)  # 学習するネットワーク
        self.qNetwork = self.try_gpu(self.qNetwork)
        self.tNetwork = QNetwork(in_features=input_size, out_features=self.action_size)  # (trainable=False)  # 学習しないネットワーク
        self.tNetwork = self.try_gpu(self.tNetwork)
        self.optimizer = optim.RMSprop(self.qNetwork.parameters(), lr=self.learning_rate)
        self.copy_weight()

    def copy_weight(self, ):
        self.tNetwork = copy.deepcopy(self.qNetwork)

    def train_model(self, x, y):
        t_x = torch.tensor(x, dtype=torch.float32)
        t_y = torch.tensor(y, dtype=torch.float32)
        t_x = self.try_gpu(t_x)
        t_y = self.try_gpu(t_y)

        loss_fn = nn.MSELoss()
        optimizer = self.optimizer
        optimizer.zero_grad()
        y_pred = self.qNetwork(t_x)
        loss = loss_fn(y_pred, t_y)
        loss.backward()
        optimizer.step()

        self.current_loss = loss.item()

    def memory_tank(self, transition): # 現在の状態・行動・報酬・次の状態・次の状態から移動できる座標・エージェントの番号・ゴールしたか否か
        if transition[0][2] == 0 and transition[0][3] == 0:  # already at goal
            return
        self.memory.append(transition)
        if len(self.memory) > self.memory_max:
            self.memory.pop(0)

    def burn_in_memory(self):
        print('Start burn in...')
        self.Gride.reset_grid(burnin=True)
        observation = [0] * self.path_number
        action = [0] * self.path_number
        next_state = [0] * self.path_number
        reward = [0] * self.path_number
        is_terminal = [False] * self.path_number
        observation_next = [0] * self.path_number

        for i in range(int(self.burn_in_memory_max / self.path_number) + 1):
            partial_observation = [0] * self.path_number
            for agent in range(self.path_number):
                partial_observation[agent] = self.Gride.state_observe(agent)
            observation = copy.deepcopy(partial_observation)

            for agent in range(self.path_number):
                action[agent] = self.Gride.sample(agent)

            next_state, reward, is_terminal = self.Gride.step(action)


            partial_observation = [0] * self.path_number
            for agent in range(self.path_number):
                partial_observation[agent] = self.Gride.state_observe(agent)
            observation_next = copy.deepcopy(partial_observation)

            for agent in range(self.path_number):
                self.memory_tank([observation[agent], action[agent], reward[agent], observation_next[agent], agent,
                                  is_terminal[agent]])
            if all(is_terminal):
                self.Gride.reset_grid(burnin=True)

        print('Burn in finished.')

    def replay_sample_batch(self, batch_size):
        index = np.random.randint(len(self.memory), size=batch_size)
        batch = [self.memory[i] for i in index]
        return batch

    def epsilon_greedy_policy(self, q_value, agent):
        if self.Gride.state[agent] == self.Gride.goal_point[agent]:
            return 4
        else:
            rnd = np.random.rand()
            if self.epsilon >= rnd:
                action = np.random.randint(5)
            else:
                action = np.argmax(q_value)
            return action

    def train(self, episodes, times):
        self.Gride.reset_episode_count()
        self.Gride.reset_only_grid()
        self.Gride.reset_route()

        for e in range(episodes):
            """
            if e>0:
                path_logger(episode=e, route_info=self.Gride.give_route())
            
            """
            self.Gride.reset_grid(burnin=False)
            episode_is_terminal = False

            if e % 50 == 0:
                self.copy_weight()

            iter = 0
            while not episode_is_terminal:
                iter += 1
                observation = [0] * self.path_number
                action = [0] * self.path_number
                next_state = [0] * self.path_number
                reward = [0] * self.path_number
                is_terminal = [False] * self.path_number
                observation_next = [0] * self.path_number

                partial_observation = [0] * self.path_number
                for p in range(self.path_number):
                    partial_observation[p] = self.Gride.state_observe(p)
                observation = copy.deepcopy(partial_observation)

                net_input_list = []
                net_input_list = observation
                net_input_array = np.array(net_input_list)
                x = torch.tensor(net_input_array, dtype=torch.float32)
                x = self.try_gpu(x)
                q_values = self.qNetwork(x).to('cpu').detach().numpy().copy()

                for agent in range(self.path_number):
                    action[agent] = int(self.epsilon_greedy_policy(q_values[agent], agent))

                next_state, reward, is_terminal = self.Gride.step(action)

                partial_observation = [0] * self.path_number
                for agent in range(self.path_number):
                    partial_observation[agent] = self.Gride.state_observe(agent)
                observation_next = partial_observation

                # update episode record rewards
                for agent in range(self.path_number):
                    self.memory_tank([observation[agent], action[agent], reward[agent], observation_next[agent], agent,
                                      is_terminal[agent]])
                # ここまでone step完了

                # ここからNNの学習
                batch = copy.deepcopy(self.replay_sample_batch(self.batch_size))

                batch_net_input = []
                batch_action = []
                batch_reward = []
                batch_net_input_next = []
                batch_is_terminal = []

                batch_net_input_array = None
                batch_net_input_next_array = None
                batch_is_terminal_array = None
                batch_reward_array = None

                for x in batch:
                    batch_net_input.append(x[0])
                    batch_action.append(x[1])
                    batch_reward.append(x[2])
                    batch_net_input_next.append(x[3])
                    batch_is_terminal.append(x[5])

                batch_net_input_array = np.array(batch_net_input)
                batch_net_input_next_array = np.array(batch_net_input_next)
                batch_is_terminal_array = np.array(batch_is_terminal)
                batch_reward_array = np.array(batch_reward)

                batch_x = torch.tensor(batch_net_input_array, dtype=torch.float32)
                batch_x = self.try_gpu(batch_x)
                q_batch = self.qNetwork(batch_x).to('cpu').detach().numpy().copy()

                batch_next_x = torch.tensor(batch_net_input_next_array, dtype=torch.float32)
                batch_next_x = self.try_gpu(batch_next_x)
                q_batch_next = self.tNetwork(batch_next_x).to('cpu').detach().numpy().copy()
                y_batch = batch_reward_array + self.gamma * (1 - batch_is_terminal_array) * np.max(q_batch_next, axis=1)

                targetQ = q_batch.copy()
                targetQ[np.arange(self.batch_size), batch_action] = y_batch

                self.train_model(batch_net_input_array, targetQ)

                if iter > times:
                    episode_is_terminal = True

        # print solution
        self.Gride.plot_reward()
        self.Gride.print_best_route()
        self.Gride.print_best_reward()
        self.Gride.print_best_grid()

    def try_gpu(self, e):
        if torch.cuda.is_available():
            return e.to('cuda:0')
        return e

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
times = 50  # step number per episode
score = 0


def MADQN(grid_list, start_point, goal_point, capacity, seed, reward, overflow_penalty, sparce_reward):
    penalty = overflow_penalty
    number_of_path = len(start_point)
    # build environment
    map = Grid_map(grid_list, start_point, goal_point, reward, penalty, overflow_penalty, capacity, sparce_reward, seed)
    # make agents
    dql_solver = MADQN_Solver(state_size=2, action_size=5, batch_size=batch_size, path_number=number_of_path, Gride=map)
    # burn in memory
    dql_solver.burn_in_memory()
    # train agent
    dql_solver.train(episodes, times)
    # get solution
    train_reward, best_route, best_reward, best_grid, best_episode, log = dql_solver.solution()

    goal_count = 0
    for item in best_route:
        check = item[-1]
        if check == 1:
            goal_count += 1
        item.pop(-1)
        for i in range(len(start_point)):
            try:
                if item[-1] == goal_point[i]:
                    item.insert(0, start_point[i])
            except:
                pass
    return best_route, best_grid, goal_count, train_reward, best_episode, log

class Solution():

    def __init__(self, train_reward, best_route, best_reward, best_grid, capacity):

        self.train_reward = train_reward
        self.best_route = best_route
        self.best_reward = best_reward
        self.best_grid = best_grid
        self.capacity = (capacity + 1) * (-1)

    def wire_length(self):
        wire_length = 0
        for i in range(len(self.best_route)):
            wire_length += len(self.best_route[i]) - 1
        return wire_length

    def heat_map(self):
        a = len(self.best_grid) - 2
        heat_map = np.zeros([a, a])
        for i in range(a):
            for j in range(a):
                if self.best_grid[i + 1][j + 1] == "#":
                    heat_map[i][j] = -self.capacity - 1
                else:
                    heat_map[i][j] = -self.best_grid[i + 1][j + 1] - 1

        plt.figure()

        plt.imshow(heat_map, interpolation='nearest', vmin=0, vmax=-self.capacity - 1, cmap='jet')
        plt.colorbar()
        # save as png
        plt.savefig('Solution_MADQN_mod3//heatmap.png')
        # plt.show()

    def show_route_solution(self):
        for i in range(len(self.best_route)):
            print(self.best_route[i])
