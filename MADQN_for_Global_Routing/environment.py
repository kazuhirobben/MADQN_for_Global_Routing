import numpy as np
import random
import copy
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from operator import mul, add


import time

class Problem_generator():

    def __init__(self, size, path_num, capacity, seed):
        random.seed(seed)
        self.size = size + 2
        self.path_num = path_num
        self.capacity = capacity+1
        self.path_combo = []
        self.start_point = [0] * self.path_num
        self.goal_point = [0] * self.path_num
        self.maze_list = [0] * self.size
        for i in range(self.size):
            self.maze_list[i] = [-1] * self.size

    def generate_path_combo(self):
        ns = []
        ns_combo = [0] * self.path_num
        k = self.path_num * 2
        while len(ns) < k:
            n = [random.randint(1, self.size - 2) for r in range(2)]
            if not n in ns:
                ns.append(n)

        for i in range(self.path_num):
            ns_combo[i] = [ns[i], ns[i + self.path_num]]
        return ns_combo

    def insert_blocks(self, k, s_r, e_r):
        b_y = random.randint(1, self.size - 2)
        b_x = random.randint(1, self.size - 2)
        if [b_y, b_x] == [1, s_r] or [b_y, b_x] == [self.size - 2, e_r]:
            k = k - 1
        else:
            self.maze_list[b_y][b_x] = "#"

    def generate_grid_info(self):
        #  make barrier
        for i in range(self.size):
            if i == 0 or i == self.size - 1:
                for k in range(self.size):
                    self.maze_list[i][k] = "#"
            else:
                self.maze_list[i][0] = "#"
                self.maze_list[i][self.size - 1] = "#"

        #  make path_combo[[start],[goal]]
        self.path_combo = self.generate_path_combo()

        #  make start/goal list
        for i in range(self.path_num):
            self.start_point[i] = self.path_combo[i][0]
            self.goal_point[i] = self.path_combo[i][1]

        # implement start/goal to maze_list
        for i in range(self.path_num):
            s_row = self.start_point[i][0]
            s_column = self.start_point[i][1]
            g_row = self.goal_point[i][0]
            g_column = self.goal_point[i][1]

            self.maze_list[s_row][s_column] -= 1
            self.maze_list[g_row][g_column] -= 1

        return self.maze_list, self.start_point, self.goal_point

    def show_generated_grid(self, point=None):
        field_data = copy.deepcopy(self.maze_list)

        #  implement to maze
        for i in range(self.path_num):
            s_row = self.path_combo[i][0][0]
            s_column = self.path_combo[i][0][1]
            g_row = self.path_combo[i][1][0]
            g_column = self.path_combo[i][1][1]

            field_data[s_row][s_column] = "s" + str(i)
            field_data[g_row][g_column] = "g" + str(i)

        if not point is None:
            y, x = point
            field_data[y][x] = "@@"
        else:
            point = ""
        for line in field_data:
            print("\t" + "%3s " * len(line) % tuple(line))

class Grid_map():

    def __init__(self, grid_list, start_point, goal_point, capacity, reward, penalty, sparce_reward, movable_vector):
        self.original_grid_list = grid_list
        self.current_grid_list = copy.deepcopy(self.original_grid_list)
        self.best_grid_list = []
        self.best_episode = 0
        self.start_point = start_point
        self.goal_point = goal_point
        self.price = reward
        self.penalty = penalty
        self.sparce_reward = sparce_reward
        self.capacity = (capacity + 1) * -1
        self.r_capacity = capacity
        self.path_num = len(self.start_point)

        self.path_count = 0
        self.episode_count = 0
        self.episode_reward_list = []
        self.train_reward_list = []
        self.episode_list = []
        if movable_vector == 5:
            self.movable_vec = [[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]]  # 動ける方向ベクトル(down, up, right, left, stay)
        if movable_vector == 4:
            self.movable_vec = [[1, 0], [-1, 0], [0, 1], [0, -1]]  # 動ける方向ベクトル(down, up, right, left, stay)

        self.state = [0] * self.path_num
        self.is_terminal = [False] * self.path_num

        self.route = []

        self.route_combo = []
        self.best_route = []
        self.pre_reward = -10000000

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
            self.train_reward_list.append(sum(self.episode_reward_list))
            self.episode_list.append(self.episode_count)
            self.get_best_route()
            # self.reset_route_combo()
            self.episode_reward_list.clear()

        self.current_grid_list = copy.deepcopy(self.original_grid_list)
        self.reset_route()
        for i in range(self.path_num):
            self.state[i] = self.start_point[i]

    def get_route(self, agent):

        if len(self.route[agent]) > 1:
            if self.route[agent][-2] == self.state[agent]:
                return

            if len(self.route[agent]) > 2:
                if self.route[agent][-3] == self.state[agent]:
                    row, column = self.route[agent][-2]
                    if self.current_grid_list[row][column] == "#":
                        self.current_grid_list[row][column] = self.capacity + 1
                    else:
                        self.current_grid_list[row][column] += 1
                    self.route[agent].pop(-2)
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

        for agent in range(self.path_num):  # if agent is already at the goal
            bool = self.state[agent] == self.goal_point[agent]
            is_terminal[agent] = bool

        for agent in range(self.path_num):
            if is_terminal[agent] == True:  # if already goal
                next_state[agent] = self.state[agent]

            elif action[agent] == 4:  # if "stay"
                next_state[agent] = self.state[agent]

            else:
                row_step, column_step = self.get_actions(action[agent])
                row = self.state[agent][0] + row_step
                column = self.state[agent][1] + column_step
                next_state[agent] = [row, column]
                if next_state[agent] == self.goal_point[agent]:  # if a agent reaches to goal
                    is_terminal[agent] = True
                elif self.current_grid_list[row][column] == "#":  # if bumped in to blockage
                    reward[agent] += self.penalty
                    next_state[agent] = self.state[agent]
                else:
                    self.update_grid_info(next_state[agent], agent)


            self.state[agent] = next_state[agent]
            self.get_route(agent)

        sum_reward = np.sum(reward)
        False_count = is_terminal.count(False)

        sum_reward += -1*self.sparce_reward  # sparse team reward
        sum_reward += -1*self.sparce_reward*False_count  # parial team reward

        self.episode_reward_list.append(sum_reward)
        self.is_terminal = is_terminal.count(True)

        joint_reward = sum_reward
        return next_state, joint_reward, is_terminal

    def update_grid_info(self, next_state, agent):

        if len(self.route[agent])>2 and next_state == self.route[agent][-2]:
            row, column = self.state[agent]
            self.current_grid_list[row][column] += 1
        else:
            row, column = next_state
            self.current_grid_list[row][column] -= 1

        if self.current_grid_list[row][column] == self.capacity:
            self.current_grid_list[row][column] = '#'

    def bulk_update_grid_info(self, route, path_count):
        for i in range(len(route)):
            row = route[i][0]
            column = route[i][1]
            if route[i] != self.start_point[path_count] \
                    and route[i] != self.goal_point[path_count] \
                    and self.current_grid_list[row][column] != '#':
                self.current_grid_list[row][column] -= 1  # increment of cost

                if self.current_grid_list[row][column] == self.capacity:
                    self.current_grid_list[row][column] = '#'

    def state_observe(self, agent):
        point_ob = []
        d_y = self.state[agent][0] - self.goal_point[agent][0]
        d_x = self.state[agent][1] - self.goal_point[agent][1]
        for item in self.movable_vec:
            neighbor = map(add, self.state[agent], item)
            #row = self.state[agent][0] + self.movable_vec[v][0]
            #column = self.state[agent][1] + self.movable_vec[v][1]
            point_ob.append(self.get_val(neighbor, agent))
        observe = [self.state[agent][0], self.state[agent][1], d_y, d_x] + point_ob
        return observe

    def get_actions(self, action):
        # (up, down, right, left, stay)
        row = self.movable_vec[action][0]
        column = self.movable_vec[action][1]
        return row, column

    def sample(self, agent):
        if self.state[agent] == self.goal_point[agent]:
            return 4
        else:
            return np.random.randint(len(self.movable_vec))

    def get_val(self, state, agent):
        y, x = state

        if state == self.goal_point[agent]:
            return self.price
        elif self.current_grid_list[y][x] == "#":
            return self.penalty
        else:
            return self.current_grid_list[y][x] * 0.1

    def plot_reward(self):
        plt.plot(self.train_reward_list)
        # save_as_png
        plt.savefig('Solution_MADQN_mod3//reward.png')

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
        best_grid = self.best_grid_list
        return best_grid

    def give_best_episode(self):
        return self.best_episode

class Environment():
    def __init__(self, grid_size, number_of_path, capacity, reward, penalty, sparce_reward):
        self.grid_size = grid_size
        self.number_of_path = number_of_path
        self.capacity = capacity
        self.reward = reward
        self.penalty = penalty
        self.sparce_reward = sparce_reward

    def build_environment(self, movalble_vec, seed):
        random.seed(seed)
        size = self.grid_size + 2
        problem = Problem_generator(size, self.number_of_path, self.capacity)
        maze_list, start_point, goal_point = problem.generate_grid_info()
        problem.show_generated_grid()
        map = \
            Grid_map(maze_list, start_point, goal_point, self.capacity, self.reward, self.penalty, self.sparce_reward, movalble_vec)
        map.display()
        return start_point, goal_point, map

    def evaluator(self, solution, grid):
        length = 0
        counter = [0]*self.capacity

        for item in solution:
            length += (len(item) - 1)

        for item in grid:
            c = Counter(item)
            # count "#"
            counter[0] += c['#']
            c.pop('#')
            # count others
            for k, v in c.items():
                counter[k] += v
        counter[0] -= (self.grid_size*4 + 4)
        print(length)
        print(counter)
        return length, counter




