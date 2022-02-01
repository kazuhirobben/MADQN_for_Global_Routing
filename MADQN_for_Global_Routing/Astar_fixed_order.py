
import numpy as np
import random
import copy
from matplotlib import pyplot as plt
from Astar_solver import astar_sover

class Grid_map():
    def __init__(self, grid_list, start_point, goal_point, capacity):
        self.original_grid_list = grid_list
        self.current_grid_list = copy.deepcopy(self.original_grid_list)
        self.best_grid_list = []
        self.best_episode = 0
        self.original_start_point = start_point
        self.original_goal_point = goal_point
        self.start_point = start_point
        self.goal_point = goal_point
        self.capacity = (capacity + 1) * -1
        self.r_capacity = capacity
        self.state = start_point[0]
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

    def reset_grid(self):
        self.get_route_combo()
        self.path_count += 1
        if self.path_count > len(self.start_point) - 1:  # if all path gets connected
            self.episode_count += 1
            # self.display()
            self.path_count = 0
            print('Episode' + str(self.episode_count))
            print('Reward' + str(sum(self.episode_reward_list)))
            self.train_reward_list.append(sum(self.episode_reward_list))
            self.episode_list.append(self.episode_count)
            self.get_best_route()
            self.reset_route_combo()
            self.episode_reward_list.clear()
            self.current_grid_list = copy.deepcopy(self.original_grid_list)
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

    def update_grid_info(self, route, path_count):
        for i in range(len(route)):
            row = route[i][0]
            column = route[i][1]
            if route[i] != self.start_point[path_count] \
                    and route[i] != self.goal_point[path_count] \
                    and self.current_grid_list[row][column] != '#':
                self.current_grid_list[row][column] -= 1  # increment of cost

                """
                if self.current_grid_list[row][column] == self.capacity:
                    self.current_grid_list[row][column] = '#'
                """


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
            point_ob.append(self.get_val([row, column]))

        return [self.state[0], self.state[1], d_y, d_x] + point_ob

    def get_actions(self, action):
        # (up, down, right, left)
        row = self.movable_vec[action][0]
        column = self.movable_vec[action][1]
        return row, column

    def sample(self):
        return np.random.randint(4)

    def get_val(self, state):
        y, x = state
        if state == self.goal_point[self.path_count]:
            return self.reward
        elif self.current_grid_list[y][x] == "#":
            return self.penalty
        else:
            return self.current_grid_list[y][x]*0.1

    def plot_reward(self):
        plt.plot(self.train_reward_list)
        # save_as_png
        plt.savefig('Solution_Single_agent_mod3//reward.png')

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

# Generate a maze
def suffle_order(start_point, goal_point):
    random.seed()
    l = list(range(len(start_point)))
    random.shuffle(l)
    new_start_point = [0] * len(start_point)
    new_goal_point = [0] * len(goal_point)
    for i in range(len(start_point)):
        new_start_point[i] = start_point[l[i]]
        new_goal_point[i] = goal_point[l[i]]

    return new_start_point, new_goal_point

def get_wire_length(solution):
    length = 0
    for i in range(len(solution)):
        length += len(solution[i])-1

    return length

def organaize_order(start_point, goal_point, descent):
    dim1, dim2 = (2, len(start_point))
    numbering_list = [[0 for i in range(dim1)] for j in range(dim2)]

    order_list = []
    for i in range(len(start_point)):
        y_s, x_s = start_point[i][0], start_point[i][1]
        y_g, x_g = goal_point[i][0], goal_point[i][1]
        d_x = abs(x_g - x_s)
        d_y = abs(y_g - y_s)
        d = d_y + d_x
        order_list.append(d)
        numbering_list[i][0], numbering_list[i][1] = i, d
    order_list = sorted(order_list, reverse=descent)
    new_order = []
    for item in order_list:
        for idx in numbering_list:
            if idx[1] == item:
                new_order.append(idx[0])
                numbering_list.remove(idx)
    new_start_point = []
    new_goal_point = []
    for idx in new_order:
        new_start_point.append(start_point[idx])
        new_goal_point.append(goal_point[idx])

    return new_start_point, new_goal_point


# # Solving the maze with A-star algorithm

def Astar(grid_list, start_point, goal_point, capacity, trial, descent):
    start_point, goal_point = organaize_order(start_point, goal_point, descent)
    map = Grid_map(grid_list, start_point, goal_point, capacity)
    solution = []
    best_solution = []
    best_length = 100000
    best_map = None

    fail_count = 0
    for t in range(trial):
        try:
            for i in range(len(start_point)):
                st = start_point[i]
                gl = goal_point[i]
                route = astar_sover(map.current_grid_list, st, gl, capacity)
                solution.append(route)
                map.update_grid_info(route, i)

            length = get_wire_length(solution)

            if t == 0:
                best_length = length
                best_solution = solution
                best_map = map.current_grid_list

            if length < best_length:
                best_solution = solution
                best_map = map.current_grid_list
        except:
            fail_count += 1
            pass
        solution = []
        map.reset_only_grid()
        #start_point, goal_point = suffle_order(start_point, goal_point)
    print("Astar fail count:" + str(fail_count))

    return best_solution, best_map, fail_count

