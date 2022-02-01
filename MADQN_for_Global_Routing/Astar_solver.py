
import numpy as np
import random
import copy
from matplotlib import pyplot as plt
import time

class Field(object):
    def __init__(self, maze, start_point, goal_point):
        self.maze = maze
        self.start_point = start_point
        self.goal_point = goal_point
        self.movable_vec = [[1,0],[-1,0],[0,1],[0,-1]]

    def display(self, point=None):
        field_data = copy.deepcopy(self.maze)
        if not point is None:
                y, x = point
                field_data[y][x] = "@@"
        else:
                point = ""
        for line in field_data:
                print ("\t" + "%3s " * len(line) % tuple(line))

    def get_actions(self, state):
        movables = []
        for v in self.movable_vec:
            y = state[0] + v[0]
            x = state[1] + v[1]
            if not self.maze[y][x] != "#":
                continue
            movables.append([y, x])
        if len(movables) != 0:
            return movables
        else:
            return None

    def get_val(self, state):
        if state == self.goal_point:
            return 0, True
        else:
            return 0, False

class Node(object):
    def __init__(self, state, start_point, goal_point):
        self.state = state
        self.start_point = start_point
        self.goal_point = goal_point
        self.hs = (self.state[0] - self.goal_point[0]) ** 2 + (self.state[1] - self.goal_point[1]) ** 2
        self.fs = 0
        self.parent_node = None

    def confirm_goal(self):
        if self.goal_point == self.state:
            return True
        else:
            return False

class NodeList(list):
    def find_nodelist(self, state):
        node_list = [t for t in self if t.state == state]
        return node_list[0] if node_list != [] else None

    def remove_from_nodelist(self, node):
        del self[self.index(node)]

class Astar_Solver(object):
    def __init__(self, maze, start_point, goal_point, capacity):
        self.Field = maze
        self.start_point = start_point
        self.goal_point = goal_point
        self.capacity = capacity
        self.open_list = NodeList()
        self.close_list = NodeList()
        self.route = []
        self.steps = 0
        self.score = 0

        self.overflow_cost = 1000

    def set_initial_node(self):
        node = Node(self.start_point, self.start_point, self.goal_point)
        node.start_point = self.start_point
        node.goal_point = self.goal_point
        return node

    def get_gs(self, action, node_gs, dist):
        y = action[0]
        x = action[1]
        if self.Field.maze[y][x] < -1*(self.capacity):
            return node_gs + dist + self.overflow_cost
        else:
            return node_gs + dist

    def go_next(self, next_actions, node):
        node_gs = node.fs - node.hs
        for action in next_actions:
            open_list = self.open_list.find_nodelist(action)
            dist = (node.state[0] - action[0]) ** 2 + (node.state[1] - action[1]) ** 2
            gs = self.get_gs(action, node_gs, dist)
            if open_list:
                if open_list.fs > gs + open_list.hs:
                    open_list.fs = gs + open_list.hs
                    open_list.parent_node = node
            else:
                open_list = self.close_list.find_nodelist(action)
                if open_list:
                    if open_list.fs > gs + open_list.hs:
                        open_list.fs = gs + open_list.hs
                        open_list.parent_node = node
                        self.open_list.append(open_list)
                        self.close_list.remove_from_nodelist(open_list)
                else:
                    open_list = Node(action, self.start_point, self.goal_point)
                    open_list.fs = gs + open_list.hs
                    open_list.parent_node = node
                    self.open_list.append(open_list)

    def get_path(self, node):
        path = []
        current_node = node
        while True:
            path.append(current_node.state)
            current_node = current_node.parent_node
            if current_node == None:
                path.reverse()
                return path

    def solve_maze(self):
        node = self.set_initial_node()
        node.fs = node.hs
        self.open_list.append(node)

        while True:
            node = min(self.open_list, key=lambda node: node.fs)
            #print("current state:  {0}".format(node.state))
            self.route.append(node.state)

            reward, tf = self.Field.get_val(node.state)
            self.score = self.score + reward
            #print("current step: {0} \t score: {1} \n".format(self.steps, self.score))
            self.steps += 1
            if tf == True:
                #print("Goal!")
                path = self.get_path(node)
                return path  # self.route

            self.open_list.remove_from_nodelist(node)
            self.close_list.append(node)

            next_actions = self.Field.get_actions(node.state)
            self.go_next(next_actions, node)


def astar_sover(grid_info, start_point, goal_point, capacity):

    maze_field = Field(grid_info, start_point, goal_point)
    astar_Solver = Astar_Solver(maze_field, start_point, goal_point, capacity)
    route = astar_Solver.solve_maze()
    return route



