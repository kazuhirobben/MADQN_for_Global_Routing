import copy
import numpy as np
import matplotlib.pyplot as plt
import os



def plot_route(start_point, goal_point, solution, dir_name, grid_size, capacity):
    grid_size += 2

    os.mkdir(dir_name+"/route")
    file_name = dir_name+"/route/"
    # organize solution
    route_list = []
    for i in start_point:
        for k in solution:
            if i == k[0]:
                route_list.append(k)

    # make grid

    grid_base_list = [[0] * grid_size for i in range(grid_size)]
    capacity_base_list = [[capacity] * grid_size for i in range(grid_size)]


    # make route figure
    pin_num = 0
    for i in route_list:
        pin_num += 1
        grid_list = copy.deepcopy(grid_base_list)
        for step in i:
            row = step[0]
            column = step[1]
            grid_list[row][column] += 1

        # make_route_figure
        grid_array = np.array(grid_list)
        fig, ax = plt.subplots()
        ax.imshow(grid_array)
        im = ax.imshow(grid_array)
        fig.colorbar(im, ax=ax)
        fig.savefig(file_name+"img_"+"pin_num_"+str(pin_num)+".png")

    # make congestion map
    capacity_list = copy.deepcopy(capacity_base_list)
    for i in route_list:
        pin_num += 1
        for step in i:
            row = step[0]
            column = step[1]
            capacity_list[row][column] -= 1

    # make_route_figure
    os.mkdir(dir_name + "/congestion")
    file_name = dir_name + "/congestion/"
    capacity_array = np.array(capacity_list)
    fig, ax = plt.subplots()
    ax.imshow(capacity_array)
    im = ax.imshow(capacity_array, cmap="Greys_r", vmin=-1, vmax=capacity)
    fig.colorbar(im, ax=ax)
    fig.savefig(file_name+"congestion_map" + ".png")

