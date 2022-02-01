import os
import copy
path = "path_log/"
def path_logger(episode, route_info):
    file_name = path + "episode" + str(episode)
    route = copy.deepcopy(route_info)
    if not os.path.isfile(file_name):
        with open(file_name, mode='w') as f:
            write(f, route)
    else:
        with open(file_name, mode='w') as f:
            write(f, route)

def write(f, route):
    for i in range(len(route)):
        route[i].pop(-1)
        for item in route[i]:
            f.write(" " + str(item[0]) + " " + str(item[1]))
        f.write("\n")
    f.close()