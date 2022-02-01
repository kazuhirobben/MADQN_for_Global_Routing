import os

def Task_Logger(file_name, seed, start_point, goal_point):
    if not os.path.isfile(file_name):
        with open(file_name, mode='w') as f:
            write(f, seed, start_point, goal_point)
    else:
        with open(file_name, mode='a') as f:
            write(f, seed, start_point, goal_point)


def write(f, seed, start_point, goal_point):
    f.write("\n" + "\n" + str(seed))
    for i in range(len(start_point)):
        f.write("\n")
        f.write(str(start_point[i]) + ", " + str(goal_point[i]))
    f.close()