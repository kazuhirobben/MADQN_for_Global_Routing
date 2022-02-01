import os

def Write_to_text(file_name, seed, length, counter, goal_check):

    if not os.path.isfile(file_name):
        with open(file_name, mode='w') as f:
            write(f, seed, length, counter, goal_check)
    else:
        with open(file_name, mode='a') as f:
            write(f, seed, length, counter, goal_check)

def write(f, seed, length, counter, goal_check):
    f.write("\n" + str(seed) + "," + str(length))
    for item in counter:
        f.write("," + str(item))

    f.write("," + str(goal_check))
    f.close()

def Write_reward_text(file_name, seed, rewards, best_episode):
    if not os.path.isfile(file_name):
        with open(file_name, mode='w') as f:
            write_reward(f, seed, rewards, best_episode)
    else:
        with open(file_name, mode='a') as f:
            write_reward(f, seed, rewards, best_episode)

def write_reward(f, seed, reward, best_episode):
    f.write("\n" + "\n" + str(seed) + "," + str(best_episode))
    for item in reward:
        f.write("\n" + str(item))
    f.close()

def Write_log_text(file_name, seed, log):
    if not os.path.isfile(file_name):
        with open(file_name, mode='w') as f:
            write_log(f, seed, log)
    else:
        with open(file_name, mode='a') as f:
            write_log(f, seed, log)

def write_log(f, seed, log):
    f.write("\n" + "\n" + str(seed))
    for items in log:
        f.write("\n")
        for elements in items:
            f.write(str(elements) + ",")
    f.close()