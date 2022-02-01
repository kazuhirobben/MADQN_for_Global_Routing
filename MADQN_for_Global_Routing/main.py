import os

from environment import Problem_generator
from evaluator import Evaluator

from DQN_random_order import DQN_random_order
import DQN_fixed_order

from Astar_random_order import Grid_map, Astar
import Astar_fixed_order
import MADQN

import Plot_route
import datetime

from write_to_text import Write_to_text, Write_reward_text, Write_log_text
from task_info_logger import Task_Logger

methods_list = ["Astar(ascent)", "Astar(descent)", "Astar(random)", "Single_agent(ascent)",
                "Single_agent(descent)", "Single_agent(random)", "MADQN"]

bench_mark = [0]*3
reward = [100]  # [100,100,100,100,100,100,100]
penalty = [-10]  # [-10,-10,-10,-1,-1,-1,0]
sparce_reward = [-0.1]  # [-1, -0.1, -0.01, -0.1, -0.01, -0.001, -1]
bench_mark[0] = reward
bench_mark[1] = penalty
bench_mark[2] = sparce_reward

dt_now = datetime.datetime.now()
# make file for results
dir_name = str(dt_now)  # "file_name"   # write file name or use「str(dt_now)」
os.mkdir(dir_name)

grid_size = 11  # size of grid
number_of_path = 50  # number of pin pairs
capacity = 5  # capacity

def main(seed):

    # problem generate
    problem = Problem_generator(grid_size, number_of_path, capacity, seed=seed)
    grid_list, start_point, goal_point = problem.generate_grid_info()

    # task info log
    Task_Logger(dir_name + "/task_info", seed, start_point, goal_point)

    for i in range(1):
        trial = 10000  # Astar trial

        reward_dir = "RewardNO." + str(i + 1)
        reward = bench_mark[0][i]
        overflow_penalty = bench_mark[1][i]
        sparce_reward = bench_mark[2][i]

        # Astar_ascent
        log_dir = dir_name + "/" + methods_list[0] + "/" + reward_dir
        seed_dir = dir_name + "/" + methods_list[0] + "/" + reward_dir + "/seed" + str(seed)

        Astar2_Solution, Astar2_grid, fail_count2 = Astar_fixed_order.Astar(grid_list, start_point, goal_point,
                                                                            capacity, trial, descent=False)
        # evaluate
        length, counter = Evaluator(Astar2_Solution, Astar2_grid, capacity, grid_size)
        Write_to_text(log_dir + "/Astar_ascent_result", seed, length, counter, fail_count2)

        Plot_route.plot_route(start_point, goal_point, Astar2_Solution, seed_dir, grid_size, capacity)

        # Astar_descent
        log_dir = dir_name + "/" + methods_list[1] + "/" + reward_dir
        seed_dir = dir_name + "/" + methods_list[1] + "/" + reward_dir + "/seed" + str(seed)

        Astar2_Solution, Astar2_grid, fail_count2 = Astar_fixed_order.Astar(grid_list, start_point, goal_point,
                                                                            capacity, trial, descent=True)
        # evaluate
        length, counter = Evaluator(Astar2_Solution, Astar2_grid, capacity, grid_size)
        Write_to_text(log_dir + "/Astar_descent_result", seed, length, counter, fail_count2)

        Plot_route.plot_route(start_point, goal_point, Astar2_Solution, seed_dir, grid_size, capacity)

        # Astar_random
        log_dir = dir_name + "/" + methods_list[2] + "/" + reward_dir
        seed_dir = dir_name + "/" + methods_list[2] + "/" + reward_dir + "/seed" + str(seed)

        Astar1_Solution, Astar1_grid, fail_count1 = Astar(grid_list, start_point, goal_point, capacity, trial)

        # Astar_evaluate
        length, counter = Evaluator(Astar1_Solution, Astar1_grid, capacity, grid_size)
        Write_to_text(log_dir + "/Astar_random_order_result", seed, length, counter, fail_count1)

        Plot_route.plot_route(start_point, goal_point, Astar1_Solution, seed_dir, grid_size, capacity)

        # DQN_fixed_order(ascent)
        log_dir = dir_name + "/" + methods_list[3] + "/" + reward_dir
        seed_dir = dir_name + "/" + methods_list[3] + "/" + reward_dir + \
                   "/" + "seed" + str(seed)

        DQN1_Solution, DQN1_grid, DQN1_goal_count, DQN1_train_reward, DQN1_best_episode, DQN1_log = DQN_fixed_order.DQN_fixed_order(
            grid_list, start_point,
            goal_point, capacity,
            seed, descent=False, reward=reward, overflow_penalty=overflow_penalty, sparce_reward=sparce_reward)

        # DQN_evaluate
        length, counter = Evaluator(DQN1_Solution, DQN1_grid, capacity, grid_size)
        Write_to_text(log_dir + "/DQN_fixed_order(ascent)_result", seed, length, counter, DQN1_goal_count)
        Write_reward_text(log_dir + "/DQN_fixed_order(ascent)_reward", seed, DQN1_train_reward, DQN1_best_episode)
        Write_log_text(log_dir + "/DQN_fixed_order(ascent)_log", seed, DQN1_log)

        Plot_route.plot_route(start_point, goal_point, DQN1_Solution, seed_dir,grid_size, capacity)
        # DQN_fixed_order(descent)
        log_dir = dir_name + "/" + methods_list[4] + "/" + reward_dir
        seed_dir = dir_name + "/" + methods_list[4] + "/" + reward_dir + \
                   "/" + "seed" + str(seed)

        DQN1_Solution, DQN1_grid, DQN1_goal_count, DQN1_train_reward, DQN1_best_episode, DQN1_log = DQN_fixed_order.DQN_fixed_order(
            grid_list, start_point,
            goal_point, capacity,
            seed, descent=True, reward=reward, overflow_penalty=overflow_penalty, sparce_reward=sparce_reward)
        # DQN_evaluate
        length, counter = Evaluator(DQN1_Solution, DQN1_grid, capacity, grid_size)
        Write_to_text(log_dir + "/DQN_fixed_order(descent)_result", seed, length, counter, DQN1_goal_count)
        Write_reward_text(log_dir + "/DQN_fixed_order(descent)_reward", seed, DQN1_train_reward, DQN1_best_episode)
        Write_log_text(log_dir + "/DQN_fixed_order(descent)_log", seed, DQN1_log)

        Plot_route.plot_route(start_point, goal_point, DQN1_Solution, seed_dir, grid_size, capacity)

        # DQN_random_order
        log_dir = dir_name + "/" + methods_list[5] + "/" + reward_dir
        seed_dir = dir_name + "/" + methods_list[5] + "/" + reward_dir + "/seed" + str(seed)

        DQN1_Solution, DQN1_grid, DQN1_goal_count, DQN1_train_reward, DQN1_best_episode, DQN1_log = DQN_random_order(
            grid_list, start_point,
            goal_point, capacity,
            seed, reward, overflow_penalty, sparce_reward)
        # DQN_evaluate
        length, counter = Evaluator(DQN1_Solution, DQN1_grid, capacity, grid_size)
        Write_to_text(log_dir + "/DQN_random_order_result", seed, length, counter, DQN1_goal_count)
        Write_reward_text(log_dir + "/DQN_random_order_reward", seed, DQN1_train_reward, DQN1_best_episode)
        Write_log_text(log_dir + "/DQN_random_order_log", seed, DQN1_log)

        Plot_route.plot_route(start_point, goal_point, DQN1_Solution, seed_dir, grid_size, capacity)

        # MADQN_2
        log_dir = dir_name + "/" + methods_list[6] + "/" + reward_dir
        seed_dir = dir_name + "/" + methods_list[6] + "/" + reward_dir + "/seed" + str(seed)

        MADQN2_Solution, MADQN2_grid, MADQN2_goal_count, MADQN2_train_reward, MADQN2_best_episode, MADQN2_log = MADQN.MADQN(
            grid_list, start_point, goal_point, capacity, seed, reward, overflow_penalty, sparce_reward)
        # MADQN_evaluate
        length, counter = Evaluator(MADQN2_Solution, MADQN2_grid, capacity, grid_size)
        Write_to_text(log_dir + "/MADQN_result", seed, length, counter, MADQN2_goal_count)
        Write_reward_text(log_dir + "/MADQN_reward", seed, MADQN2_train_reward, MADQN2_best_episode)
        Write_log_text(log_dir + "/MADQN_log", seed, MADQN2_log)

        Plot_route.plot_route(start_point, goal_point, MADQN2_Solution, seed_dir, grid_size, capacity)


if __name__ == "__main__":
    seed_num = list(range(0, 100, 100))
    print("seed: " + str(seed_num))
    seeds = seed_num

    for m in methods_list:
        os.mkdir(dir_name + "/" + m)
        for i in range(1):
            os.mkdir(dir_name + "/" + m + "/RewardNO." + str(i + 1))
            for s in seeds:
                os.mkdir(dir_name + "/" + m + "/RewardNO." + str(i + 1) +
                         "/seed" + str(s))

    for item in seeds:
        main(item)
