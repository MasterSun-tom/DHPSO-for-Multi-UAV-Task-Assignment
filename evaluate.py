import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import copy
from multiprocessing import Pool
from hpso import HPSO
from iilpso import IILPSO
from mpso import MPSO
from dhpso import DHPSO
from slpso import SLPSO
import json
import os

def get_files_name(size):
    folder_path = "case"
    file_names = os.listdir(folder_path)
    count = sum(1 for item in file_names if size in item) + 1
    file_name = f"{folder_path}/{size}_{count}.json"
    return file_name

def write_file(file_map, size):
    file_name = get_files_name(size)
    with open(file_name, 'w') as f:
        json.dump(file_map, f)

def read_file(filename):
    folder_name = "case"
    filename = f"{folder_name}/{filename}"
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

class Env:
    def __init__(self, vehicle_num, target_num, map_size, file_name="", visualized=True, time_cost=None, repeat_cost=None):
        self.vehicle_num = vehicle_num
        self.target_num = target_num
        self.vehicles_position = np.zeros(vehicle_num, dtype=np.int32)
        self.vehicles_speed = np.zeros(vehicle_num, dtype=np.int32)
        self.targets = np.zeros(shape=(target_num+1, 4), dtype=np.int32)
        
        if vehicle_num == 5:
            self.size = 'small'
        elif vehicle_num == 10:
            self.size = 'medium'
        elif vehicle_num == 15:
            self.size = 'large'
            
        self.map_size = map_size
        self.speed_range = [10, 15, 30]
        self.time_lim = self.map_size / self.speed_range[1]
        self.vehicles_lefttime = np.ones(vehicle_num, dtype=np.float32) * self.time_lim
        self.distant_mat = np.zeros((target_num+1, target_num+1), dtype=np.float32)
        self.total_reward = 0
        self.reward = 0
        self.visualized = visualized
        self.time = 0
        self.time_cost = time_cost
        self.repeat_cost = repeat_cost
        self.end = False
        self.assignment = [[] for _ in range(vehicle_num)]
        self.task_generator(file_name)

    def task_generator(self, file_name=""):
        self.vehicles_speed = np.zeros(self.vehicles_speed.shape[0], dtype=np.int32)
        self.targets = np.zeros(shape=(self.target_num + 1, 4), dtype=np.int32)
        self.distant_mat = np.zeros((self.target_num + 1, self.target_num + 1), dtype=np.float32)
        
        if len(file_name) == 0:
            for i in range(self.vehicles_speed.shape[0]):
                choose = random.randint(0, 2)
                self.vehicles_speed[i] = self.speed_range[choose]
            
            for i in range(self.targets.shape[0]-1):
                self.targets[i+1, 0] = random.randint(1, self.map_size) - 0.5 * self.map_size
                self.targets[i+1, 1] = random.randint(1, self.map_size) - 0.5 * self.map_size
                
                if self.target_num > 60:
                    self.targets[i+1, 2] = random.randint(15, 20)
                    self.targets[i+1, 3] = random.randint(15, 30)
                elif self.target_num > 30:
                    self.targets[i+1, 2] = random.randint(10, 15)
                    self.targets[i+1, 3] = random.randint(10, 25)
                else:
                    self.targets[i+1, 2] = random.randint(5, 10)
                    self.targets[i+1, 3] = random.randint(5, 20)
            
            for i in range(self.targets.shape[0]):
                for j in range(self.targets.shape[0]):
                    self.distant_mat[i, j] = np.linalg.norm(self.targets[i, :2] - self.targets[j, :2])
            
            self.targets_value = copy.deepcopy(self.targets[:, 2])
            tasks = [[] for _ in range(self.vehicle_num)]
            total_value = np.sum(self.targets_value)
            
            for i in range(1, self.target_num+1):
                best_task = -1
                best_task_score = 0
                for j in range(self.vehicle_num):
                    time_to_target = np.linalg.norm(self.targets[i, :2]) / self.vehicles_speed[j] + self.targets[i, 3]
                    if time_to_target <= self.time_lim:
                        task_score = self.targets_value[i] / time_to_target
                        if task_score > best_task_score:
                            best_task_score = task_score
                            best_task = j
                if best_task != -1:
                    tasks[best_task].append(i)
                    assigned_value = sum(self.targets_value[t] for t in tasks[best_task])
                    self.targets_value[i] *= (1 + assigned_value / total_value)
            
            self.assignment = tasks
            file_map = {
                "speed": self.vehicles_speed.tolist(),
                "targets": self.targets.tolist(),
                "tasks": tasks,
                "map": self.distant_mat.tolist()
            }
            write_file(file_map, self.size)
        else:
            file_map = read_file(file_name)
            self.vehicles_speed = np.array(file_map["speed"])
            self.targets = np.array(file_map["targets"])
            self.assignment = file_map["tasks"]
            self.distant_mat = np.array(file_map["map"])

    def step(self, action):
        count = 0
        for j in range(len(action)):
            k = action[j]
            delta_time = self.distant_mat[self.vehicles_position[j], k] / self.vehicles_speed[j] + self.targets[k, 3]
            self.vehicles_lefttime[j] = self.vehicles_lefttime[j] - delta_time
            if self.vehicles_lefttime[j] < 0:
                count += 1
                continue
            else:
                if k == 0:
                    self.reward = -self.repeat_cost
                else:
                    self.reward = self.targets[k, 2] - delta_time * self.time_cost + self.targets[k, 2]
                    if self.targets[k, 2] == 0:
                        self.reward = self.reward - self.repeat_cost
                    self.vehicles_position[j] = k
                    self.targets[k, 2] = 0
                self.total_reward += self.reward
            self.assignment[j].append(action)
        if count == len(action):
            self.end = True

    def run(self, assignment, algorithm, play, rond):
        self.assignment = assignment
        self.algorithm = algorithm
        self.play = play
        self.rond = rond
        self.get_total_reward()
        if self.visualized:
            self.visualize()

    def reset(self):
        self.vehicles_position = np.zeros(self.vehicles_position.shape[0], dtype=np.int32)
        self.vehicles_lefttime = np.ones(self.vehicles_position.shape[0], dtype=np.float32) * self.time_lim
        self.targets[:, 2] = self.targets_value
        self.total_reward = 0
        self.reward = 0
        self.end = False

    def get_total_reward(self):
        for i in range(len(self.assignment)):
            speed = self.vehicles_speed[i]
            for j in range(len(self.assignment[i])):
                position = self.targets[self.assignment[i][j], :4]
                self.total_reward += position[2]
                if j == 0:
                    self.vehicles_lefttime[i] -= np.linalg.norm(position[:2]) / speed + position[3]
                else:
                    self.vehicles_lefttime[i] -= np.linalg.norm(position[:2] - position_last[:2]) / speed + position[3]
                position_last = position
                if self.vehicles_lefttime[i] > self.time_lim:
                    self.end = True
                    break
            if self.end:
                self.total_reward = 0
                break

    def visualize(self):
        if self.assignment is None:
            plt.scatter(0, 0, s=200, c='k')
            plt.scatter(self.targets[1:, 0], self.targets[1:, 1], s=self.targets[1:, 2]*5, c='r', alpha=0.7)
            plt.title('Target distribution')
            plt.savefig(f'task_pic/{self.size}/{self.algorithm}-{self.play}-{self.rond}.png')
            plt.cla()
        else:
            plt.title(f'Task assignment by {self.algorithm}, total reward: {self.total_reward}')
            plt.scatter(0, 0, s=200, c='k')
            plt.scatter(self.targets[1:, 0], self.targets[1:, 1], s=self.targets[1:, 2]*5, c='r', alpha=0.7)

            for i, assignment in enumerate(self.assignment):
                color = plt.cm.jet(i / len(self.assignment))
                trajectory = np.array([[0, 0, 20]])
                
                for j, task in enumerate(assignment):
                    position = self.targets[task, :3]
                    trajectory = np.insert(trajectory, j+1, values=position, axis=0)

                for k in range(len(trajectory) - 1):
                    p0 = trajectory[k, :2]
                    p2 = trajectory[k+1, :2]
                    p1 = (p0 + p2) / 2
                    
                    t = np.linspace(0, 1, 100)[:, np.newaxis]
                    bezier = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2

                    plt.plot(bezier[:, 0], bezier[:, 1], '-', color=color, markersize=5)
                    
                    mid_point = bezier[len(bezier)//2]
                    dx = p2[0] - p0[0]
                    dy = p2[1] - p0[1]
                    direction = np.array([dx, dy], dtype=float)
                    direction /= np.linalg.norm(direction)
                    
                    plt.arrow(mid_point[0], mid_point[1], direction[0]*0.1, direction[1]*0.1, 
                             head_width=0.05, head_length=0.1, fc='k', ec='k')

            plt.savefig(f'task_pic/{self.size}/{self.algorithm}-{self.play}-{self.rond}.png')
            plt.cla()

def evaluate(vehicle_num, target_num, map_size, file_name=""):
    if vehicle_num == 5:
        size = 'small'
    elif vehicle_num == 10:
        size = 'medium'
    elif vehicle_num == 15:
        size = 'large'
        
    re_hpso = [[] for _ in range(10)]
    re_iilpso = [[] for _ in range(10)]
    re_mpso = [[] for _ in range(10)]
    re_pso2 = [[] for _ in range(10)]
    re_slpso = [[] for _ in range(10)]
    
    for i in range(10):
        env = Env(vehicle_num, target_num, map_size, file_name=file_name, visualized=True)
        for j in range(30):
            p = Pool(5)
            
            hpso = HPSO(vehicle_num, target_num, env.targets, env.vehicles_speed, env.time_lim)
            iilpso = IILPSO(vehicle_num, target_num, env.targets, env.vehicles_speed, env.time_lim, 20)
            mpso = MPSO(vehicle_num, target_num, env.targets, env.vehicles_speed, env.time_lim)
            pso2 = DHPSO(vehicle_num, target_num, env.targets, env.vehicles_speed, env.time_lim)
            slpso = SLPSO(vehicle_num, target_num, env.targets, env.vehicles_speed, env.time_lim)
            
            print(f"Running experiment {i+1}, round {j+1} for {size} size.")
            
            hpso_result = p.apply_async(hpso.run)
            iilpso_result = p.apply_async(iilpso.run)
            mpso_result = p.apply_async(mpso.run)
            pso2_result = p.apply_async(pso2.run)
            slpso_result = p.apply_async(slpso.run)
            p.close()
            p.join()
            
            ga_task_assignment = hpso_result.get()[0]
            env.run(ga_task_assignment, 'HPSO', i+1, j+1)
            re_hpso[i].append((env.total_reward, hpso_result.get()[1]))
            env.reset()
            
            aco_task_assignment = iilpso_result.get()[0]
            env.run(aco_task_assignment, 'IILPSO', i+1, j+1)
            re_iilpso[i].append((env.total_reward, iilpso_result.get()[1]))
            env.reset()
            
            pso_task_assignment = mpso_result.get()[0]
            env.run(pso_task_assignment, 'MPSO', i+1, j+1)
            re_mpso[i].append((env.total_reward, mpso_result.get()[1]))
            env.reset()
            
            pso2_task_assignment = pso2_result.get()[0]
            env.run(pso2_task_assignment, 'DHPSO', i+1, j+1)
            re_pso2[i].append((env.total_reward, pso2_result.get()[1]))
            env.reset()

            slpso_task_assignment = slpso_result.get()[0]
            env.run(slpso_task_assignment, 'SLPSO', i+1, j+1)
            re_slpso[i].append((env.total_reward, slpso_result.get()[1]))
            env.reset()
    
    x_index = np.arange(10)
    ymax11, ymax12, ymax21, ymax22 = [], [], [], []
    ymax31, ymax32, ymax41, ymax42 = [], [], [], []
    ymax51, ymax52 = [], []
    
    ymean11, ymean12, ymean21, ymean22 = [], [], [], []
    ymean31, ymean32, ymean41, ymean42 = [], [], [], []
    ymean51, ymean52 = [], []

    for i in range(10):
        tmp1 = [re_hpso[i][j][0] for j in range(len(re_hpso[i]))]
        tmp2 = [re_hpso[i][j][1] for j in range(len(re_hpso[i]))]
        ymax11.append(np.amax(tmp1))
        ymax12.append(np.amax(tmp2))
        ymean11.append(np.mean(tmp1))
        ymean12.append(np.mean(tmp2))
        
        tmp1 = [re_iilpso[i][j][0] for j in range(len(re_iilpso[i]))]
        tmp2 = [re_iilpso[i][j][1] for j in range(len(re_iilpso[i]))]
        ymax21.append(np.amax(tmp1))
        ymax22.append(np.amax(tmp2))
        ymean21.append(np.mean(tmp1))
        ymean22.append(np.mean(tmp2))
        
        tmp1 = [re_mpso[i][j][0] for j in range(len(re_mpso[i]))]
        tmp2 = [re_mpso[i][j][1] for j in range(len(re_mpso[i]))]
        ymax31.append(np.amax(tmp1))
        ymax32.append(np.amax(tmp2))
        ymean31.append(np.mean(tmp1))
        ymean32.append(np.mean(tmp2))
        
        tmp1 = [re_pso2[i][j][0] for j in range(len(re_pso2[i]))]
        tmp2 = [re_pso2[i][j][1] for j in range(len(re_pso2[i]))]
        ymax41.append(np.amax(tmp1))
        ymax42.append(np.amax(tmp2))
        ymean41.append(np.mean(tmp1))
        ymean42.append(np.mean(tmp2))
        
        tmp1 = [re_slpso[i][j][0] for j in range(len(re_slpso[i]))]
        tmp2 = [re_slpso[i][j][1] for j in range(len(re_slpso[i]))]
        ymax51.append(np.amax(tmp1))
        ymax52.append(np.amax(tmp2))
        ymean51.append(np.mean(tmp1))
        ymean52.append(np.mean(tmp2))
    
    plt.rcParams.update({'legend.fontsize': 'x-small'})
    
    plt.bar(x_index, ymax11, width=0.08, color='lightskyblue', label='MPSO_max_reward')
    plt.bar(x_index+0.08, ymax21, width=0.08, color='b', label='IILPSO_max_reward')
    plt.bar(x_index+0.16, ymax31, width=0.08, color='yellowgreen', label='HPSO_max_reward')
    plt.bar(x_index+0.24, ymax41, width=0.08, color='r', label='DHPSO_max_reward')
    plt.bar(x_index+0.32, ymax51, width=0.08, color='orange', label='SLPSO_max_reward')
    
    plt.xticks(x_index+0.2, x_index)
    plt.legend(bbox_to_anchor=(1.1, 1.1), loc='upper right')
    plt.title(f'max_reward_for_{size}_size')
    plt.savefig(f'max_reward_{size}.png')
    plt.cla()
    
    plt.bar(x_index, ymax12, width=0.08, color='lightskyblue', label='MPSO_max_time')
    plt.bar(x_index+0.08, ymax22, width=0.08, color='b', label='IILPSO_max_time')
    plt.bar(x_index+0.16, ymax32, width=0.08, color='yellowgreen', label='HPSO_max_time')
    plt.bar(x_index+0.24, ymax42, width=0.08, color='r', label='DHPSO_max_time')
    plt.bar(x_index+0.32, ymax52, width=0.08, color='orange', label='SLPSO_max_time')
    
    plt.xticks(x_index+0.2, x_index)
    plt.legend(bbox_to_anchor=(1.1, 1.1), loc='upper right')
    plt.title(f'max_time_for_{size}_size')
    plt.savefig(f'max_time_{size}.png')
    plt.cla()
    
    plt.bar(x_index, ymean11, width=0.08, color='lightskyblue', label='MPSO_mean_reward')
    plt.bar(x_index+0.08, ymean21, width=0.08, color='b', label='IILPSO_mean_reward')
    plt.bar(x_index+0.16, ymean31, width=0.08, color='yellowgreen', label='HPSO_mean_reward')
    plt.bar(x_index+0.24, ymean41, width=0.08, color='r', label='DHPSO_mean_reward')
    plt.bar(x_index+0.32, ymean51, width=0.08, color='orange', label='SLPSO_mean_reward')
    
    plt.xticks(x_index+0.2, x_index)
    plt.legend(bbox_to_anchor=(1.1, 1.1), loc='upper right')
    plt.title(f'mean_reward_for_{size}_size')
    plt.savefig(f'mean_reward_{size}.png')
    plt.cla()
    
    plt.bar(x_index, ymean12, width=0.08, color='lightskyblue', label='MPSO_mean_time')
    plt.bar(x_index+0.08, ymean22, width=0.08, color='b', label='IILPSO_mean_time')
    plt.bar(x_index+0.16, ymean32, width=0.08, color='yellowgreen', label='HPSO_mean_time')
    plt.bar(x_index+0.24, ymean42, width=0.08, color='r', label='DHPSO_mean_time')
    plt.bar(x_index+0.32, ymean52, width=0.08, color='orange', label='SLPSO_mean_time')
    
    plt.xticks(x_index+0.2, x_index)
    plt.legend(bbox_to_anchor=(1.1, 1.1), loc='upper right')
    plt.title(f'mean_time_{size}_size')
    plt.savefig(f'mean_time_{size}.png')
    plt.cla()
    
    t_hpso, r_hpso = [], []
    t_iilpso, r_iilpso = [], []
    t_mpso, r_mpso = [], []
    t_pso2, r_pso2 = [], []
    t_slpso, r_slpso = [], []

    for i in range(10):
        for j in range(30):
            if j < len(re_hpso[i]):
                t_hpso.append(re_hpso[i][j][1])
                r_hpso.append(re_hpso[i][j][0])
            if j < len(re_iilpso[i]):
                t_iilpso.append(re_iilpso[i][j][1])
                r_iilpso.append(re_iilpso[i][j][0])
            if j < len(re_mpso[i]):
                t_mpso.append(re_mpso[i][j][1])
                r_mpso.append(re_mpso[i][j][0])
            if j < len(re_pso2[i]):
                t_pso2.append(re_pso2[i][j][1])
                r_pso2.append(re_pso2[i][j][0])
            if j < len(re_slpso[i]):
                t_slpso.append(re_slpso[i][j][1])
                r_slpso.append(re_slpso[i][j][0])
            
    dataframe = pd.DataFrame({
        'hpso_time': t_hpso, 'hpso_reward': r_hpso, 
        'iilpso_time': t_iilpso, 'iilpso_reward': r_iilpso, 
        'mpso_time': t_mpso, 'mpso_reward': r_mpso,
        'dhpso_time': t_pso2, 'dhpso_reward': r_pso2,
        'slpso_time': t_slpso, 'slpso_reward': r_slpso
    })
    dataframe.to_csv(f'{size}_baseline_result.csv', sep=',')

if __name__ == '__main__':
    evaluate(5, 30, 5e3)
    evaluate(10, 60, 1e4)

    evaluate(15, 90, 1.5e4)
