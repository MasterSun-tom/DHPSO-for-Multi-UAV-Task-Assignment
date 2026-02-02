import numpy as np
import random
import math
import cmath
import time
import os

class DHPSO:
    def __init__(self, uav_num, target_num, targets, vehicles_speed, time_lim):
        self.uav_num = uav_num
        self.dim = target_num
        self.targets = targets
        self.vehicles_speed = vehicles_speed
        self.time_all = time_lim
        
        self.pN = 200
        self.max_iter = 230
        self.Distance = np.zeros((target_num + 1, target_num + 1))
        self.Value = np.zeros(target_num + 1)
        self.Stay_time = []
        
        self.w = 0.7
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.7
        self.r2 = 0.3
        self.k = 0
        self.wini = 0.9
        self.wend = 0.4
        
        self.X = np.zeros((self.pN, self.dim + self.uav_num - 1))
        self.V = np.zeros((self.pN, self.dim + self.uav_num - 1))
        self.pbest = np.zeros((self.pN, self.dim + self.uav_num - 1))
        self.gbest = np.zeros((1, self.dim + self.uav_num - 1))
        self.gbest_ring = np.zeros((self.pN, self.dim + self.uav_num - 1))
        self.p_fit = np.zeros(self.pN)
        self.fit = 0
        self.ring = [[] for _ in range(self.pN)]
        self.ring_fit = np.zeros(self.pN)
        
        self.p1 = 0.4
        self.p2 = 0.4
        self.p3 = 0.4
        self.uav_best = []
        self.time_out = np.zeros(self.uav_num)
        self.cal_time = 0
        
        self.topology = 'ring'
        self.topology_change_iter = 25

    def fun_get_initial_parameter(self):
        self.max_iter = 230
        
        Targets = self.targets
        self.Stay_time = Targets[:, 3]
        self.Distance = np.zeros((self.dim + 1, self.dim + 1))
        self.Value = np.zeros(self.dim + 1)
        
        for i in range(self.dim + 1):
            self.Value[i] = Targets[i, 2]
            for j in range(i):
                dist = (Targets[i, 0] - Targets[j, 0])**2 + (Targets[i, 1] - Targets[j, 1])**2
                self.Distance[i][j] = math.sqrt(dist)
                self.Distance[j][i] = self.Distance[i][j]

    def fun_Transfer(self, X):
        X1 = X[0:self.dim]
        X_path = []
        l1 = len(X1)
        for i in range(l1):
            m = X1[i] * (self.dim - i)
            m = math.floor(m)
            X_path.append(m)

        X2 = X[self.dim:]
        l1 = len(X2)
        X_rank = []
        for i in range(l1):
            m = X2[i] * (self.dim + 1)
            m1 = math.floor(m)
            X_rank.append(m1)

        c = sorted(X_rank)
        l1 = len(c)
        Rank = [0]
        for i in range(l1):
            Rank.append(c[i])
        Rank.append(self.dim)

        Sep = []
        for i in range(l1 + 1):
            sep = Rank[i + 1] - Rank[i]
            Sep.append(sep)

        return X_path, Sep

    def position(self, X):
        Position_All = list(range(1, self.dim + 1))
        X2 = []
        for i in range(self.dim):
            m1 = int(X[i])
            X2.append(Position_All[m1])
            del Position_All[m1]
        return X2

    def function(self, X):
        X_path, Sep = self.fun_Transfer(X)
        X = self.position(X_path)
        UAV = []
        l = 0
        for i in range(self.uav_num):
            UAV.append([])
            k = Sep[i]
            for j in range(k):
                UAV[i].append(X[l])
                l = l + 1

        fitness = 0
        for i in range(self.uav_num):
            k = Sep[i]
            t = 0
            m2 = 0
            for j in range(k):
                m1 = UAV[i][j]
                if j == 0:
                    t = t + self.Distance[0, m1] / self.vehicles_speed[i] + self.Stay_time[m1]
                else:
                    m2 = UAV[i][j - 1]
                    t = t + self.Distance[m2][m1] / self.vehicles_speed[i] + self.Stay_time[m1]
                if t <= self.time_all:
                    fitness += self.Value[m1] / t
        return fitness

    def variation_fun(self):
        p1 = np.random.uniform(0, 1)
        if p1 < self.p1:
            for i in range(self.pN):
                p2 = np.random.uniform(0, 1)
                if p2 < self.p2:
                    m = int(self.p3 * (self.dim + self.uav_num - 1))
                    for j in range(m):
                        replace_position = math.floor(np.random.uniform(0, 1) * (self.dim + self.uav_num - 1))
                        replace_value = np.random.uniform(0, 1)
                        self.X[i][replace_position] = replace_value

            for i in range(self.pN):
                temp = self.function(self.X[i])
                self.ring_fit[i] = temp
                if temp > self.p_fit[i]:
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]
                    if self.p_fit[i] > self.fit:
                        self.gbest = self.X[i]
                        self.fit = self.p_fit[i]

    def update_topology(self, iter_num):
        if iter_num % self.topology_change_iter == 0:
            if self.topology == 'ring':
                self.topology = 'fully_connected'
            else:
                self.topology = 'ring'
            self.update_ring_topology() if self.topology == 'ring' else self.update_fully_connected_topology()

    def update_ring_topology(self):
        self.ring = [[] for _ in range(self.pN)]
        for i in range(self.pN):
            for j in range(1, self.uav_num + 1):
                self.ring[i].append((i + j) % self.pN)

    def update_fully_connected_topology(self):
        self.ring = [[] for _ in range(self.pN)]
        for i in range(self.pN):
            for j in range(self.pN):
                if i != j:
                    self.ring[i].append(j)

    def init_Population(self):
        for i in range(self.pN):
            x = np.random.uniform(0, 1, self.dim + self.uav_num - 1)
            self.X[i, :] = x
            v = np.random.uniform(0, 0.4, self.dim + self.uav_num - 1)
            self.V[i, :] = v
            self.pbest[i] = self.X[i]

            tmp = self.function(self.X[i])
            self.p_fit[i] = tmp
            if tmp > self.fit:
                self.fit = tmp
                self.gbest = self.X[i]

        phi = self.c1 + self.c2
        k = abs(phi * phi - 4 * phi)
        k = cmath.sqrt(k)
        k = abs(2 - phi - k)
        k = 2 / k
        self.k = k

        self.update_ring_topology()

    def iterator(self):
        fitness = []
        fitness_old = 0
        k = 0
        
        for t in range(self.max_iter):
            self.update_topology(t)
            
            linear_component = (self.wini - self.wend) / 2 * (self.max_iter - t) / self.max_iter
            exponential_component = np.exp(-(self.wini - self.wend) * t)
            w = linear_component + exponential_component
            self.w = w
            
            self.variation_fun()
            l1 = len(self.ring[0])
            
            if l1 < self.pN:
                if not (t % 2):
                    k = k + 1
                    for i in range(self.pN):
                        m1 = i - k
                        if m1 < 0:
                            m1 = self.pN + m1
                        m2 = i + k
                        if m2 > self.pN - 1:
                            m2 = m2 - self.pN
                        self.ring[i].append(m1)
                        self.ring[i].append(m2)

                l_ring = len(self.ring[0])
                for i in range(self.pN):
                    fitness1 = 0
                    for j in range(l_ring):
                        m1 = self.ring[i][j]
                        fitness2 = self.ring_fit[m1]
                        if fitness2 > fitness1:
                            self.gbest_ring[i] = self.X[m1]
                            fitness1 = fitness2

                for i in range(self.pN):
                    self.V[i] = self.k * (self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i])) + \
                                self.c2 * self.r2 * (self.gbest_ring[i] - self.X[i])
                    self.X[i] = self.X[i] + self.V[i]

            else:
                for i in range(self.pN):
                    self.V[i] = self.k * (self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i])) + \
                                self.c2 * self.r2 * (self.gbest - self.X[i])
                    self.X[i] = self.X[i] + self.V[i]

            for i in range(self.pN):
                for j in range(self.dim + self.uav_num - 1):
                    if self.X[i][j] >= 1:
                        self.X[i][j] = 0.999
                    if self.X[i][j] < 0:
                        self.X[i][j] = 0

            for i in range(self.pN):
                temp = self.function(self.X[i])
                self.ring_fit[i] = temp
                if temp > self.p_fit[i]:
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]
                    if self.p_fit[i] > self.fit:
                        self.gbest = self.X[i]
                        self.fit = self.p_fit[i]
                        self.uav_best = self.fun_Data()

            fitness.append(self.fit)
            if self.fit == fitness_old:
                continue
            else:
                fitness_old = self.fit
        return fitness

    def fun_Data(self):
        X_path, Sep = self.fun_Transfer(self.gbest)
        X = self.position(X_path)

        UAV = []
        l = 0
        for i in range(self.uav_num):
            UAV.append([])
            k = Sep[i]
            for j in range(k):
                UAV[i].append(X[l])
                l = l + 1

        UAV_Out = []
        for i in range(self.uav_num):
            k = Sep[i]
            t = 0
            UAV_Out.append([])
            for j in range(k):
                m1 = UAV[i][j]
                if j == 0:
                    t = t + self.Distance[0, m1] / self.vehicles_speed[i] + self.Stay_time[m1]
                else:
                    m2 = UAV[i][j - 1]
                    t = t + self.Distance[m2][m1] / self.vehicles_speed[i] + self.Stay_time[m1]
                if t <= self.time_all:
                    UAV_Out[i].append(m1)
                    self.time_out[i] = t
        return UAV_Out

    def run(self):
        print(f"swpso start, pid: {os.getpid()}")
        start_time = time.time()
        self.fun_get_initial_parameter()
        self.init_Population()
        fitness = self.iterator()
        end_time = time.time()
        print(f"swpso result: {self.uav_best}")
        print(f"swpso time: {end_time - start_time}")
        return self.uav_best, end_time - start_time