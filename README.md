# DHPSO Algorithm (Dynamic Topology Hybrid PSO Algorithm)

## 1. Algorithm Introduction

DHPSO (Dynamic Topology Hybrid PSO Algorithm) is an improved particle swarm optimization algorithm designed specifically for multi-UAV task allocation problems. The algorithm combines distributed computing concepts and hybrid optimization strategies, enabling efficient solution of task allocation scenarios of different scales.

### 1.1 Core Features

- **Dynamic Topology Adjustment**: The algorithm switches between ring topology and fully connected topology during iterations to balance local and global search capabilities
- **Hybrid Inertia Weight Strategy**: Uses a combination of linear and exponential inertia weight adjustment to improve convergence speed and search precision
- **Mutation Operation**: Introduces mutation mechanism to enhance population diversity and avoid local optima
- **Distributed Computing**: Supports multi-process parallel computing to improve solving efficiency in large-scale scenarios

## 2. Directory Structure

```
DHPSO/
├── case/                  # Task scenario configuration files
│   ├── large_1.json       # Large scenario configuration 1
│   ├── large_2.json       # Large scenario configuration 2
│   ├── medium_1.json      # Medium scenario configuration 1
│   ├── medium_2.json      # Medium scenario configuration 2
│   ├── small_1.json       # Small scenario configuration 1
│   └── small_2.json       # Small scenario configuration 2
├── dhpso.py               # DHPSO algorithm implementation
├── evaluate.py            # Evaluation system implementation
└── task_pic/              # Task allocation visualization results (generated after running)
    ├── small/             # Small scenario allocation diagrams
    ├── medium/            # Medium scenario allocation diagrams
    └── large/             # Large scenario allocation diagrams
```

## 3. System Architecture

### 3.1 Algorithm Module

The core implementation of the DHPSO algorithm is located in the `dhpso.py` file, which mainly includes parameter initialization, topology management, fitness calculation, particle update, mutation operation, and result extraction.

### 3.2 Evaluation System

The evaluation system is implemented in the `evaluate.py` file, which mainly includes task scenario generation, multi-algorithm comparison, result visualization, and data statistics.

## 4. Algorithm Principles

### 4.1 Encoding Method

DHPSO uses real-number encoding, where each particle consists of two parts: task sequence encoding and UAV allocation encoding, which determine the task execution order and allocation scheme respectively.

### 4.2 Core Mechanisms

- **Fitness Function**: Considers the trade-off between task value and execution time to optimize task allocation schemes
- **Dynamic Topology Adjustment**: Switches between ring topology and fully connected topology to balance local and global search capabilities
- **Hybrid Inertia Weight Strategy**: Uses a combination of linear and exponential inertia weight adjustment to improve convergence speed and search precision
- **Mutation Operation**: Introduces mutation mechanism to enhance population diversity and avoid local optima

## 5. Evaluation System Usage

### 5.1 Scenario Configuration

The evaluation system supports three scales of task scenarios:

| Scenario Scale | Number of UAVs | Number of Targets | Map Size |
|---------------|----------------|-------------------|----------|
| Small         | 5              | 30                | 5000     |
| Medium        | 10             | 60                | 10000    |
| Large         | 15             | 90                | 15000    |

### 5.2 Running Evaluation

Command to run the evaluation system:

```bash
python evaluate.py
```

The system will automatically generate three scales of scenarios and test and compare all algorithms.

### 5.3 Result Output

After evaluation, the system will generate the following outputs:

1. **Task Allocation Diagrams**: Saved in the `task_pic` directory, categorized by scenario scale
2. **Performance Comparison Charts**: Including maximum reward, average reward, maximum time, average time and other indicators
3. **CSV Data Files**: Saving detailed performance data of each algorithm

## 6. Summary

DHPSO is an efficient multi-UAV task allocation algorithm with the following features:

1. **Dynamic Topology Adjustment**: Balances local and global search capabilities by switching topologies
2. **Hybrid Inertia Weight Strategy**: Improves algorithm convergence speed and search precision
3. **Mutation Operation**: Enhances population diversity and avoids local optima
4. **Distributed Computing**: Supports parallel computing to improve solving efficiency
5. **Multi-scale Adaptation**: Effectively handles various task scenarios from small to large scales

Through comparison with other algorithms, DHPSO has demonstrated good performance in both task allocation quality and computational efficiency, providing an effective solution for multi-UAV collaborative task allocation problems.


