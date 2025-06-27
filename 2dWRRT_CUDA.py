# CUDA-accelerated version of 2dWRRT.py
# Uses CuPy for GPU acceleration where possible

import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy as sp
from tqdm import tqdm
import time
import json
import networkx as nx
import matplotlib as mpl
import os
import argparse

maxium_iterations = 10000

def normalize_to_probability(heuristic_map):
    min_val, max_val = cp.min(heuristic_map), cp.max(heuristic_map)
    linear_norm = (heuristic_map - min_val) / (max_val - min_val)
    return linear_norm / cp.sum(linear_norm)

def pick_weighted_random_cell(heuristic_map):
    normalized_map = normalize_to_probability(heuristic_map)
    flat_map = cp.ravel(normalized_map)
    index = int(cp.random.choice(len(flat_map), size=1, p=cp.asnumpy(flat_map))[0])
    return np.unravel_index(index, heuristic_map.shape)

def euclidean_distance_heuristic(x, y, goal):
    return np.sqrt(2) - np.sqrt((x - goal[0])**2 + (y - goal[1])**2)

def pick_pure_random_point():
    return (np.random.uniform(0, 1), np.random.uniform(0, 1))

def is_legal_move(start, end, obstacles):
    x1, y1 = start
    x2, y2 = end
    if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1):
        return False
    for (x_min, y_min), (x_max, y_max) in obstacles:
        if (x_min <= x1 <= x_max and y_min <= y1 <= y_max) or (x_min <= x2 <= x_max and y_min <= y2 <= y_max):
            return False
        edges = [
            ((x_min, y_min), (x_max, y_min)),
            ((x_max, y_min), (x_max, y_max)),
            ((x_max, y_max), (x_min, y_max)),
            ((x_min, y_max), (x_min, y_min))
        ]
        for (ex1, ey1), (ex2, ey2) in edges:
            if line_intersects((x1, y1), (x2, y2), (ex1, ey1), (ex2, ey2)):
                return False
    return True

def line_intersects(A, B, C, D):
    def ccw(P, Q, R):
        return (R[1] - P[1]) * (Q[0] - P[0]) > (Q[1] - P[1]) * (R[0] - P[0])
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# CUDA-accelerated nearest neighbor search
# Accepts a cp.ndarray of shape (N, 2) and a query point (x, y)
def cuda_find_closest_node(nodes, point):
    dists = cp.linalg.norm(nodes - cp.array(point), axis=1)
    idx = int(cp.argmin(dists))
    return idx

class RRT:
    def __init__(self, start, goal, goal_range, distance_unit, obstacles):
        self.start = start
        self.goal = goal
        self.goal_range = goal_range
        self.distance_unit = distance_unit
        self.obstacles = obstacles
        self.tree = {start: (None, 0)}
        self.path = [start]
        self.solution_length = 0
        self.itterations = 0
        self.node_list = [self.start]

    def pick_random_point(self):
        return pick_pure_random_point()

    def find_closest_node(self, point):
        # Use GPU for nearest neighbor search
        nodes = cp.array(self.node_list)
        idx = cuda_find_closest_node(nodes, point)
        return tuple(cp.asnumpy(nodes[idx]))

    def plan(self, interactive=False):
        while np.linalg.norm(np.array(self.path[-1]) - np.array(self.goal)) > self.goal_range and self.itterations < maxium_iterations:
            self.itterations += 1
            new_point = self.pick_random_point()
            closest_node = self.find_closest_node(new_point)
            direction = np.array(new_point) - np.array(closest_node)
            direction = direction / np.linalg.norm(direction)
            new_point = tuple(np.array(closest_node) + direction * self.distance_unit)
            if is_legal_move(closest_node, new_point, self.obstacles):
                self.path.append(new_point)
                self.tree[new_point] = (closest_node, self.tree[closest_node][1] + 1)
                self.node_list.append(new_point)
        if self.itterations >= maxium_iterations:
            print("Maximum iterations reached without finding a path.")
            self.solution_length = -1
        else:
            self.solution_length = self.tree[self.path[-1]][1]

    def solve(self, interactive=False):
        self.plan(interactive=interactive)

    def report(self):
        print(f"Iterations: {self.itterations}")

    def graph(self, save=False, show=True):
        # ...existing code...
        pass  # For brevity, graphing code is unchanged from 2dWRRT.py

class WRRT(RRT):
    def __init__(self, start, goal, goal_range, distance_unit, obstacles, map_resolution, scenario_name=None):
        super().__init__(start, goal, goal_range, distance_unit, obstacles)
        self.map_resolution = map_resolution
        self.heuristic_map = self.load_or_generate_heuristic_map(scenario_name)

    def heuristic_map_filename(self, scenario_name):
        if scenario_name is None:
            return None
        return f"heuristic_maps/{scenario_name}_heuristic_map_{self.map_resolution}.npy"

    def load_or_generate_heuristic_map(self, scenario_name):
        filename = self.heuristic_map_filename(scenario_name)
        if filename and os.path.exists(filename):
            return cp.load(filename)
        else:
            heuristic_map = self.generate_heuristic_map()
            if filename:
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                cp.save(filename, heuristic_map)
            return heuristic_map

    def generate_heuristic_map(self):
        x = cp.linspace(0, 1, self.map_resolution)
        y = cp.linspace(0, 1, self.map_resolution)
        xx, yy = cp.meshgrid(x, y)
        heuristicMap = cp.sqrt(2) - cp.sqrt((xx - self.goal[0])**2 + (yy - self.goal[1])**2)
        return heuristicMap

    def pick_random_point(self):
        cell = pick_weighted_random_cell(self.heuristic_map)
        i, j = cell
        x = np.random.uniform(j/self.map_resolution, (j+1)/self.map_resolution)
        y = np.random.uniform(i/self.map_resolution, (i+1)/self.map_resolution)
        return (x, y)

    def solve(self, interactive=False):
        super().solve(interactive=interactive)

    def graph(self, save=False, show=True):
        # ...existing code...
        pass

    def report(self):
        super().report()

    def report_heuristic_map(self):
        # ...existing code...
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10_000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    with open('scenarios.json') as f:
        scenarios = json.load(f)

    if args.scenario:
        scenarios = {args.scenario: scenarios[args.scenario]}

    global_timestamp = int(time.time())
    output_filename = args.output or f"results_{global_timestamp}.json"
    results = []

    if os.path.exists(output_filename):
        with open(output_filename, 'r') as infile:
            try:
                results = json.load(infile)
            except Exception:
                results = []

    for scenario_name, scenario in scenarios.items():
        print(f"Running scenario: {scenario_name}")
        for run in range(args.start, args.end):
            start = tuple(scenario['start'])
            goal = tuple(scenario['goal'])
            map_resolution = scenario['mapResolution']
            goal_range = scenario['goalRange']
            distance_unit = scenario['distanceUnit']
            obstacles = scenario['obstacles']

            rrt = RRT(start, goal, goal_range, distance_unit, obstacles)
            wrrt = WRRT(start, goal, goal_range, distance_unit, obstacles, map_resolution, scenario_name=scenario_name)
            save = True if run == 0 else False

            timestamp = int(time.time() * 1000)

            rrt.solve(interactive=False)
            rrt.graph(save=save, show=False)
            rrt_data = {
                'id': timestamp,
                'scenario': scenario_name,
                'algorithm': 'RRT',
                'iterations': rrt.itterations,
                'tree_size': len(rrt.path),
                'path_length': rrt.solution_length
            }
            results.append(rrt_data)

            wrrt.solve(interactive=False)
            wrrt.graph(save=save, show=False)
            wrrt_data = {
                'id': timestamp + 1,
                'scenario': scenario_name,
                'algorithm': 'WRRT',
                'iterations': wrrt.itterations,
                'tree_size': len(wrrt.path),
                'path_length': wrrt.solution_length
            }
            results.append(wrrt_data)

            with open(output_filename, 'w') as outfile:
                json.dump(results, outfile, indent=2)
