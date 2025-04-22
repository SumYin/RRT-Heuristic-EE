# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy as sp
from tqdm import tqdm
import time
import json
import networkx as nx
import matplotlib as mpl

maxium_iterations = 5000 

# Define utility functions
def normalize_to_probability(heuristic_map):
    min_val, max_val = np.min(heuristic_map), np.max(heuristic_map)
    linear_norm = (heuristic_map - min_val) / (max_val - min_val)
    return linear_norm / np.sum(linear_norm)

def pick_weighted_random_cell(heuristic_map):
    normalized_map = normalize_to_probability(heuristic_map)
    flat_map = normalized_map.flatten()
    index = np.random.choice(len(flat_map), p=flat_map)
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

# Define RRT class
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

    def pick_random_point(self):
        return pick_pure_random_point()

    def plan(self, interactive=False):
        if interactive:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal', adjustable='box')
            ax.plot(self.start[0], self.start[1], 'go', markersize=8)  
            ax.plot(self.goal[0], self.goal[1], 'ro', markersize=8) 
            goal_area = plt.Circle(self.goal, self.goal_range, color='red', alpha=0.1)
            ax.add_artist(goal_area)

            # Plot obstacles
            for (x_min, y_min), (x_max, y_max) in self.obstacles:
                ax.fill([x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max], 'k', alpha=0.5)

        while np.linalg.norm(np.array(self.path[-1]) - np.array(self.goal)) > self.goal_range and self.itterations < maxium_iterations:
            self.itterations += 1
            new_point = self.pick_random_point()
            # Find the closest node from all nodes in the tree
            closest_node = min(self.tree.keys(), key=lambda node: np.linalg.norm(np.array(new_point) - np.array(node)))
            direction = np.array(new_point) - np.array(closest_node)
            direction = direction / np.linalg.norm(direction)
            new_point = tuple(np.array(closest_node) + direction * self.distance_unit)
            if is_legal_move(closest_node, new_point, self.obstacles):
                self.path.append(new_point)
                self.tree[new_point] = (closest_node, self.tree[closest_node][1] + 1)
                if interactive:
                    ax.plot([closest_node[0], new_point[0]], [closest_node[1], new_point[1]], 'b-', linewidth=0.5)
        
        if interactive:
            plt.ioff() 
            plt.close()  

        if self.itterations >= maxium_iterations:
            print("Maximum iterations reached without finding a path.")
            self.solution_length = -1

        else:    
            self.solution_length = self.tree[self.path[-1]][1]

    def solve(self, interactive=False):
        start_time = time.time()
        self.plan( interactive=interactive)
        end_time = time.time()
        self.execution_time = end_time - start_time

    def report(self):
        print(f"Execution Time: {self.execution_time:.6f} seconds")
        print(f"Iterations: {self.itterations}")
        print(f"Tree Size: {len(self.path)}")
        print(f"Path Length: {self.solution_length}")
        
        return [self.execution_time, self.itterations,len(self.path), self.solution_length]

    def graph(self, save=False, show=True):

        # Create a directed graph using networkx
        G = nx.DiGraph()
        for node, (parent, _) in self.tree.items():
            if parent is not None:
                G.add_edge(parent, node)

        pos = {node: node for node in G.nodes()}  
        node_sizes = [5 for _ in G.nodes()]  
        edge_colors = range(len(G.edges()))
        cmap = plt.cm.magma

        plt.figure(figsize=(16, 16))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect('equal', adjustable='box')

        # Plot start and goal points
        plt.plot(self.start[0], self.start[1], 'go', markersize=8) 
        plt.plot(self.goal[0], self.goal[1], 'ro', markersize=8) 
        # plot the goal area
        goal_area = plt.Circle(self.goal, self.goal_range, color='red', alpha=0.1)
        plt.gca().add_artist(goal_area)

        # Plot obstacles
        for (x_min, y_min), (x_max, y_max) in self.obstacles:
            plt.fill([x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max], 'k', alpha=0.5)

        # Draw nodes and edges (simple straight lines for all edges)
        edges = list(G.edges())
        for i, (start, end) in enumerate(edges):
            plt.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=cmap(i / len(edges)),
            linewidth=1
            )

        current = self.path[-1]
        if np.linalg.norm(np.array(current) - np.array(self.goal)) <= self.goal_range:
            path_edges = []
            while current is not None:
                parent = self.tree[current][0]
                if parent is not None:
                    path_edges.append((parent, current))
                current = parent
            for start, end in path_edges:
                plt.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    color="#00FA9A",
                    linewidth=1.5 
                )


        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="indigo")


        # Add colorbar
        pc = mpl.collections.PatchCollection([], cmap=cmap)
        pc.set_array(edge_colors)
        ax = plt.gca()
        cbar = plt.colorbar(pc, ax=ax, label="Edge Index", fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)

        plt.xlabel('X', fontsize=14)
        plt.ylabel('Y', fontsize=14)
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.4)
        plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        plt.title(f'{self.__class__.__name__} Path', fontsize=16, fontweight='bold')
        timestamp = int(time.time())
        if show:
            plt.show()
        if save:
            plt.savefig(f'graphs/{self.__class__.__name__}_path_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

# Define WRRT class
class WRRT(RRT):
    def __init__(self, start, goal, goal_range, distance_unit, obstacles, map_resolution):
        super().__init__(start, goal, goal_range, distance_unit, obstacles)
        self.map_resolution = map_resolution
        self.heuristic_map = self.generate_heuristic_map()

    def generate_heuristic_map(self):
        heuristicMap = np.zeros((self.map_resolution, self.map_resolution))
        for i in range(self.map_resolution):
            for j in range(self.map_resolution):
                heuristicMap[i, j] = euclidean_distance_heuristic(j/self.map_resolution, i/self.map_resolution, self.goal)
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
        super().graph(save=save, show=show)

    def report(self):
        super().report()

    def report_heuristic_map(self):
        plt.figure(figsize=(16, 16))
        plt.imshow(self.heuristic_map, cmap='magma', interpolation='nearest', extent=[0, 1, 0, 1], origin='lower')
        
        # Plot obstacles
        for (x_min, y_min), (x_max, y_max) in self.obstacles:
            plt.fill([x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max], 'k', alpha=0.5)

        # Plot start and goal
        plt.plot(self.start[0], self.start[1], 'go', label='Start')  
        plt.plot(self.goal[0], self.goal[1], 'ro', label='Goal')
        goal_area = plt.Circle(self.goal, self.goal_range, color='red', alpha=0.1)
        plt.gca().add_artist(goal_area)

        # Add colorbar and labels
        plt.colorbar(label='Heuristic Value', fraction=0.046, pad=0.04)
        plt.title(f'Heuristic Map', fontsize=16, fontweight='bold')
        plt.xlabel('X', fontsize=14)
        plt.ylabel('Y', fontsize=14)
        plt.legend(fontsize=12)

        # Save and close the plot
        timestamp = int(time.time())
        # plt.show()
        plt.savefig(f'graphs/WRRT_heuristic_map_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

# Main execution
if __name__ == "__main__":
    with open('scenarios.json') as f:
        scenarios = json.load(f)

    # single runs for each scenario and testing
    for scenario_name, scenario in scenarios.items():
        for run in range(2):
            print(f"Run {run + 1} for scenario: {scenario_name}")
            start = tuple(scenario['start'])
            goal = tuple(scenario['goal'])
            map_resolution = scenario['mapResolution']
            goal_range = scenario['goalRange']
            distance_unit = scenario['distanceUnit']
            obstacles = scenario['obstacles']

            # Initialize RRT and WRRT
            rrt = RRT(start, goal, goal_range, distance_unit, obstacles)
            wrrt = WRRT(start, goal, goal_range, distance_unit, obstacles, map_resolution)

            # Run RRT
            print(f"Running RRT for scenario: {scenario_name}")
            rrt.solve(interactive=False)
            rrt.report()
            rrt.graph(save=True, show=False)
            
            # Run WRRT
            print(f"Running WRRT for scenario: {scenario_name}")
            wrrt.solve(interactive=False)
            wrrt.report()
            wrrt.graph(save=True, show=False)
        # wrrt.report_heuristic_map()
    # 