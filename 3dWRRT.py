# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm
import time
import json
import networkx as nx
import matplotlib as mpl

maxium_iterations = 5000  

def normalize_to_probability(heuristic_map):
    """Normalizes a 3D heuristic map to a probability distribution."""
    min_val, max_val = np.min(heuristic_map), np.max(heuristic_map)
    if max_val == min_val:
        num_elements = heuristic_map.size
        if num_elements > 0:
            return np.ones_like(heuristic_map) / num_elements
        else:
            return heuristic_map 
    linear_norm = (heuristic_map - min_val) / (max_val - min_val)
    sum_norm = np.sum(linear_norm)
    if sum_norm == 0:
        non_zero_indices = np.nonzero(linear_norm)
        if len(non_zero_indices[0]) > 0:
            prob_map = np.zeros_like(linear_norm)
            prob_map[non_zero_indices] = 1.0 / len(non_zero_indices[0])
            return prob_map
        else: 
             num_elements = heuristic_map.size
             if num_elements > 0:
                 return np.ones_like(heuristic_map) / num_elements
             else:
                 return heuristic_map
    return linear_norm / sum_norm

def pick_weighted_random_cell(heuristic_map):
    """Picks a random cell (i, j, k) from a 3D map based on weights."""
    if np.any(heuristic_map < 0) or np.sum(heuristic_map) <= 0:
         print("Warning: Invalid heuristic map encountered. Falling back to uniform sampling.")
         idx = np.random.choice(heuristic_map.size)
         return np.unravel_index(idx, heuristic_map.shape)

    normalized_map = normalize_to_probability(heuristic_map)
    flat_map = normalized_map.flatten()
    flat_map /= flat_map.sum()
    try:
        index = np.random.choice(len(flat_map), p=flat_map)
    except ValueError as e:
        print(f"Error in np.random.choice: {e}")
        print(f"Sum of probabilities: {np.sum(flat_map)}")
        index = np.random.choice(len(flat_map))

    return np.unravel_index(index, heuristic_map.shape)

def euclidean_distance_heuristic_3d(x, y, z, goal):
    """Calculates a heuristic value based on 3D Euclidean distance to the goal."""
    max_dist = np.sqrt(3)
    distance = np.sqrt((x - goal[0])**2 + (y - goal[1])**2 + (z - goal[2])**2)
    return max_dist - distance

def pick_pure_random_point_3d():
    """Picks a random point within the unit cube [0,1]x[0,1]x[0,1]."""
    return (np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))

def is_point_inside_obstacle(point, obstacle):
    """Checks if a 3D point is inside a cubic obstacle."""
    x, y, z = point
    (x_min, y_min, z_min), (x_max, y_max, z_max) = obstacle
    return x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max

def line_segment_intersects_cube(p1, p2, cube_min, cube_max):
    """Checks if a line segment (p1, p2) intersects with a 3D cube (AABB)."""
    p1 = np.array(p1)
    p2 = np.array(p2)
    cube_min = np.array(cube_min)
    cube_max = np.array(cube_max)
    direction = p2 - p1
    t_near = -np.inf
    t_far = np.inf

    for i in range(3):
        if abs(direction[i]) < 1e-9: 
            if p1[i] < cube_min[i] or p1[i] > cube_max[i]:
                return False
        else:
            t1 = (cube_min[i] - p1[i]) / direction[i]
            t2 = (cube_max[i] - p1[i]) / direction[i]

            if t1 > t2:
                t1, t2 = t2, t1

            t_near = max(t_near, t1)
            t_far = min(t_far, t2)

            if t_near > t_far:
                return False

    # The line intersects the cube's volume. Now check if the intersection
    # overlaps with the line segment's parameter range [0, 1].
    # Intersection occurs if the interval [t_near, t_far] overlaps with [0, 1]
    return t_near <= 1 and t_far >= 0


def is_legal_move_3d(start, end, obstacles):
    """Checks if a move from start to end point is legal in 3D space."""
    x1, y1, z1 = start
    x2, y2, z2 = end

    if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= z1 <= 1 and
            0 <= x2 <= 1 and 0 <= y2 <= 1 and 0 <= z2 <= 1):
        return False

    for obs_min, obs_max in obstacles:
        if is_point_inside_obstacle(start, (obs_min, obs_max)):
            return False
        if start != end and is_point_inside_obstacle(end, (obs_min, obs_max)):
            return False

    for obs_min, obs_max in obstacles:
        if line_segment_intersects_cube(start, end, obs_min, obs_max):
            return False

    return True

def draw_cube(ax, center, size, color='k', alpha=0.1):
    ox, oy, oz = center
    l = size / 2.0
    points = np.array([
        [ox-l, oy-l, oz-l], [ox+l, oy-l, oz-l], [ox+l, oy+l, oz-l], [ox-l, oy+l, oz-l],
        [ox-l, oy-l, oz+l], [ox+l, oy-l, oz+l], [ox+l, oy+l, oz+l], [ox-l, oy+l, oz+l]
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    # Plot the edges
    for i, j in edges:
        ax.plot([points[i, 0], points[j, 0]],
                [points[i, 1], points[j, 1]],
                [points[i, 2], points[j, 2]], color=color, alpha=alpha*2, linewidth=0.5) 


def draw_obstacles_3d(ax, obstacles, color='k', alpha=0.2):
    """Draws all obstacles (cubes) on the 3D plot."""
    for (x_min, y_min, z_min), (x_max, y_max, z_max) in obstacles:
        center = ((x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2)
        verts = [
            (x_min, y_min, z_min), (x_max, y_min, z_min), (x_max, y_max, z_min), (x_min, y_max, z_min),
            (x_min, y_min, z_max), (x_max, y_min, z_max), (x_max, y_max, z_max), (x_min, y_max, z_max)
        ]
        faces = [
            [verts[0], verts[1], verts[2], verts[3]], # Bottom
            [verts[4], verts[5], verts[6], verts[7]], # Top
            [verts[0], verts[1], verts[5], verts[4]], # Front
            [verts[2], verts[3], verts[7], verts[6]], # Back
            [verts[1], verts[2], verts[6], verts[5]], # Right
            [verts[0], verts[3], verts[7], verts[4]]  # Left
        ]
        poly = Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='k', alpha=alpha)
        ax.add_collection3d(poly)


# --- RRT Class (Adapted for 3D) ---
class RRT3D:
    def __init__(self, start, goal, goal_range, distance_unit, obstacles):
        self.start = tuple(start)
        self.goal = tuple(goal) 
        self.goal_range = goal_range
        self.distance_unit = distance_unit
        self.obstacles = obstacles
        self.tree = {self.start: (None, 0)}
        self.nodes = [self.start]
        self.solution_node = None 
        self.solution_length = -1
        self.iterations = 0
        self.execution_time = 0

    def pick_random_point(self):
        return pick_pure_random_point_3d()

    def find_closest_node(self, point):
        point_arr = np.array(point)
        min_dist = float('inf')
        closest_node = None
        for node in self.tree:
            dist = np.linalg.norm(point_arr - np.array(node))
            if dist < min_dist:
                min_dist = dist
                closest_node = node
        return closest_node

    def steer(self, from_node, to_point):
        """Steers from 'from_node' towards 'to_point' by 'distance_unit'."""
        from_arr = np.array(from_node)
        to_arr = np.array(to_point)
        direction = to_arr - from_arr
        dist = np.linalg.norm(direction)

        if dist < 1e-9:
            return tuple(from_arr) 

        direction = direction / dist

        # Calculate the new point
        step_dist = min(self.distance_unit, dist)
        new_point_arr = from_arr + direction * step_dist
        return tuple(new_point_arr)

    def plan(self, interactive=False):
        """Plans the path using the RRT algorithm."""
        ax = None
        if interactive:
            plt.ion()
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{self.__class__.__name__} Planning')
            ax.scatter(*self.start, c='g', marker='o', s=100, label='Start')
            ax.scatter(*self.goal, c='r', marker='*', s=100, label='Goal')
            ax.scatter(*self.goal, s=(self.goal_range*100)**2, c='r', alpha=0.1)

            # Plot obstacles
            draw_obstacles_3d(ax, self.obstacles, alpha=0.3)
            plt.legend()


        goal_reached = False
        for i in range(maxium_iterations):
            self.iterations += 1
            random_point = self.pick_random_point()

            closest_node = self.find_closest_node(random_point)

            new_point = self.steer(closest_node, random_point)

            if is_legal_move_3d(closest_node, new_point, self.obstacles):
                self.tree[new_point] = (closest_node, self.tree[closest_node][1] + 1)
                self.nodes.append(new_point)

                if interactive and ax:
                    ax.plot([closest_node[0], new_point[0]],
                            [closest_node[1], new_point[1]],
                            [closest_node[2], new_point[2]], 'b-', linewidth=0.5, alpha=0.7)
                    plt.draw()

                if np.linalg.norm(np.array(new_point) - np.array(self.goal)) <= self.goal_range:
                    self.solution_node = new_point
                    self.solution_length = self.tree[new_point][1]
                    goal_reached = True
                    print(f"Goal reached after {self.iterations} iterations.")
                    break 

        if interactive:
            plt.ioff()
            if not goal_reached:
                 print("Interactive mode finished: Goal not reached within iterations.")
            plt.show() 
            plt.close(fig)

        if not goal_reached:
            print(f"Maximum iterations ({maxium_iterations}) reached without finding a path.")
            self.solution_length = -1 


    def reconstruct_path(self):
        """Reconstructs the path from goal to start if a solution was found."""
        path = []
        if self.solution_node:
            current = self.solution_node
            while current is not None:
                path.append(current)
                parent, _ = self.tree[current]
                current = parent
            path.reverse()
        return path

    def solve(self, interactive=False):
        """Runs the planning process and measures time."""
        start_time = time.time()
        self.plan(interactive=interactive)
        end_time = time.time()
        self.execution_time = end_time - start_time

    def report(self):
        """Prints a report of the RRT execution results."""
        print(f"--- {self.__class__.__name__} Report ---")
        print(f"Execution Time: {self.execution_time:.6f} seconds")
        print(f"Iterations: {self.iterations}")
        print(f"Tree Size (nodes): {len(self.tree)}")
        if self.solution_node:
            print(f"Path Found: Yes")
            print(f"Path Length (steps): {self.solution_length}")
            path = self.reconstruct_path()
            path_dist = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path)-1))
            print(f"Path Euclidean Distance: {path_dist:.4f}")
        else:
            print(f"Path Found: No")
            print(f"Path Length (steps): {self.solution_length}")

        return [self.execution_time, self.iterations, len(self.tree), self.solution_length]

    def graph(self, save=False, show=True):
        """Visualizes the final RRT tree and path in 3D."""
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_box_aspect([1,1,1]) 


        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{self.__class__.__name__} Path', fontsize=16, fontweight='bold')

        ax.scatter(*self.start, c='lime', marker='o', s=150, label='Start', depthshade=False, edgecolors='k')
        ax.scatter(*self.goal, c='red', marker='*', s=200, label='Goal', depthshade=False, edgecolors='k')
        ax.scatter(*self.goal, s=(self.goal_range*150)**2, c='red', alpha=0.05, depthshade=False)

        draw_obstacles_3d(ax, self.obstacles, color='dimgray', alpha=0.4)

        edges = []
        for node, (parent, _) in self.tree.items():
            if parent is not None:
                edges.append((parent, node))

        cmap = plt.cm.magma
        for i, (parent, node) in enumerate(edges):
             ax.plot([parent[0], node[0]],
                     [parent[1], node[1]],
                     [parent[2], node[2]],
                     color=cmap(i / len(edges)), linewidth=0.6, alpha=0.8)

        solution_path = self.reconstruct_path()
        if solution_path:
            path_x = [p[0] for p in solution_path]
            path_y = [p[1] for p in solution_path]
            path_z = [p[2] for p in solution_path]
            ax.plot(path_x, path_y, path_z, color='#00FF00', linewidth=2.5, label='Solution Path', alpha=1.0, zorder=10)

        norm = mpl.colors.Normalize(vmin=0, vmax=len(edges))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # cbar = plt.colorbar(sm, ax=ax, label="Edge Addition Order", fraction=0.03, pad=0.1) # Adjust fraction/pad
        # cbar.ax.tick_params(labelsize=10)

        ax.legend()
        ax.view_init(elev=20., azim=-65)

        if save:
            timestamp = int(time.time())
            filename = f'graphs/{self.__class__.__name__}_path_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved graph to {filename}")
        if show:
            plt.show()

        plt.close(fig)


class WRRT3D(RRT3D):
    def __init__(self, start, goal, goal_range, distance_unit, obstacles, map_resolution):
        super().__init__(start, goal, goal_range, distance_unit, obstacles)
        self.map_resolution = map_resolution
        print("Generating heuristic map...")
        self.heuristic_map = self.generate_heuristic_map()
        print("Heuristic map generated.")

    def generate_heuristic_map(self):
        """Generates a 3D heuristic map based on distance to the goal."""
        res = self.map_resolution
        heuristicMap = np.zeros((res, res, res))

        cell_centers_x = (np.arange(res) + 0.5) / res
        cell_centers_y = (np.arange(res) + 0.5) / res
        cell_centers_z = (np.arange(res) + 0.5) / res

        for i in tqdm(range(res), desc="Heuristic Map Gen"): # x index
            for j in range(res): # y index
                for k in range(res): # z index
                    cell_x = cell_centers_x[i]
                    cell_y = cell_centers_y[j]
                    cell_z = cell_centers_z[k]
                    heuristicMap[i, j, k] = euclidean_distance_heuristic_3d(cell_x, cell_y, cell_z, self.goal)

        heuristicMap = np.maximum(heuristicMap, 0)

        return heuristicMap

    def pick_random_point(self):
        """Picks a random point biased by the heuristic map."""
        try:
             i, j, k = pick_weighted_random_cell(self.heuristic_map) # (x, y, z) indices
        except Exception as e:
             print(f"Error picking weighted cell: {e}. Falling back to uniform random point.")
             return pick_pure_random_point_3d()

        res = self.map_resolution
        x = np.random.uniform(i / res, (i + 1) / res)
        y = np.random.uniform(j / res, (j + 1) / res)
        z = np.random.uniform(k / res, (k + 1) / res)

        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        z = np.clip(z, 0.0, 1.0)

        return (x, y, z)

    def report_heuristic_map(self, save=True):
        """Visualizes the 3D heuristic map."""
        if self.heuristic_map is None:
            print("Heuristic map not generated.")
            return

        res = self.map_resolution
        x = np.linspace(0, 1, res)
        y = np.linspace(0, 1, res)
        z = np.linspace(0, 1, res)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        normalized_map = self.heuristic_map / np.max(self.heuristic_map)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

        scatter = ax.scatter(
            X.flatten(), Y.flatten(), Z.flatten(),
            c=normalized_map.flatten(), cmap='viridis', alpha=0.6, s=10
        )

        for (x_min, y_min, z_min), (x_max, y_max, z_max) in self.obstacles:
            draw_cube(ax, center=((x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2),
                      size=(x_max - x_min), color='k', alpha=0.3)

        ax.scatter(*self.start, c='lime', marker='o', s=150, label='Start', depthshade=False, edgecolors='k')
        ax.scatter(*self.goal, c='red', marker='*', s=200, label='Goal', depthshade=False, edgecolors='k')

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label('Heuristic Value', fontsize=12)

        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title('3D Heuristic Map', fontsize=16, fontweight='bold')

        ax.view_init(elev=20., azim=-65)

        if save:
            timestamp = int(time.time())
            filename = f'graphs/WRRT3D_heuristic_map_3d_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved 3D heuristic map to {filename}")

        plt.legend()
        plt.show()
        plt.close()

if __name__ == "__main__":
    try:
        with open('scenarios_3d.json') as f:
            scenarios = json.load(f)
    except FileNotFoundError:
        print("data.json file not found. Please ensure the file exists in the current directory.")
        exit(1)

    import os
    if not os.path.exists('graphs'):
        os.makedirs('graphs')

    num_runs = 1 # Number of runs per scenario per algorithm

    for scenario_name, scenario in scenarios.items():
        print(f"\n===== Running Scenario: {scenario_name} =====")
        start = tuple(scenario['start'])
        goal = tuple(scenario['goal'])
        map_resolution = scenario['mapResolution']
        goal_range = scenario['goalRange']
        distance_unit = scenario['distanceUnit']
        obstacles = [(tuple(obs[0]), tuple(obs[1])) for obs in scenario['obstacles']]

        for run in range(num_runs):
            print(f"\n--- Run {run + 1}/{num_runs} for {scenario_name} ---")

            rrt = RRT3D(start, goal, goal_range, distance_unit, obstacles)
            wrrt = WRRT3D(start, goal, goal_range, distance_unit, obstacles, map_resolution)

            # Run RRT3D
            print(f"\nRunning RRT3D...")
            rrt.solve(interactive=(run==0 and False)) 
            rrt.report()
            rrt.graph(save=True, show=True)

            # Run WRRT3D
            print(f"\nRunning WRRT3D...")
            wrrt.solve(interactive=(run==0 and False))
            wrrt.report()
            wrrt.graph(save=True, show=True) 
            if run == 0:
                wrrt.report_heuristic_map(save=True)


    print("\n===== All Scenarios Completed =====")