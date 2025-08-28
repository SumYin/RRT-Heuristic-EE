# RRT with Goal-Based Heuristics

This project investigates the impact of using a goal-based Euclidean distance heuristic to guide the sampling process of the Rapidly-Exploring Random Tree (RRT) algorithm. It compares the performance of a standard RRT with a weighted variant (WRRT) across various 2D and 3D scenarios.


### 2D Performance Metrics

| Average Iterations | Average Tree Size |
| :---: | :---: |
| ![Average Iterations](visualization_graphics/2d_avg_iterations.png) | ![Average Tree Size](visualization_graphics/2d_avg_tree_size.png) |
| **Average Path Length** | **Success Rate** |
| ![Average Path Length](visualization_graphics/2d_avg_path_length.png) | ![Success Rate](visualization_graphics/2d_success_rate.png) |
## 3D Visualizations and Results

Below are the results and visualizations from the 3D experiments.

### 3D Performance Metrics

| Average Iterations | Average Tree Size |
| :---: | :---: |
| ![Average Iterations](visualization_graphics/3d_avg_iterations.png) | ![Average Tree Size](visualization_graphics/3d_avg_tree_size.png) |
| **Average Path Length** | **Success Rate** |
| ![Average Path Length](visualization_graphics/3d_avg_path_length.png) | ![Success Rate](visualization_graphics/3d_success_rate.png) |

### Default 3D Scenario

| Scenario | RRT Path | WRRT Path |
| :---: | :---: | :---: |
| ![Default 3D Scenario](visualization_graphics/scenario_default_3d_rotating.gif) | ![RRT on Default 3D](visualization_graphics/RRT3D_path_default_3d_rotating.gif) | ![WRRT on Default 3D](visualization_graphics/WRRT3D_path_default_3d_rotating.gif) |

### Center Hole 3D Scenario

| Scenario | RRT Path | WRRT Path |
| :---: | :---: | :---: |
| ![Center Hole Scenario](visualization_graphics/scenario_center_hole_rotating.gif) | ![RRT on Center Hole](visualization_graphics/RRT3D_path_center_hole_rotating.gif) | ![WRRT on Center Hole](visualization_graphics/WRRT3D_path_center_hole_rotating.gif) |