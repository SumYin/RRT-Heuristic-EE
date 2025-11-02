# RRT with Goal-Based Heuristics

### Scenario summaries

| Iterations (per-scenario) | Tree size (per-scenario) |
| :---: | :---: |
| ![2D iterations by scenario](AnalysisV2/Graphics2d/2D_scenario_bar_iterations.png) | ![2D tree size by scenario](AnalysisV2/Graphics2d/2D_scenario_bar_tree_size.png) |
| Path length (per-scenario) | |
| ![2D path length by scenario](AnalysisV2/Graphics2d/2D_scenario_bar_path_length.png) | |

### Improvement heatmaps

| Iterations improvement | Tree size improvement | Path length improvement |
| :---: | :---: | :---: |
| ![2D iterations improvement](AnalysisV2/Graphics2d/2D_iterations_improvement_heatmap.png) | ![2D tree size improvement](AnalysisV2/Graphics2d/2D_tree_size_improvement_heatmap.png) | ![2D path length improvement](AnalysisV2/Graphics2d/2D_path_length_improvement_heatmap.png) |

### More 2D plots

<details>
<summary>Open extra 2D figures (boxplots, CDFs, scatter)</summary>

![2D boxplot iterations](AnalysisV2/Excess%20Plots/2D_boxplot_iterations.png)
![2D boxplot path length](AnalysisV2/Excess%20Plots/2D_boxplot_path_length.png)
![2D boxplot tree size](AnalysisV2/Excess%20Plots/2D_boxplot_tree_size.png)

![2D CDF iterations](AnalysisV2/Excess%20Plots/2D_cdf_iterations.png)
![2D CDF path length](AnalysisV2/Excess%20Plots/2D_cdf_path_length.png)
![2D CDF tree size](AnalysisV2/Excess%20Plots/2D_cdf_tree_size.png)

![2D scatter tree vs iterations](AnalysisV2/Excess%20Plots/2D_scatter_tree_vs_iterations.png)

</details>


## 3D visualizations and results

### Default 3D scenario

| Scenario | RRT Path | WRRT Path |
| :---: | :---: | :---: |
| ![Default 3D Scenario](AnimatedVisuals/scenario_default_3d_rotating.gif) | ![RRT on Default 3D](AnimatedVisuals/RRT3D_path_default_3d_rotating.gif) | ![WRRT on Default 3D](AnimatedVisuals/WRRT3D_path_default_3d_rotating.gif) |

### Center hole 3D scenario

| Scenario | RRT Path | WRRT Path |
| :---: | :---: | :---: |
| ![Center Hole Scenario](AnimatedVisuals/scenario_center_hole_rotating.gif) | ![RRT on Center Hole](AnimatedVisuals/RRT3D_path_center_hole_rotating.gif) | ![WRRT on Center Hole](AnimatedVisuals/WRRT3D_path_center_hole_rotating.gif) |

### 3D summary figures
| Iterations (per-scenario) | Tree size (per-scenario) | Path length (per-scenario) |
| :---: | :---: | :---: |
| ![3D iterations by scenario](AnalysisV2/Graphics3d/3D_scenario_bar_iterations.png) | ![3D tree size by scenario](AnalysisV2/Graphics3d/3D_scenario_bar_tree_size.png) | ![3D path length by scenario](AnalysisV2/Graphics3d/3D_scenario_bar_path_length.png) |

| Iterations improvement | Tree size improvement | Path length improvement |
| :---: | :---: | :---: |
| ![3D iterations improvement](AnalysisV2/Graphics3d/3D_iterations_improvement_heatmap.png) | ![3D tree size improvement](AnalysisV2/Graphics3d/3D_tree_size_improvement_heatmap.png) | ![3D path length improvement](AnalysisV2/Graphics3d/3D_path_length_improvement_heatmap.png) |

<details>
<summary>Open extra 3D figures (boxplots, CDFs, scatter)</summary>

![3D boxplot iterations](AnalysisV2/Excess%20Plots/3D_boxplot_iterations.png)
![3D boxplot path length](AnalysisV2/Excess%20Plots/3D_boxplot_path_length.png)
![3D boxplot tree size](AnalysisV2/Excess%20Plots/3D_boxplot_tree_size.png)

![3D CDF iterations](AnalysisV2/Excess%20Plots/3D_cdf_iterations.png)
![3D CDF path length](AnalysisV2/Excess%20Plots/3D_cdf_path_length.png)
![3D CDF tree size](AnalysisV2/Excess%20Plots/3D_cdf_tree_size.png)

![3D scatter iterations vs runtime](AnalysisV2/Excess%20Plots/3D_scatter_iterations_vs_runtime.png)
![3D scatter tree size vs runtime](AnalysisV2/Excess%20Plots/3D_scatter_tree_size_vs_runtime.png)
![3D scatter (iterations×tree) vs runtime](AnalysisV2/Excess%20Plots/3D_scatter_iterations_times_tree_size_vs_runtime.png)
![3D scatter tree vs iterations](AnalysisV2/Excess%20Plots/3D_scatter_tree_vs_iterations.png)

</details>


## Heuristic maps and example paths
<details>
<summary>Open heuristic maps and paths</summary>

![WRRT heuristic map](graphs/WRRT_heuristic_map_16_(0,%200)_(0.2,%200.2).png)
![WRRT heuristic map](graphs/WRRT_heuristic_map_16_(0,%200)_(0.95,%200.95).png)
![WRRT heuristic map](graphs/WRRT_heuristic_map_16_(0,%201)_(0.95,%200.5).png)
![WRRT heuristic map](graphs/WRRT_heuristic_map_16_(0.1,%200.5)_(0.9,%200.5).png)
![WRRT heuristic map](graphs/WRRT_heuristic_map_16_(0.1,%200.7)_(0.9,%200.7).png)
![WRRT heuristic map](graphs/WRRT_heuristic_map_16_(0.2,%200.4)_(0.95,%200.95).png)

![RRT path](graphs/RRT_path_1752698112.png)
![RRT path](graphs/RRT_path_1752698113.png)
![RRT path](graphs/RRT_path_1752698116.png)

![WRRT path](graphs/WRRT_path_1752698113.png)
![WRRT path](graphs/WRRT_path_1752698114.png)
![WRRT path](graphs/WRRT_path_1752698116.png)
![WRRT path](graphs/WRRT_path_1752698118.png)

</details>