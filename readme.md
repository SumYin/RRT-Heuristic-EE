# RRT-Heuristic-EE

## Introduction

This project implements the Rapidly-exploring Random Tree (RRT) algorithm for path planning in a 2D environment. The goal is to navigate from a start point to a goal point while avoiding obstacles. The environment is represented as a 1x1 grid, and the heuristic map is used to guide the path planning process. The project includes functions to generate heuristic maps, normalize them to probability distributions, and select points based on these distributions to add a greedy aspect to the path planning algorithm, improving its efficiency.

We aim to introduce a heuristic approach to the RRT algorithm, which will help in selecting the next point to explore based on the heuristic map. This will allow the algorithm to focus on areas of the environment that are more likely to lead to a successful path to the goal.

This is for a IB Computer Science EE project, and the code is written in Python. The project is structured to allow for easy modification and testing of different heuristic approaches.