import tqdm as tqdm
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# read results from all JSON files in the "Results 100k" directory
results_dir = './Results 100k'
results = []
for filename in os.listdir(results_dir):
    if filename.endswith('.json'):
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
                results.extend(data)  # assuming each file contains a list of results
            except json.JSONDecodeError as e:
                print(f"Error reading {filename}: {e}")
# Data structures to aggregate results
stats = defaultdict(lambda: defaultdict(list))  # stats[scenario][algorithm] = list of runs

for entry in results:
    scenario = entry['scenario']
    algorithm = entry['algorithm']
    stats[scenario][algorithm].append(entry)

print('--- Success Rate, Average Iterations, Path Length, Tree Size ---')
for scenario in stats:
    print(f'\nScenario: {scenario}')
    for algorithm in stats[scenario]:
        runs = stats[scenario][algorithm]
        total = len(runs)
        successes = [r for r in runs if r['path_length'] > 0]
        success_rate = len(successes) / total if total > 0 else 0
        avg_iterations = np.mean([r['iterations'] for r in successes]) if successes else 0
        avg_path_length = np.mean([r['path_length'] for r in successes]) if successes else 0
        avg_tree_size = np.mean([r['tree_size'] for r in successes]) if successes else 0
        print(f'  Algorithm: {algorithm}')
        print(f'    Success Rate: {success_rate:.2%} ({len(successes)}/{total})')
        print(f'    Avg Iterations: {avg_iterations:.2f}')
        print(f'    Avg Path Length: {avg_path_length:.2f}')
        print(f'    Avg Tree Size: {avg_tree_size:.2f}')

# Prepare data for plotting
scenarios = sorted(stats.keys())
algorithms = sorted({alg for s in stats for alg in stats[s]})

# Data containers
success_rates = {alg: [] for alg in algorithms}
avg_iterations = {alg: [] for alg in algorithms}
avg_path_length = {alg: [] for alg in algorithms}
avg_tree_size = {alg: [] for alg in algorithms}

for scenario in scenarios:
    for alg in algorithms:
        runs = stats[scenario].get(alg, [])
        total = len(runs)
        successes = [r for r in runs if r['path_length'] > 0]
        success_rate = len(successes) / total if total > 0 else 0
        avg_iter = np.mean([r['iterations'] for r in successes]) if successes else 0
        avg_path = np.mean([r['path_length'] for r in successes]) if successes else 0
        avg_tree = np.mean([r['tree_size'] for r in successes]) if successes else 0
        success_rates[alg].append(success_rate)
        avg_iterations[alg].append(avg_iter)
        avg_path_length[alg].append(avg_path)
        avg_tree_size[alg].append(avg_tree)

x = np.arange(len(scenarios))
width = 0.35 if len(algorithms) == 2 else 0.8 / len(algorithms)

# Plotting helper
def plot_bar(data_dict, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    for i, alg in enumerate(algorithms):
        plt.bar(x + i*width, data_dict[alg], width, label=alg)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x + width/2, scenarios, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

plot_bar(success_rates, 'Success Rate', 'Success Rate by Scenario and Algorithm', 'success_rate.png')
plot_bar(avg_iterations, 'Average Iterations', 'Average Iterations by Scenario and Algorithm', 'avg_iterations.png')
plot_bar(avg_path_length, 'Average Path Length', 'Average Path Length by Scenario and Algorithm', 'avg_path_length.png')
plot_bar(avg_tree_size, 'Average Tree Size', 'Average Tree Size by Scenario and Algorithm', 'avg_tree_size.png')

