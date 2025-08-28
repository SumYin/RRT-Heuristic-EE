import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

results_dir = './3d results'
results = []
for filename in os.listdir(results_dir):
    if filename.endswith('.json'):
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
                results.extend(data)  
            except json.JSONDecodeError as e:
                print(f"Error reading {filename}: {e}")

stats = defaultdict(lambda: defaultdict(list)) 

for entry in results:
    scenario = entry['scenario']
    algorithm = entry['algorithm']
    stats[scenario][algorithm].append(entry)

summary_stats = defaultdict(dict)
for scenario in stats:
    for algorithm in stats[scenario]:
        runs = stats[scenario][algorithm]
        total = len(runs)
        successes = [r for r in runs if r['path_length'] > 0]
        success_rate = len(successes) / total if total > 0 else 0
        avg_iterations = np.mean([r['iterations'] for r in successes]) if successes else 0
        avg_path_length = np.mean([r['path_length'] for r in successes]) if successes else 0
        avg_tree_size = np.mean([r['tree_size'] for r in successes]) if successes else 0
        summary_stats[scenario][algorithm] = {
            'Success Rate': f'{success_rate:.2%} ({len(successes)}/{total})',
            'Avg Iterations': f'{avg_iterations:.2f}',
            'Avg Path Length': f'{avg_path_length:.2f}',
            'Avg Tree Size': f'{avg_tree_size:.2f}'
        }

overall_stats = defaultdict(list)
for scenario in stats:
    for algorithm in stats[scenario]:
        successes = [r for r in stats[scenario][algorithm] if r['path_length'] > 0]
        overall_stats[algorithm].extend(successes)

overall_summary = {}
for algorithm in sorted(overall_stats.keys()):
    runs = overall_stats[algorithm]
    total_runs = sum(len(stats[scenario].get(algorithm, [])) for scenario in stats)
    success_rate = len(runs) / total_runs if total_runs > 0 else 0
    avg_iterations = np.mean([r['iterations'] for r in runs]) if runs else 0
    avg_path_length = np.mean([r['path_length'] for r in runs]) if runs else 0
    avg_tree_size = np.mean([r['tree_size'] for r in runs]) if runs else 0

    overall_summary[algorithm] = {
        'Overall Success Rate': f'{success_rate:.2%} ({len(runs)}/{total_runs})',
        'Overall Avg Iterations': f'{avg_iterations:.2f}',
        'Overall Avg Path Length': f'{avg_path_length:.2f}',
        'Overall Avg Tree Size': f'{avg_tree_size:.2f}'
    }

with open('postprocessing_3d_log.json', 'w') as f:
    json.dump({'per_scenario': summary_stats, 'overall': overall_summary}, f, indent=4)

scenarios = sorted(stats.keys())
algorithms = sorted({alg for s in stats for alg in stats[s]})

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


def plot_bar(data_dict, ylabel, title, filename):
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  # Professional palette
    
    bars = []
    for i, alg in enumerate(algorithms):
        bar = plt.bar(x + i * width, data_dict[alg], width, label=alg, color=colors[i % len(colors)])
        bars.append(bar)
        
        # Add value labels on top of bars
        for j, (bar_rect, value) in enumerate(zip(bar, data_dict[alg])):
            height = bar_rect.get_height()
            if 'Success Rate' in title:
                label_text = f'{value:.1%}'
            else:
                label_text = f'{value:.1f}'
            
            plt.text(bar_rect.get_x() + bar_rect.get_width()/2., height + max(data_dict[alg]) * 0.01,
                    label_text, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(x + width / 2, scenarios, rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.4)
    
    y_max = max(max(values) for values in data_dict.values())
    plt.ylim(0, y_max * 1.15)
    
    plt.tight_layout()
    plt.savefig(f'3d results/{filename}', dpi=300, bbox_inches='tight')
    plt.show()

plot_bar(success_rates, 'Success Rate', '3D RRT Success Rate by Scenario and Algorithm', '3d_success_rate.png')
plot_bar(avg_iterations, 'Average Iterations', '3D RRT Average Iterations by Scenario and Algorithm', '3d_avg_iterations.png')
plot_bar(avg_path_length, 'Average Path Length (Steps)', '3D RRT Average Path Length by Scenario and Algorithm', '3d_avg_path_length.png')
plot_bar(avg_tree_size, 'Average Tree Size', '3D RRT Average Tree Size by Scenario and Algorithm', '3d_avg_tree_size.png')

print("\n3D Results analysis complete. Plots saved to '3d results' directory.")
