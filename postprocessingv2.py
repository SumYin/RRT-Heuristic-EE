from __future__ import annotations

import json
import os
import math
import csv
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_2D_DIR = os.path.join(ROOT, "Results 100k")
RESULTS_3D_DIR = os.path.join(ROOT, "3d results")

OutpuDir = os.path.join(ROOT, "AnalysisV2")
out2d = os.path.join(OutpuDir, "Graphics2d")
out3d = os.path.join(OutpuDir, "Graphics3d")
apendixDir = os.path.join(OutpuDir, "Appendix")

plt.rcParams.update({
	"savefig.dpi": 300,
	"figure.dpi": 300,
	"font.size": 11,
	"axes.titlesize": 13,
	"axes.labelsize": 12,
	"xtick.labelsize": 10,
	"ytick.labelsize": 10,
	"legend.fontsize": 10,
})


def ensure_dirs():
	for d in [OutpuDir, out2d, out3d, apendixDir]:
		os.makedirs(d, exist_ok=True)


def read_json_lines(path: str) -> List[dict]:
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)


def load_results_2d() -> List[dict]:
	if not os.path.isdir(RESULTS_2D_DIR):
		return []
	rows: List[dict] = []
	for name in os.listdir(RESULTS_2D_DIR):
		if not name.endswith(".json"):
			continue
		path = os.path.join(RESULTS_2D_DIR, name)
		try:
			items = read_json_lines(path)
		except Exception:
			continue
		for it in items:
			alg = it.get("algorithm", "").upper()
			if alg not in ("RRT", "WRRT"):
				continue
			rows.append({
				"domain": "2D",
				"scenario": it.get("scenario", "unknown"),
				"algorithm": alg,  # RRT or WRRT
				"iterations": it.get("iterations"),
				"tree_size": it.get("tree_size"),
				"path_length": it.get("path_length"),
			})
	return rows


def load_results_3d() -> List[dict]:
	if not os.path.isdir(RESULTS_3D_DIR):
		return []
	rows: List[dict] = []
	for name in os.listdir(RESULTS_3D_DIR):
		if not name.endswith(".json"):
			continue
		path = os.path.join(RESULTS_3D_DIR, name)
		try:
			items = read_json_lines(path)
		except Exception:
			continue
		for it in items:
			alg = it.get("algorithm", "").upper()
			if alg == "RRT3D":
				norm_alg = "RRT"
			elif alg == "WRRT3D":
				norm_alg = "WRRT"
			else:
				continue
			rows.append({
				"domain": "3D",
				"scenario": it.get("scenario", "unknown"),
				"algorithm": norm_alg,
				"iterations": it.get("iterations"),
				"tree_size": it.get("tree_size"),
				"path_length": it.get("path_length"),
				"execution_time": it.get("execution_time"),
				"path_distance": it.get("path_distance"),
			})
	return rows


def to_numpy(arr: List[float]) -> np.ndarray:
	a = np.array([x for x in arr if x is not None], dtype=float)
	return a

def mean_ci(x: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
	if x.size == 0:
		return float("nan"), float("nan"), float("nan")
	m = float(np.mean(x))
	s = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
	se = s / math.sqrt(x.size) if x.size > 0 else float("nan")
	z = stats.norm.ppf(1 - alpha / 2)
	return m, m - z * se, m + z * se


def summarize_by(key_fields: Tuple[str, ...], rows: List[dict]) -> List[dict]:
	groups: Dict[Tuple, List[dict]] = defaultdict(list)
	for r in rows:
		key = tuple(r.get(k) for k in key_fields)
		groups[key].append(r)

	out: List[dict] = []
	for key, grp in groups.items():
		iterations = to_numpy([g.get("iterations") for g in grp])
		tree_size = to_numpy([g.get("tree_size") for g in grp])
		path_length = to_numpy([g.get("path_length") for g in grp])

		m_iter, l_iter, u_iter = mean_ci(iterations)
		m_tree, l_tree, u_tree = mean_ci(tree_size)
		m_path, l_path, u_path = mean_ci(path_length)

		rec = {k: v for k, v in zip(key_fields, key)}
		rec.update({
			"n": len(grp),
			"iterations_mean": m_iter, "iterations_ci_low": l_iter, "iterations_ci_high": u_iter,
			"tree_size_mean": m_tree, "tree_size_ci_low": l_tree, "tree_size_ci_high": u_tree,
			"path_length_mean": m_path, "path_length_ci_low": l_path, "path_length_ci_high": u_path,
			"iterations_median": float(np.median(iterations)) if iterations.size else float("nan"),
			"tree_size_median": float(np.median(tree_size)) if tree_size.size else float("nan"),
			"path_length_median": float(np.median(path_length)) if path_length.size else float("nan"),
		})
		out.append(rec)
	return out


def pairwise_improvement(per_scenario: List[dict], domain: str) -> List[dict]:
	idx: Dict[Tuple[str, str, str], dict] = {}
	for r in per_scenario:
		idx[(r.get("domain"), r.get("scenario"), r.get("algorithm"))] = r

	scenarios = sorted({r.get("scenario") for r in per_scenario if r.get("domain") == domain})
	out: List[dict] = []
	for sc in scenarios:
		rrt = idx.get((domain, sc, "RRT"))
		wrrt = idx.get((domain, sc, "WRRT"))
		if not rrt or not wrrt:
			continue
		def pct(delta_base: Tuple[float, float]) -> float:
			base, new = delta_base
			if base is None or new is None or not np.isfinite(base) or base == 0:
				return float("nan")
			return 100.0 * (base - new) / base

		out.append({
			"domain": domain,
			"scenario": sc,
			"iterations_pct_gain": pct((rrt.get("iterations_mean"), wrrt.get("iterations_mean"))),
			"tree_size_pct_gain": pct((rrt.get("tree_size_mean"), wrrt.get("tree_size_mean"))),
			"path_length_pct_change": pct((rrt.get("path_length_mean"), wrrt.get("path_length_mean"))),
			"n_rrt": rrt.get("n", 0),
			"n_wrrt": wrrt.get("n", 0),
		})
	return out


def write_csv(path: str, rows: List[dict]):
	if not rows:
		return
	keys = sorted({k for r in rows for k in r.keys()})
	with open(path, "w", newline="", encoding="utf-8") as f:
		w = csv.DictWriter(f, fieldnames=keys)
		w.writeheader()
		for r in rows:
			w.writerow(r)


def boxplot_metric(rows: List[dict], domain: str, metric: str, out_path: str):
	data_rrt = to_numpy([r[metric] for r in rows if r["domain"] == domain and r["algorithm"] == "RRT" and r.get(metric) is not None])
	data_wrrt = to_numpy([r[metric] for r in rows if r["domain"] == domain and r["algorithm"] == "WRRT" and r.get(metric) is not None])

	# remember to increase height!! cuts off right now
	plt.figure(figsize=(8.5, 7.5))
	plt.boxplot([data_rrt, data_wrrt], labels=["RRT", "WRRT"], showmeans=True)
	plt.title(f"{domain}: {metric} distribution by algorithm")
	plt.ylabel(metric)
	plt.grid(True, axis='y', alpha=0.3)
	plt.tight_layout()
	plt.savefig(out_path, bbox_inches='tight', dpi=300)
	plt.close()


def bar_means_per_scenario(per_scenario: List[dict], domain: str, metric: str, out_path: str):
	scen = sorted({(r["scenario"]) for r in per_scenario if r["domain"] == domain and r["algorithm"] in ("RRT", "WRRT")})
	rrt_means, wrrt_means = [], []
	rrt_err, wrrt_err = [], []
	for s in scen:
		rrt = next((r for r in per_scenario if r["domain"] == domain and r["scenario"] == s and r["algorithm"] == "RRT"), None)
		wrrt = next((r for r in per_scenario if r["domain"] == domain and r["scenario"] == s and r["algorithm"] == "WRRT"), None)
		if not rrt or not wrrt:
			continue
		rrt_means.append(rrt[f"{metric}_mean"]) ; wrrt_means.append(wrrt[f"{metric}_mean"]) 
		# use half CI range as error bar
		rrt_err.append(abs(rrt[f"{metric}_ci_high"] - rrt[f"{metric}_mean"]))
		wrrt_err.append(abs(wrrt[f"{metric}_ci_high"] - wrrt[f"{metric}_mean"]))

	x = np.arange(len(rrt_means))
	width = 0.38

	# too tall
	plt.figure(figsize=(max(10, len(x) * 0.7), 4.5))
	plt.bar(x - width/2, rrt_means, width, yerr=rrt_err, label="RRT", alpha=0.8)
	plt.bar(x + width/2, wrrt_means, width, yerr=wrrt_err, label="WRRT", alpha=0.8)
	plt.title(f"{domain}: Mean {metric} by scenario (95% CI)")
	plt.ylabel(metric)
	plt.xticks(x, scen, rotation=45, ha='right')
	plt.grid(True, axis='y', alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_path, bbox_inches='tight', dpi=300)
	plt.close()


def heatmap_improvement(improve_rows: List[dict], domain: str, metric: str, out_path: str):

	key_map = {
		"iterations": ("iterations_pct_gain", "% gain in iterations (WRRT vs RRT)", "WRRT % improvement in iterations by scenario"),
		"tree_size": ("tree_size_pct_gain", "% gain in tree_size (WRRT vs RRT)", "WRRT % improvement in tree_size by scenario"),
		"path_length": ("path_length_pct_change", "% change in path_length (WRRT vs RRT)", "WRRT % change in path_length by scenario"),
	}
	pct_key, cbar_label, title_suffix = key_map[metric]
	rows = [r for r in improve_rows if r["domain"] == domain and np.isfinite(r.get(pct_key, np.nan))]
	if not rows:
		return
	scen = [r["scenario"] for r in rows]
	vals = np.array([r[pct_key] for r in rows], dtype=float)
	vmin, vmax = -20, 60
	# too low
	plt.figure(figsize=(max(10, len(scen) * 0.7), 5))
	im = plt.imshow(vals.reshape(1, -1), aspect='auto', cmap='RdYlGn', vmin=vmin, vmax=vmax)
	plt.colorbar(im, label=cbar_label)
	plt.yticks([])
	plt.xticks(np.arange(len(scen)), scen, rotation=45, ha='right')
	plt.title(f"{domain}: {title_suffix}")
	threshold = (vmin + vmax) / 2.0
	for j, v in enumerate(vals):
		color = 'white' if v > threshold else 'black'
		plt.text(j, 0, f"{v:+.1f}%", ha='center', va='center', color=color, fontsize=10, fontweight='bold')
	plt.tight_layout()
	plt.savefig(out_path, bbox_inches='tight', dpi=300)
	plt.close()


def scatter_tree_vs_iter(rows: List[dict], domain: str, out_path: str, max_points_per_series: int = 1500):
	domain_rows = [r for r in rows if r["domain"] == domain and r.get("iterations") is not None and r.get("tree_size") is not None]
	scenarios = sorted({r["scenario"] for r in domain_rows})
	cmap = plt.get_cmap('tab20')
	colors = {sc: cmap(i % 20) for i, sc in enumerate(scenarios)}

	def series_sample(series: List[dict]) -> List[dict]:
		n = len(series)
		if n <= max_points_per_series:
			return series
		step = max(1, n // max_points_per_series)
		return series[::step]

	plt.figure(figsize=(12, 10))
	for sc in scenarios:
		for alg, marker, label_suffix in [("RRT", 'o', "RRT"), ("WRRT", '^', "WRRT")]:
			srows = [r for r in domain_rows if r["scenario"] == sc and r["algorithm"] == alg]
			srows = series_sample(srows)
			if not srows:
				continue
			x = [r["iterations"] for r in srows]
			y = [r["tree_size"] for r in srows]
			plt.scatter(x, y, s=7, alpha=0.25, marker=marker, color=colors[sc], edgecolors='none', label=f"{sc} • {label_suffix}")

	plt.xlabel("iterations")
	plt.ylabel("tree_size")
	plt.title(f"{domain}: tree_size vs iterations by scenario and algorithm")
	plt.grid(True, alpha=0.3)

	from matplotlib.lines import Line2D
	scenario_handles = [Line2D([0], [0], marker='o', color='w', label=sc, markerfacecolor=colors[sc], markersize=7) for sc in scenarios]
	alg_handles = [Line2D([0], [0], marker='o', color='k', label='RRT', markerfacecolor='none', markersize=7),
				   Line2D([0], [0], marker='^', color='k', label='WRRT', markerfacecolor='none', markersize=7)]
	leg1 = plt.legend(handles=scenario_handles, title="Scenario", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
	plt.gca().add_artist(leg1)
	plt.legend(handles=alg_handles, title="Algorithm", loc='lower right')
	plt.tight_layout()
	plt.savefig(out_path, bbox_inches='tight', dpi=300)
	plt.close()


def scatter_runtime_relations_3d(rows: List[dict], out_dir: str, max_points: int = 5000):
	domain_rows = [
		{
			"iterations": r.get("iterations"),
			"tree_size": r.get("tree_size"),
			"execution_time": r.get("execution_time"),
			"scenario": r.get("scenario", "unknown"),
			"algorithm": r.get("algorithm", ""),
		}
		for r in rows
		if r.get("domain") == "3D"
		and r.get("execution_time") is not None
		and r.get("iterations") is not None
		and r.get("tree_size") is not None
	]

	if not domain_rows:
		return

	def subsample(items: List[dict]) -> List[dict]:
		n = len(items)
		if n <= max_points:
			return items
		step = max(1, n // max_points)
		return items[::step]

	def do_scatter(x_key: str, x_label: str, filename: str, transform=None):
		pts = domain_rows
		if transform is not None:
			x_vals = []
			y_vals = []
			for r in pts:
				try:
					x_vals.append(float(transform(r)))
					y_vals.append(float(r["execution_time"]))
				except Exception:
					continue
			pairs = [{"x": x, "y": y} for x, y in zip(x_vals, y_vals) if np.isfinite(x) and np.isfinite(y)]
		else:
			pairs = []
			for r in pts:
				vx = r.get(x_key)
				vy = r.get("execution_time")
				if vx is None or vy is None:
					continue
				try:
					vx = float(vx); vy = float(vy)
					if np.isfinite(vx) and np.isfinite(vy):
						pairs.append({"x": vx, "y": vy})
				except Exception:
					continue

		pairs = subsample(pairs)
		if not pairs:
			return

		x = [p["x"] for p in pairs]
		y = [p["y"] for p in pairs]

		plt.figure(figsize=(8.5, 7.5))
		plt.scatter(x, y, s=8, alpha=0.35, edgecolors='none')
		plt.xlabel(x_label)
		plt.ylabel("runtime (s)")
		plt.title(f"3D: {x_label} vs runtime")
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight', dpi=300)
		plt.close()

	do_scatter("iterations", "iterations", "3D_scatter_iterations_vs_runtime.png")
	do_scatter("tree_size", "tree_size", "3D_scatter_tree_size_vs_runtime.png")
	do_scatter(
		"product",
		"iterations × tree_size",
		"3D_scatter_iterations_times_tree_size_vs_runtime.png",
		transform=lambda r: float(r.get("iterations", 0)) * float(r.get("tree_size", 0)),
	)




def main():
	ensure_dirs()

	rows_2d = load_results_2d()
	rows_3d = load_results_3d()
	rows = rows_2d + rows_3d

	with open(os.path.join(apendixDir, "merged_rows.json"), "w", encoding="utf-8") as f:
		json.dump(rows, f)

	per_scenario = summarize_by(("domain", "scenario", "algorithm"), rows)
	write_csv(os.path.join(apendixDir, "per_scenario_summary.csv"), per_scenario)

	per_domain_alg = summarize_by(("domain", "algorithm"), rows)
	write_csv(os.path.join(apendixDir, "per_domain_algorithm_summary.csv"), per_domain_alg)

	improve_2d = pairwise_improvement(per_scenario, domain="2D")
	improve_3d = pairwise_improvement(per_scenario, domain="3D")
	write_csv(os.path.join(apendixDir, "improvements_2d.csv"), improve_2d)
	write_csv(os.path.join(apendixDir, "improvements_3d.csv"), improve_3d)


	for domain, out in [("2D", out2d), ("3D", out3d)]:
		boxplot_metric(rows, domain, "iterations", os.path.join(out, f"{domain}_boxplot_iterations.png"))
		boxplot_metric(rows, domain, "tree_size", os.path.join(out, f"{domain}_boxplot_tree_size.png"))
		boxplot_metric(rows, domain, "path_length", os.path.join(out, f"{domain}_boxplot_path_length.png"))

		bar_means_per_scenario(per_scenario, domain, "iterations", os.path.join(out, f"{domain}_scenario_bar_iterations.png"))
		bar_means_per_scenario(per_scenario, domain, "tree_size", os.path.join(out, f"{domain}_scenario_bar_tree_size.png"))
		bar_means_per_scenario(per_scenario, domain, "path_length", os.path.join(out, f"{domain}_scenario_bar_path_length.png"))

		improve_rows = improve_2d if domain == "2D" else improve_3d
		heatmap_improvement(improve_rows, domain, "iterations", os.path.join(out, f"{domain}_iterations_improvement_heatmap.png"))
		heatmap_improvement(improve_rows, domain, "tree_size", os.path.join(out, f"{domain}_tree_size_improvement_heatmap.png"))
		heatmap_improvement(improve_rows, domain, "path_length", os.path.join(out, f"{domain}_path_length_improvement_heatmap.png"))

		scatter_tree_vs_iter(rows, domain, os.path.join(out, f"{domain}_scatter_tree_vs_iterations.png"))

		if domain == "3D":
			scatter_runtime_relations_3d(rows, out)

	overall_summary = {
		"per_domain_algorithm": per_domain_alg,
		"improvement_2d": improve_2d,
		"improvement_3d": improve_3d,
	}
	with open(os.path.join(apendixDir, "overall_summary.json"), "w", encoding="utf-8") as f:
		json.dump(overall_summary, f, indent=2)

	print("AnalysisV2 complete. Artifacts written to:", OutpuDir)


if __name__ == "__main__":
	main()

