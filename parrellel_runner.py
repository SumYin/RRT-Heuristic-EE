import subprocess
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

SCENARIOS_FILE = "scenarios.json"
RUNS_PER_SCENARIO = 10_000
CHUNK_SIZE = 1000  # Number of runs per process (adjust as needed)

def run_chunk(scenario_name, chunk_start, chunk_end):
    # Use the .venv Python interpreter
    venv_python = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
    output_file = f"results_{scenario_name}_{chunk_start}_{chunk_end}.json"
    cmd = [
        venv_python, "2dWRRT_CUDA.py",  # Use CUDA-accelerated version
        "--scenario", scenario_name,
        "--start", str(chunk_start),
        "--end", str(chunk_end),
        "--output", output_file
    ]
    print(f"Launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return output_file

def main():
    # Load scenarios
    with open(SCENARIOS_FILE) as f:
        scenarios = json.load(f)

    scenario_names = list(scenarios.keys())
    jobs = []
    for scenario_name in scenario_names:
        for chunk_start in range(0, RUNS_PER_SCENARIO, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, RUNS_PER_SCENARIO)
            jobs.append((scenario_name, chunk_start, chunk_end))

    print("="*50)
    print("Parallel RRT/WRRT Experiment Runner")
    print("="*50)
    print(f"Scenarios loaded: {len(scenario_names)}")
    print(f"Scenario names: {', '.join(scenario_names)}")
    print(f"Runs per scenario: {RUNS_PER_SCENARIO}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Total jobs to run: {len(jobs)}")
    print(f"Each job will output to: results_<scenario>_<start>_<end>.json")
    print("="*50)
    proceed = input("Proceed with these settings? (y/n): ").strip().lower()
    if proceed != "y":
        print("Aborted by user.")
        return

    # Run in parallel
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(run_chunk, *job) for job in jobs]
        for future in as_completed(futures):
            try:
                output_file = future.result()
                print(f"Completed: {output_file}")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()