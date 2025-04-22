
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import json
import os

class MapMakerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("2D RRT Scenario Maker")

        self.obstacles = []
        self.start_point = None
        self.goal_point = None
        self.temp_rect = None
        self.start_coords = None # For drawing rectangles

        # --- Parameters Frame ---
        param_frame = tk.Frame(master, bd=2, relief=tk.GROOVE)
        param_frame.pack(pady=10, padx=10, fill=tk.X)

        tk.Label(param_frame, text="Scenario Name:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.scenario_name_entry = tk.Entry(param_frame, width=20)
        self.scenario_name_entry.grid(row=0, column=1, padx=5, pady=2)
        self.scenario_name_entry.insert(0, "MyScenario")

        tk.Label(param_frame, text="Goal Range:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.goal_range_entry = tk.Entry(param_frame, width=10)
        self.goal_range_entry.grid(row=1, column=1, padx=5, pady=2, sticky="w")
        self.goal_range_entry.insert(0, "0.05") # Default value

        tk.Label(param_frame, text="Distance Unit:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.distance_unit_entry = tk.Entry(param_frame, width=10)
        self.distance_unit_entry.grid(row=2, column=1, padx=5, pady=2, sticky="w")
        self.distance_unit_entry.insert(0, "0.02") # Default value

        tk.Label(param_frame, text="Map Resolution (WRRT):").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.map_resolution_entry = tk.Entry(param_frame, width=10)
        self.map_resolution_entry.grid(row=3, column=1, padx=5, pady=2, sticky="w")
        self.map_resolution_entry.insert(0, "50") # Default value

        # --- Canvas Frame ---
        canvas_frame = tk.Frame(master)
        canvas_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        self.canvas_width = 500
        self.canvas_height = 500
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height, bg="white", bd=1, relief=tk.SUNKEN)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Obstacle List Frame ---
        list_frame = tk.Frame(canvas_frame, bd=1, relief=tk.SUNKEN)
        list_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5,0))
        tk.Label(list_frame, text="Obstacles:").pack(pady=2)
        self.obstacle_listbox = tk.Listbox(list_frame, height=20, width=30)
        self.obstacle_listbox.pack(fill=tk.Y, expand=True)
        delete_obstacle_button = tk.Button(list_frame, text="Delete Selected", command=self.delete_obstacle)
        delete_obstacle_button.pack(pady=5)


        # --- Controls Frame ---
        controls_frame = tk.Frame(master)
        controls_frame.pack(pady=5, padx=10, fill=tk.X)

        self.mode_var = tk.StringVar(value="obstacle") # Modes: obstacle, start, goal
        tk.Radiobutton(controls_frame, text="Add Obstacle", variable=self.mode_var, value="obstacle").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(controls_frame, text="Set Start", variable=self.mode_var, value="start").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(controls_frame, text="Set Goal", variable=self.mode_var, value="goal").pack(side=tk.LEFT, padx=5)

        clear_button = tk.Button(controls_frame, text="Clear All", command=self.clear_all)
        clear_button.pack(side=tk.LEFT, padx=10)

        save_button = tk.Button(controls_frame, text="Save Scenario", command=self.save_scenario)
        save_button.pack(side=tk.RIGHT, padx=10)


        # --- Canvas Bindings ---
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # --- Status Bar ---
        self.status_var = tk.StringVar()
        self.status_bar = tk.Label(master, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var.set("Mode: Add Obstacle. Click and drag to draw obstacles.")


    def update_status(self):
        mode = self.mode_var.get()
        if mode == "obstacle":
            self.status_var.set("Mode: Add Obstacle. Click and drag to draw obstacles.")
        elif mode == "start":
            self.status_var.set("Mode: Set Start. Click to place the start point.")
        elif mode == "goal":
            self.status_var.set("Mode: Set Goal. Click to place the goal point.")

    def coords_to_normalized(self, x, y):
        """Convert canvas coordinates to normalized [0, 1] range."""
        norm_x = max(0.0, min(1.0, x / self.canvas_width))
        norm_y = max(0.0, min(1.0, 1.0 - (y / self.canvas_height))) # Invert Y for typical plot origin
        return norm_x, norm_y

    def normalized_to_coords(self, norm_x, norm_y):
        """Convert normalized [0, 1] coordinates to canvas coordinates."""
        x = norm_x * self.canvas_width
        y = (1.0 - norm_y) * self.canvas_height # Invert Y back
        return x, y

    def on_press(self, event):
        self.update_status()
        mode = self.mode_var.get()
        x, y = event.x, event.y
        norm_x, norm_y = self.coords_to_normalized(x, y)

        if mode == "obstacle":
            self.start_coords = (x, y)
            # Create a temporary rectangle for visual feedback
            self.temp_rect = self.canvas.create_rectangle(x, y, x, y, outline="red", dash=(2, 2))
        elif mode == "start":
            if self.start_point:
                self.canvas.delete(self.start_point) # Delete old start point visual
            self.start_point = self.canvas.create_oval(x-4, y-4, x+4, y+4, fill="green", outline="black", tags="start")
            self.start_coords_normalized = (norm_x, norm_y)
            self.status_var.set(f"Start set at ({norm_x:.2f}, {norm_y:.2f})")
        elif mode == "goal":
            if self.goal_point:
                self.canvas.delete(self.goal_point) # Delete old goal point visual
            self.goal_point = self.canvas.create_rectangle(x-4, y-4, x+4, y+4, fill="red", outline="black", tags="goal")
            self.goal_coords_normalized = (norm_x, norm_y)
            self.status_var.set(f"Goal set at ({norm_x:.2f}, {norm_y:.2f})")

    def on_drag(self, event):
        mode = self.mode_var.get()
        if mode == "obstacle" and self.temp_rect and self.start_coords:
            x1, y1 = self.start_coords
            x2, y2 = event.x, event.y
            self.canvas.coords(self.temp_rect, x1, y1, x2, y2)

    def on_release(self, event):
        mode = self.mode_var.get()
        if mode == "obstacle" and self.temp_rect and self.start_coords:
            x1, y1 = self.start_coords
            x2, y2 = event.x, event.y
            self.canvas.delete(self.temp_rect) # Remove temporary rectangle

            # Ensure min coords are first
            final_x1, final_y1 = min(x1, x2), min(y1, y2)
            final_x2, final_y2 = max(x1, x2), max(y1, y2)

            # Avoid zero-sized obstacles
            if final_x1 == final_x2 or final_y1 == final_y2:
                self.status_var.set("Obstacle too small, ignored.")
                return

            # Convert to normalized coordinates
            norm_x1, norm_y1 = self.coords_to_normalized(final_x1, final_y1)
            norm_x2, norm_y2 = self.coords_to_normalized(final_x2, final_y2)

            # Store obstacle (min_x, min_y), (max_x, max_y) - note y-inversion handling
            obstacle_data = ((norm_x1, norm_y2), (norm_x2, norm_y1)) # Store min Y first
            self.obstacles.append(obstacle_data)

            # Draw permanent obstacle
            self.canvas.create_rectangle(final_x1, final_y1, final_x2, final_y2, fill="black", tags="obstacle")
            self.update_obstacle_listbox()
            self.status_var.set(f"Obstacle added: ({norm_x1:.2f},{norm_y2:.2f}) to ({norm_x2:.2f},{norm_y1:.2f})")

        self.start_coords = None
        self.temp_rect = None

    def update_obstacle_listbox(self):
        self.obstacle_listbox.delete(0, tk.END)
        for i, obs in enumerate(self.obstacles):
            (x1, y1), (x2, y2) = obs
            self.obstacle_listbox.insert(tk.END, f"{i}: ({x1:.2f},{y1:.2f}) - ({x2:.2f},{y2:.2f})")

    def delete_obstacle(self):
        selected_indices = self.obstacle_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Delete Error", "No obstacle selected.")
            return

        # Delete from list and redraw canvas
        # Iterate in reverse to avoid index issues after deletion
        for index in sorted(selected_indices, reverse=True):
            del self.obstacles[index]

        self.redraw_canvas()
        self.update_obstacle_listbox()
        self.status_var.set("Selected obstacle(s) deleted.")


    def redraw_canvas(self):
        # Clear only obstacles, keep start/goal
        self.canvas.delete("obstacle")
        # Redraw remaining obstacles
        for obs in self.obstacles:
            (norm_x1, norm_y1), (norm_x2, norm_y2) = obs
            # Convert back to canvas coords
            cx1, cy1 = self.normalized_to_coords(norm_x1, norm_y1) # Top-left
            cx2, cy2 = self.normalized_to_coords(norm_x2, norm_y2) # Bottom-right
            # Canvas uses top-left and bottom-right, handle potential y-inversion mismatch
            canvas_y1 = min(cy1, cy2)
            canvas_y2 = max(cy1, cy2)
            self.canvas.create_rectangle(cx1, canvas_y1, cx2, canvas_y2, fill="black", tags="obstacle")


    def clear_all(self):
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear everything?"):
            self.obstacles = []
            self.start_point = None
            self.goal_point = None
            self.start_coords_normalized = None
            self.goal_coords_normalized = None
            self.canvas.delete("all") # Clear canvas completely
            self.obstacle_listbox.delete(0, tk.END)
            self.status_var.set("Cleared all elements. Ready for new scenario.")

    def save_scenario(self):
        scenario_name = self.scenario_name_entry.get().strip()
        if not scenario_name:
            messagebox.showerror("Save Error", "Please enter a scenario name.")
            return

        if not self.start_coords_normalized:
            messagebox.showerror("Save Error", "Please set a start point.")
            return

        if not self.goal_coords_normalized:
            messagebox.showerror("Save Error", "Please set a goal point.")
            return

        try:
            goal_range = float(self.goal_range_entry.get())
            distance_unit = float(self.distance_unit_entry.get())
            map_resolution = int(self.map_resolution_entry.get())
            if goal_range <= 0 or distance_unit <= 0 or map_resolution <= 0:
                raise ValueError("Values must be positive.")
        except ValueError as e:
            messagebox.showerror("Save Error", f"Invalid parameter value: {e}")
            return

        scenario_data = {
            "start": list(self.start_coords_normalized),
            "goal": list(self.goal_coords_normalized),
            "mapResolution": map_resolution,
            "goalRange": goal_range,
            "distanceUnit": distance_unit,
            "obstacles": self.obstacles # Already in the correct format [((min_x, min_y), (max_x, max_y)), ...]
        }

        # --- File Handling ---
        filename = "scenarios.json"
        filepath = os.path.join(os.path.dirname(__file__), filename) # Assume scenarios.json is in the same dir

        scenarios = {}
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    scenarios = json.load(f)
            except json.JSONDecodeError:
                messagebox.showwarning("File Warning", f"{filename} is corrupted. A new file will be created.")
            except Exception as e:
                 messagebox.showerror("File Error", f"Error reading {filename}: {e}")
                 return

        if scenario_name in scenarios:
            if not messagebox.askyesno("Overwrite Confirmation", f"Scenario '{scenario_name}' already exists. Overwrite?"):
                return

        scenarios[scenario_name] = scenario_data

        try:
            with open(filepath, 'w') as f:
                json.dump(scenarios, f, indent=4)
            messagebox.showinfo("Save Success", f"Scenario '{scenario_name}' saved to {filename}")
            self.status_var.set(f"Scenario '{scenario_name}' saved.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not write to {filename}: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MapMakerApp(root)
    root.mainloop()
