import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class MapMaker3DApp:
    def __init__(self, master):
        self.master = master
        self.master.title("3D RRT Scenario Maker")
        self.master.geometry("1200x800")
        
        self.obstacles = []
        self.start_point = None
        self.goal_point = None
        self.selected_obstacle = None
        
        # Current mode and action
        self.mode = "view"  # modes: view, add_obstacle, move_obstacle
        self.current_action = None  # actions: drawing, moving
        self.temp_obstacle = None
        self.start_coords = None
        self.selected_point = None

        # Create main frames
        left_frame = tk.Frame(master, width=450)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10, expand=False)
        
        right_frame = tk.Frame(master)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10, expand=True)

        # --- Parameters Frame ---
        param_frame = tk.LabelFrame(left_frame, text="Scenario Settings", padx=10, pady=10)
        param_frame.pack(fill=tk.X, pady=10)

        tk.Label(param_frame, text="Scenario Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.scenario_name_entry = tk.Entry(param_frame, width=20)
        self.scenario_name_entry.grid(row=0, column=1, padx=5, pady=5)
        self.scenario_name_entry.insert(0, "MyScenario3D")

        tk.Label(param_frame, text="Goal Range:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.goal_range_entry = tk.Entry(param_frame, width=10)
        self.goal_range_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.goal_range_entry.insert(0, "0.1")  # Default value

        tk.Label(param_frame, text="Distance Unit:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.distance_unit_entry = tk.Entry(param_frame, width=10)
        self.distance_unit_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.distance_unit_entry.insert(0, "0.05")  # Default value

        tk.Label(param_frame, text="Map Resolution:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.map_resolution_entry = tk.Entry(param_frame, width=10)
        self.map_resolution_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.map_resolution_entry.insert(0, "16")  # Default value

        # --- Obstacle Frame ---
        obstacle_frame = tk.LabelFrame(left_frame, text="Obstacle Controls", padx=10, pady=10)
        obstacle_frame.pack(fill=tk.X, pady=10)
        
        # Add obstacle section
        tk.Label(obstacle_frame, text="Add New Obstacle:").grid(row=0, column=0, columnspan=2, pady=(0,5), sticky="w")
        
        # Min point (x,y,z)
        tk.Label(obstacle_frame, text="Min Point (x,y,z):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        min_point_frame = tk.Frame(obstacle_frame)
        min_point_frame.grid(row=1, column=1, padx=5, pady=2)
        
        self.min_x = tk.Entry(min_point_frame, width=5)
        self.min_x.pack(side=tk.LEFT, padx=2)
        self.min_x.insert(0, "0.0")
        
        self.min_y = tk.Entry(min_point_frame, width=5)
        self.min_y.pack(side=tk.LEFT, padx=2)
        self.min_y.insert(0, "0.0")
        
        self.min_z = tk.Entry(min_point_frame, width=5)
        self.min_z.pack(side=tk.LEFT, padx=2)
        self.min_z.insert(0, "0.0")
        
        # Max point (x,y,z)
        tk.Label(obstacle_frame, text="Max Point (x,y,z):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        max_point_frame = tk.Frame(obstacle_frame)
        max_point_frame.grid(row=2, column=1, padx=5, pady=2)
        
        self.max_x = tk.Entry(max_point_frame, width=5)
        self.max_x.pack(side=tk.LEFT, padx=2)
        self.max_x.insert(0, "0.0")
        
        self.max_y = tk.Entry(max_point_frame, width=5)
        self.max_y.pack(side=tk.LEFT, padx=2)
        self.max_y.insert(0, "0.0")
        
        self.max_z = tk.Entry(max_point_frame, width=5)
        self.max_z.pack(side=tk.LEFT, padx=2)
        self.max_z.insert(0, "0.0")
        
        add_obstacle_button = tk.Button(obstacle_frame, text="Add Obstacle", command=self.add_obstacle)
        add_obstacle_button.grid(row=3, column=0, columnspan=2, pady=5)
        
        # --- Start/Goal Frame ---
        points_frame = tk.LabelFrame(left_frame, text="Set Start & Goal", padx=10, pady=10)
        points_frame.pack(fill=tk.X, pady=10)
        
        # Start point
        tk.Label(points_frame, text="Start point (x,y,z):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        start_point_frame = tk.Frame(points_frame)
        start_point_frame.grid(row=0, column=1, padx=5, pady=5)
        
        self.start_x = tk.Entry(start_point_frame, width=5)
        self.start_x.pack(side=tk.LEFT, padx=2)
        self.start_x.insert(0, "0.1")
        
        self.start_y = tk.Entry(start_point_frame, width=5)
        self.start_y.pack(side=tk.LEFT, padx=2)
        self.start_y.insert(0, "0.1")
        
        self.start_z = tk.Entry(start_point_frame, width=5)
        self.start_z.pack(side=tk.LEFT, padx=2)
        self.start_z.insert(0, "0.1")
        
        set_start_button = tk.Button(points_frame, text="Set Start", command=self.set_start)
        set_start_button.grid(row=0, column=2, padx=5)
        
        # Goal point
        tk.Label(points_frame, text="Goal point (x,y,z):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        goal_point_frame = tk.Frame(points_frame)
        goal_point_frame.grid(row=1, column=1, padx=5, pady=5)
        
        self.goal_x = tk.Entry(goal_point_frame, width=5)
        self.goal_x.pack(side=tk.LEFT, padx=2)
        self.goal_x.insert(0, "0.9")
        
        self.goal_y = tk.Entry(goal_point_frame, width=5)
        self.goal_y.pack(side=tk.LEFT, padx=2)
        self.goal_y.insert(0, "0.9")
        
        self.goal_z = tk.Entry(goal_point_frame, width=5)
        self.goal_z.pack(side=tk.LEFT, padx=2)
        self.goal_z.insert(0, "0.9")
        
        set_goal_button = tk.Button(points_frame, text="Set Goal", command=self.set_goal)
        set_goal_button.grid(row=1, column=2, padx=5)

        # --- Obstacle List Frame ---
        list_frame = tk.LabelFrame(left_frame, text="Obstacle List", padx=10, pady=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.obstacle_listbox = tk.Listbox(list_frame, height=12, width=40)
        self.obstacle_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self.obstacle_listbox.bind('<<ListboxSelect>>', self.on_obstacle_select)

        scrollbar = tk.Scrollbar(list_frame, orient="vertical")
        scrollbar.config(command=self.obstacle_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.obstacle_listbox.config(yscrollcommand=scrollbar.set)
        
        button_frame = tk.Frame(list_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        delete_obstacle_button = tk.Button(button_frame, text="Delete Selected", command=self.delete_obstacle)
        delete_obstacle_button.pack(side=tk.LEFT, padx=5)
        
        # --- Bottom controls ---
        controls_frame = tk.Frame(left_frame)
        controls_frame.pack(fill=tk.X, pady=10)

        clear_button = tk.Button(controls_frame, text="Clear All", command=self.clear_all)
        clear_button.pack(side=tk.LEFT, padx=10)

        save_button = tk.Button(controls_frame, text="Save Scenario", command=self.save_scenario)
        save_button.pack(side=tk.RIGHT, padx=10)

        # --- 3D Plot Frame ---
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Make canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = tk.Frame(right_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        # --- Status Bar ---
        self.status_var = tk.StringVar()
        self.status_bar = tk.Label(master, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var.set("Ready. Create a 3D scenario.")
        
        # Initialize the 3D plot
        self.init_plot()

    def init_plot(self):
        self.ax.clear()
        self.ax.set_title("3D Environment", fontsize=14)
        self.ax.set_xlabel('X', fontsize=10)
        self.ax.set_ylabel('Y', fontsize=10)
        self.ax.set_zlabel('Z', fontsize=10)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_zlim(0, 1)
        self.ax.grid(True)
        self.ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        
        # Add a reference cube
        self.draw_reference_cube()
        
        # Draw obstacles, start, and goal
        self.redraw_plot()
        
    def draw_reference_cube(self):
        # Draw a transparent reference cube to show the boundaries
        r = [0, 1]
        for s, e in [(0, 1), (0, 0), (1, 1), (1, 0)]:
            self.ax.plot3D([s, s], [e, e], r, color="gray", alpha=0.3)
            self.ax.plot3D([s, s], r, [e, e], color="gray", alpha=0.3)
            self.ax.plot3D(r, [s, s], [e, e], color="gray", alpha=0.3)

    def redraw_plot(self):
        # Clear but keep the reference cube
        for artist in self.ax.collections:
            artist.remove()
            
        # Draw obstacles
        for i, obs in enumerate(self.obstacles):
            min_point, max_point = obs
            self.draw_cube(min_point, max_point, color="blue", alpha=0.5, highlight=(i == self.selected_obstacle))
            
        # Draw start and goal points
        if self.start_point:
            self.ax.scatter(*self.start_point, color='green', s=100, marker='o', label='Start')
            
        if self.goal_point:
            self.ax.scatter(*self.goal_point, color='red', s=100, marker='*', label='Goal')
            
        # Show legend
        if self.start_point or self.goal_point:
            self.ax.legend()
            
        self.canvas.draw()
        
    def draw_cube(self, min_point, max_point, color="blue", alpha=0.5, highlight=False):
        # Create vertices of the cube
        x_min, y_min, z_min = min_point
        x_max, y_max, z_max = max_point
        
        # Define the 8 vertices of the cube
        vertices = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max]
        ])
        
        # Define the 6 faces using indices of vertices
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left face
            [vertices[1], vertices[2], vertices[6], vertices[5]]   # Right face
        ]
        
        # Create a Poly3DCollection object
        poly = Poly3DCollection(faces, alpha=alpha)
        edge_color = 'red' if highlight else 'black'
        line_width = 2 if highlight else 1
        
        poly.set_facecolor(color)
        poly.set_edgecolor(edge_color)
        poly.set_linewidth(line_width)
        
        # Add the collection to the axes
        self.ax.add_collection3d(poly)

    def add_obstacle(self):
        try:
            min_x = float(self.min_x.get())
            min_y = float(self.min_y.get())
            min_z = float(self.min_z.get())
            max_x = float(self.max_x.get())
            max_y = float(self.max_y.get())
            max_z = float(self.max_z.get())
            
            # Validate the coordinates
            if min_x >= max_x or min_y >= max_y or min_z >= max_z:
                messagebox.showerror("Invalid Dimensions", "Max values must be greater than min values.")
                return
                
            if min_x < 0 or min_y < 0 or min_z < 0 or max_x > 1 or max_y > 1 or max_z > 1:
                messagebox.showwarning("Out of Bounds", "Coordinates should be between 0 and 1.")
                
            # Create the obstacle
            min_point = [min_x, min_y, min_z]
            max_point = [max_x, max_y, max_z]
            obstacle = [min_point, max_point]
            
            self.obstacles.append(obstacle)
            self.update_obstacle_listbox()
            self.redraw_plot()
            
            self.status_var.set(f"Obstacle added: {min_point} to {max_point}")
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values for coordinates.")

    def set_start(self):
        try:
            x = float(self.start_x.get())
            y = float(self.start_y.get())
            z = float(self.start_z.get())
            
            if x < 0 or y < 0 or z < 0 or x > 1 or y > 1 or z > 1:
                messagebox.showwarning("Out of Bounds", "Coordinates should be between 0 and 1.")
            
            self.start_point = [x, y, z]
            self.redraw_plot()
            self.status_var.set(f"Start point set at {self.start_point}")
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values for coordinates.")

    def set_goal(self):
        try:
            x = float(self.goal_x.get())
            y = float(self.goal_y.get())
            z = float(self.goal_z.get())
            
            if x < 0 or y < 0 or z < 0 or x > 1 or y > 1 or z > 1:
                messagebox.showwarning("Out of Bounds", "Coordinates should be between 0 and 1.")
            
            self.goal_point = [x, y, z]
            self.redraw_plot()
            self.status_var.set(f"Goal point set at {self.goal_point}")
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values for coordinates.")
            
    def update_obstacle_listbox(self):
        self.obstacle_listbox.delete(0, tk.END)
        for i, obs in enumerate(self.obstacles):
            min_point, max_point = obs
            self.obstacle_listbox.insert(
                tk.END, 
                f"{i+1}: [{min_point[0]:.2f}, {min_point[1]:.2f}, {min_point[2]:.2f}] to [{max_point[0]:.2f}, {max_point[1]:.2f}, {max_point[2]:.2f}]"
            )
    
    def on_obstacle_select(self, event):
        selected_indices = self.obstacle_listbox.curselection()
        if selected_indices:
            self.selected_obstacle = selected_indices[0]
            self.redraw_plot()
            
            # Update the entry fields with the selected obstacle's values
            min_point, max_point = self.obstacles[self.selected_obstacle]
            
            self.min_x.delete(0, tk.END)
            self.min_x.insert(0, f"{min_point[0]}")
            self.min_y.delete(0, tk.END)
            self.min_y.insert(0, f"{min_point[1]}")
            self.min_z.delete(0, tk.END)
            self.min_z.insert(0, f"{min_point[2]}")
            
            self.max_x.delete(0, tk.END)
            self.max_x.insert(0, f"{max_point[0]}")
            self.max_y.delete(0, tk.END)
            self.max_y.insert(0, f"{max_point[1]}")
            self.max_z.delete(0, tk.END)
            self.max_z.insert(0, f"{max_point[2]}")
        else:
            self.selected_obstacle = None

    def delete_obstacle(self):
        if self.selected_obstacle is not None:
            del self.obstacles[self.selected_obstacle]
            self.selected_obstacle = None
            self.update_obstacle_listbox()
            self.redraw_plot()
            self.status_var.set("Obstacle deleted")
        else:
            messagebox.showwarning("Delete Error", "No obstacle selected.")
    
    def clear_all(self):
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear everything?"):
            self.obstacles = []
            self.start_point = None
            self.goal_point = None
            self.selected_obstacle = None
            self.update_obstacle_listbox()
            self.init_plot()
            self.status_var.set("Cleared all elements. Ready for new scenario.")

    def save_scenario(self):
        scenario_name = self.scenario_name_entry.get().strip()
        if not scenario_name:
            messagebox.showerror("Save Error", "Please enter a scenario name.")
            return

        if not self.start_point:
            messagebox.showerror("Save Error", "Please set a start point.")
            return

        if not self.goal_point:
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
            "start": self.start_point,
            "goal": self.goal_point,
            "mapResolution": map_resolution,
            "goalRange": goal_range,
            "distanceUnit": distance_unit,
            "obstacles": self.obstacles
        }

        # --- File Handling ---
        filename = "scenarios_3d.json"
        filepath = os.path.join(os.path.dirname(__file__), filename)

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
    app = MapMaker3DApp(root)
    root.mainloop()