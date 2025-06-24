import taichi as ti
import numpy as np
import time
from simulator import Simulator # Assuming simulator.py is in the same directory or accessible
# import os # Removed as it's not used for output directories anymore

def rgb_to_hex(rgb_tuple):
    """Converts an RGB tuple (float 0-1) to a hex integer (0xRRGGBB)."""
    r, g, b = rgb_tuple
    return (int(r * 255) << 16) + (int(g * 255) << 8) + int(b * 255)

@ti.data_oriented
class SimulationGUI_2D(object):
    def __init__(self, sim: Simulator, title: str = "2D APIC Fluid Simulator (ti.GUI)", 
                 window_resolution=(800, 800), save_output: bool = False):
        
        self.sim = sim
        self.window_resolution = window_resolution
        
        # Simulation starts automatically
        self.sim.paused = False 

        self.particle_color_rgb = (0.3, 0.6, 0.9) 
        self.particle_color_hex = rgb_to_hex(self.particle_color_rgb)

        self.boundary_color_rgb = (0.6, 0.6, 0.6) 
        self.boundary_color_hex = rgb_to_hex(self.boundary_color_rgb) 
        
        self.background_color_rgb = (0.1, 0.12, 0.15) 
        self.background_color_hex = rgb_to_hex(self.background_color_rgb)
        
        self.grid_size = self.sim.grid_size
        x_grid, y_grid = np.meshgrid(
            np.linspace(0, 1, self.grid_size[0]), 
            np.linspace(0, 1, self.grid_size[1]),  
            indexing='ij'
        )
        self.grid_pos = np.column_stack((x_grid.flatten(), y_grid.flatten()))
        
        self.particle_world_radius = self.sim.dx * 0.25 # Particle radius in world units
        
        self.gui = ti.GUI(title, self.window_resolution, background_color=self.background_color_hex)
        
        self.video_manager = None
        self.output_dir = None
        if save_output:
            file_date = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
            self.output_dir = f"./output/{file_date}"
            self.video_manager = ti.tools.VideoManager(output_dir=self.output_dir,
                                framerate=25,
                                automatic_build=False)
        
    def render_content(self, show_pressure = True):
        world_w = self.sim.grid_extent.x
        world_h = self.sim.grid_extent.y

        # Draw boundaries
        self.gui.line(begin=(0.0, 0.0), end=(1.0, 0.0), radius=1, color=self.boundary_color_hex)
        self.gui.line(begin=(0.0, 1.0), end=(1.0, 1.0), radius=1, color=self.boundary_color_hex)
        self.gui.line(begin=(0.0, 0.0), end=(0.0, 1.0), radius=1, color=self.boundary_color_hex)
        self.gui.line(begin=(1.0, 0.0), end=(1.0, 1.0), radius=1, color=self.boundary_color_hex)

        particles_pos_world = self.sim.particles_position.to_numpy()
        
        # particles_pos_world should be (N,2) from sim.get_particle_positions_numpy()
        # If it's (N,2), then particles_pos_world[:, 0] is (N,)
        # Ensure pos_normalized_x and pos_normalized_y are 1D arrays (shape (N,))
        pos_normalized_x = (particles_pos_world[:, 0] / world_w).flatten()
        pos_normalized_y = (particles_pos_world[:, 1] / world_h).flatten()
        
        # Stack the 1D arrays to get an (N, 2) array
        # axis=1 will stack them as columns: [[x0,y0], [x1,y1], ...]
        normalized_particles_2d_coords = np.stack((pos_normalized_x, pos_normalized_y), axis=1).astype(np.float32)
        # At this point, normalized_particles_2d_coords.shape should be (N, 2)
        
        # Reshape to (N, 2, 1) to satisfy the assert pos.shape[2] == 1 in gui.circles
        final_pos_for_gui = np.expand_dims(normalized_particles_2d_coords, axis=2)
        
        pixel_radius = max(1, int(self.particle_world_radius / world_h * self.window_resolution[1]))
        
        if show_pressure:
            pressure = self.sim.pressure.to_numpy().flatten()  # Flatten to 1D array
            min_pressure, max_pressure = np.min(pressure), np.max(pressure)
            
            if max_pressure > min_pressure:
                pressure = (pressure - min_pressure) / (max_pressure - min_pressure)
            else:
                pressure = np.zeros_like(pressure)

            r = (pressure * 255).astype(np.int32) << 16
            g = np.zeros_like(r)  # 绿色分量为0
            b = ((1.0 - pressure) * 255).astype(np.int32)
            
            # 组合成最终的颜色整数数组
            colors = r + g + b

            self.gui.circles(self.grid_pos, 
                            radius=2.5*pixel_radius, 
                            color=colors)
            self.gui.circles(final_pos_for_gui, 
                            radius=0.7*pixel_radius, 
                            color=self.particle_color_hex)
        else:
            self.gui.circles(final_pos_for_gui, 
                            radius=pixel_radius, 
                            color=self.particle_color_hex)
        

    def run(self):
        steps_per_render_frame = 4 
        frame = 0
        total_frames = 500
        while self.gui.running and not self.gui.get_event(ti.GUI.ESCAPE) and frame <= total_frames:
            for _ in range(steps_per_render_frame):
                self.sim.step()
            
            self.gui.clear(self.background_color_hex) 
            self.render_content() 
            frame += 1
            if self.video_manager:
                self.video_manager.write_frame(self.gui.get_image())
                print(f"frame:{frame - 1}/{total_frames}", end='\r')
            else:
                self.gui.show()
        if self.video_manager:
            self.video_manager.make_video(gif=True, mp4=True)
            print(f"Video saved to {self.output_dir}")
            
    def run2(self, show_pressure=True):
        steps_per_render_frame = 4 
        frame = 0
        # total_frames = 500
        total_frames = 200
        save_frame_list = [125, 138, 150, 162, 175, 182]
        save_frame_num = 0
        while self.gui.running and not self.gui.get_event(ti.GUI.ESCAPE) and frame <= total_frames:
            for _ in range(steps_per_render_frame):
                self.sim.step()
            
            self.gui.clear(self.background_color_hex) 
            
            frame += 1
            print(f"frame:{frame - 1}/{total_frames}; save frame:{save_frame_num}/{len(save_frame_list)}", end='\r')
            if frame in save_frame_list:
                self.render_content(show_pressure=show_pressure) 
                self.video_manager.write_frame(self.gui.get_image())
                save_frame_num += 1
            


