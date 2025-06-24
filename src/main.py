# from simulator import Simulator
from simulator import Simulator
from gui import SimulationGUI_2D 
import taichi as ti

# Configuration
WINDOW_RESOLUTION = (1024,1024) # GUI Window resolution

if __name__ == "__main__":
    ti.init(arch=ti.cuda, device_memory_fraction=0.8)

    sim = Simulator() 
    # gui = SimulationGUI_2D(sim, window_resolution=WINDOW_RESOLUTION, save_output=True)
    # gui = SimulationGUI_2D(sim, window_resolution=WINDOW_RESOLUTION)
    
    # gui.run()
    
    gui = SimulationGUI_2D(sim, window_resolution=WINDOW_RESOLUTION, save_output=True)
    # gui.run2()
    gui.run2(False)
    
