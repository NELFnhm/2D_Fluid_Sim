import taichi as ti

# Note: all physical properties are in SI units (s for time, m for length, kg for mass, etc.)
global_params = {
    'mode' : 'apic',                            # pic, apic, flip
    'dt' : 0.01,                                # Time step
    'g' : (0.0, -9.8),                          # Body force
    'rho': 1000.0,                              # Density of the fluid
    'grid_size' : (128, 128),                   # Grid size (integer)
    'cell_extent': 0.1,                         # Extent of a single cell. grid_extent equals to the product of grid_size and cell_extent

    'num_jacobi_iter' : 20,                     # Number of iterations for pressure solving
    'damped_jacobi_weight' : 1.0,               # Damping weighte in damped jacobi

    'particles_per_cell': 4,                    # Number of particles per cell
}

FLUID = 0
AIR = 1
SOLID = 2

@ti.data_oriented
class Simulator(object):
    def __init__(self, params : dict = global_params):
        def get_param(key:str, default_val=None):
            return params[key] if key in params else default_val

        # Time step
        self.dt = get_param('dt')
        # Body force (gravity)
        self.g = ti.Vector(get_param('g'), dt=ti.f32)

        self.paused = True

        # parameters that are fixed (changing these after starting the simulatin will not have an effect!)
        self.grid_size = ti.Vector(get_param('grid_size'), dt=ti.i32)
        self.cell_extent = get_param('cell_extent')
        self.grid_extent = self.grid_size * self.cell_extent
        self.dx = self.cell_extent

        self.rho = get_param('rho')

        self.num_jacobi_iter = get_param('num_jacobi_iter')
        self.damped_jacobi_weight = get_param('damped_jacobi_weight')

        # simulation state
        self.cur_step = 0
        self.t = 0.0

        # friction coefficient
        self.mu = 0.6
        # boundary friction coefficient
        self.b_mu  = 0.8

        self.init_fields()

        self.scene_init = get_param('scene_init')
        self.particles_per_cell = get_param('particles_per_cell')
            
        self.init_particles((0.1, 0.1), (0.5, 0.5))

        self.reset()
        print(f"Initialized simulator successfully")

    def init_fields(self):
        # MAC grid
        self.pressure = ti.field(ti.f32, shape=self.grid_size)

        # mark each grid as FLUID = 0, AIR = 1 or SOLID = 2
        self.cell_type = ti.field(ti.i32, shape=self.grid_size)
        self.cell_type.fill(AIR)

        for i in range(self.grid_size[0]):
            self.cell_type[i, 0] = SOLID
            self.cell_type[i, self.grid_size[1]-1] = SOLID
        for j in range(self.grid_size[1]):
            self.cell_type[0, j] = SOLID
            self.cell_type[self.grid_size[0]-1, j] = SOLID

        self.grid_velocity_x = ti.field(ti.f32, shape=(self.grid_size[0] + 1, self.grid_size[1]))
        self.grid_velocity_y = ti.field(ti.f32, shape=(self.grid_size[0], self.grid_size[1] + 1))

        self.grid_velocity_x_last = ti.field(ti.f32, shape=(self.grid_size[0] + 1, self.grid_size[1]))
        self.grid_velocity_y_last = ti.field(ti.f32, shape=(self.grid_size[0], self.grid_size[1] + 1))

        self.grid_weight_x = ti.field(ti.f32, shape=(self.grid_size[0] + 1, self.grid_size[1]))
        self.grid_weight_y = ti.field(ti.f32, shape=(self.grid_size[0], self.grid_size[1] + 1))

        self.clear_grid()
        self.divergence = ti.field(ti.f32, shape=self.grid_size)
        self.new_pressure = ti.field(ti.f32, shape=self.grid_size)

    def init_particles(self, range_min, range_max):
        gird_size_num, _ = self.grid_size
        min_x, min_y = range_min
        max_x, max_y = range_max
        range_min = ti.Vector([min_x*gird_size_num, min_y*gird_size_num], dt=ti.i32)
        range_max = ti.Vector([max_x*gird_size_num, max_y*gird_size_num], dt=ti.i32)
        print(f'particles range: {range_min} - {range_max}')
        range_min = ti.max(range_min, 1)
        range_max = ti.min(range_max, self.grid_size-1)

        # Number of particles
        range_size = range_max - range_min
        self.num_particles = range_size.x * range_size.y
        print(f'total particel num:{self.num_particles}')

        # Particles
        self.particles_position = ti.Vector.field(2, dtype=ti.f32, shape=self.num_particles * 4)
        self.particles_velocity = ti.Vector.field(2, dtype=ti.f32, shape=self.num_particles * 4)
        self.particles_affine_C = ti.Matrix.field(2, 2, dtype=ti.f32, shape=self.num_particles * 4)

        self.init_particles_kernel(range_min[0], range_min[1], range_max[0], range_max[1])

    @ti.kernel
    def init_particles_kernel(self, range_min_x:ti.i32, range_min_y:ti.i32,range_max_x:ti.i32,range_max_y:ti.i32):
        particles_per_cell = self.particles_per_cell
        for p in self.particles_position:
            p1 = p // particles_per_cell
            j = p1 % (range_max_y - range_min_y) + range_min_y
            i = (p1 // (range_max_y - range_min_y)) % (range_max_x - range_min_x) + range_min_x
            if self.cell_type[i, j] != SOLID:
                self.cell_type[i, j] = FLUID
            self.particles_position[p] = (ti.Vector([i,j]) + ti.Vector([ti.random(), ti.random()])) * self.cell_extent
    
    def reset(self):
        # Reset simulation state
        self.cur_step = 0
        self.t = 0.0

    def step(self):
        # print(f"Sim step {self.cur_step} at time {self.t:.2f}s")
        self.cur_step += 1
        self.t += self.dt

        # Clear the grid for each step
        self.clear_grid()

        # Scatter properties (mainly velocity) from particle to grid
        self.p2g()

        # Clear solid boundary velocity
        self.enforce_boundary_condition()

        # Apply body force
        self.apply_force()

        # Compute velocity divergence
        self.compute_divergence()

        # Solve the poisson equation to get pressure
        self.solve_pressure()

        # Accelerate velocity using the solved pressure
        self.project_velocity()

        # Gather properties (mainly velocity) from grid to particle
        self.g2p()

        # Advect particles
        self.advect_particles()

        # Mark grid cell type as FLUID, AIR or SOLID (boundary)
        self.mark_cell_type()

    # Clear grid velocities and weights to 0
    def clear_grid(self):
        self.grid_velocity_x.fill(0.0)
        self.grid_velocity_y.fill(0.0)

        self.grid_weight_x.fill(0.0)
        self.grid_weight_y.fill(0.0)


    # Helper
    @ti.func
    def is_valid(self, i, j):
        return 0 <= i < self.grid_size[0] and 0 <= j < self.grid_size[1]

    @ti.func
    def is_fluid(self, i, j):
        return self.is_valid(i, j) and self.cell_type[i, j] == FLUID

    @ti.func
    def is_air(self, i, j):
        return self.is_valid(i, j) and self.cell_type[i, j] == AIR

    @ti.func
    def is_solid(self, i, j):
        return self.is_valid(i, j) and self.cell_type[i, j] == SOLID
    # ###########################################################
    # Kernels launched in one step
    @ti.kernel
    def apply_force(self):
        for i, j in self.grid_velocity_y:
            if j > 1:
                self.grid_velocity_y[i, j] -= 9.8 * self.dt

    @ti.kernel
    def p2g(self):
        for p in self.particles_position:
            xp = self.particles_position[p]
            vp = self.particles_velocity[p]
            cp = self.particles_affine_C[p]
            idx = xp/self.dx
            base = ti.cast(ti.floor(idx), dtype=ti.i32)
            frac = idx - base
            self.interp_grid(base, frac, vp, cp)

        for i, j in self.grid_velocity_x:
            v = self.grid_velocity_x[i, j]
            w = self.grid_weight_x[i,j]
            self.grid_velocity_x[i, j] = v / w if w > 0.0 else 0.0
        for i, j in self.grid_velocity_y:
            v = self.grid_velocity_y[i, j]
            w = self.grid_weight_y[i,j]
            self.grid_velocity_y[i, j] = v / w if w > 0.0 else 0.0

    @ti.kernel
    def g2p(self):
        for p in self.particles_position:
            xp = self.particles_position[p]
            idx = xp/self.dx
            base = ti.cast(ti.floor(idx), dtype=ti.i32)
            frac = idx - base
            self.interp_particle(base, frac, p)

    def solve_pressure(self):
        for i in range(self.num_jacobi_iter):
            self.jacobi_iter()
            self.pressure.copy_from(self.new_pressure)

    @ti.kernel
    def project_velocity(self):
        scale = self.dt / (self.rho * self.dx)
        for i, j in ti.ndrange(self.grid_size[0], self.grid_size[1]):
            if self.is_fluid(i - 1, j) or self.is_fluid(i, j):
                if self.is_solid(i - 1, j) or self.is_solid(i, j):
                    self.grid_velocity_x[i, j] = 0 # u_solid = 0
                else:
                    self.grid_velocity_x[i, j] -= scale * (self.pressure[i, j] - self.pressure[i - 1, j])

            if self.is_fluid(i, j - 1) or self.is_fluid(i, j):
                if self.is_solid(i, j - 1) or self.is_solid(i, j):
                    self.grid_velocity_y[i, j] = 0
                else:
                    self.grid_velocity_y[i, j] -= scale * (self.pressure[i, j] - self.pressure[i, j - 1])

    @ti.kernel
    def advect_particles(self):
        # Forward Euler
        for p in self.particles_position:
            self.particles_position[p] += self.particles_velocity[p] * self.dt

        for p in self.particles_position:
            pos = self.particles_position[p]
            v = self.particles_velocity[p]

            for i in ti.static(range(2)):
                if pos[i] <= self.cell_extent:
                    pos[i] = self.cell_extent
                    v[i] = 0
                if pos[i] >= self.grid_extent[i]-self.cell_extent:
                    pos[i] = self.grid_extent[i]-self.cell_extent
                    v[i] = 0

                if pos[0] <= self.cell_extent * 2 or pos[0] >= self.grid_extent[0]-2 * self.cell_extent:
                    vn = v[0]
                    v[0] = 0.0
                    vt = ti.Vector([v[0], v[1]], dt=ti.f32)
                    vt = ti.max(0, 1 - self.b_mu * ti.abs(vn) / vt.norm()) * vt
                    v[1] = vt[1]

                if pos[1] <= self.cell_extent * 2 or pos[1] >= self.grid_extent[1]-2 * self.cell_extent:
                    vn = v[1]
                    v[1] = 0.0
                    vt = ti.Vector([v[0], v[1]], dt=ti.f32)
                    vt = ti.max(0, 1 - self.b_mu * ti.abs(vn) / vt.norm()) * vt
                    v[0] = vt[0]

            self.particles_position[p] = pos
            self.particles_velocity[p] = v

    @ti.kernel
    def enforce_boundary_condition(self):
        for i in ti.ndrange(self.grid_size[0]):
            self.grid_velocity_y[i, 0] = 0
            self.grid_velocity_y[i, 1] = 0
            self.grid_velocity_y[i, self.grid_size[1]-1] = 0
            self.grid_velocity_y[i, self.grid_size[1]] = 0

        for j in ti.ndrange(self.grid_size[1]):
            self.grid_velocity_x[0, j] = 0
            self.grid_velocity_x[1, j] = 0
            self.grid_velocity_x[self.grid_size[0]-1, j] = 0
            self.grid_velocity_x[self.grid_size[0], j] = 0
    

    @ti.kernel
    def mark_cell_type(self):
        for i, j in self.cell_type:
            if not self.is_solid(i, j):
                self.cell_type[i, j] = AIR

        for p in self.particles_position:
            xp = self.particles_position[p]
            idx = ti.cast(ti.floor(xp / self.dx), ti.i32)

            if not self.is_solid(idx[0], idx[1]):
                self.cell_type[idx] = FLUID

    # ###########################################################
    # Funcs called by kernels

    # Spline functions for interpolation
    # Input x should be non-negative (abs)

    # Quadratic B-spline
    # 0.75-x^2,         |x| in [0, 0.5)
    # 0.5*(1.5-|x|)^2,  |x| in [0.5, 1.5)
    # 0,                |x| in [1.5, inf)
    @ti.func
    def quadratic_kernel(self, x):
        w = ti.Vector([0.0 for _ in ti.static(range(2))])
        for i in ti.static(range(2)):  # todo: maybe we should not assume x.n==3 here
            if x[i] < 0.5:
                w[i] = 0.75 - x[i]**2
            elif x[i] < 1.5:
                w[i] = 0.5 * (1.5 - x[i])**2
            else:
                w[i] = 0.0
        return w


    @ti.func
    def interp_grid(self, base, frac, vp, cp):
        # Quadratic

        # Index on sides
        idx_side = [base-1, base, base+1, base+2]
        # Weight on sides
        w_side = [
            self.quadratic_kernel(1.0+frac), 
            self.quadratic_kernel(frac), 
            self.quadratic_kernel(1.0-frac), 
            self.quadratic_kernel(2.0-frac)
        ]
        # Index on centers
        idx_center = [base-1, base, base+1]
        # Weight on centers
        w_center = [
            self.quadratic_kernel(0.5+frac), 
            self.quadratic_kernel(ti.abs(0.5-frac)), 
            self.quadratic_kernel(1.5-frac)
        ]

        for i in ti.static(range(4)):
            for j in ti.static(range(3)):
                w = w_side[i].x * w_center[j].y
                idx = (idx_side[i].x, idx_center[j].y)
                self.grid_velocity_x[idx] += vp.x * w
                dpos = (ti.Vector([i-1, j-0.5]) - frac) * self.dx
                self.grid_velocity_x[idx] += w * (cp @ dpos).x
                self.grid_weight_x[idx] += w

        for i in ti.static(range(3)):
            for j in ti.static(range(4)):
                w = w_center[i].x * w_side[j].y
                idx = (idx_center[i].x, idx_side[j].y)
                self.grid_velocity_y[idx] += vp.y * w
                dpos = (ti.Vector([i-0.5, j-1]) - frac) * self.dx
                self.grid_velocity_y[idx] += w * (cp @ dpos).y
                self.grid_weight_y[idx] += w

    @ti.func
    def interp_particle(self, base, frac, p):
        # Index on sides
        idx_side = [base-1, base, base+1, base+2]
        # Weight on sides
        w_side = [
            self.quadratic_kernel(1.0+frac), 
            self.quadratic_kernel(frac), 
            self.quadratic_kernel(1.0-frac), 
            self.quadratic_kernel(2.0-frac)
        ]
        # Index on centers
        idx_center = [base-1, base, base+1]
        # Weight on centers
        w_center = [
            self.quadratic_kernel(0.5+frac), 
            self.quadratic_kernel(ti.abs(0.5-frac)), 
            self.quadratic_kernel(1.5-frac)
        ]

        wx, wy, vx, vy = 0.0, 0.0, 0.0, 0.0
        C_x, C_y = ti.Matrix.zero(ti.f32, 2), ti.Matrix.zero(ti.f32, 2)

        for i in ti.static(range(4)):
            for j in ti.static(range(3)):
                w = w_side[i].x * w_center[j].y
                idx = (idx_side[i].x, idx_center[j].y)
                vtemp = self.grid_velocity_x[idx] * w
                vx += vtemp
                dpos = ti.Vector([i-1, j-0.5]) - frac
                C_x += 4 * vtemp * dpos  / self.dx
                wx += w

        for i in ti.static(range(3)):
            for j in ti.static(range(4)):
                w = w_center[i].x * w_side[j].y
                idx = (idx_center[i].x, idx_side[j].y)
                vtemp = self.grid_velocity_y[idx] * w
                vy += vtemp
                dpos = ti.Vector([i-0.5, j-1]) - frac
                C_y += 4 * vtemp * dpos / self.dx
                wy += w

        self.particles_velocity[p] = ti.Vector([vx/wx, vy/wy])
        self.particles_affine_C[p] = ti.Matrix.rows([C_x/wx, C_y/wy])

    @ti.kernel
    def compute_divergence(self):
        for i, j in self.divergence:
            if not self.is_solid(i, j):
                self.divergence[i, j] = (
                    (self.grid_velocity_x[i + 1, j] - self.grid_velocity_x[i, j]) 
                    + (self.grid_velocity_y[i, j + 1] - self.grid_velocity_y[i, j])
                )
            else:
                self.divergence[i, j] = 0.0
            self.divergence[i, j] /= self.dx

    @ti.kernel
    def jacobi_iter(self):
        for i, j in self.pressure:
            if self.is_fluid(i, j):
                div = self.divergence[i, j]

                p_x1 = self.pressure[i - 1, j]
                p_x2 = self.pressure[i + 1, j]
                p_y1 = self.pressure[i, j - 1]
                p_y2 = self.pressure[i, j + 1]
                n = 4
                if self.is_solid(i-1, j):
                    p_x1 = 0.0
                    n -= 1
                if self.is_solid(i+1, j):
                    p_x2 = 0.0
                    n -= 1
                if self.is_solid(i, j-1):
                    p_y1 = 0.0
                    n -= 1
                if self.is_solid(i, j+1):
                    p_y2 = 0.0
                    n -= 1

                self.new_pressure[i, j] = (
                    (1 - self.damped_jacobi_weight) * self.pressure[i, j]
                    + self.damped_jacobi_weight * (p_x1 + p_x2 + p_y1 + p_y2 - div * self.rho / self.dt * (self.dx ** 2)) / n
                    )                               
            else:
                self.new_pressure[i, j] = 0.0


