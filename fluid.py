import taichi as ti
import numpy as np
import time
import random
import argparse


ti.init(arch=ti.cpu, default_fp=ti.f32)


@ti.data_oriented
class World:
    FLUID = 0
    AIR = 1
    SOLID = 2

    def __init__(self, height, width, cell_res, blend_ratio, p_count, density):
        self.height = height
        self.width = width
        self.cell_size = 1.0 / cell_res
        self.x_count = width * cell_res
        self.y_count = height * cell_res
        self.p_count = p_count
        self.blend_ratio = blend_ratio
        self.density = density

        self.u = ti.field(dtype=ti.f32, shape=(self.x_count + 1, self.y_count))
        self.u_prev = ti.field(dtype=ti.f32, shape=(self.x_count + 1, self.y_count))
        self.u_weight = ti.field(dtype=ti.f32, shape=(self.x_count + 1, self.y_count))

        self.v = ti.field(dtype=ti.f32, shape=(self.x_count, self.y_count + 1))
        self.v_prev = ti.field(dtype=ti.f32, shape=(self.x_count, self.y_count + 1))
        self.v_weight = ti.field(dtype=ti.f32, shape=(self.x_count, self.y_count + 1))

        self.p = ti.field(dtype=ti.f32, shape=(self.x_count, self.y_count))

        self.cell_type = ti.field(dtype=ti.f32, shape=(self.x_count, self.y_count))

        self.particle_pos = ti.Vector.field(2, dtype=ti.f32,
                                            shape=(self.x_count, self.y_count, self.p_count, self.p_count))
        self.particle_v = ti.Vector.field(2, dtype=ti.f32,
                                          shape=(self.x_count, self.y_count, self.p_count, self.p_count))
        self.particle_type = ti.field(dtype=ti.f32, shape=(self.x_count, self.y_count, self.p_count, self.p_count))

        self.b = ti.field(dtype=ti.f32, shape=(self.x_count, self.y_count))
        self.Adiag = ti.field(dtype=ti.f32, shape=(self.x_count, self.y_count))
        self.Ax = ti.field(dtype=ti.f32, shape=(self.x_count, self.y_count))
        self.Ay = ti.field(dtype=ti.f32, shape=(self.x_count, self.y_count))

        self.pressue = ti.field(dtype=ti.f32, shape=(self.x_count, self.y_count))
        self.res = ti.field(dtype=ti.f32, shape=(self.x_count, self.y_count))
        self.s = ti.field(dtype=ti.f32, shape=(self.x_count, self.y_count))
        self.As = ti.field(dtype=ti.f32, shape=(self.x_count, self.y_count))



    @ti.func
    def init(self):
        for i, j in self.u:
            self.u[i, j] = 0.0
            self.u_prev[i, j] = 0.0

        for i, j in self.v:
            self.v[i, j] = 0.0
            self.v_prev[i, j] = 0.0

        for i, j in self.p:
            self.p[i, j] = 0.0

        for i, j, x, y in self.particle_pos:
            if self.cell_type[i, j] == World.FLUID:
                self.particle_type[i, j, x, y] = 1
            else:
                self.particle_type[i, j, x, y] = 0

            px = i * self.cell_size + (x + random.random()) * self.cell_size / self.p_count
            py = j * self.cell_size + (y + random.random()) * self.cell_size / self.p_count

            self.particle_pos[i, j, x, y] = ti.Vector([px, py])
            self.particle_v[i, j, x, y] = ti.Vector([0.0, 0.0])

    @ti.func
    def is_wall(self, i, j):
        return i == 0 or i == self.x_count - 1 or j == 0 or j == self.y_count - 1

    @ti.kernel
    def setup_dambreak(self, x: ti.f32, y: ti.f32):
        cellx = int(x / self.cell_size)
        celly = int(y / self.cell_size)
        for i, j in self.cell_type:
            if self.is_wall(i, j):
                self.cell_type[i, j] = World.SOLID
            else:
                if i < cellx and j < celly:
                    self.cell_type[i, j] = World.FLUID
                else:
                    self.cell_type[i, j] = World.AIR
        self.init()

    @ti.kernel
    def setup_2dambreak(self, x: ti.f32, y: ti.f32):
        cellx = int(x / self.cell_size)
        celly = int(y / self.cell_size)
        for i, j in self.cell_type:
            if self.is_wall(i, j):
                self.cell_type[i, j] = World.SOLID
            else:
                if (i < cellx/2 or i > self.x_count - cellx/2)and j < celly:
                    self.cell_type[i, j] = World.FLUID
                else:
                    self.cell_type[i, j] = World.AIR
        self.init()

    @ti.kernel
    def setup_stepdown(self, x: ti.f32, y: ti.f32):
        cellx = int(x / self.cell_size)
        celly = int(y / self.cell_size)
        for i, j in self.cell_type:
            if self.is_wall(i, j) or (j == self.y_count//3 and i > self.x_count//2) or (j == self.y_count//3 * 2 and i < self.x_count//2):
                self.cell_type[i, j] = World.SOLID
            else:
                if i < cellx and celly < j < self.y_count - 10:
                    self.cell_type[i, j] = World.FLUID
                else:
                    self.cell_type[i, j] = World.AIR
        self.init()

    @ti.kernel
    def setup_fountain(self):
        for i, j in self.cell_type:
            if self.is_wall(i, j):
                self.cell_type[i, j] = World.SOLID
            else:
                if j < self.y_count / 4:
                    self.cell_type[i, j] = World.FLUID
                else:
                    self.cell_type[i, j] = World.AIR
        self.init()
        for i,j in self.v:
            if not self.is_wall(i,j) and j < self.y_count/4 and self.x_count/3 < i < self.x_count/1.5:
                self.v[i,j] = 0

    @ti.kernel
    def boundary_condition(self):
        for i, j in self.u:
            if self.cell_type[i - 1, j] == World.SOLID or self.cell_type[i, j] == World.SOLID:
                self.u[i, j] = 0.0

        for i, j in self.v:
            if self.cell_type[i, j - 1] == World.SOLID or self.cell_type[i, j] == World.SOLID:
                self.v[i, j] = 0.0

    @ti.kernel
    def external_force(self, dt: ti.f32):
        for i, j in self.v:
            self.v[i, j] += -9.8 * dt

    @ti.kernel
    def advect(self, dt: ti.f32):
        for p in ti.grouped(self.particle_pos):
            if self.particle_type[p] == 1:
                position = self.particle_pos[p]
                velocity = self.particle_v[p]

                position += velocity * dt

                if position[0] <= self.cell_size:
                    position[0] = self.cell_size
                    velocity[0] = 0
                if position[0] >= self.width - self.cell_size:
                    position[0] = self.width - self.cell_size
                    velocity[0] = 0
                if position[1] <= self.cell_size:
                    position[1] = self.cell_size
                    velocity[1] = 0
                if position[1] >= self.height - self.cell_size:
                    position[1] = self.height - self.cell_size
                    velocity[1] = 0

                self.particle_pos[p] = position
                self.particle_v[p] = velocity

    @ti.kernel
    def label(self):
        for i, j in self.cell_type:
            if self.cell_type[i, j] != World.SOLID:
                self.cell_type[i, j] = World.AIR

        for i, j, x, y in self.particle_pos:
            if self.particle_type[i, j, x, y] == 1:
                pxy = self.particle_pos[i, j, x, y]
                cell_pos = ti.cast(pxy / self.cell_size, int)
                if self.cell_type[cell_pos] != World.SOLID:
                    self.cell_type[cell_pos] = World.FLUID

    @ti.kernel
    def grid_to_particle(self):
        for p in ti.grouped(self.particle_pos):
            if self.particle_type[p] == 1:
                pxy = self.particle_pos[p]
                u_cell_pos = ti.cast(pxy / self.cell_size - ti.Vector([0.5, 1]), int)
                v_cell_pos = ti.cast(pxy / self.cell_size - ti.Vector([1, 0.5]), int)
                u_offset = pxy / self.cell_size - (u_cell_pos + ti.Vector([0.0, 0.5]))
                v_offset = pxy / self.cell_size - (v_cell_pos + ti.Vector([0.5, 0.0]))
                weight_u = [0.5 * (1.5 - u_offset) ** 2, 0.75 - (u_offset - 1) ** 2, 0.5 * (u_offset - 0.5) ** 2]
                weight_v = [0.5 * (1.5 - v_offset) ** 2, 0.75 - (v_offset - 1) ** 2, 0.5 * (v_offset - 0.5) ** 2]
                u_pic = 0.0
                v_pic = 0.0
                u_flip = 0.0
                v_flip = 0.0
                for i in ti.static(range(3)):
                    for j in ti.static(range(3)):
                        offset = ti.Vector([i, j])
                        u_cell = u_cell_pos + offset
                        v_cell = v_cell_pos + offset
                        w_u = weight_u[i][0] * weight_u[j][1]
                        w_v = weight_v[i][0] * weight_v[j][1]
                        u_pic += w_u * self.u[u_cell]
                        v_pic += w_v * self.v[v_cell]
                        u_flip += w_u * (self.u[u_cell] - self.u_prev[u_cell])
                        v_flip += w_v * (self.v[v_cell] - self.v_prev[v_cell])

                self.particle_v[p] = self.blend_ratio * (self.particle_v[p] + ti.Vector([u_flip, v_flip])) + (
                        1 - self.blend_ratio) * ti.Vector([u_pic, v_pic])

    @ti.kernel
    def particle_to_grid(self):
        for p in ti.grouped(self.particle_pos):
            if self.particle_type[p] == 1:
                pxy = self.particle_pos[p]
                u_cell_pos = ti.cast(pxy / self.cell_size - ti.Vector([0.5, 1]), int)
                v_cell_pos = ti.cast(pxy / self.cell_size - ti.Vector([1, 0.5]), int)
                u_offset = pxy / self.cell_size - (u_cell_pos + ti.Vector([0.0, 0.5]))
                v_offset = pxy / self.cell_size - (v_cell_pos + ti.Vector([0.5, 0.0]))
                weight_u = [0.5 * (1.5 - u_offset) ** 2, 0.75 - (u_offset - 1) ** 2, 0.5 * (u_offset - 0.5) ** 2]
                weight_v = [0.5 * (1.5 - v_offset) ** 2, 0.75 - (v_offset - 1) ** 2, 0.5 * (v_offset - 0.5) ** 2]

                for i in ti.static(range(3)):
                    for j in ti.static(range(3)):
                        offset = ti.Vector([i, j])
                        u_cell = u_cell_pos + offset
                        v_cell = v_cell_pos + offset
                        w_u = weight_u[i][0] * weight_u[j][1]
                        w_v = weight_v[i][0] * weight_v[j][1]
                        self.u[u_cell] += w_u * self.particle_v[p][0]
                        self.u_weight[u_cell] += w_u
                        self.v[v_cell] += w_v * self.particle_v[p][1]
                        self.v_weight[v_cell] += w_v
        for i, j in self.u:
            if self.u_weight[i,j] > 0:
                self.u[i, j] /= self.u_weight[i, j]
        for i, j in self.v:
            if self.v_weight[i,j] > 0:
                self.v[i, j] /= self.v_weight[i, j]

    @ti.pyfunc
    def product(self, p, q):
        total = 0.0
        for i, j in ti.ndrange(self.x_count, self.y_count):
            if self.cell_type[i, j] == World.FLUID:
                total += p[i, j] * q[i, j]
        return total

    @ti.kernel
    def setup_CG(self, dt: ti.f32):
        # rhs of linear system
        scale_r = 1 / self.cell_size

        for i, j in self.b:
            if self.cell_type[i, j] == World.FLUID:
                self.b[i, j] = -1 * scale_r * (self.u[i + 1, j] - self.u[i, j] + self.v[i, j + 1] - self.v[i, j])
                if self.cell_type[i - 1, j] == World.SOLID:
                    self.b[i, j] -= scale_r * self.u[i, j]
                if self.cell_type[i + 1, j] == World.SOLID:
                    self.b[i, j] += scale_r * self.u[i + 1, j]
                if self.cell_type[i, j - 1] == World.SOLID:
                    self.b[i, j] -= scale_r * self.v[i, j]
                if self.cell_type[i, j + 1] == World.SOLID:
                    self.b[i, j] += scale_r * self.v[i, j + 1]

        # lhs of linear system
        scale_l = dt / (self.density * self.cell_size * self.cell_size)
        for i, j in self.Adiag:
            if self.cell_type[i, j] == World.FLUID:
                if self.cell_type[i - 1, j] == World.FLUID:
                    self.Adiag[i, j] += scale_l
                if self.cell_type[i + 1, j] == World.FLUID:
                    self.Adiag[i, j] += scale_l
                    self.Ax[i, j] = - scale_l
                elif self.cell_type[i + 1, j] == World.AIR:
                    self.Adiag[i, j] += scale_l

                if self.cell_type[i, j - 1] == World.FLUID:
                    self.Adiag[i, j] += scale_l
                if self.cell_type[i, j + 1] == World.FLUID:
                    self.Adiag[i, j] += scale_l
                    self.Ay[i, j] = - scale_l
                elif self.cell_type[i, j + 1] == World.AIR:
                    self.Adiag[i, j] += scale_l

    @ti.kernel
    def CG_step(self, diff:ti.f32) -> ti.f32:
        for i, j in self.As:
            if self.cell_type[i, j] == World.FLUID:
                self.As[i, j] = self.Adiag[i, j] * self.s[i, j] + self.Ax[i - 1, j] * self.s[i - 1, j] + self.Ax[
                    i, j] * self.s[i + 1, j] + self.Ay[i, j - 1] * self.s[i, j - 1] + self.Ay[i, j] * self.s[
                                    i, j + 1]

        sAs = self.product(self.s, self.As)
        alpha = diff / sAs

        for i, j in self.pressue:
            if self.cell_type[i, j] == World.FLUID:
                self.pressue[i, j] += alpha * self.s[i, j]
                self.res[i, j] -= alpha * self.As[i, j]

        diff_new = self.product(self.res, self.res)
        beta = diff_new / diff

        for i, j in self.s:
            if self.cell_type[i, j] == World.FLUID:
                self.s[i, j] = self.res[i, j] + beta * self.s[i, j]

        diff = diff_new
        return diff

    @ti.kernel
    def update_pressure(self, dt: ti.f32):
        scale = dt / (self.density * self.cell_size)
        for i, j in self.p:
            if self.cell_type[i - 1, j] == World.FLUID or self.cell_type[i, j] == World.FLUID:
                if self.cell_type[i - 1, j] == World.SOLID or self.cell_type[i, j] == World.SOLID:
                    self.u[i, j] = 0
                else:
                    self.u[i, j] -= scale * (self.p[i, j] - self.p[i - 1, j])
            if self.cell_type[i, j - 1] == World.FLUID or self.cell_type[i, j] == World.FLUID:
                if self.cell_type[i, j - 1] == World.SOLID or self.cell_type[i, j] == World.SOLID:
                    self.v[i, j] = 0
                else:
                    self.v[i, j] -= scale * (self.p[i, j] - self.p[i, j - 1])

    def pressure_project(self, dt, max_iter):
        self.b.fill(0.0)
        self.Adiag.fill(0.0)
        self.Ax.fill(0.0)
        self.Ay.fill(0.0)
        self.pressue.fill(0.0)
        self.As.fill(0.0)
        self.setup_CG(dt)
        self.res.copy_from(self.b)
        self.s.copy_from(self.res)
        tol = 1e-12

        diff = self.product(self.res, self.res)
        old_diff = diff
        for iter in range(max_iter):
            if iter > max_iter or diff < tol * old_diff:
                break
            diff = self.CG_step(diff)
        print("Converged to {} in {} iterations".format(diff, iter))
        self.p.copy_from(self.pressue)
        self.update_pressure(dt)

    def step(self, dt: ti.f32):
        self.external_force(dt)
        self.boundary_condition()
        self.pressure_project(dt, 500)
        self.boundary_condition()
        self.grid_to_particle()
        self.advect(dt)
        self.label()

        self.u.fill(0.0)
        self.v.fill(0.0)
        self.u_weight.fill(0.0)
        self.v_weight.fill(0.0)

        self.particle_to_grid()
        np.save("p2gu", self.u.to_numpy())
        np.save("p2gv", self.v.to_numpy())
        np.save("p2gp", self.p.to_numpy())
        self.u_prev.copy_from(self.u)
        self.v_prev.copy_from(self.v)


@ti.data_oriented
class Viewer:
    def __init__(self, screen_res, scene: World, dump=True, result_dir="./results"):
        self.dump = dump
        self.scene = scene
        self.screen_res = (screen_res, screen_res * scene.height // scene.width)
        self.buffer = ti.Vector.field(3, dtype=ti.f32, shape=self.screen_res)
        self.gui = ti.GUI("fluid", self.screen_res)
        if self.dump:
            self.video_manager = ti.VideoManager(
                output_dir=result_dir, framerate=24, automatic_build=False)

    def render_particles(self):
        bg_color = 0x112f41
        particle_color = 0x068587
        particle_radius = 1.0

        pf = self.scene.particle_type.to_numpy()
        np_type = pf.copy()
        np_type = np.reshape(np_type, -1)

        pos = self.scene.particle_pos.to_numpy()
        np_pos = np.reshape(pos, (-1, 2))
        np_pos = np_pos[np.where(np_type == 1)]

        for i in range(np_pos.shape[0]):
            np_pos[i][0] /= self.scene.width
            np_pos[i][1] /= self.scene.height
        self.gui.clear(bg_color)
        self.gui.circles(np_pos, radius=particle_radius, color=particle_color)

    @ti.kernel
    def render_pixel(self):
        for x, y in self.buffer:
            i = int(x * self.scene.x_count / self.screen_res[0])
            j = int(y * self.scene.y_count / self.screen_res[1])
            if self.scene.cell_type[i, j] == World.SOLID:
                self.buffer[x, y] = ti.Vector([1.0, 1.0, 1.0])
            elif self.scene.cell_type[i, j] == World.FLUID:
                self.buffer[x, y] = ti.Vector([.1, 0.6, 0.75]) * 0.6
            else:
                self.buffer[x, y] = ti.Vector([.1, 0.6, 0.75]) * 0.8

    def draw(self):
        # self.render_pixel()
        # img = self.buffer.to_numpy()
        # self.gui.set_image(img)
        self.render_particles()

        if self.dump:
            self.video_manager.write_frame(self.gui.get_image())
            #self.video_manager.write_frame(img)
        self.gui.show()

    def make_video(self):
        self.video_manager.make_video(gif=True, mp4=True)


def main(max_step, max_time, substep, type, blend):
    scene = World(10, 10, 25, blend, 2, 1000)
    viewer = Viewer(800, scene, result_dir="./results/blend_{}_type_{}".format(blend, type))
    if type == "dam":
        scene.setup_dambreak(4, 4)
    elif type == "2dam":
        scene.setup_2dambreak(6,4)
    elif type == "step":
        scene.setup_stepdown(5,6.6)
    elif type == "fountain":
        scene.setup_fountain()
    t0 = time.time()
    dt = 0.01
    t = 0
    step = 1
    while step < max_step and t < max_time:
        for i in range(substep):

            scene.step(dt)
            t += dt
        viewer.draw()
        step += 1
    t1 = time.time()
    viewer.make_video()
    print("simulation elapsed time = {} seconds".format(t1 - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PIC FLIP Fluid')
    parser.add_argument('--type', type=str, default='dam')
    parser.add_argument('--blend', type=float, default=0.0)

    args = parser.parse_args()
    main(500, 500, 4, args.type, args.blend)
