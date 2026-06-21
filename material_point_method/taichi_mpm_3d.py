"""
3D MLS-MPM (Moving Least Squares Material Point Method) solver, simulating
three elastic "jello" cubes entering a shared scene in sequence and
colliding under gravity.

Targets a 3D MPM solver with >100k particles producing non-trivial,
artifact-free dynamics. The three originally near-identical
reset_jelloN() kernels are consolidated into one parameterized
reset_jello().

Pipeline per substep:
  1. Clear the background grid.
  2. P2G: particles scatter mass/momentum to the grid, and contribute an
     elastic force computed from the fixed-corotated (FCR) Kirchhoff stress.
  3. Grid update: convert momentum to velocity, apply gravity and
     boundary collisions.
  4. G2P: particles gather the updated grid velocity (via APIC, using the
     affine velocity field C) and advect; the deformation gradient F is
     updated from the velocity gradient.

Run with:
    python3 taichi_mpm_3d.py
Output: render_output/jello_*.ply (one frame per simulation step), which
can be loaded as a point cache in Houdini -- see the project README.
"""

import os

import numpy as np
import taichi as ti

# With arch=ti.gpu, Taichi tries CUDA first and falls back to Metal/Vulkan
# depending on platform. No code changes are needed to move between GPUs.
ti.init(arch=ti.gpu)

# --- Scalar parameters ------------------------------------------------------

QUALITY = 1  # bump this for higher-resolution simulations (cost scales ~quality^2)
N_PARTICLES = 150_000 * QUALITY**2
N_GRID = 128 * QUALITY
NUM_JELLOS = 3

DX = 0.5 / N_GRID
INV_DX = float(N_GRID)
DT = 1e-4 / QUALITY
P_VOL = (DX * 0.5) ** 2
P_RHO = 1.0
P_MASS = P_VOL * P_RHO

YOUNGS_MODULUS = 2e3
POISSON_RATIO = 0.2
MU_0 = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO))
LAMBDA_0 = YOUNGS_MODULUS * POISSON_RATIO / ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO))

# --- Fields ------------------------------------------------------------------

x = ti.Vector.field(3, dtype=float, shape=N_PARTICLES)       # particle positions
v = ti.Vector.field(3, dtype=float, shape=N_PARTICLES)       # particle velocities
C = ti.Matrix.field(3, 3, dtype=float, shape=N_PARTICLES)    # APIC affine velocity field
F = ti.Matrix.field(3, 3, dtype=float, shape=N_PARTICLES)    # deformation gradient
Jp = ti.field(dtype=float, shape=N_PARTICLES)                # tracked plastic volume change

grid_v = ti.Vector.field(3, dtype=float, shape=(N_GRID, N_GRID, N_GRID))  # node momentum -> velocity
grid_m = ti.field(dtype=float, shape=(N_GRID, N_GRID, N_GRID))            # node mass
gravity = ti.Vector.field(3, dtype=float, shape=())


@ti.func
def kirchhoff_fcr(F_p, R, J, mu, la):
    """Kirchhoff stress for the fixed-corotated elasticity model.

    tau = P F^T, where P is the first Piola-Kirchhoff stress of the FCR
    energy: P = 2*mu*(F - R) + lambda*J*(J - 1)*F^{-T}. Substituting and
    using J*F^{-T} = cofactor(F) (here folded into the identity term since
    we only need the isotropic pressure part) gives the form below.
    """
    return 2 * mu * (F_p - R) @ F_p.transpose() + ti.Matrix.identity(float, 3) * la * J * (J - 1)


@ti.kernel
def substep():
    # 1. Clear grid.
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]
        grid_m[i, j, k] = 0

    # 2. P2G: scatter mass, momentum, and elastic force to the grid.
    for p in x:
        base = (x[p] * INV_DX - 0.5).cast(int)
        fx = x[p] * INV_DX - base.cast(float)

        # Quadratic B-spline kernel weights and their derivatives, evaluated
        # at offsets {-1, 0, 1} relative to fx. See mpm.graphics, Eq. 123.
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        dw = [fx - 1.5, -2.0 * (fx - 1), fx - 0.5]

        mu, la = MU_0, LAMBDA_0

        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(3)):
            J *= sig[d, d]

        kirchhoff = kirchhoff_fcr(F[p], U @ V.transpose(), J, mu, la)

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * DX
            weight = w[i][0] * w[j][1] * w[k][2]

            dweight = ti.Vector.zero(float, 3)
            dweight[0] = INV_DX * dw[i][0] * w[j][1] * w[k][2]
            dweight[1] = INV_DX * w[i][0] * dw[j][1] * w[k][2]
            dweight[2] = INV_DX * w[i][0] * w[j][1] * dw[k][2]

            force = -P_VOL * kirchhoff @ dweight

            grid_v[base + offset] += P_MASS * weight * (v[p] + C[p] @ dpos)  # momentum
            grid_m[base + offset] += weight * P_MASS
            grid_v[base + offset] += DT * force  # still momentum; divide by mass below

    # 3. Grid update: momentum -> velocity, gravity, boundary collisions.
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] = (1 / grid_m[i, j, k]) * grid_v[i, j, k]
            grid_v[i, j, k] += DT * gravity[None] * 30

            if i < 3 and grid_v[i, j, k][0] < 0:           grid_v[i, j, k][0] = 0
            if i > N_GRID - 3 and grid_v[i, j, k][0] > 0:  grid_v[i, j, k][0] = 0
            if j < 3 and grid_v[i, j, k][1] < 0:           grid_v[i, j, k][1] = 0
            if j > N_GRID - 3 and grid_v[i, j, k][1] > 0:  grid_v[i, j, k][1] = 0
            if k < 3 and grid_v[i, j, k][2] < 0:           grid_v[i, j, k][2] = 0
            if k > N_GRID - 3 and grid_v[i, j, k][2] > 0:  grid_v[i, j, k][2] = 0

    # 4. G2P: gather velocity/affine field, advect, update F.
    for p in x:
        base = (x[p] * INV_DX - 0.5).cast(int)
        fx = x[p] * INV_DX - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        dw = [fx - 1.5, -2.0 * (fx - 1), fx - 0.5]

        new_v = ti.Vector.zero(float, 3)
        new_C = ti.Matrix.zero(float, 3, 3)
        new_F = ti.Matrix.zero(float, 3, 3)

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            dpos = ti.Vector([i, j, k]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j, k])]
            weight = w[i][0] * w[j][1] * w[k][2]

            dweight = ti.Vector.zero(float, 3)
            dweight[0] = INV_DX * dw[i][0] * w[j][1] * w[k][2]
            dweight[1] = INV_DX * w[i][0] * dw[j][1] * w[k][2]
            dweight[2] = INV_DX * w[i][0] * w[j][1] * dw[k][2]

            new_v += weight * g_v
            new_C += 4 * INV_DX * weight * g_v.outer_product(dpos)
            new_F += g_v.outer_product(dweight)

        v[p], C[p] = new_v, new_C
        x[p] += DT * v[p]
        F[p] = (ti.Matrix.identity(float, 3) + DT * new_F) @ F[p]


@ti.kernel
def reset_jello(jello_index: int, group_size: int, origin: ti.types.vector(3, float),
                 initial_velocity: ti.types.vector(3, float)):
    """(Re-)initializes one group of particles as a unit-cube jello block.

    Replaces the original reset_jello1()/reset_jello2()/reset_jello3(),
    which differed only in which particle range they touched, where they
    spawned, and the initial velocity -- all three are now parameters.
    """
    start = jello_index * group_size
    end = N_PARTICLES if jello_index == NUM_JELLOS - 1 else start + group_size
    for i in range(start, end):
        x[i] = [
            ti.random() * 0.2 + origin[0],
            ti.random() * 0.2 + origin[1],
            ti.random() * 0.2 + origin[2],
        ]
        v[i] = initial_velocity
        F[i] = ti.Matrix.identity(float, 3)
        Jp[i] = 1
        C[i] = ti.Matrix.zero(float, 3, 3)


def dump_frame(frame, series_prefix):
    np_pos = x.to_numpy().reshape(N_PARTICLES, 3)
    dirname = os.path.dirname(series_prefix)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    writer = ti.tools.PLYWriter(num_vertices=N_PARTICLES)
    writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
    writer.export_frame_ascii(frame, series_prefix)


def main():
    gravity[None] = [0, -9.8, 0]
    series_prefix = "render_output/jello.ply"
    group_size = N_PARTICLES // NUM_JELLOS
    substeps_per_frame = int(2e-3 // DT)

    # Spawn schedule: each jello enters at a different frame, origin, and
    # initial velocity, matching the original three-cube collision scene.
    spawn_schedule = [
        (0, (0.4, 0.45, 0.7), (0.0, 0.0, 0.0)),
        (30, (0.4, 0.75, 0.8), (0.0, 0.0, 0.0)),
        (45, (0.3, 0.25, 0.0), (0.0, 0.0, 30.0)),
    ]
    total_frames = 240

    next_spawn = 0
    for frame in range(total_frames):
        if next_spawn < len(spawn_schedule) and frame == spawn_schedule[next_spawn][0]:
            _, origin, init_v = spawn_schedule[next_spawn]
            reset_jello(next_spawn, group_size, ti.Vector(origin), ti.Vector(init_v))
            next_spawn += 1

        for _ in range(substeps_per_frame):
            substep()

        dump_frame(frame, series_prefix)


if __name__ == "__main__":
    main()
