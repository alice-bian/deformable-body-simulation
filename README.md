# Deformable Body Simulation

Simulates elastic and continuum materials under four numerical schemes:
explicit (forward) integration, implicit (backward) integration via
Newton's method, XPBD (a constraint-based alternative to both), and the
material point method (MPM) for continuum mechanics. The same underlying
physics — spring/damping forces for the mass-spring scenes, elastic
stress for MPM — is solved multiple different ways, which is what makes
the comparisons below (stability, stiffness, cost) meaningful: it's the
same problem, different solvers.

`material_point_method/` is a separate, self-contained Python/Taichi
script with no relationship to the C++ build below other than the shared
underlying physics — it has its own dependencies (`taichi`, `numpy`) and
its own run command, since MPM's particle/grid representation has nothing
in common with the mass-spring code's node/segment representation.

## Mass-spring systems

A mass-spring system represents a deformable body as a set of point
masses connected by springs. For a segment connecting nodes A and B with
rest length `l0` and current length `l = ‖x_A - x_B‖`, the spring energy
is

```
E_spring = (1/2) · l0 · k · (l/l0 - 1)²
```

where `k` is the segment's Young's modulus. The force on A is the
negative gradient:

```
f_A = -∂E_spring/∂x_A = -k · (l/l0 - 1) · n_AB,   n_AB = (x_A - x_B)/l
```

and `f_B = -f_A` by symmetry. Damping is a dashpot resisting relative
velocity *along* the spring direction (not the full relative velocity,
which would also damp rotation):

```
E_damping = (1/2) · dt · c · (n_AB · (v_A - v_B))²
f_A_damping = -c · (n_AB · (v_A - v_B)) · n_AB
```

### Explicit (symplectic Euler) integration

```
v_{n+1} = v_n + dt · (f(x_n) / m + g)
x_{n+1} = x_n + dt · v_{n+1}
```

Forces are evaluated once at the start of the step, so no system needs to
be solved — but stability requires the timestep to resolve the system's
stiffest mode. For a single spring of stiffness `k` and mass `m`, the
linear stability limit is approximately

```
dt < 2 √(m/k)
```

Exceed it and the spring's oscillation amplitude grows every step instead
of staying bounded — the simulation diverges. This is why the explicit
bunny scene needs `dt ≈ 0.0001` despite the implicit version handling the
identical mesh at `dt = 1/24`.

### Implicit (backward) Euler via Newton's method

Backward Euler defines the new state implicitly:

```
v_{n+1} = v_n + dt · (f(x_{n+1}) / m + g)
x_{n+1} = x_n + dt · v_{n+1}
```

Substituting and writing `dx = x_{n+1} - x_n`, this is equivalent to
minimizing the incremental potential

```
E(dx) = Σ_i (1/2) m_i ‖dx_i - v_i·dt - g·dt²‖²   (inertia)
      + dt² · Σ_segments (E_spring + E_damping)   (elasticity/damping)
      + dt² · Σ_i E_collision(x_i + dx_i)          (penalty collisions)
```

Newton's method finds the minimizer iteratively: at each iterate, solve

```
H(dx) · Δ(dx) = ∇E(dx)
```

for the step `Δ(dx)`, where `H` is the Hessian of `E` and `∇E` its
gradient, then update `dx ← dx - α·Δ(dx)` with a backtracking line search
on `α` to guarantee the energy decreases. Iterate until `‖Δ(dx)‖` is small
relative to the mesh scale.

**PD projection.** The per-segment Hessian block

```
K_S = k · [(1/l0 - 1/l)·(I - n nᵗ) + (1/l0)· n nᵗ]
```

is not positive-definite everywhere (springs under compression can
contribute a negative eigenvalue). Each segment's local Hessian is
projected to the nearest PSD matrix (negative eigenvalues clamped to zero)
before assembly, guaranteeing the global system stays solvable via
Cholesky (`SimplicialLDLT`) every iteration — this is the same
projected-Newton approach widely used in production cloth/FEM solvers,
not a simplification specific to this implementation.

**Collision energy.** The sphere collision is a smooth one-sided penalty,
active only once a node is within `1.1×` the sphere radius:

```
E_collision = (1/2) · k_c · (1 - d/r)²     for d < r,   0 otherwise
```

where `d` is the node's distance from the sphere center and `r` the
radius. Its gradient and Hessian are added directly into the global
system alongside the spring/damping terms — collisions don't need
special-case handling in the solver, only an extra energy term.

### XPBD (Extended Position-Based Dynamics)

XPBD replaces forces and energy with **constraints solved directly in
position space** — no global linear system is ever assembled. Each
distance constraint between nodes A and B is

```
C(p_A, p_B) = ‖p_A - p_B‖ - l0
```

and is driven toward zero by repeatedly applying

```
α̃ = α / dt²
Δλ = (-C - α̃λ) / (w_A + w_B + α̃)
p_A += w_A · Δλ · n,   p_B -= w_B · Δλ · n
```

where `w = 1/m` is inverse mass, `n` is the constraint's current
direction, `λ` is a Lagrange multiplier accumulated across a substep's
iterations (reset to zero at the start of every substep), and
`α = 1/k` is the constraint's **compliance** — XPBD's term for
inverse-stiffness. Compliance is the "Extended" part of PBD: plain
position-based dynamics ties effective stiffness to substep count and
iteration count in a way that makes it hard to reason about physically;
XPBD's `α̃` term decouples them, so compliance keeps the same physical
meaning regardless of how many substeps or iterations the solver uses.

A full substep is: **predict** new positions under external forces
(`p_i = x_i + dt·v_i + dt²·g`, which is the only place gravity enters —
the constraint projection above only ever sees positions), **reset**
every constraint's `λ`, **project** every constraint some number of
times (Gauss-Seidel — each constraint uses whatever positions the
previous constraint in the same iteration already updated), then
**update** `v_i = (p_i - x_i)/dt` and `x_i = p_i`.

This split between "predict under forces" and "project constraints" is
what makes `compliance = 1/youngs_modulus` mean the same thing physically
as the corresponding force-based spring's stiffness: at equilibrium under
a static load, both reach the same steady-state stretch
(`l = l0 + load/k`), which is the equivalence this repo's tests verify
directly (a hanging chain settling to the force-balance-predicted
length). Unlike explicit Euler, constraint projection has no linear
stability limit tied to stiffness and mass, so XPBD tolerates the same
large substeps implicit Euler does, without needing to assemble or solve
a Hessian.

## Material point method (MLS-MPM)

MPM tracks the material as a set of Lagrangian particles, each carrying a
deformation gradient `F`, but routes all spatial-derivative computation
(stress divergence, velocity gradient) through a background Eulerian grid
that's reset every substep. The key transfers:

**P2G (particle → grid).** Particle `p` contributes mass, momentum, and
elastic force to every grid node within its quadratic B-spline kernel
support (a 3×3×3 neighborhood):

```
(mv)_i += w_ip · m_p · (v_p + C_p · (x_i - x_p))     [APIC affine term]
m_i    += w_ip · m_p
(mv)_i += dt · f_i,     f_i = -V_p · τ_p · ∇w_ip
```

`C_p` is the APIC affine velocity matrix — carrying it per-particle lets
MPM reconstruct a locally-linear (rather than just locally-constant)
velocity field during G2P, which substantially reduces numerical
dissipation compared to the original (non-APIC) MPM formulation.

**Elasticity: fixed corotated (FCR) model.** Given the polar decomposition
`F = RΣ` (via SVD, `F = UΣVᵗ`, `R = UVᵗ`) and `J = det(F)`, the Kirchhoff
stress is

```
τ = 2μ(F - R)Fᵗ + λ·J·(J - 1)·I
```

The first term penalizes deviation from a pure rotation (shear/shape
change); the second penalizes volume change. `μ` and `λ` are the Lamé
parameters, related to Young's modulus `E` and Poisson's ratio `ν` by the
standard conversion `μ = E / (2(1+ν))`, `λ = Eν / ((1+ν)(1-2ν))`.

**Grid update.** Momentum is converted to velocity (`v_i = (mv)_i / m_i`),
gravity is applied, and simple slip boundary conditions zero out velocity
components pointing into domain walls.

**G2P (grid → particle).** Each particle gathers velocity and its local
affine field `C_p` from the same kernel neighborhood, advects its
position, and updates its deformation gradient using the (now
grid-resolved) velocity gradient:

```
F_p ← (I + dt · ∇v_p) · F_p
```

No tetrahedral or spring topology is needed at all — particles only
interact indirectly, through the shared grid, which is what lets MPM
handle large deformation and multiple colliding bodies without
remeshing.

**Backend and output.** `ti.init(arch=ti.gpu)` requests CUDA first,
falling back to Metal/Vulkan depending on platform — no code changes
needed across machines, but expect this to be slow on a CPU-only fallback
at full particle count. The `QUALITY` constant near the top of
`taichi_mpm_3d.py` scales resolution and particle count together
(`~QUALITY²` cost) if you want faster, lower-fidelity runs while
iterating. Output is `render_output/jello_*.ply`, one point cloud per
frame (240 frames total) — load the sequence into Houdini via
`material_point_method/JelloRender.hipnc` as a `Frame_Sequence` point
cache, or use any other `.ply`-sequence-aware renderer.

## Building (Linux / WSL)

```bash
sudo apt install build-essential cmake libeigen3-dev   # one-time
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces a single binary, `build/bin/deformable_sim`, dispatched by
subcommand. **Every subcommand must be run from the repo root** — `data/`
(mesh input) and `output/` (per-frame `.poly` output) are resolved as
relative paths, not relative to the binary's own location, so running
from anywhere else fails trying to open `data/points.txt` or
`data/cells.txt`:

```bash
cd deformable-body-simulation   # repo root, not build/
./build/bin/deformable_sim explicit cloth
./build/bin/deformable_sim explicit bunny
./build/bin/deformable_sim implicit bunny 0.1     # Young's modulus sweep:
./build/bin/deformable_sim implicit bunny 1       #   run all five for the
./build/bin/deformable_sim implicit bunny 10      #   five-stiffness demo
./build/bin/deformable_sim implicit bunny 100
./build/bin/deformable_sim implicit bunny 1000
./build/bin/deformable_sim implicit brush
./build/bin/deformable_sim xpbd cloth
```

Each writes its own `output/<case>/*.poly` sequence (one file per frame,
starting at `0.poly` for the rest pose):

| Command | Output folder | Surface mesh to deform in Houdini |
|---|---|---|
| `explicit cloth` | `output/cloth/` | `data/cloth.obj` (written by this same run) |
| `explicit bunny` | `output/bunny/` | `data/bunny.obj` |
| `implicit bunny <k>` | `output/bunny_<k>/` (e.g. `bunny_100.00`) | `data/bunny.obj` |
| `implicit brush` | `output/brush/` | none — bristles render directly from the `.poly` segments |
| `xpbd cloth` | `output/cloth_xpbd/` | `data/cloth.obj` (same file `explicit cloth` wrote) |

`xpbd cloth` reuses `data/cloth.obj`, so run `explicit cloth` at least
once first if you haven't — it's the only command that writes that file.

The material point method has no C++ build — it's pure Python/Taichi:

```bash
pip install taichi numpy
cd material_point_method
python3 taichi_mpm_3d.py
```

## Demo

**Explicit cloth & bunny** — a flag-like cloth grid is pulled and
released at two pinned corners, and a volumetric bunny mesh has its tail
pulled and let go. Both demonstrate the explicit integrator's spring
forces and damping actually conserving a recognizable shape under
stretching rather than the mesh tearing apart or oscillating
unboundedly — the core risk explicit integration runs whenever stiffness
and timestep aren't matched (see the stability discussion above).

**XPBD cloth** *(placeholder — fill in once recorded)* — the identical
cloth mesh and pin/drag sequence as the explicit demo above, run through
the constraint-based solver instead, at a substep size ~100× larger than
explicit Euler tolerates on this mesh. The comparison to watch for: does
the same boundary motion produce visually comparable cloth behavior
despite the completely different solver underneath, and where does it
visibly differ (XPBD's compliance-driven softness vs. explicit's
force-driven response) as solver_iterations or compliance are varied.

**Implicit brush vs. sphere** — a grid of bristles is lowered onto a
sphere, showing the bristles individually deflecting around the sphere's
surface on contact rather than passing through it or getting stuck. This
demonstrates the penalty-based collision response (gradient/Hessian
terms folded directly into the Newton solve) handling many simultaneous
per-bristle contacts without any special-case collision logic.

**Implicit bunny, five stiffnesses side by side** — the identical bunny
mesh and impulse, run independently at five Young's modulus values, shown
together. This isolates stiffness as the only varied parameter and shows
the resulting range of behavior directly: visibly floppy and
slow-to-recover at the lowest value, progressively tighter and
faster-recovering at each step up, near-rigid at the highest — the same
qualitative range a real material parameter sweep would produce.

**MPM jello collision** — three elastic cubes enter the scene at
different times and trajectories, colliding mid-air. This demonstrates
MPM's core advantage over the mass-spring scenes above: there's no shared
mesh between the three cubes, no special handling for the moment they
touch, and no remeshing — each cube is just a particle cloud, and
collision response (and the resulting bounce/separation) falls directly
out of all particles sharing one background grid during P2G/G2P.

## Running the tests

```bash
cd build && ctest --output-on-failure
```

This automatically checks that the analytic spring/damping force and
Hessian match central finite differences to a tight tolerance, and that
XPBD's compliance parameter reaches the same equilibrium a force-based
spring of the equivalent stiffness would.

## Future work

Committed next steps:
- **Material comparison grid** — a second MPM constitutive model (snow
  plasticity, or a near-incompressible fluid) compared side by side
  against the current fixed-corotated elastic model.

Other ideas, not yet committed:
- An XPBD bunny scene (same volumetric mesh as the explicit/implicit
  bunny cases, reusing the existing tet-edge-to-constraint extraction).
- Ground-plane and self-collision for the cloth and bunny scenes.
- A lightweight in-repo viewer so the `.poly`/`.ply` output can be
  previewed without Houdini.
