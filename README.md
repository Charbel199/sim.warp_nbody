# sim.warp_nbody

GPU-accelerated N-body gravitational simulation in NVIDIA Omniverse, powered by [WARP](https://github.com/NVIDIA/warp) kernels.

![demo](docs/galaxy_disk_regular_gravity.gif)

## How it works

All physics runs on the GPU through WARP kernels. Each frame, gravity is computed between all pairs of bodies (O(N^2)), velocities and positions are integrated, and overlapping bodies merge. Nothing leaves the GPU. Positions get synced to USD/Fabric for rendering through a zero-copy bridge so the simulation loop never touches the CPU (almost).

## Kernels

The core of the simulation is just a few WARP kernels.

### Gravity

Brute-force O(N^2). Every body computes the gravitational pull from every other body. Softening prevents singularities when bodies get close.

```python
@wp.kernel
def kernel_forces(
    positions:   wp.array(dtype=wp.vec3),
    masses:      wp.array(dtype=float),
    active:      wp.array(dtype=int),
    forces:      wp.array(dtype=wp.vec3),
    G:           float,
    softening_sq: float,
    n:           int,
):
    i = wp.tid()
    if active[i] == 0:
        return
    f  = wp.vec3(0.0, 0.0, 0.0)
    pi = positions[i]
    mi = masses[i]
    for j in range(n):
        if j == i or active[j] == 0:
            continue
        r        = positions[j] - pi
        dist_sq  = wp.dot(r, r) + softening_sq
        inv_dist3 = 1.0 / (dist_sq * wp.sqrt(dist_sq))
        f = f + r * (G * mi * masses[j] * inv_dist3)
    forces[i] = f
```

### Integration

Simple Euler integration. Acceleration from the force kernel gets applied to velocity, then velocity to position.

```python
@wp.kernel
def kernel_integrate(
    positions:  wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    forces:     wp.array(dtype=wp.vec3),
    masses:     wp.array(dtype=float),
    active:     wp.array(dtype=int),
    dt:         float,
):
    i = wp.tid()
    if active[i] == 0:
        return
    acc         = forces[i] * (1.0 / masses[i])
    velocities[i] = velocities[i] + acc * dt
    positions[i]  = positions[i]  + velocities[i] * dt
```

### Accretion (merging)

Two-pass kernel. First pass checks if two bodies overlap and marks the lighter one for merging. Second pass transfers mass and deactivates the absorbed body, then recomputes the radius based on the new mass.

```python
@wp.kernel
def kernel_accrete_pass1(
    positions:  wp.array(dtype=wp.vec3),
    masses:     wp.array(dtype=float),
    radii:      wp.array(dtype=float),
    active:     wp.array(dtype=int),
    merge_into: wp.array(dtype=int),
    n:          int,
):
    i = wp.tid()
    if active[i] == 0:
        return
    merge_into[i] = -1
    pi = positions[i]
    mi = masses[i]
    for j in range(n):
        if j == i or active[j] == 0:
            continue
        dist = wp.length(positions[j] - pi)
        if dist < radii[i] + radii[j]:
            if masses[j] > mi or (masses[j] == mi and j < i):
                merge_into[i] = j
                return

@wp.kernel
def kernel_accrete_pass2(
    masses:     wp.array(dtype=float),
    radii:      wp.array(dtype=float),
    active:     wp.array(dtype=int),
    merge_into: wp.array(dtype=int),
    base_mass:   float,
    base_radius: float,
):
    i = wp.tid()
    if active[i] == 0 or merge_into[i] == -1:
        return
    j = merge_into[i]
    wp.atomic_add(masses, j, masses[i])
    active[i] = 0
    radii[j]  = base_radius * wp.pow(masses[j] / base_mass, 1.0 / 3.0)
```

### Colorization

Colors are computed on GPU too. Each body gets a color based on its mass and speed relative to the current max. Heavy/fast bodies shift from blue to orange.

```python
@wp.kernel
def kernel_colorize(
    masses:        wp.array(dtype=float),
    velocities:    wp.array(dtype=wp.vec3),
    active:        wp.array(dtype=int),
    colors:        wp.array(dtype=wp.vec3),
    max_mass_arr:  wp.array(dtype=float),
    max_speed_arr: wp.array(dtype=float),
):
    i = wp.tid()
    if active[i] == 0:
        colors[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    max_mass  = wp.max(max_mass_arr[0],  float(1e-6))
    max_speed = wp.max(max_speed_arr[0], float(1e-6))
    t_mass  = wp.min(masses[i] / max_mass, 1.0)
    t_speed = wp.min(wp.length(velocities[i]) / max_speed, 1.0)
    t = t_mass * 0.7 + t_speed * 0.3
    if t < 0.5:
        s = t * 2.0
        colors[i] = wp.vec3(s * 0.9 + 0.1, s * 0.9 + 0.1, 1.0)
    else:
        s = (t - 0.5) * 2.0
        colors[i] = wp.vec3(1.0, 1.0 - s * 0.6, 1.0 - s)
```

## Fabric Bridge (keeping it on GPU)

The simulation never copies data back to CPU for rendering. Omniverse's Fabric API (via USDRT) lets us write GPU buffers directly into USD attributes. The `mark_dirty` method runs every frame:

```python
def mark_dirty(self) -> None:
    with wp.ScopedDevice("cuda:0"):
        # GPU -> GPU: copy sim positions into scratch buffer
        wp.copy(self._pos_wp, self._sim.positions)

        # compute scales on GPU
        wp.launch(kernel_compute_scales, dim=self._n, device="cuda:0", inputs=[
            self._sim.radii, self._sim.active, self._scales_wp,
            self._visual_scale, self._visual_cap,
        ])

        # compute colors on GPU
        self._colorizer.compute_colors(self._sim, self._colors_wp)

        # push to Fabric (GPU -> GPU copies via USDRT)
        self._pos_attr.Set(Vt.Vec3fArray(self._pos_wp))
        self._scale_attr.Set(Vt.Vec3fArray(self._scales_wp))
        self._color_attr.Set(Vt.Vec3fArray(self._colors_wp))
```

Warp arrays go into USDRT `Vt` arrays which live on the same device. No `cuda.memcpy` to host, no numpy, no CPU staging buffers. The renderer picks up the updated Fabric attributes directly.

## Presets

### Galaxy Disk

Flat disk of bodies orbiting a central mass. Here's what happens when you crank up the gravitational constant:

![galaxy disk high gravity](docs/galaxy_disk_high_gravity.gif)

### Black Hole

Dense inner ring with particles spiraling inward. The central body has a massive mass ratio over everything else.

![blackhole](docs/blackhole.gif)

### Sphere

Equal-mass bodies in a uniform random sphere. No central mass, just everything pulling on everything else.

![sphere](docs/circle_high_gravity.gif)

### Others

- **Solar System** - star, 8 planets, and an asteroid belt
- **Random** - bodies scattered in a box
- **Binary Galaxy** - two galaxy disks colliding

## [EXPERIMENTAL] AI Physics

The idea here comes from surrogate modeling in scientific computing. Instead of solving the full equations every timestep, you train a neural network to approximate the solver and use that as a cheap stand-in. NVIDIA's NeMo Physics does this at scale for things like CFD and weather prediction, training models (FourCastNet, MeshGraphNet, etc.) on simulation data and then running inference orders of magnitude faster than the original PDE solver.

This is the same concept applied to N-body gravity. The classical O(N^2) force computation is expensive, so we train a GNN on recorded simulation data and see if it can predict the per-particle accelerations well enough to be useful. It won't match NeMo Physics in accuracy or scale, but it's a good sandbox to play with the idea of surrogate models inside a live simulation.

The GNN runs next to the classical simulation for comparison. Blue particles are the classical solver, orange particles are the neural one. Both start from the same initial conditions.

![neural physics](docs/galaxy_disk_neural_physics.gif)

This is experimental. The network learns decent force approximations per preset but it's not a replacement for the real solver. Errors accumulate over time and you can see the neural side slowly drift away from the classical one.

### Pipeline

Each preset has its own data/training/checkpoint pipeline since the dynamics and mass scales are very different between setups.

1. Pick a preset in the UI
2. Generate data - run the classical sim and record positions, velocities, masses, and accelerations to HDF5
3. Train - train a GNS-style GNN on the recorded data
4. Infer - load the model, predict forces through a zero-copy Warp-to-PyTorch bridge

### Architecture

Node features (velocity + mass) and edge features (relative position, distance, relative velocity) go into 3 message-passing layers with residual connections, then get decoded to per-particle accelerations.

### Tunable parameters

| Parameter | Effect |
|---|---|
| Cutoff radius | Controls graph connectivity. Smaller = faster but less accurate |
| Inference interval | Run the GNN every K frames, reuse forces in between |

### Known limitations

- Models are per-preset and don't generalize across them (intentional, the mass scales are too different)
- High mass ratio systems (Solar System, Black Hole) are harder to learn than uniform ones (Sphere, Random)
- The model drifts over long rollouts since errors accumulate. Dual-stream mode makes this easy to see

## Project Structure

```
sim/warp_nbody/
  extension.py          Omniverse Kit extension entry point
  simulation.py         main simulation loop (classical + neural dual-stream)
  fabric_bridge.py      USDRT/Fabric zero-copy GPU<->USD sync
  instancer.py          USD particle instancers (classical + neural)
  spawner.py            initial condition generators
  colorizer.py          per-particle color assignment
  kernels/
    physics.py          Warp N-body force + integration kernels
    visual.py           Warp kernels for color/scale updates
  neural/
    model.py            GNS-style GNN (PyTorch Geometric)
    inference.py        Warp<->PyTorch zero-copy bridge
    data_gen.py         HDF5 training data generation
    train.py            training loop
  ui/
    panel.py            omni.ui panel (simulation + neural controls)
```

## Requirements

- NVIDIA Omniverse Kit
- NVIDIA WARP
- CUDA GPU
- PyTorch (CUDA build), PyTorch Geometric, torch_cluster, h5py (for neural features)

## Profiling with Nsight

```
nsys launch \
  --trace=cuda,nvtx \
  .../kit-app-template/_build/linux-x86_64/release/kit/kit \
  .../kit-app-template/_build/linux-x86_64/release/apps/my_company.my_usd_composer.kit

# Once the sim is running:
nsys start
# wait a few seconds...
nsys stop

QT_QPA_PLATFORM=xcb nsys-ui .../report1.nsys-rep
```
