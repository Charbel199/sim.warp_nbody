# sim.warp_nbody

GPU-accelerated N-body gravitational simulation running inside NVIDIA Omniverse, powered by [WARP](https://github.com/NVIDIA/warp).

![demo](sim_trimmed.gif)

## How it works

All physics runs on the GPU using WARP kernels. Every frame, gravity is computed between all pairs of bodies (O(N^2)), velocities and positions are integrated, and overlapping bodies merge together. The goal is for nothing to leave the GPU.

## [EXPERIMENTAL] Neural Force Field

> **This feature is experimental.** The GNN can learn reasonable force approximations for individual presets, but accuracy varies across configurations and it is not yet a drop-in replacement for the classical solver. Treat it as a research sandbox for exploring learned physics.

A GNN (Graph Neural Network) attempts to approximate the N-body gravitational forces and runs side-by-side with the classical simulation for comparison. The idea is to see whether a lightweight graph network can learn the force field well enough to be useful as a fast surrogate - trading exactness for speed.

**Pipeline (per-preset):**

Each preset (Galaxy Disk, Sphere, Solar System, etc.) has its own data/training/checkpoint pipeline, since the dynamics and mass scales differ significantly between setups.

1. **Select a preset** in the UI
2. **Generate data** - runs the classical simulation for that preset and records positions, velocities, masses, and accelerations to HDF5 (`data/nbody_{preset}.h5`)
3. **Train** - trains a GNS-style GNN on the recorded data, saving checkpoints to `checkpoints/{preset}/`
4. **Inference** - load the trained model and the GNN predicts forces via zero-copy Warp-to-PyTorch bridge

**Architecture:** Node encoder (vel + mass) and edge encoder (relative pos + distance + relative vel) feed into 3 message-passing layers with residual connections, decoded to per-particle accelerations.

**Dual-stream mode:** When enabled, both classical (blue) and neural (orange) particles spawn from identical initial conditions. The position error between them is logged every 100 frames.

**Tunable parameters:**
- **Cutoff radius** - controls the radius graph size (smaller = faster, less accurate)
- **Inference interval** - run the GNN every K frames and reuse cached forces in between

**Known limitations:**
- Models trained on one preset don't generalize to others (by design - each preset has different mass scales and dynamics)
- High-mass-ratio systems (Solar System, Black Hole) are harder to learn than uniform-mass setups (Sphere, Random)
- The model can drift over long rollouts since errors accumulate; the dual-stream comparison makes this visible

## Presets

- **Galaxy Disk** - disk of bodies orbiting a central mass
- **Sphere** - random uniform sphere
- **Solar System** - star, 8 planets, and an asteroid belt
- **Random** - random box of bodies
- **Binary Galaxy** - two colliding galaxy disks
- **Black Hole** - dense inner ring with outer spiral drift

## Project Structure

```
sim/warp_nbody/
  extension.py          Omniverse Kit extension entry point
  simulation.py         main simulation loop (classical + neural dual-stream)
  fabric_bridge.py      USDRT/Fabric zero-copy GPU<>USD sync
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
