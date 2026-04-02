# sim.warp_nbody

GPU-accelerated N-body gravitational simulation running inside NVIDIA Omniverse, powered by [WARP](https://github.com/NVIDIA/warp).

![demo](sim_trimmed.gif)

## How it works

All physics runs on the GPU using WARP kernels. Every frame, gravity is computed between all pairs of bodies, velocities and positions are integrated, and overlapping bodies merge together. The goal is for nothing leave the GPU (so far, we copy the arrays once to CPU per simulation step).

## Presets

- **Galaxy Disk** - disk of bodies orbiting a central mass
- **Sphere** - random uniform sphere
- **Solar System** - star, 8 planets, and an asteroid belt
- **Random** - random box of bodies
- **Binary Galaxy** - two colliding galaxy disks
- **Black Hole** - dense inner ring with outer spiral drift

## Requirements

- NVIDIA Omniverse Kit
- NVIDIA WARP
- CUDA GPU



To nsys


  nsys launch \
    --trace=cuda,nvtx \
    .../kit-app-template/_build/linux-x86_64/release/kit/kit \
    .../kit-app-template/_build/linux-x86_64/release/apps/my_company.my_usd_composer.kit

  Then once the sim is running:

  nsys start
  # wait a few seconds...
  nsys start

  QT_QPA_PLATFORM=xcb nsys-ui .../report1.nsys-rep