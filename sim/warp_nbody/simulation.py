import numpy as np
import warp as wp

from .kernels.physics import (
    kernel_forces,
    kernel_integrate,
    kernel_accrete_pass1,
    kernel_accrete_pass2,
    kernel_reset_int,
    kernel_count_active,
)

BASE_MASS:   float = 1.0
BASE_RADIUS: float = 0.3


class NBodySimulation:

    def __init__(self):
        self.positions:  wp.array | None = None
        self.velocities: wp.array | None = None
        self.masses:     wp.array | None = None
        self.radii:      wp.array | None = None
        self.active:     wp.array | None = None
        self.forces:     wp.array | None = None

        self._n:            int = 0
        self._frame:        int = 0
        self._active_count: wp.array | None = None

        self.G:                  float = 0.001
        self.softening:          float = 0.05
        self.dt:                 float = 0.01
        self.accretion_enabled:  bool  = True
        self.accretion_interval: int   = 5

    def allocate(self, positions_np, velocities_np, masses_np) -> None:
        n        = len(masses_np)
        self._n  = n
        self._frame = 0

        self.positions  = wp.array(positions_np,  dtype=wp.vec3, device="cuda")
        self.velocities = wp.array(velocities_np, dtype=wp.vec3, device="cuda")
        self.masses     = wp.array(masses_np,     dtype=float,   device="cuda")
        self.forces     = wp.zeros(n, dtype=wp.vec3, device="cuda")
        self.active     = wp.ones(n,  dtype=int,   device="cuda")

        radii_np   = (BASE_RADIUS * (masses_np / BASE_MASS) ** (1.0 / 3.0)).astype(np.float32)
        self.radii = wp.array(radii_np, dtype=float, device="cuda")

        self._active_count = wp.zeros(1, dtype=int, device="cuda")

    def free(self) -> None:
        self.positions     = None
        self.velocities    = None
        self.masses        = None
        self.radii         = None
        self.active        = None
        self.forces        = None
        self._active_count = None
        self._n            = 0
        self._frame        = 0

    def count_active(self) -> int:
        # returns the number of active bodies. copies only 1 int from GPU -> CPU
        if self.active is None:
            return 0
        wp.launch(kernel_reset_int,    dim=1,       device="cuda", inputs=[self._active_count])
        wp.launch(kernel_count_active, dim=self._n, device="cuda", inputs=[
            self.active, self._active_count,
        ])
        return int(self._active_count.numpy()[0])

    def step(self) -> None:
        if self.positions is None:
            return
        self._run_forces()
        self._run_integrate()
        if self.accretion_enabled and self._frame % self.accretion_interval == 0:
            self._run_accrete()
        self._frame += 1

    def _run_forces(self) -> None:
        wp.launch(kernel_forces, dim=self._n, device="cuda", inputs=[
            self.positions, self.masses, self.active, self.forces,
            self.G, self.softening ** 2, self._n,
        ])

    def _run_integrate(self) -> None:
        wp.launch(kernel_integrate, dim=self._n, device="cuda", inputs=[
            self.positions, self.velocities, self.forces, self.masses, self.active, self.dt,
        ])

    def _run_accrete(self) -> None:
        merge_into = wp.full(self._n, -1, dtype=int, device="cuda")
        wp.launch(kernel_accrete_pass1, dim=self._n, device="cuda", inputs=[
            self.positions, self.masses, self.radii, self.active, merge_into, self._n,
        ])
        wp.launch(kernel_accrete_pass2, dim=self._n, device="cuda", inputs=[
            self.masses, self.radii, self.active, merge_into, BASE_MASS, BASE_RADIUS,
        ])
