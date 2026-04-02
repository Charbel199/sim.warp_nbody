import time
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

        self.neural_mode:       bool = False
        self.pos_neural:        wp.array | None = None
        self.vel_neural:        wp.array | None = None
        self.forces_neural:     wp.array | None = None
        self._neural_ff = None
        self.neural_inference_interval: int = 10
        self.neural_cutoff: float = 2.0

    def set_neural_mode(self, enabled: bool, checkpoint_path: str | None = None) -> None:
        self.neural_mode = enabled
        if enabled and self._neural_ff is None and checkpoint_path:
            try:
                from .neural import NeuralForceField
                self._neural_ff = NeuralForceField(checkpoint_path, cutoff=self.neural_cutoff)
            except Exception as e:
                import carb
                carb.log_warn(f"[warp_nbody] Failed to load neural model: {e}")
                self.neural_mode = False

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

        if self.neural_mode:
            self.pos_neural    = wp.array(positions_np,  dtype=wp.vec3, device="cuda")
            self.vel_neural    = wp.array(velocities_np, dtype=wp.vec3, device="cuda")
            self.forces_neural = wp.zeros(n, dtype=wp.vec3, device="cuda")

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
        self.pos_neural    = None
        self.vel_neural    = None
        self.forces_neural = None

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

        t0 = time.perf_counter()

        self._run_forces()
        wp.synchronize()
        t_forces = time.perf_counter()

        self._run_integrate()
        if self.accretion_enabled and self._frame % self.accretion_interval == 0:
            self._run_accrete()
        wp.synchronize()
        t_integrate = time.perf_counter()

        t_neural = t_integrate
        if self.neural_mode and self._neural_ff is not None and self.pos_neural is not None:
            if self._frame % self.neural_inference_interval == 0:
                self.forces_neural = self._neural_ff.compute_forces(
                    self.pos_neural, self.vel_neural, self.masses,
                )
            wp.launch(kernel_integrate, dim=self._n, device="cuda", inputs=[
                self.pos_neural, self.vel_neural, self.forces_neural,
                self.masses, self.active, self.dt,
            ])
            wp.synchronize()
            t_neural = time.perf_counter()

        self._frame += 1

        self.last_classical_forces_ms = (t_forces - t0) * 1000
        self.last_classical_integrate_ms = (t_integrate - t_forces) * 1000
        self.last_neural_ms = (t_neural - t_integrate) * 1000
        self.last_total_ms = (t_neural - t0) * 1000

        if self._frame % 100 == 0:
            msg = (f"[warp_nbody] Frame {self._frame} | "
                   f"forces: {self.last_classical_forces_ms:.1f}ms | "
                   f"integrate: {self.last_classical_integrate_ms:.1f}ms")
            if self.neural_mode:
                cached = "infer" if (self._frame - 1) % self.neural_inference_interval == 0 else "cached"
                msg += f" | neural: {self.last_neural_ms:.1f}ms ({cached})"
                msg += f" | pos_error: {self.get_position_error():.4f}"
            msg += f" | total: {self.last_total_ms:.1f}ms"
            print(msg)

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

    def get_neural_positions(self) -> wp.array | None:
        return self.pos_neural

    def get_position_error(self) -> float:
        if self.pos_neural is None or self.positions is None:
            return 0.0
        import torch
        pos_c = wp.to_torch(self.positions)
        pos_n = wp.to_torch(self.pos_neural)
        return torch.norm(pos_c - pos_n, dim=-1).mean().item()
