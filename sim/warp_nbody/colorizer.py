import warp as wp

from .kernels.visual import (
    kernel_reduce_max_mass_speed,
    kernel_colorize,
    _kernel_clear_float,
)


class ColorManager:

    def __init__(self):
        self._max_mass_wp  = None
        self._max_speed_wp = None

    def allocate(self, n: int) -> None:
        self._max_mass_wp  = wp.zeros(1, dtype=float, device="cuda")
        self._max_speed_wp = wp.zeros(1, dtype=float, device="cuda")

    def compute_colors(self, sim, fabric_colors: wp.array) -> None:
        wp.launch(_kernel_clear_float, dim=1, device="cuda", inputs=[self._max_mass_wp])
        wp.launch(_kernel_clear_float, dim=1, device="cuda", inputs=[self._max_speed_wp])
        wp.launch(kernel_reduce_max_mass_speed, dim=sim._n, device="cuda", inputs=[
            sim.masses, sim.velocities, sim.active,
            self._max_mass_wp, self._max_speed_wp,
        ])
        wp.launch(kernel_colorize, dim=sim._n, device="cuda", inputs=[
            sim.masses, sim.velocities, sim.active,
            fabric_colors, self._max_mass_wp, self._max_speed_wp,
        ])

    def free(self) -> None:
        self._max_mass_wp  = None
        self._max_speed_wp = None
