import time
import traceback

import omni.kit.pipapi

_BSP = "--break-system-packages"

omni.kit.pipapi.install("nvtx", extra_args=[_BSP])
omni.kit.pipapi.install("h5py", extra_args=[_BSP])

# PyTorch + PyG need CUDA-specific index URLs
_TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu124"
_PYG_INDEX = "https://data.pyg.org/whl/torch-2.6.0+cu124.html"

omni.kit.pipapi.install(
    "torch==2.6.0",
    extra_args=["--index-url", _TORCH_CUDA_INDEX, _BSP],
)
omni.kit.pipapi.install(
    "torch_geometric",
    extra_args=["--find-links", _PYG_INDEX, _BSP],
)
omni.kit.pipapi.install(
    "torch_scatter",
    extra_args=["--find-links", _PYG_INDEX, _BSP],
)
omni.kit.pipapi.install(
    "torch_sparse",
    extra_args=["--find-links", _PYG_INDEX, _BSP],
)
omni.kit.pipapi.install(
    "torch_cluster",
    extra_args=["--find-links", _PYG_INDEX, _BSP],
)
import warp as wp
import carb
import omni.ext
import omni.kit.app

from .simulation import NBodySimulation
from .fabric_bridge import FabricBridge
from .colorizer import ColorManager
from .instancer import create_instancer, destroy_instancer, create_neural_instancer, destroy_neural_instancer
from .ui.panel import NBodyPanel, SPAWN_FNS


class NBodyExtension(omni.ext.IExt):

    def on_startup(self, _ext_id: str) -> None:
        try:
            self._sim        = NBodySimulation()
            self._bridge     = FabricBridge()
            self._panel      = NBodyPanel()
            self._colorizer  = None
            self._update_sub = None
            self._running    = False
            self._spawn_time: float | None = None
            self._initial_n: int = 0

            self._panel.build(on_spawn=self._on_spawn, on_stop=self._on_stop)
        except Exception as e:
            carb.log_error(f"[warp_nbody] on_startup failed: {e}\n{traceback.format_exc()}")

    def _on_spawn(self, preset, n, G, softening, dt, spread, body_mass, accretion) -> None:
        try:
            if self._running:
                self._on_stop()

            positions_np, velocities_np, masses_np = SPAWN_FNS[preset](n, G, spread, body_mass)

            create_instancer(n)

            neural_enabled = self._panel.get_neural_enabled()
            if neural_enabled:
                checkpoint = self._panel.get_checkpoint_path()
                self._sim.neural_cutoff = self._panel.get_neural_cutoff()
                self._sim.neural_inference_interval = self._panel.get_neural_interval()
                self._sim.set_neural_mode(True, checkpoint)
                carb.log_info(f"[warp_nbody] Neural mode enabled, cutoff={self._sim.neural_cutoff}, "
                              f"interval={self._sim.neural_inference_interval}, checkpoint: {checkpoint}")
            else:
                self._sim.neural_mode = False

            self._sim.G                 = G
            self._sim.softening         = softening
            self._sim.dt                = dt
            self._sim.accretion_enabled = accretion
            self._sim.allocate(positions_np, velocities_np, masses_np)

            self._colorizer = ColorManager()
            self._colorizer.allocate(n)

            self._bridge.bind(self._sim, n, self._colorizer)

            if self._sim.neural_mode:
                create_neural_instancer(n)
                self._bridge.bind_neural(n)
                carb.log_info(f"[warp_nbody] Neural instancer created with {n} particles")

            self._running    = True
            self._spawn_time = time.monotonic()
            self._initial_n  = n

            self._update_sub = (
                omni.kit.app.get_app()
                .get_update_event_stream()
                .create_subscription_to_pop(self._on_update, name="NBodyUpdate")
            )
        except Exception as e:
            carb.log_error(f"[warp_nbody] spawn failed: {e}\n{traceback.format_exc()}")

    def _on_stop(self) -> None:
        self._update_sub = None
        self._running    = False
        self._bridge.unbind()
        if self._colorizer:
            self._colorizer.free()
            self._colorizer = None
        self._sim.free()
        destroy_instancer()
        destroy_neural_instancer()

    def _on_update(self, _event) -> None:
        if not self._running:
            return
        try:
            self._sim.step()
            self._bridge.mark_dirty()
            if self._sim.pos_neural is not None:
                self._bridge.write_neural(self._sim)
            self._refresh_stats()
        except Exception as e:
            carb.log_error(f"[warp_nbody] update error: {e}\n{traceback.format_exc()}")
            self._running = False

    def _refresh_stats(self) -> None:
        active   = self._sim.count_active()
        merges   = self._initial_n - active
        sim_time = time.monotonic() - self._spawn_time

        neural_active = 0
        neural_merges = 0
        pos_error = 0.0
        if self._sim.neural_mode and self._sim.pos_neural is not None:
            neural_active = self._sim.count_active_neural()
            neural_merges = self._initial_n - neural_active
            pos_error = self._sim.get_position_error()

        self._panel.update_stats(active, merges, sim_time, neural_active, neural_merges, pos_error)

    def on_shutdown(self) -> None:
        if getattr(self, "_running", False):
            self._on_stop()
        if getattr(self, "_panel", None):
            self._panel.destroy()
