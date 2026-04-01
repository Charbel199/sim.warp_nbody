import time
import traceback

import carb
import omni.ext
import omni.kit.app

from .simulation import NBodySimulation
from .fabric_bridge import FabricBridge
from .colorizer import ColorManager
from .instancer import create_instancer, destroy_instancer
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

    def _on_spawn(self, preset, n, G, softening, dt, accretion) -> None:
        try:
            if self._running:
                self._on_stop()

            positions_np, velocities_np, masses_np = SPAWN_FNS[preset](n, G)

            create_instancer(n)

            self._sim.G                 = G
            self._sim.softening         = softening
            self._sim.dt                = dt
            self._sim.accretion_enabled = accretion
            self._sim.allocate(positions_np, velocities_np, masses_np)

            self._colorizer = ColorManager()
            self._colorizer.allocate(n)

            self._bridge.bind(self._sim, n, self._colorizer)

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

    def _on_update(self, _event) -> None:
        if not self._running:
            return
        try:
            self._sim.step()
            self._bridge.mark_dirty()
            self._refresh_stats()
        except Exception as e:
            carb.log_error(f"[warp_nbody] update error: {e}\n{traceback.format_exc()}")
            self._running = False

    def _refresh_stats(self) -> None:
        active   = self._sim.count_active()
        merges   = self._initial_n - active
        sim_time = time.monotonic() - self._spawn_time
        self._panel.update_stats(active, merges, sim_time)

    def on_shutdown(self) -> None:
        if getattr(self, "_running", False):
            self._on_stop()
        if getattr(self, "_panel", None):
            self._panel.destroy()
