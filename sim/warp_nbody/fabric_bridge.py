import warp as wp
from usdrt import Usd, Sdf, Vt
import omni.usd

from .instancer import INSTANCER_PATH

_VISUAL_SCALE_REF = 3.0   # scale at reference body count
_VISUAL_CAP_REF   = 15.0  # cap at reference body count
_N_REF            = 1000  # reference body count


@wp.kernel
def kernel_compute_scales(
    radii:   wp.array(dtype=float),
    active:  wp.array(dtype=int),
    scales:  wp.array(dtype=wp.vec3),
    v_scale: float,
    v_cap:   float,
):
    i = wp.tid()
    if active[i] == 0:
        scales[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    r = wp.min(radii[i] * v_scale, v_cap)
    scales[i] = wp.vec3(r, r, r)


class FabricBridge:

    def __init__(self):
        self._sim        = None
        self._n          = 0
        self._colorizer  = None
        self._rt_stage   = None
        self._pos_attr   = None
        self._scale_attr = None
        self._color_attr = None
        self._pos_wp     = None  # GPU scratch buffers
        self._scales_wp  = None
        self._colors_wp  = None

    def bind(self, sim, n_bodies: int, colorizer) -> None:
        self._sim       = sim
        self._n         = n_bodies
        self._colorizer = colorizer

        density_factor      = (_N_REF / n_bodies) ** (1.0 / 3.0)
        self._visual_scale  = _VISUAL_SCALE_REF * density_factor
        self._visual_cap    = _VISUAL_CAP_REF   * density_factor

        self._pos_wp    = wp.zeros(n_bodies, dtype=wp.vec3, device="cuda:0")
        self._scales_wp = wp.zeros(n_bodies, dtype=wp.vec3, device="cuda:0")
        self._colors_wp = wp.zeros(n_bodies, dtype=wp.vec3, device="cuda:0")

        stage_id = omni.usd.get_context().get_stage_id()
        self._rt_stage = Usd.Stage.Attach(stage_id)

        prim = self._rt_stage.GetPrimAtPath(Sdf.Path(INSTANCER_PATH))
        self._pos_attr   = prim.GetAttribute("positions")
        self._scale_attr = prim.GetAttribute("scales")
        self._color_attr = prim.GetAttribute("primvars:displayColor")

        # push initial GPU buffers into Fabric (GPU -> GPU copy).
        with wp.ScopedDevice("cuda:0"):
            self._pos_attr.Set(Vt.Vec3fArray(sim.positions))
            self._scale_attr.Set(Vt.Vec3fArray(self._scales_wp))
            self._color_attr.Set(Vt.Vec3fArray(self._colors_wp))
         #   wp.synchronize_device("cuda:0")

    def mark_dirty(self) -> None:
        if self._sim is None:
            return

        with wp.ScopedDevice("cuda:0"):
            # GPU -> GPU: copy sim positions into scratch buffer
            wp.copy(self._pos_wp, self._sim.positions)

            # compute scales on GPU into scratch buffer
            wp.launch(kernel_compute_scales, dim=self._n, device="cuda:0", inputs=[
                self._sim.radii, self._sim.active, self._scales_wp,
                self._visual_scale, self._visual_cap,
            ])

            # compute colors on GPU into scratch buffer (no CPU involved)
            self._colorizer.compute_colors(self._sim, self._colors_wp)

            # push all scratch buffers to Fabric (GPU -> GPU copies via USDRT) #TODO: we do not need it for positions as we already have a buffer for it
            self._pos_attr.Set(Vt.Vec3fArray(self._pos_wp))
            self._scale_attr.Set(Vt.Vec3fArray(self._scales_wp))
            self._color_attr.Set(Vt.Vec3fArray(self._colors_wp))

        #wp.synchronize_device("cuda:0")

    def unbind(self) -> None:
        self._sim        = None
        self._n          = 0
        self._colorizer  = None
        self._rt_stage   = None
        self._pos_attr   = None
        self._scale_attr = None
        self._color_attr = None
        self._pos_wp     = None
        self._scales_wp  = None
        self._colors_wp  = None
