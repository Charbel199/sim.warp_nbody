import torch
import warp as wp

from .model import NBodyGNN


class NeuralForceField:

    def __init__(self, checkpoint_path: str, device: str = "cuda", cutoff: float = 2.0):
        self._device = device
        self._model = NBodyGNN(cutoff=cutoff)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self._model.load_state_dict(checkpoint)
        self._model.to(device)
        self._model.eval()
        self._loaded = True

    def compute_forces(
        self,
        pos_wp: wp.array,
        vel_wp: wp.array,
        mass_wp: wp.array,
    ) -> wp.array:
        wp.synchronize()

        pos_t = wp.to_torch(pos_wp)
        vel_t = wp.to_torch(vel_wp)
        mass_t = wp.to_torch(mass_wp).unsqueeze(-1)

        with torch.no_grad():
            acc_t = self._model(pos_t, vel_t, mass_t)
            forces_t = acc_t * mass_t

        torch.cuda.synchronize()
        return wp.from_torch(forces_t.contiguous(), dtype=wp.vec3)

    def is_loaded(self) -> bool:
        return self._loaded
