import dataclasses
import pathlib

import numpy as np
import h5py
import warp as wp

from ..kernels.physics import kernel_forces, kernel_integrate
from ..spawner import (
    spawn_binary_galaxy,
    spawn_galaxy_disk,
    spawn_random,
    spawn_sphere,
    spawn_solar_system,
)

_DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "data"

PRESET_SPAWNERS = {
    "Sphere":        lambda n, G: spawn_sphere(n, radius=50.0, body_mass=1.0, speed_scale=0.5),
    "Galaxy Disk":   lambda n, G: spawn_galaxy_disk(n, radius=50.0, body_mass=1.0, G=G),
    "Binary Galaxy": lambda n, G: spawn_binary_galaxy(n, radius=40.0, body_mass=1.0, G=G),
    "Solar System":  lambda n, G: spawn_solar_system(n, G=G),
    "Random":        lambda n, G: spawn_random(n, extent=50.0, body_mass=1.0, speed_scale=0.5),
}


def dataset_path_for_preset(preset: str) -> str:
    slug = preset.lower().replace(" ", "_")
    return str(_DATA_DIR / f"nbody_{slug}.h5")


@dataclasses.dataclass
class DataGenConfig:
    PRESET: str = "Sphere"
    N_PARTICLES: int = 1000
    N_EPISODES: int = 200
    N_STEPS: int = 500
    DT: float = 0.01
    G: float = 0.001
    SOFTENING: float = 0.05


def generate_dataset(config: DataGenConfig) -> None:
    wp.init()

    if config.PRESET not in PRESET_SPAWNERS:
        raise ValueError(f"Unknown preset '{config.PRESET}'. Available: {list(PRESET_SPAWNERS)}")

    spawner = PRESET_SPAWNERS[config.PRESET]
    n = config.N_PARTICLES
    total_frames = config.N_EPISODES * config.N_STEPS

    all_positions = np.empty((total_frames, n, 3), dtype=np.float32)
    all_velocities = np.empty((total_frames, n, 3), dtype=np.float32)
    all_masses = np.empty((total_frames, n, 1), dtype=np.float32)
    all_accelerations = np.empty((total_frames, n, 3), dtype=np.float32)

    softening_sq = config.SOFTENING ** 2
    frame_idx = 0

    for episode in range(config.N_EPISODES):
        positions_np, velocities_np, masses_np = spawner(n, config.G)

        rng = np.random.default_rng(seed=episode)
        positions_np += rng.normal(0, 1.0, positions_np.shape).astype(np.float32)
        velocities_np += rng.normal(0, 0.1, velocities_np.shape).astype(np.float32)

        pos_wp = wp.array(positions_np, dtype=wp.vec3, device="cuda")
        vel_wp = wp.array(velocities_np, dtype=wp.vec3, device="cuda")
        mass_wp = wp.array(masses_np, dtype=float, device="cuda")
        forces_wp = wp.zeros(n, dtype=wp.vec3, device="cuda")
        active_wp = wp.ones(n, dtype=int, device="cuda")

        for step in range(config.N_STEPS):
            wp.launch(kernel_forces, dim=n, device="cuda", inputs=[
                pos_wp, mass_wp, active_wp, forces_wp,
                config.G, softening_sq, n,
            ])

            pos_t = pos_wp.numpy()
            vel_t = vel_wp.numpy()
            forces_t = forces_wp.numpy()
            mass_np = mass_wp.numpy()

            acc_t = forces_t / mass_np[:, np.newaxis]

            all_positions[frame_idx] = pos_t
            all_velocities[frame_idx] = vel_t
            all_masses[frame_idx] = mass_np[:, np.newaxis]
            all_accelerations[frame_idx] = acc_t

            wp.launch(kernel_integrate, dim=n, device="cuda", inputs=[
                pos_wp, vel_wp, forces_wp, mass_wp, active_wp, config.DT,
            ])

            frame_idx += 1

        if (episode + 1) % 10 == 0:
            print(f"  [{config.PRESET}] Episode {episode + 1}/{config.N_EPISODES} done")

    output_path = pathlib.Path(dataset_path_for_preset(config.PRESET))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(output_path), "w") as f:
        f.create_dataset("positions", data=all_positions, compression="gzip")
        f.create_dataset("velocities", data=all_velocities, compression="gzip")
        f.create_dataset("masses", data=all_masses, compression="gzip")
        f.create_dataset("accelerations", data=all_accelerations, compression="gzip")

    print(f"Dataset saved to {output_path} - {total_frames} frames, {n} particles each")


if __name__ == "__main__":
    import sys
    preset = sys.argv[1] if len(sys.argv) > 1 else "Sphere"
    cfg = DataGenConfig(PRESET=preset)
    print(f"Generating [{preset}]: {cfg.N_EPISODES} episodes x {cfg.N_STEPS} steps x {cfg.N_PARTICLES} particles")
    generate_dataset(cfg)
