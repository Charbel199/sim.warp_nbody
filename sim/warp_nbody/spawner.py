import numpy as np

# spawn a big planet and a disk of smaller objects around
def spawn_galaxy_disk(n, radius, central_mass, body_mass, G=0.001):
    rng = np.random.default_rng(seed=42)

    positions  = np.zeros((n, 3), dtype=np.float32)
    velocities = np.zeros((n, 3), dtype=np.float32)
    masses     = np.full(n, body_mass, dtype=np.float32)
    masses[0]  = central_mass

    angles = rng.uniform(0, 2 * np.pi, n - 1)
    radii  = rng.uniform(2.0, radius, n - 1)
    positions[1:, 0] = radii * np.cos(angles)
    positions[1:, 2] = radii * np.sin(angles)
    positions[1:, 1] = rng.uniform(-0.5, 0.5, n - 1)

    orbital_speeds = np.sqrt(G * central_mass / (radii + 1e-6))
    velocities[1:, 0] = -orbital_speeds * np.sin(angles)
    velocities[1:, 2] =  orbital_speeds * np.cos(angles)

    return positions, velocities, masses

# spawn sphere of objects 
def spawn_sphere(n, radius, body_mass, speed_scale):
    rng = np.random.default_rng(seed=42)

    positions = np.zeros((n, 3), dtype=np.float32)
    count = 0
    while count < n:
        pts    = rng.uniform(-1, 1, (n * 2, 3))
        inside = pts[np.linalg.norm(pts, axis=1) <= 1.0]
        take   = min(len(inside), n - count)
        positions[count:count + take] = inside[:take] * radius
        count += take

    velocities = rng.uniform(-speed_scale, speed_scale, (n, 3)).astype(np.float32)
    masses     = np.full(n, body_mass, dtype=np.float32)
    return positions, velocities, masses

# spawn a solar system
def spawn_solar_system(n, G=0.001):
    rng = np.random.default_rng(seed=42)

    # star mass chosen so physics collision radius (~6.5 u) sits well inside
    # the innermost planet orbit (20 u), keeping all planets alive.
    star_mass = 10_000.0

    # 8 planets: 4 rocky inner, 4 gas/ice outer (masses scaled for visibility) TODO: Will add some more examples
    planet_orbits = np.array([20, 30, 42, 55, 88, 118, 150, 180], dtype=np.float32)
    planet_masses = np.array([0.4, 1.0, 1.2, 0.6, 22.0, 16.0, 7.0, 6.5], dtype=np.float32)
    n_planets     = len(planet_orbits)
    n_asteroids   = max(0, n - 1 - n_planets)

    positions  = np.zeros((n, 3), dtype=np.float32)
    velocities = np.zeros((n, 3), dtype=np.float32)
    masses     = np.ones(n, dtype=np.float32)
    masses[0]  = star_mass

    for i in range(n_planets):
        r     = planet_orbits[i]
        angle = rng.uniform(0, 2 * np.pi)
        positions[1 + i, 0]  = r * np.cos(float(angle))
        positions[1 + i, 2]  = r * np.sin(float(angle))
        positions[1 + i, 1]  = rng.uniform(-0.5, 0.5)
        masses[1 + i]        = planet_masses[i]
        v = np.sqrt(G * star_mass / r)
        velocities[1 + i, 0] = -v * np.sin(float(angle))
        velocities[1 + i, 2] =  v * np.cos(float(angle))

    if n_asteroids > 0:
        s      = 1 + n_planets
        angles = rng.uniform(0, 2 * np.pi, n_asteroids).astype(np.float32)
        radii  = rng.uniform(65.0, 80.0,   n_asteroids).astype(np.float32)
        positions[s:, 0]  = radii * np.cos(angles)
        positions[s:, 2]  = radii * np.sin(angles)
        positions[s:, 1]  = rng.uniform(-1.5, 1.5, n_asteroids).astype(np.float32)
        masses[s:]        = 0.3
        v_ast = np.sqrt(G * star_mass / radii).astype(np.float32)
        velocities[s:, 0] = -v_ast * np.sin(angles)
        velocities[s:, 2] =  v_ast * np.cos(angles)

    return positions, velocities, masses

# spawn small objects randomly
def spawn_random(n, extent, body_mass, speed_scale):
    rng = np.random.default_rng(seed=42)
    positions  = rng.uniform(-extent, extent, (n, 3)).astype(np.float32)
    velocities = rng.uniform(-speed_scale, speed_scale, (n, 3)).astype(np.float32)
    masses     = np.full(n, body_mass, dtype=np.float32)
    return positions, velocities, masses

# spawn 2 galaxy disks
def spawn_binary_galaxy(n, radius=40.0, central_mass=1e6, body_mass=1.0, G=0.001):
    rng  = np.random.default_rng(seed=42)
    half = n // 2
    rest = n - half

    def _disk(n_disk, center_x, bulk_vx):
        pos  = np.zeros((n_disk, 3), dtype=np.float32)
        vel  = np.zeros((n_disk, 3), dtype=np.float32)
        mass = np.full(n_disk, body_mass, dtype=np.float32)
        mass[0]   = central_mass
        pos[0, 0] = center_x

        angles = rng.uniform(0, 2 * np.pi, n_disk - 1)
        radii  = rng.uniform(2.0, radius, n_disk - 1)
        pos[1:, 0] = center_x + radii * np.cos(angles)
        pos[1:, 2] = radii * np.sin(angles)
        pos[1:, 1] = rng.uniform(-0.5, 0.5, n_disk - 1)

        orbital_speeds = np.sqrt(G * central_mass / (radii + 1e-6))
        vel[1:, 0] = -orbital_speeds * np.sin(angles)
        vel[1:, 2] =  orbital_speeds * np.cos(angles)
        vel[:, 0] += bulk_vx
        return pos, vel, mass

    v_approach           = 0.3 * float(np.sqrt(G * central_mass / 60.0))
    pos_a, vel_a, mass_a = _disk(half, -60.0, +v_approach)
    pos_b, vel_b, mass_b = _disk(rest, +60.0, -v_approach)

    return (
        np.concatenate([pos_a, pos_b], axis=0),
        np.concatenate([vel_a, vel_b], axis=0),
        np.concatenate([mass_a, mass_b], axis=0),
    )

# spawn a backhole, not working yet, will be a cool concept once everything is setup
def spawn_black_hole(n, bh_mass=1e9, body_mass=1.0, max_radius=80.0, G=0.001):
    rng        = np.random.default_rng(seed=42)
    positions  = np.zeros((n, 3), dtype=np.float32)
    velocities = np.zeros((n, 3), dtype=np.float32)
    masses     = np.full(n, body_mass, dtype=np.float32)
    masses[0]  = bh_mass

    inner = n // 2
    outer = n - 1 - inner

    in_r = rng.uniform(2.0, 15.0, inner).astype(np.float32)
    in_a = rng.uniform(0, 2 * np.pi, inner).astype(np.float32)
    positions[1:inner + 1, 0]  = in_r * np.cos(in_a)
    positions[1:inner + 1, 2]  = in_r * np.sin(in_a)
    v_in = np.sqrt(G * bh_mass / (in_r + 1e-6)).astype(np.float32)
    velocities[1:inner + 1, 0] = -v_in * np.sin(in_a)
    velocities[1:inner + 1, 2] =  v_in * np.cos(in_a)

    out_r = rng.uniform(15.0, max_radius, outer).astype(np.float32)
    out_a = rng.uniform(0, 2 * np.pi, outer).astype(np.float32)
    positions[inner + 1:, 0]  = out_r * np.cos(out_a)
    positions[inner + 1:, 2]  = out_r * np.sin(out_a)
    v_out = np.sqrt(G * bh_mass / (out_r + 1e-6)).astype(np.float32)
    velocities[inner + 1:, 0] = -v_out * np.sin(out_a) + 0.3 * v_out * np.cos(out_a)
    velocities[inner + 1:, 2] =  v_out * np.cos(out_a) + 0.3 * v_out * np.sin(out_a)

    return positions, velocities, masses
