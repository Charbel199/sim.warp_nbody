import warp as wp


# positions, masses, forces -> compute forces
@wp.kernel
def kernel_forces(
    positions:   wp.array(dtype=wp.vec3),
    masses:      wp.array(dtype=float),
    active:      wp.array(dtype=int),
    forces:      wp.array(dtype=wp.vec3),
    G:           float,
    softening_sq: float,
    n:           int,
):
    i = wp.tid()
    if active[i] == 0:
        return
    f  = wp.vec3(0.0, 0.0, 0.0)
    pi = positions[i]
    mi = masses[i]
    for j in range(n):
        if j == i or active[j] == 0:
            continue
        r        = positions[j] - pi
        dist_sq  = wp.dot(r, r) + softening_sq
        inv_dist3 = 1.0 / (dist_sq * wp.sqrt(dist_sq))
        f = f + r * (G * mi * masses[j] * inv_dist3)
    forces[i] = f


# positions, velocities, forces, masses -> compute new velocities and positions
@wp.kernel
def kernel_integrate(
    positions:  wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    forces:     wp.array(dtype=wp.vec3),
    masses:     wp.array(dtype=float),
    active:     wp.array(dtype=int),
    dt:         float,
):
    i = wp.tid()
    if active[i] == 0:
        return
    acc         = forces[i] * (1.0 / masses[i])
    velocities[i] = velocities[i] + acc * dt
    positions[i]  = positions[i]  + velocities[i] * dt


# 2 passes to account for object merging
# pass 1: Check which objects get merged by which objects (if merge_into[i] = -1 -> Remains, if merge_into[i] = j, i is eaten by j)
# pass 2: Compute new radius for objects that ate other objects
@wp.kernel
def kernel_accrete_pass1(
    positions:  wp.array(dtype=wp.vec3),
    masses:     wp.array(dtype=float),
    radii:      wp.array(dtype=float),
    active:     wp.array(dtype=int),
    merge_into: wp.array(dtype=int),
    n:          int,
):
    i = wp.tid()
    if active[i] == 0:
        return
    merge_into[i] = -1
    pi = positions[i]
    mi = masses[i]
    for j in range(n):
        if j == i or active[j] == 0:
            continue
        dist = wp.length(positions[j] - pi)
        if dist < radii[i] + radii[j]:
            if masses[j] > mi or (masses[j] == mi and j < i):
                merge_into[i] = j
                return
@wp.kernel
def kernel_accrete_pass2(
    masses:     wp.array(dtype=float),
    radii:      wp.array(dtype=float),
    active:     wp.array(dtype=int),
    merge_into: wp.array(dtype=int),
    base_mass:   float,
    base_radius: float,
):
    i = wp.tid()
    if active[i] == 0 or merge_into[i] == -1:
        return
    j = merge_into[i]
    wp.atomic_add(masses, j, masses[i])
    active[i] = 0
    radii[j]  = base_radius * wp.pow(masses[j] / base_mass, 1.0 / 3.0)


@wp.kernel  # TODO: look into a warp.fill function
def kernel_reset_int(arr: wp.array(dtype=int)):
    arr[0] = int(0)


@wp.kernel  # TODO: Check tile sum or some sort of reduction tree in WARP
def kernel_count_active(active: wp.array(dtype=int), count: wp.array(dtype=int)):
    i = wp.tid()
    if active[i] != 0:
        wp.atomic_add(count, 0, 1)


