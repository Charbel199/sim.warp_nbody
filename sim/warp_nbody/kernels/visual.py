import warp as wp


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


@wp.kernel
def kernel_reduce_max_mass_speed(
    masses:     wp.array(dtype=float),
    velocities: wp.array(dtype=wp.vec3),
    active:     wp.array(dtype=int),
    max_mass:   wp.array(dtype=float),
    max_speed:  wp.array(dtype=float),
):
    i = wp.tid()
    if active[i] == 0:
        return
    wp.atomic_max(max_mass,  0, masses[i])
    wp.atomic_max(max_speed, 0, wp.length(velocities[i]))


# max_mass_arr and max_speed_arr are 1-element GPU arrays populated by kernel_reduce_max_mass_speed
@wp.kernel
def kernel_colorize(
    masses:        wp.array(dtype=float),
    velocities:    wp.array(dtype=wp.vec3),
    active:        wp.array(dtype=int),
    colors:        wp.array(dtype=wp.vec3),
    max_mass_arr:  wp.array(dtype=float),
    max_speed_arr: wp.array(dtype=float),
):
    i = wp.tid()
    if active[i] == 0:
        colors[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    max_mass  = wp.max(max_mass_arr[0],  float(1e-6))
    max_speed = wp.max(max_speed_arr[0], float(1e-6))
    t_mass  = wp.min(masses[i] / max_mass, 1.0)
    t_speed = wp.min(wp.length(velocities[i]) / max_speed, 1.0)
    t = t_mass * 0.7 + t_speed * 0.3
    if t < 0.5:
        s = t * 2.0
        colors[i] = wp.vec3(s * 0.9 + 0.1, s * 0.9 + 0.1, 1.0)
    else:
        s = (t - 0.5) * 2.0
        colors[i] = wp.vec3(1.0, 1.0 - s * 0.6, 1.0 - s)


@wp.kernel  # TODO: check wp fill
def _kernel_clear_float(arr: wp.array(dtype=float)):
    arr[wp.tid()] = float(0.0)
