"""
tunneling.py
============
Trajectory math and tunneling metrics for pitch pairings.

For current pitches: uses measured Statcast acceleration values (ax, ay, az)
which already include all observed forces (Magnus + drag + gravity + SSW).

For hypothetical optimized shapes: derives acceleration from target pfx_x/pfx_z
using Magnus physics. Less accurate for sinkers (SSW deviation) but reasonable
for most pitch types.

Coordinate convention (Statcast):
  y = 0 at front of plate, y = 60.5 at rubber
  x positive toward first base (catcher's POV)
  z positive upward
  
Tunnel point: commonly 23 ft from release, where elite hitters commit.
With release_extension ~6 ft, that means y ~= 60.5 - 6 - 23 = ~31.5 ft from plate.
We'll use the actual release_y per pitch for accuracy.
"""

import numpy as np


# Standard tunnel point — distance from release where hitters commit
TUNNEL_DIST_FROM_RELEASE = 23.0  # ft

# Gravity (z-direction, ft/s²)
GRAVITY = -32.174


def trajectory_at_y(release_x, release_y, release_z,
                    vx0, vy0, vz0, ax, ay, az, y_target):
    """
    Compute pitch position (x, z) at a given y-coordinate.
    
    Statcast convention: vx0/vy0/vz0/ax/ay/az are measured at y=50 ft frame.
    The release position (release_pos_x/y/z) is the actual 3D location at release,
    typically at y between 51-55 ft (50.5 + release_extension - 6 for a pitcher
    with 6ft extension would put it at y=50.5 + 0 = 50.5; actually y_release = 60.5 - extension).
    
    To compute position at any y, we solve the quadratic from the y=50 frame:
      y(t) = 50 + vy0*t + 0.5*ay*t²  (t=0 at the y=50 crossing)
    
    Then x(t) and z(t) use the same time t.
    
    To anchor x and z properly, we use the release point: at t = t_release
    (which is negative since pitch was at y=50 AFTER release), x = release_x.
    """
    import numpy as np
    
    # Solve for t when y(t) = y_target, starting from y=50 frame
    a = 0.5 * ay
    b = vy0  # negative for pitch moving toward plate
    c = 50.0 - y_target
    
    disc = b*b - 4*a*c
    if disc < 0:
        return None, None
    
    sqrt_disc = np.sqrt(disc)
    # Two roots; for a pitch moving in -y direction with positive drag (ay > 0),
    # the pitch passes through y=target twice mathematically, but the physical
    # one is the smaller |t|. Since vy0 < 0 and ay > 0:
    #   - At y_target above 50: both roots are negative (pitch was there before y=50)
    #     Take the larger (less negative) one
    #   - At y_target below 50: roots have opposite signs; take positive
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)
    
    # Pick the smaller absolute value root (closest in time to y=50 crossing)
    t = t1 if abs(t1) < abs(t2) else t2
    
    # Time from y=50 frame to release point
    c_r = 50.0 - release_y
    disc_r = b*b - 4*a*c_r
    if disc_r < 0:
        return None, None
    sqrt_disc_r = np.sqrt(disc_r)
    tr1 = (-b - sqrt_disc_r) / (2*a)
    tr2 = (-b + sqrt_disc_r) / (2*a)
    t_release = tr1 if abs(tr1) < abs(tr2) else tr2
    
    # Anchor x0/z0 at y=50 frame using release point
    # release_x = x0_50 + vx0*t_release + 0.5*ax*t_release²
    x0_50 = release_x - vx0*t_release - 0.5*ax*t_release**2
    z0_50 = release_z - vz0*t_release - 0.5*az*t_release**2
    
    # Now compute position at target y
    x_at = x0_50 + vx0*t + 0.5*ax*t**2
    z_at = z0_50 + vz0*t + 0.5*az*t**2
    
    return float(x_at), float(z_at)


def compute_tunnel_metrics(pitch_a, pitch_b):
    """
    Compute tunneling metrics for a pitch pair.
    
    Each pitch dict needs: release_pos_x, release_pos_y, release_pos_z,
                           vx0, vy0, vz0, ax, ay, az
    
    Returns dict with:
      tunnel_sep_in:    inches separation at tunnel point (closer = better tunnel)
      plate_sep_in:     inches separation at the plate
      late_break_ratio: plate_sep / tunnel_sep (higher = more late deception)
      tunnel_y:         y-coord of tunnel point (ft from plate)
    """
    # Tunnel point — use average release_y from both pitches for fairness
    avg_release_y = (pitch_a['release_pos_y'] + pitch_b['release_pos_y']) / 2
    tunnel_y = avg_release_y - TUNNEL_DIST_FROM_RELEASE
    
    # Position at tunnel point
    ax_t, az_t = trajectory_at_y(
        pitch_a['release_pos_x'], pitch_a['release_pos_y'], pitch_a['release_pos_z'],
        pitch_a['vx0'], pitch_a['vy0'], pitch_a['vz0'],
        pitch_a['ax'], pitch_a['ay'], pitch_a['az'],
        tunnel_y
    )
    bx_t, bz_t = trajectory_at_y(
        pitch_b['release_pos_x'], pitch_b['release_pos_y'], pitch_b['release_pos_z'],
        pitch_b['vx0'], pitch_b['vy0'], pitch_b['vz0'],
        pitch_b['ax'], pitch_b['ay'], pitch_b['az'],
        tunnel_y
    )
    
    # Position at plate (y=0)
    ax_p, az_p = trajectory_at_y(
        pitch_a['release_pos_x'], pitch_a['release_pos_y'], pitch_a['release_pos_z'],
        pitch_a['vx0'], pitch_a['vy0'], pitch_a['vz0'],
        pitch_a['ax'], pitch_a['ay'], pitch_a['az'],
        1.417  # front of plate is ~1.4ft for break calc; use plate front
    )
    bx_p, bz_p = trajectory_at_y(
        pitch_b['release_pos_x'], pitch_b['release_pos_y'], pitch_b['release_pos_z'],
        pitch_b['vx0'], pitch_b['vy0'], pitch_b['vz0'],
        pitch_b['ax'], pitch_b['ay'], pitch_b['az'],
        1.417
    )
    
    if None in (ax_t, az_t, bx_t, bz_t, ax_p, az_p, bx_p, bz_p):
        return None
    
    # Convert ft separations to inches
    tunnel_sep_in = np.sqrt((ax_t - bx_t)**2 + (az_t - bz_t)**2) * 12
    plate_sep_in  = np.sqrt((ax_p - bx_p)**2 + (az_p - bz_p)**2) * 12
    
    late_break_ratio = plate_sep_in / tunnel_sep_in if tunnel_sep_in > 0.01 else 0
    
    return {
        'tunnel_sep_in':    round(float(tunnel_sep_in), 2),
        'plate_sep_in':     round(float(plate_sep_in), 2),
        'late_break_ratio': round(float(late_break_ratio), 2),
        'tunnel_y':         round(float(tunnel_y), 2),
    }


def derive_accelerations_from_shape(pfx_x_in, pfx_z_in, release_speed_mph,
                                     release_x, release_y, release_z,
                                     vx0_actual=None, vy0_actual=None, vz0_actual=None,
                                     ay_actual=None):
    """
    Derive ax, ay, az for a hypothetical pitch shape.

    For predicting tunneling of an optimized shape, we want to keep the pitcher's
    actual release direction and speed (vx0, vy0, vz0) and only modify the
    Magnus accelerations (ax, az) to produce the target movement. This way the
    comparison isolates the effect of the shape change rather than confounding
    it with aim differences.

    If vx0_actual/vy0_actual/vz0_actual are provided, use them. Otherwise fall
    back to estimating from release position assuming aim at strike-zone center.
    """
    # Speed in ft/s (1 mph = 1.467 ft/s)
    speed_fps = release_speed_mph * 1.467

    # Use actual release velocities if provided
    if vx0_actual is not None and vy0_actual is not None and vz0_actual is not None:
        vx0 = vx0_actual
        vy0 = vy0_actual
        vz0 = vz0_actual
    else:
        # Fallback: estimate aiming at center of plate
        flight_dist = release_y
        vy0 = -speed_fps
        vertical_angle_rad = np.arctan2(release_z - 2.5, flight_dist)
        vz0 = -speed_fps * np.sin(vertical_angle_rad)
        horizontal_angle_rad = np.arctan2(-release_x, flight_dist)
        vx0 = speed_fps * np.sin(horizontal_angle_rad)

    # Time of flight from y=50 to plate (approx)
    # vy0 is at y=50 frame, pitch travels to y=1.417 (~48.6 ft to cover)
    # With drag, average speed is roughly |vy0| - 5 ft/s
    avg_vy = abs(vy0) - 5.0
    t_flight = (50.0 - 1.417) / avg_vy

    # Required Magnus accelerations to produce specified break
    # pfx is induced break in inches; convert to ft
    # break = 0.5 * a * t² => a = 2 * break / t²
    pfx_x_ft = pfx_x_in / 12
    pfx_z_ft = pfx_z_in / 12

    ax = 2 * pfx_x_ft / (t_flight ** 2)
    az_induced = 2 * pfx_z_ft / (t_flight ** 2)
    az = az_induced + GRAVITY  # pfx_z excludes gravity; add it back for total az

    # Use actual ay if provided (drag is mostly determined by velocity, doesn't change with shape)
    if ay_actual is not None:
        ay = ay_actual
    else:
        ay = -0.0023 * speed_fps**2 / (release_speed_mph / 95)

    return vx0, vy0, vz0, ax, ay, az


def make_pitch_dict_from_optimized(pfx_x_in, pfx_z_in, release_speed_mph,
                                    release_x, release_y, release_z,
                                    vx0_actual=None, vy0_actual=None, vz0_actual=None,
                                    ay_actual=None):
    """
    Build a pitch dict for an optimized hypothetical shape that can be
    used with compute_tunnel_metrics.

    Optional: pass current pitch's actual vx0/vy0/vz0/ay to keep release
    direction stable so tunneling comparisons isolate the shape change.
    """
    vx0, vy0, vz0, ax, ay, az = derive_accelerations_from_shape(
        pfx_x_in, pfx_z_in, release_speed_mph,
        release_x, release_y, release_z,
        vx0_actual, vy0_actual, vz0_actual, ay_actual
    )
    return {
        'release_pos_x': release_x,
        'release_pos_y': release_y,
        'release_pos_z': release_z,
        'vx0': vx0, 'vy0': vy0, 'vz0': vz0,
        'ax':  ax,  'ay':  ay,  'az':  az,
    }


# ── Vertical Approach Angle (VAA) ──────────────────────────────────────────
def vaa_from_trajectory(vy0, vz0, ay, az, y0=50.0, y_plate=1.417):
    """
    Compute vertical approach angle (degrees) at the plate from measured
    trajectory components. Validated against Statcast/Trackman VertApprAngle
    to ~0.06 deg mean error.

    VAA = arctan(vz_plate / |vy_plate|), where velocities at the plate come
    from constant-acceleration kinematics in the y=50 ft frame.
    """
    a = 0.5 * ay
    b = vy0
    c = y0 - y_plate
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    # pitch travels toward plate; take the smaller positive time
    t = t1 if (t1 > 0 and (t1 < t2 or t2 <= 0)) else t2
    if t <= 0:
        return None
    vz_plate = vz0 + az * t
    vy_plate = vy0 + ay * t
    return float(np.degrees(np.arctan2(vz_plate, abs(vy_plate))))


def vaa_from_shape(pfx_x_in, pfx_z_in, release_speed_mph,
                   release_x, release_y, release_z,
                   vx0_actual=None, vy0_actual=None, vz0_actual=None, ay_actual=None):
    """
    Compute VAA for a hypothetical shape by deriving accelerations from the
    shape (Magnus) and then applying the trajectory VAA formula.

    Used with the delta method: the absolute value carries a small IVB-dependent
    bias (~0.06 deg at low IVB to ~0.54 deg at high IVB), but the DIFFERENCE
    between two shapes computed this way is accurate, so we use it to compute how
    VAA changes from current to optimized and apply that delta to the measured VAA.
    """
    vx0, vy0, vz0, ax, ay, az = derive_accelerations_from_shape(
        pfx_x_in, pfx_z_in, release_speed_mph,
        release_x, release_y, release_z,
        vx0_actual, vy0_actual, vz0_actual, ay_actual
    )
    return vaa_from_trajectory(vy0, vz0, ay, az)


def optimized_vaa(current_vaa_measured, current_pfx_x, current_pfx_z,
                  opt_pfx_x, opt_pfx_z, release_speed_mph,
                  release_x, release_y, release_z,
                  vx0_actual=None, vy0_actual=None, vz0_actual=None, ay_actual=None):
    """
    Compute the VAA of an optimized shape using the delta method:
      optimized_VAA = measured_current_VAA + (vaa_shape(opt) - vaa_shape(current))

    The shape-derivation bias cancels in the delta, so the result stays on the
    measured-VAA scale (what the stuff model trained on). Returns None if either
    shape VAA cannot be computed or no measured current VAA is available.
    """
    if current_vaa_measured is None:
        # No measured baseline — fall back to raw shape-derived VAA for optimized
        return vaa_from_shape(opt_pfx_x, opt_pfx_z, release_speed_mph,
                              release_x, release_y, release_z,
                              vx0_actual, vy0_actual, vz0_actual, ay_actual)
    v_cur = vaa_from_shape(current_pfx_x, current_pfx_z, release_speed_mph,
                           release_x, release_y, release_z,
                           vx0_actual, vy0_actual, vz0_actual, ay_actual)
    v_opt = vaa_from_shape(opt_pfx_x, opt_pfx_z, release_speed_mph,
                           release_x, release_y, release_z,
                           vx0_actual, vy0_actual, vz0_actual, ay_actual)
    if v_cur is None or v_opt is None:
        return current_vaa_measured
    return float(current_vaa_measured + (v_opt - v_cur))


# ── Vectorized VAA (batch) ─────────────────────────────────────────────────
def vaa_from_shape_batch(pfx_z_arr, release_speed_mph,
                         release_y, release_z,
                         vy0_actual=None, vz0_actual=None, ay_actual=None,
                         y0=50.0, y_plate=1.417):
    """
    Vectorized version of vaa_from_shape over an array of candidate pfx_z
    values. VAA does not depend on horizontal break, so only pfx_z is needed.
    Mirrors the scalar path exactly: derive az from the shape (Magnus + gravity),
    keep vy0/vz0/ay from the pitcher's actual trajectory when available, then
    apply the plate-crossing kinematics. Returns an array of VAA in degrees,
    or None if the trajectory has no valid plate crossing (matches scalar API).
    """
    pfx_z_arr = np.asarray(pfx_z_arr, dtype=float)
    speed_fps = release_speed_mph * 1.467

    if vy0_actual is not None and vz0_actual is not None:
        vy0, vz0 = vy0_actual, vz0_actual
    else:
        flight_dist = release_y
        vy0 = -speed_fps
        vz0 = -speed_fps * np.sin(np.arctan2(release_z - 2.5, flight_dist))

    avg_vy   = abs(vy0) - 5.0
    t_flight = (50.0 - 1.417) / avg_vy
    az = 2 * (pfx_z_arr / 12.0) / (t_flight ** 2) + GRAVITY

    if ay_actual is not None:
        ay = ay_actual
    else:
        ay = -0.0023 * speed_fps ** 2 / (release_speed_mph / 95)

    # Kinematic time to the plate — depends only on vy0/ay, so it is a scalar
    # shared by every candidate shape (identical to vaa_from_trajectory).
    a = 0.5 * ay
    b = vy0
    c = y0 - y_plate
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    t = t1 if (t1 > 0 and (t1 < t2 or t2 <= 0)) else t2
    if t <= 0:
        return None

    vz_plate = vz0 + az * t
    vy_plate = vy0 + ay * t
    return np.degrees(np.arctan2(vz_plate, abs(vy_plate)))


def optimized_vaa_batch(current_vaa_measured, current_pfx_z, cand_pfx_z_arr,
                        release_speed_mph, release_y, release_z,
                        vy0_actual=None, vz0_actual=None, ay_actual=None):
    """
    Vectorized delta-method VAA for an array of candidate shapes:
      vaa[i] = measured_current_VAA + (vaa_shape(cand[i]) - vaa_shape(current))
    Same semantics as optimized_vaa, evaluated for all candidates at once.
    Returns an array; falls back to the measured value (or raw shape VAA when
    no measurement exists) exactly like the scalar version.
    """
    cand_pfx_z_arr = np.asarray(cand_pfx_z_arr, dtype=float)
    v_cand = vaa_from_shape_batch(cand_pfx_z_arr, release_speed_mph,
                                  release_y, release_z,
                                  vy0_actual, vz0_actual, ay_actual)
    if current_vaa_measured is None:
        return v_cand  # may be None; caller handles fallback
    if v_cand is None:
        return np.full(cand_pfx_z_arr.shape, float(current_vaa_measured))
    v_cur = vaa_from_shape_batch(np.array([current_pfx_z]), release_speed_mph,
                                 release_y, release_z,
                                 vy0_actual, vz0_actual, ay_actual)
    if v_cur is None:
        return np.full(cand_pfx_z_arr.shape, float(current_vaa_measured))
    return current_vaa_measured + (v_cand - v_cur[0])
