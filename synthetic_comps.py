
def find_comps_for_synthetic(syn_profile, mlb_profiles, sensitivity_radii):
    """
    Find mechanical comps for a synthetic (uploaded) pitcher by matching against
    the MLB profiles' mechanics. Computes achievable movement bounds per pitch type.

    Mirrors the notebook's comp-finding logic but operates on loaded profiles
    rather than the raw mech_df.

    Returns a comp_bounds dict {pt: {pfx_x_lo, pfx_x_hi, pfx_z_lo, pfx_z_hi,
                                     comp_pitchers, n_comps, ...}}
    """
    import numpy as np

    DEFAULT_RADII = {
        'release_pos_x': 0.10, 'release_pos_z': 0.15,
        'release_extension': 0.30, 'release_spin_rate': 200,
    }
    MIN_COMP_N = 15

    syn_throws = syn_profile.get('p_throws', 'R')
    comp_bounds = {}

    for pt, info in syn_profile['pitches'].items():
        fixed = info['fixed']
        my_rpx = fixed.get('release_pos_x')
        my_rpz = fixed.get('release_pos_z')
        my_ext = fixed.get('release_extension')
        my_spin = info.get('semi_fixed', {}).get('spin_rate_mean')

        if my_rpx is None or my_rpz is None or my_ext is None:
            continue

        radii = sensitivity_radii.get(pt, DEFAULT_RADII) if sensitivity_radii else DEFAULT_RADII

        def collect_comps(mult):
            comps = []
            rx = radii.get('release_pos_x', DEFAULT_RADII['release_pos_x']) * mult
            rz = radii.get('release_pos_z', DEFAULT_RADII['release_pos_z']) * mult
            re = radii.get('release_extension', DEFAULT_RADII['release_extension']) * mult
            rs = radii.get('release_spin_rate', DEFAULT_RADII['release_spin_rate']) * mult

            for mlb_pid, mlb_p in mlb_profiles.items():
                if mlb_p.get('is_synthetic'):
                    continue
                mlb_pt_info = mlb_p.get('pitches', {}).get(pt)
                if not mlb_pt_info:
                    continue
                mfx = mlb_pt_info.get('fixed', {})
                m_rpx = mfx.get('release_pos_x')
                m_rpz = mfx.get('release_pos_z')
                m_ext = mfx.get('release_extension')
                if None in (m_rpx, m_rpz, m_ext):
                    continue

                mlb_throws = mlb_p.get('p_throws', 'R')
                mirrored = (mlb_throws != syn_throws)
                # Mirror arm slot for opposite handedness
                cmp_rpx = -m_rpx if mirrored else m_rpx

                if abs(cmp_rpx - my_rpx) > rx: continue
                if abs(m_rpz - my_rpz) > rz: continue
                if abs(m_ext - my_ext) > re: continue
                if my_spin is not None:
                    m_spin = mlb_pt_info.get('semi_fixed', {}).get('spin_rate_mean')
                    if m_spin is not None and abs(m_spin - my_spin) > rs:
                        continue

                # Movement (mirror horizontal for opposite hand)
                m_opt = mlb_pt_info.get('optimizable', {})
                m_pfx_x = m_opt.get('pfx_x_mean')
                m_pfx_z = m_opt.get('pfx_z_mean')
                if m_pfx_x is None or m_pfx_z is None:
                    continue
                adj_pfx_x = -m_pfx_x if mirrored else m_pfx_x

                comps.append({
                    'pitcher_id': int(mlb_pid),
                    'name': mlb_p.get('player_name', str(mlb_pid)),
                    'pfx_x': round(float(adj_pfx_x), 2),
                    'pfx_z': round(float(m_pfx_z), 2),
                    'is_mirrored': mirrored,
                })
            return comps

        comps = collect_comps(1.0)
        widened = False
        if len(comps) < MIN_COMP_N:
            comps = collect_comps(1.5)
            widened = True

        if len(comps) >= 5:
            xs = np.array([c['pfx_x'] for c in comps])
            zs = np.array([c['pfx_z'] for c in comps])
            comp_bounds[pt] = {
                'pfx_x_lo': round(float(np.percentile(xs, 10)), 2),
                'pfx_x_hi': round(float(np.percentile(xs, 90)), 2),
                'pfx_z_lo': round(float(np.percentile(zs, 10)), 2),
                'pfx_z_hi': round(float(np.percentile(zs, 90)), 2),
                'comp_pitchers': comps,
                'n_comps': len(comps),
                'widened': widened,
                'n_mirrored': sum(1 for c in comps if c['is_mirrored']),
            }
        else:
            # Fallback to own shape ±tight range (no comps found)
            px = info['optimizable']['pfx_x_mean']
            pz = info['optimizable']['pfx_z_mean']
            comp_bounds[pt] = {
                'pfx_x_lo': round(px - 3, 2), 'pfx_x_hi': round(px + 3, 2),
                'pfx_z_lo': round(pz - 3, 2), 'pfx_z_hi': round(pz + 3, 2),
                'comp_pitchers': [],
                'n_comps': 0, 'widened': False, 'n_mirrored': 0,
            }

    return comp_bounds


def apply_comp_bounds_to_synthetic(syn_profile, comp_bounds):
    """Fill in the optimizable pfx bounds on the synthetic profile from comp_bounds.

    The comp-derived bounds (10th-90th percentile of mechanically similar MLB
    pitchers) can occasionally land such that the pitcher's OWN current shape sits
    outside them -- e.g. a pitcher whose fastball has more ride than his mechanical
    comps. If we applied those bounds directly, the optimizer would be capped below
    the pitcher's current movement and could only recommend LOSING break, which is
    never sensible: a pitcher can always at least keep his current shape. So we
    expand the bounds as needed to always contain the current shape (plus a small
    margin), guaranteeing the current shape is reachable and the optimizer never
    recommends giving up movement the pitcher already has.
    """
    for pt, info in syn_profile['pitches'].items():
        cb = comp_bounds.get(pt)
        if not cb:
            continue
        opt = info['optimizable']
        cur_x = opt.get('pfx_x_mean')
        cur_z = opt.get('pfx_z_mean')

        x_lo, x_hi = cb['pfx_x_lo'], cb['pfx_x_hi']
        z_lo, z_hi = cb['pfx_z_lo'], cb['pfx_z_hi']

        # Expand bounds to include the current shape (with a small 0.5" margin so
        # the current shape isn't sitting exactly on the boundary).
        if cur_x is not None:
            x_lo = min(x_lo, cur_x - 0.5)
            x_hi = max(x_hi, cur_x + 0.5)
        if cur_z is not None:
            z_lo = min(z_lo, cur_z - 0.5)
            z_hi = max(z_hi, cur_z + 0.5)

        opt['pfx_x_lo'] = round(x_lo, 2)
        opt['pfx_x_hi'] = round(x_hi, 2)
        opt['pfx_z_lo'] = round(z_lo, 2)
        opt['pfx_z_hi'] = round(z_hi, 2)
    return syn_profile


def finalize_synthetic_profile(syn_profile, norm_tables, stuff_models,
                                predict_grades_fn, predict_weighted_fn):
    """
    Compute current grades for each pitch and set the primary fastball reference
    so FB-differential features work. Mutates and returns syn_profile.
    """
    pitches = syn_profile['pitches']

    # Identify primary fastball (most pitches among FF/SI/FC)
    FASTBALL_TYPES = {'FF', 'SI', 'FC'}
    fb_candidates = [(pt, info) for pt, info in pitches.items() if pt in FASTBALL_TYPES]
    if fb_candidates:
        fb_pt, fb_info = max(fb_candidates, key=lambda x: x[1].get('n_pitches', 0))
        syn_profile['primary_fb_velo']  = fb_info['semi_fixed']['velo_mean']
        syn_profile['primary_fb_pfx_x'] = fb_info['optimizable']['pfx_x_mean']
        syn_profile['primary_fb_pfx_z'] = fb_info['optimizable']['pfx_z_mean']

    # Compute grades for each pitch at its current shape
    for pt, info in pitches.items():
        opt = info['optimizable']
        semi = info['semi_fixed']
        fixed = info['fixed']
        px = opt['pfx_x_mean']
        pz = opt['pfx_z_mean']
        velo = semi['velo_mean']
        spin_eff = fixed.get('spin_efficiency') or 0.0

        # Inject pitcher-level fields onto the pitch-level info dict so
        # build_features (which reads profile_info['fixed'], primary_fb_*, etc.)
        # works correctly. build_features expects a PITCH-level dict.
        info['p_throws'] = syn_profile.get('p_throws', 'R')
        info['primary_fb_velo']  = syn_profile.get('primary_fb_velo')
        info['primary_fb_pfx_x'] = syn_profile.get('primary_fb_pfx_x')
        info['primary_fb_pfx_z'] = syn_profile.get('primary_fb_pfx_z')

        try:
            gr = predict_grades_fn(pt, px, pz, spin_eff, velo, info,
                                   norm_tables, stuff_models, stand='R')
            gl = predict_grades_fn(pt, px, pz, spin_eff, velo, info,
                                   norm_tables, stuff_models, stand='L')
            gc, _, _ = predict_weighted_fn(pt, px, pz, spin_eff, velo, info,
                                           norm_tables, stuff_models)
            info['grades_rhh'] = gr
            info['grades_lhh'] = gl
            info['grades'] = gc
        except Exception as e:
            info['grades'] = {}
            info['grades_rhh'] = {}
            info['grades_lhh'] = {}
            info['_grade_error'] = str(e)

    return syn_profile


def recompute_synthetic_profile(syn_profile, existing_profiles, sensitivity_radii,
                                norm_tables, stuff_models,
                                predict_grades_fn, predict_weighted_fn):
    """
    Run the full synthetic pipeline on a profile whose fixed/semi_fixed/optimizable
    values may have been edited: find comps -> apply bounds (clamped to current
    shape) -> recompute primary-FB fields -> recompute grades.

    This is the single entry point used by BOTH the initial upload and the
    "Edit pitcher" override flow, so edited profiles go through the identical
    computation as freshly-parsed ones. Returns (profile, comp_bounds).
    """
    # Recompute the primary fastball reference from the (possibly edited) pitches,
    # since velo_diff_fb / pfx_diff_fb features depend on it.
    _recompute_primary_fb(syn_profile)

    comp_bounds = find_comps_for_synthetic(syn_profile, existing_profiles, sensitivity_radii)
    syn_profile = apply_comp_bounds_to_synthetic(syn_profile, comp_bounds)
    syn_profile = finalize_synthetic_profile(
        syn_profile, norm_tables, stuff_models, predict_grades_fn, predict_weighted_fn
    )
    syn_profile['_comp_bounds'] = comp_bounds
    return syn_profile, comp_bounds


def _recompute_primary_fb(syn_profile):
    """
    Recompute the pitcher-level primary-fastball reference (velo, pfx_x, pfx_z)
    from the current pitch values. Priority: FF, then SI, then FC, else the
    hardest pitch by velocity. Mirrors how the notebook picks the primary FB.
    """
    pitches = syn_profile.get('pitches', {})
    primary = None
    for pref in ('FF', 'SI', 'FC'):
        if pref in pitches:
            primary = pref
            break
    if primary is None and pitches:
        # Fall back to hardest pitch
        primary = max(
            pitches.keys(),
            key=lambda pt: pitches[pt].get('semi_fixed', {}).get('velo_mean', 0) or 0
        )
    if primary is not None:
        info = pitches[primary]
        syn_profile['primary_fb_velo']  = info.get('semi_fixed', {}).get('velo_mean')
        syn_profile['primary_fb_pfx_x'] = info.get('optimizable', {}).get('pfx_x_mean')
        syn_profile['primary_fb_pfx_z'] = info.get('optimizable', {}).get('pfx_z_mean')
